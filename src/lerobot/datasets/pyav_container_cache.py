#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Process-local cache of open PyAV containers.

A v3.0 dataset concatenates every episode of a chunk into a single MP4, so
random-access training reads a handful of very long files over and over.
``av.open`` has to parse the whole MP4 sample table before the first frame can
be demuxed, which on a 53k-frame AV1 file costs about 19 ms of CPU and a burst
of small reads, while the seek-and-decode that follows costs about 4 ms. Opening
per ``__getitem__`` therefore spends most of the decode budget re-reading
container metadata, and on a shared filesystem the read burst turns into network
round-trips that scale with the number of dataloader workers.

Keeping the containers open removes both costs. The cache is per process and
per thread-safe entry: a forked dataloader worker never reuses a container that
its parent opened, and two camera threads decoding concurrently take separate
entries (one per file) or serialise on the same one.

Set ``LEROBOT_PYAV_CONTAINER_CACHE_SIZE`` to bound the number of retained
containers, or to ``0`` / ``none`` to restore the open-per-call behaviour.
"""

from __future__ import annotations

import contextlib
import os
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any

import av

DEFAULT_CONTAINER_CACHE_SIZE = 16
"""Default number of open containers retained per process.

A dataset group holds one file per camera per chunk, so a few dozen entries
cover every file a worker touches. Each entry costs one file descriptor plus the
decoder state of a single stream (a few MB), which is why the cap is small.
"""


DEFAULT_DECODE_THREADS = 1
"""Decoder threads per cached stream.

FFmpeg's automatic thread count is the machine's core count, so a dataloader
that opens one decoder per camera on a 96-core node asks libdav1d for hundreds
of threads per worker. Spawning and tearing those down dominates the system CPU
time of a training run, and with enough workers it exhausts the process's thread
limit outright. One thread per decoder is the right shape here: parallelism
comes from the dataloader workers, and a training window is a handful of small
frames. Override with ``LEROBOT_PYAV_DECODE_THREADS``; ``0`` restores FFmpeg's
automatic choice.
"""


def _default_decode_threads() -> int:
    raw = os.environ.get("LEROBOT_PYAV_DECODE_THREADS")
    if raw is None:
        return DEFAULT_DECODE_THREADS
    value = int(raw)
    if value < 0:
        raise ValueError(f"LEROBOT_PYAV_DECODE_THREADS must be non-negative; got {value}")
    return value


def configure_decoder_threads(stream: Any, thread_count: int | None = None) -> Any:
    """Pin the decoder thread count of ``stream`` before it decodes anything."""
    count = _default_decode_threads() if thread_count is None else thread_count
    if count:
        stream.thread_type = "NONE"
        stream.codec_context.thread_count = count
    return stream


def _default_max_cache_size() -> int | None:
    raw = os.environ.get("LEROBOT_PYAV_CONTAINER_CACHE_SIZE")
    if raw is None:
        return DEFAULT_CONTAINER_CACHE_SIZE
    raw = raw.strip().lower()
    if raw in ("", "none", "off", "0"):
        return None
    try:
        value = int(raw)
    except ValueError as e:
        raise ValueError(
            f"LEROBOT_PYAV_CONTAINER_CACHE_SIZE must be an integer, 'none', or '0'; got {raw!r}"
        ) from e
    if value < 0:
        raise ValueError(f"LEROBOT_PYAV_CONTAINER_CACHE_SIZE must be non-negative; got {value}")
    return value or None


@dataclass
class _Entry:
    """One cached container together with the stream a decoder reads from."""

    container: Any
    stream: Any
    lock: Lock = field(default_factory=Lock)


class PyAVContainerCache:
    """LRU of open :class:`av.container.InputContainer` objects, keyed by path.

    Args:
        max_size: Number of containers to retain. ``None`` disables caching
            entirely and every acquisition opens and closes its own container.
            Defaults to ``LEROBOT_PYAV_CONTAINER_CACHE_SIZE`` when set, otherwise
            :data:`DEFAULT_CONTAINER_CACHE_SIZE`.
    """

    _SENTINEL: Any = object()

    def __init__(self, max_size: int | None | Any = _SENTINEL):
        if max_size is PyAVContainerCache._SENTINEL:
            max_size = _default_max_cache_size()
        if max_size is not None and max_size <= 0:
            raise ValueError(f"max_size must be positive or None; got {max_size}")
        self.max_size: int | None = max_size
        self._cache: OrderedDict[str, _Entry] = OrderedDict()
        self._lock = Lock()
        self._pid = os.getpid()

    def __contains__(self, video_path: object) -> bool:
        with self._lock:
            return str(video_path) in self._cache

    def size(self) -> int:
        with self._lock:
            return len(self._cache)

    def clear(self) -> None:
        """Drop every cached container.

        Entries are released rather than closed: a container another thread is
        still decoding from stays alive until that thread lets go of it, and its
        file descriptor is closed when the last reference disappears.
        """
        with self._lock:
            self._cache.clear()

    def _entry(self, video_path: str) -> _Entry:
        with self._lock:
            # A forked dataloader worker inherits its parent's descriptors, and
            # a shared descriptor means a shared file offset: two processes
            # seeking the same container would corrupt each other's reads. Start
            # the child with an empty cache instead.
            pid = os.getpid()
            if pid != self._pid:
                self._cache.clear()
                self._pid = pid

            entry = self._cache.get(video_path)
            if entry is not None:
                self._cache.move_to_end(video_path)
                return entry

            container = av.open(video_path)
            try:
                stream = configure_decoder_threads(container.streams.video[0])
            except Exception:
                container.close()
                raise
            entry = _Entry(container=container, stream=stream)
            self._cache[video_path] = entry

            # Evicted entries are dropped, not closed: whoever is decoding from
            # one keeps it alive, and the descriptor closes with the last
            # reference.
            if self.max_size is not None:
                while len(self._cache) > self.max_size:
                    self._cache.popitem(last=False)
            return entry

    @contextlib.contextmanager
    def acquire(self, video_path: Path | str) -> Iterator[tuple[Any, Any]]:
        """Yield ``(container, stream)`` for ``video_path`` with exclusive access.

        The entry lock is held for the duration of the block because a container
        carries one demuxer position: two concurrent seeks on the same file would
        interleave.
        """
        video_path = str(video_path)
        if self.max_size is None:
            with av.open(video_path) as container:
                yield container, configure_decoder_threads(container.streams.video[0])
            return

        entry = self._entry(video_path)
        with entry.lock:
            yield entry.container, entry.stream


_default_container_cache = PyAVContainerCache()


def acquire_video_stream(
    video_path: Path | str,
    cache: PyAVContainerCache | None = None,
) -> contextlib.AbstractContextManager[tuple[Any, Any]]:
    """Acquire ``(container, stream)`` for ``video_path`` from the process cache."""
    return (cache or _default_container_cache).acquire(video_path)


def clear_container_cache() -> None:
    """Release every container held by the process-wide cache."""
    _default_container_cache.clear()
