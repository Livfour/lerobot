import os
from pathlib import Path

import av
import pytest
import torch
from fixture_contract import MAIN_KEY, WRIST_KEY

from lerobot.datasets import LeRobotDataset, pyav_container_cache as cache_module
from lerobot.datasets.pyav_container_cache import PyAVContainerCache, clear_container_cache
from lerobot.datasets.video_utils import decode_video_frames

REPO_ID = "vlaforge/data-only-v30-fixture"
TIMESTAMPS = [0.0, 0.1, 0.2]
TOLERANCE_S = 1e-4


@pytest.fixture(scope="module")
def fixture_root() -> Path:
    value = os.environ.get("LEROBOT_V30_FIXTURE")
    if value is None:
        pytest.fail("LEROBOT_V30_FIXTURE must point to the official v3.0 fixture")
    return Path(value)


@pytest.fixture(scope="module")
def video_path(fixture_root: Path) -> Path:
    dataset = LeRobotDataset(REPO_ID, root=fixture_root, video_backend="pyav", return_uint8=True)
    return fixture_root / dataset.meta.get_video_file_path(0, MAIN_KEY)


@pytest.fixture(autouse=True)
def clear_cache():
    clear_container_cache()
    yield
    clear_container_cache()


def _decode(path: Path, timestamps: list[float]) -> torch.Tensor:
    return decode_video_frames(path, timestamps, TOLERANCE_S, backend="pyav", return_uint8=True)


def test_cached_reads_match_uncached_reads(video_path: Path, monkeypatch) -> None:
    cached = _decode(video_path, TIMESTAMPS)
    monkeypatch.setenv("LEROBOT_PYAV_CONTAINER_CACHE_SIZE", "0")
    clear_container_cache()
    monkeypatch.setattr(cache_module, "_default_container_cache", PyAVContainerCache())
    uncached = _decode(video_path, TIMESTAMPS)
    assert torch.equal(cached, uncached)


def test_repeated_reads_open_the_file_once(video_path: Path, monkeypatch) -> None:
    opens = 0
    real_open = av.open

    def counting_open(*args, **kwargs):
        nonlocal opens
        opens += 1
        return real_open(*args, **kwargs)

    monkeypatch.setattr("lerobot.datasets.pyav_container_cache.av.open", counting_open)
    for _ in range(5):
        _decode(video_path, TIMESTAMPS)
    assert opens == 1
    assert video_path in cache_module._default_container_cache


def test_disabled_cache_opens_every_read(video_path: Path, monkeypatch) -> None:
    opens = 0
    real_open = av.open

    def counting_open(*args, **kwargs):
        nonlocal opens
        opens += 1
        return real_open(*args, **kwargs)

    monkeypatch.setattr("lerobot.datasets.pyav_container_cache.av.open", counting_open)
    cache = PyAVContainerCache(max_size=None)
    for _ in range(3):
        with cache.acquire(video_path) as (_container, _stream):
            pass
    assert opens == 3
    assert cache.size() == 0


def test_cache_evicts_beyond_max_size(video_path: Path, fixture_root: Path) -> None:
    dataset = LeRobotDataset(REPO_ID, root=fixture_root, video_backend="pyav", return_uint8=True)
    wrist_path = fixture_root / dataset.meta.get_video_file_path(0, WRIST_KEY)
    cache = PyAVContainerCache(max_size=1)
    with cache.acquire(video_path) as (_c, _s):
        pass
    with cache.acquire(wrist_path) as (_c, _s):
        pass
    assert cache.size() == 1
    assert wrist_path in cache
    assert video_path not in cache


def test_cache_is_dropped_after_fork(video_path: Path) -> None:
    cache = PyAVContainerCache()
    with cache.acquire(video_path) as (_c, _s):
        pass
    assert cache.size() == 1

    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:  # child
        os.close(read_fd)
        try:
            with cache.acquire(video_path) as (_c, _s):
                pass
            os.write(write_fd, str(cache.size()).encode())
        finally:
            os._exit(0)
    os.close(write_fd)
    child_size = int(os.read(read_fd, 8))
    os.close(read_fd)
    os.waitpid(pid, 0)

    # The child re-opened the file for itself rather than reusing the parent's
    # descriptor, so it holds exactly one entry of its own.
    assert child_size == 1


def test_decoder_threads_are_pinned(video_path: Path) -> None:
    cache = PyAVContainerCache()
    with cache.acquire(video_path) as (_container, stream):
        assert stream.codec_context.thread_count == 1
