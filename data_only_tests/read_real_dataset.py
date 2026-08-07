import argparse
import json
import time
from pathlib import Path

import torch

from lerobot.datasets import LeRobotDataset


def parse_delta_frames(value: str) -> tuple[int, ...]:
    return tuple(int(frame) for frame in value.split(","))


def sample_indices(dataset_length: int, sample_count: int) -> list[int]:
    if sample_count == 1:
        return [0]
    return [index * (dataset_length - 1) // (sample_count - 1) for index in range(sample_count)]


def run_check(args: argparse.Namespace) -> dict:
    metadata_dataset = LeRobotDataset(
        args.repo_id,
        root=args.root,
        video_backend=args.video_backend,
        return_uint8=True,
    )
    delta_keys = [
        key
        for key, feature in metadata_dataset.features.items()
        if (key == "action" or key.startswith("observation."))
        and feature["dtype"] in {"float32", "float64", "image", "video"}
    ]
    delta_timestamps = {
        key: [frame / metadata_dataset.fps for frame in args.delta_frames] for key in delta_keys
    }
    dataset = LeRobotDataset(
        args.repo_id,
        root=args.root,
        delta_timestamps=delta_timestamps,
        video_backend=args.video_backend,
        return_uint8=True,
    )

    requested_count = min(args.samples, len(dataset))
    if requested_count < 1:
        raise ValueError("samples must be positive and the dataset must not be empty")
    indices = sample_indices(len(dataset), requested_count)
    subset = torch.utils.data.Subset(dataset, indices)
    loader = torch.utils.data.DataLoader(
        subset,
        batch_size=min(16, requested_count),
        num_workers=args.workers,
        shuffle=False,
    )

    camera_shapes = None
    state_shape = None
    action_shape = None
    samples_read = 0
    started = time.perf_counter()
    for batch in loader:
        if camera_shapes is None:
            camera_shapes = [list(batch[key].shape[1:]) for key in sorted(dataset.meta.camera_keys)]
            if "observation.state" in batch:
                state_shape = list(batch["observation.state"].shape[1:])
            if "action" in batch:
                action_shape = list(batch["action"].shape[1:])
        samples_read += batch["index"].numel()
    elapsed_seconds = time.perf_counter() - started

    return {
        "metadata_total_episodes": dataset.meta.total_episodes,
        "metadata_total_frames": dataset.meta.total_frames,
        "samples_read": samples_read,
        "workers": args.workers,
        "camera_shapes": camera_shapes,
        "state_shape": state_shape,
        "action_shape": action_shape,
        "elapsed_seconds": elapsed_seconds,
        "samples_per_second": samples_read / elapsed_seconds,
        "errors": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--video-backend", choices=("pyav", "torchcodec"), default="pyav")
    parser.add_argument("--delta-frames", type=parse_delta_frames, default=(-1, 0, 1))
    args = parser.parse_args()
    print(json.dumps(run_check(args), sort_keys=True))


if __name__ == "__main__":
    main()
