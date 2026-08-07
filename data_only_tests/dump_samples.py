import argparse
import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fixture_contract import MAIN_KEY, SAMPLE_INDICES, WRIST_KEY

from lerobot.datasets import LeRobotDataset

REPO_ID = "vlaforge/data-only-v30-fixture"
DELTA_TIMESTAMPS = {
    "observation.state": [-0.1, 0.0, 0.1],
    "action": [-0.1, 0.0, 0.1],
    MAIN_KEY: [-0.1, 0.0, 0.1],
    WRIST_KEY: [-0.1, 0.0, 0.1],
}


def as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def dump_samples(root: Path, output: Path) -> None:
    dataset = LeRobotDataset(
        REPO_ID,
        root=root,
        delta_timestamps=DELTA_TIMESTAMPS,
        video_backend="pyav",
        return_uint8=True,
    )
    arrays: dict[str, np.ndarray] = {}
    samples = []
    for sample_number, index in enumerate(SAMPLE_INDICES):
        item = dataset[index]
        fields = {}
        for field_number, key in enumerate(sorted(item.keys() - {"task"})):
            archive_key = f"sample_{sample_number:03d}_field_{field_number:03d}"
            value = as_numpy(item[key])
            arrays[archive_key] = value
            fields[key] = {
                "archive_key": archive_key,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
        samples.append(
            {
                "requested_index": index,
                "task": item["task"],
                "fields": fields,
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output.with_suffix(".npz"), **arrays)
    manifest = {
        "package_version": importlib.metadata.version("lerobot"),
        "python_version": platform.python_version(),
        "dataset_length": len(dataset),
        "samples": samples,
    }
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dump_samples(args.root, args.output)


if __name__ == "__main__":
    main()
