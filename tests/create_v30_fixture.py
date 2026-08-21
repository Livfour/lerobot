import argparse
import json
from pathlib import Path

import numpy as np
from fixture_contract import (
    ACTION_DIM,
    EPISODE_LENGTHS,
    FPS,
    MAIN_KEY,
    STATE_DIM,
    WRIST_KEY,
    action_value,
    state_value,
)

from lerobot.configs import RGBEncoderConfig
from lerobot.datasets import LeRobotDataset

REPO_ID = "vlaforge/data-only-v30-fixture"
FEATURES = {
    "observation.state": {
        "dtype": "float32",
        "shape": (STATE_DIM,),
        "names": ["state_0", "state_1", "state_2"],
    },
    "action": {
        "dtype": "float32",
        "shape": (ACTION_DIM,),
        "names": ["action_0", "action_1"],
    },
    MAIN_KEY: {
        "dtype": "video",
        "shape": (16, 24, 3),
        "names": ["height", "width", "channels"],
    },
    WRIST_KEY: {
        "dtype": "video",
        "shape": (16, 24, 3),
        "names": ["height", "width", "channels"],
    },
}


def create_fixture(root: Path) -> None:
    dataset = LeRobotDataset.create(
        repo_id=REPO_ID,
        root=root,
        fps=FPS,
        features=FEATURES,
        use_videos=True,
        rgb_encoder=RGBEncoderConfig(vcodec="h264", crf=18, preset="veryfast"),
    )
    for episode, length in enumerate(EPISODE_LENGTHS):
        for frame in range(length):
            dataset.add_frame(
                {
                    "task": f"fixture task {episode}",
                    "observation.state": np.asarray(state_value(episode, frame), dtype=np.float32),
                    "action": np.asarray(action_value(episode, frame), dtype=np.float32),
                    MAIN_KEY: np.full((16, 24, 3), episode * 40 + frame, dtype=np.uint8),
                    WRIST_KEY: np.full((16, 24, 3), 120 + episode * 40 + frame, dtype=np.uint8),
                }
            )
        dataset.save_episode(parallel_encoding=False)
    dataset.finalize()

    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text())
    assert info["codebase_version"] == "v3.0"
    assert info["total_episodes"] == len(EPISODE_LENGTHS)
    assert info["total_frames"] == sum(EPISODE_LENGTHS)
    assert (root / "meta/tasks.parquet").is_file()
    assert (root / "meta/episodes").is_dir()
    assert (root / "data").is_dir()
    assert (root / "videos" / MAIN_KEY).is_dir()
    assert (root / "videos" / WRIST_KEY).is_dir()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    create_fixture(args.root)


if __name__ == "__main__":
    main()
