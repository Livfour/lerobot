import os
from pathlib import Path

import pytest
import torch
from fixture_contract import (
    ACTION_DIM,
    EPISODE_LENGTHS,
    MAIN_KEY,
    STATE_DIM,
    WRIST_KEY,
    action_value,
    state_value,
)

from lerobot.datasets import LeRobotDataset
from lerobot.datasets.io_utils import load_info

REPO_ID = "vlaforge/data-only-v30-fixture"


@pytest.fixture(scope="module")
def fixture_root() -> Path:
    value = os.environ.get("LEROBOT_V30_FIXTURE")
    if value is None:
        pytest.fail("LEROBOT_V30_FIXTURE must point to the official v3.0 fixture")
    return Path(value)


@pytest.fixture(scope="module")
def dataset(fixture_root: Path) -> LeRobotDataset:
    return LeRobotDataset(
        REPO_ID,
        root=fixture_root,
        video_backend="pyav",
        return_uint8=True,
    )


def test_metadata_and_integer_slice_indexing(dataset: LeRobotDataset) -> None:
    assert dataset.meta.info.codebase_version == "v3.0"
    assert dataset.meta.total_episodes == 2
    assert len(dataset) == sum(EPISODE_LENGTHS)
    assert dataset[0]["task"] == "fixture task 0"
    assert len(dataset[0:2]) == 2


@pytest.mark.parametrize(
    ("index", "episode", "frame"),
    [(0, 0, 0), (4, 0, 4), (5, 1, 0), (11, 1, 6)],
)
def test_state_action_and_camera_values(
    dataset: LeRobotDataset, index: int, episode: int, frame: int
) -> None:
    item = dataset[index]
    assert torch.equal(item["observation.state"], torch.tensor(state_value(episode, frame)))
    assert torch.equal(item["action"], torch.tensor(action_value(episode, frame)))
    assert item["observation.state"].shape == (STATE_DIM,)
    assert item["action"].shape == (ACTION_DIM,)
    assert item[MAIN_KEY].shape == (3, 16, 24)
    assert item[WRIST_KEY].shape == (3, 16, 24)
    assert item[MAIN_KEY].dtype == torch.uint8
    assert item[WRIST_KEY].dtype == torch.uint8


def test_episode_filtering(fixture_root: Path) -> None:
    selected = LeRobotDataset(
        REPO_ID,
        root=fixture_root,
        episodes=[1],
        video_backend="pyav",
        return_uint8=True,
    )
    assert len(selected) == EPISODE_LENGTHS[1]
    assert selected[0]["episode_index"].item() == 1
    assert selected[0]["index"].item() == EPISODE_LENGTHS[0]


def test_delta_windows_and_episode_edge_padding(fixture_root: Path) -> None:
    delta = {
        "observation.state": [-0.1, 0.0, 0.1],
        "action": [-0.1, 0.0, 0.1],
        MAIN_KEY: [-0.1, 0.0, 0.1],
        WRIST_KEY: [-0.1, 0.0, 0.1],
    }
    windowed = LeRobotDataset(
        REPO_ID,
        root=fixture_root,
        delta_timestamps=delta,
        video_backend="pyav",
        return_uint8=True,
    )

    first = windowed[0]
    assert first["action"].shape == (3, ACTION_DIM)
    assert first["observation.state"].shape == (3, STATE_DIM)
    assert first[MAIN_KEY].shape == (3, 3, 16, 24)
    assert first["action_is_pad"].tolist() == [True, False, False]
    assert first[f"{MAIN_KEY}_is_pad"].tolist() == [True, False, False]
    assert torch.equal(first["action"][0], first["action"][1])

    last = windowed[EPISODE_LENGTHS[0] - 1]
    assert last["action_is_pad"].tolist() == [False, False, True]
    assert last[f"{WRIST_KEY}_is_pad"].tolist() == [False, False, True]
    assert torch.equal(last["action"][1], last["action"][2])


def test_invalid_delta_timestamp_is_rejected(fixture_root: Path) -> None:
    with pytest.raises(ValueError, match="outside of tolerance"):
        LeRobotDataset(
            REPO_ID,
            root=fixture_root,
            delta_timestamps={"action": [0.15]},
            video_backend="pyav",
        )


def test_missing_local_metadata_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_info(tmp_path)
