import importlib.util
import os
import shutil
from pathlib import Path

import pytest
import torch
from fixture_contract import ACTION_DIM, MAIN_KEY, WRIST_KEY

from lerobot.datasets import LeRobotDataset
from lerobot.datasets.video_utils import FrameTimestampError, decode_video_frames

REPO_ID = "vlaforge/data-only-v30-fixture"
DELTA_TIMESTAMPS = {
    "observation.state": [-0.1, 0.0, 0.1],
    "action": [-0.1, 0.0, 0.1],
    MAIN_KEY: [-0.1, 0.0, 0.1],
    WRIST_KEY: [-0.1, 0.0, 0.1],
}


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


@pytest.fixture(scope="module")
def windowed_dataset(fixture_root: Path) -> LeRobotDataset:
    return LeRobotDataset(
        REPO_ID,
        root=fixture_root,
        delta_timestamps=DELTA_TIMESTAMPS,
        video_backend="pyav",
        return_uint8=True,
    )


def test_pyav_returns_both_uint8_camera_windows(windowed_dataset: LeRobotDataset) -> None:
    item = windowed_dataset[1]
    assert item[MAIN_KEY].shape == (3, 3, 16, 24)
    assert item[WRIST_KEY].shape == (3, 3, 16, 24)
    assert item[MAIN_KEY].dtype == torch.uint8
    assert item[WRIST_KEY].dtype == torch.uint8


def test_repeated_random_reads_are_stable(windowed_dataset: LeRobotDataset) -> None:
    order = [4, 0, 6, 1, 11, 5, 4, 0]
    first = [windowed_dataset[index]["action"].clone() for index in order]
    second = [windowed_dataset[index]["action"].clone() for index in order]
    assert all(torch.equal(left, right) for left, right in zip(first, second, strict=True))


def test_two_worker_dataloader_covers_requested_samples(dataset: LeRobotDataset) -> None:
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, num_workers=2, shuffle=False)
    assert sum(batch["index"].numel() for batch in loader) == len(dataset)


def test_explicit_unavailable_video_backend_raises_import_error(fixture_root: Path) -> None:
    if importlib.util.find_spec("torchcodec") is not None:
        pytest.skip("torchcodec is installed in this environment")
    torchcodec_dataset = LeRobotDataset(
        REPO_ID,
        root=fixture_root,
        video_backend="torchcodec",
        return_uint8=True,
    )
    with pytest.raises(ImportError):
        torchcodec_dataset[0]


def test_missing_video_raises_file_not_found(fixture_root: Path, tmp_path: Path) -> None:
    local_root = tmp_path / "fixture"
    shutil.copytree(fixture_root, local_root)
    local_dataset = LeRobotDataset(
        REPO_ID,
        root=local_root,
        video_backend="pyav",
        return_uint8=True,
    )
    video_path = local_root / local_dataset.meta.get_video_file_path(0, MAIN_KEY)
    video_path.unlink()

    with pytest.raises(FileNotFoundError):
        local_dataset[0]


def test_timestamp_outside_tolerance_raises_frame_timestamp_error(
    dataset: LeRobotDataset,
) -> None:
    video_path = dataset.root / dataset.meta.get_video_file_path(0, MAIN_KEY)
    with pytest.raises(FrameTimestampError):
        decode_video_frames(
            video_path,
            timestamps=[99.0],
            tolerance_s=1e-4,
            backend="pyav",
            return_uint8=True,
        )


def test_torchcodec_matches_pyav_when_available(fixture_root: Path) -> None:
    try:
        pytest.importorskip("torchcodec")
    except (OSError, RuntimeError) as error:
        pytest.skip(f"torchcodec runtime is unavailable: {error}")
    pyav_dataset = LeRobotDataset(
        REPO_ID,
        root=fixture_root,
        delta_timestamps=DELTA_TIMESTAMPS,
        video_backend="pyav",
        return_uint8=True,
    )
    torchcodec_dataset = LeRobotDataset(
        REPO_ID,
        root=fixture_root,
        delta_timestamps=DELTA_TIMESTAMPS,
        video_backend="torchcodec",
        return_uint8=True,
    )

    pyav_item = pyav_dataset[1]
    torchcodec_item = torchcodec_dataset[1]
    for key in (MAIN_KEY, WRIST_KEY):
        assert torchcodec_item[key].shape == pyav_item[key].shape
        assert torchcodec_item[key].dtype == pyav_item[key].dtype
        torch.testing.assert_close(torchcodec_item[key], pyav_item[key], rtol=0, atol=2)
    assert torchcodec_item["action"].shape == (3, ACTION_DIM)
