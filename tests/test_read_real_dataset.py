import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_bounded_checker_reports_only_schema_and_throughput() -> None:
    fixture_root = os.environ.get("LEROBOT_V30_FIXTURE")
    if fixture_root is None:
        pytest.fail("LEROBOT_V30_FIXTURE must point to the official v3.0 fixture")

    script = Path(__file__).with_name("read_real_dataset.py")
    process = subprocess.run(
        [
            sys.executable,
            str(script),
            "--root",
            fixture_root,
            "--repo-id",
            "local/validation",
            "--samples",
            "12",
            "--workers",
            "2",
            "--video-backend",
            "pyav",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(process.stdout)

    assert summary["samples_read"] == 12
    assert summary["workers"] == 2
    assert summary["errors"] == 0
    assert summary["camera_shapes"] == [[3, 3, 16, 24], [3, 3, 16, 24]]
    assert summary["state_shape"] == [3, 3]
    assert summary["action_shape"] == [3, 2]
    assert summary["samples_per_second"] > 0
    assert set(summary) == {
        "action_shape",
        "camera_shapes",
        "elapsed_seconds",
        "errors",
        "metadata_total_episodes",
        "metadata_total_frames",
        "samples_per_second",
        "samples_read",
        "state_shape",
        "workers",
    }
