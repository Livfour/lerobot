import importlib.metadata
import importlib.util
import json
import subprocess
import sys

import pytest

EXCLUDED_PREFIXES = (
    "lerobot.envs",
    "lerobot.motors",
    "lerobot.policies",
    "lerobot.processor",
    "lerobot.robots",
    "lerobot.datasets.streaming_dataset",
)


def _probe(source: str):
    proc = subprocess.run(
        [sys.executable, "-c", source],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(proc.stdout)


def test_native_reader_import_is_isolated():
    loaded = _probe(
        "import json,sys; from lerobot.datasets import LeRobotDataset; print(json.dumps(sorted(sys.modules)))"
    )
    assert not any(name.startswith(EXCLUDED_PREFIXES) for name in loaded)


def test_transformers_and_lerobot_import_in_both_orders():
    if importlib.util.find_spec("transformers") is None:
        pytest.skip("transformers is not part of the data-only environment")
    _probe("import transformers; from lerobot.datasets import LeRobotDataset; print('{}')")
    _probe("from lerobot.datasets import LeRobotDataset; import transformers; print('{}')")


def test_rgb_encoder_config_keeps_official_pyav_probe_available():
    from lerobot.configs import RGBEncoderConfig

    config = RGBEncoderConfig(vcodec="h264", crf=18, preset="veryfast")
    assert config.vcodec == "h264"


def test_distribution_has_no_unrelated_runtime_dependencies():
    requirements = importlib.metadata.requires("lerobot") or []
    forbidden = (
        "gymnasium",
        "safetensors",
        "draccus",
        "einops",
        "opencv-python",
        "transformers",
    )
    assert not any(req.lower().startswith(forbidden) for req in requirements)
