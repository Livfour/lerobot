import importlib.metadata
import json
import subprocess
import sys

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
    _probe("import transformers; from lerobot.datasets import LeRobotDataset; print('{}')")
    _probe("from lerobot.datasets import LeRobotDataset; import transformers; print('{}')")


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
