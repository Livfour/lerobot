# lerobot — data-only fork for Python 3.11

A trimmed fork of [huggingface/lerobot](https://github.com/huggingface/lerobot) `v0.6.1`
([`7e241bd6`](https://github.com/huggingface/lerobot/tree/v0.6.1)) that keeps **only the
LeRobotDataset read/write path** and installs on Python 3.11.

It exists so [VLAForge](https://github.com/Livfour/VLAForge) can read and write LeRobot
v2.1 / v3.0 datasets using upstream's own metadata parser, Parquet reader, and video
decoder — inside an environment pinned to Python 3.11 and PyArrow 18, and without pulling
policies, robots, motors, envs, or training into the import graph.

Everything outside the data path is **deleted, not merely unused**: policies, robots,
cameras, motors, teleoperators, envs, processor, RL, training and eval CLIs, the HF docs
site, examples, Docker images, and upstream CI.

## Layout

```
src/lerobot/
  datasets/   LeRobotDataset, LeRobotDatasetMetadata, DatasetReader, DatasetWriter,
              statistics, and video encode/decode (PyAV; TorchCodec optional)
  configs/    the feature types and video-encoder configs the dataset path needs
  utils/      constants, import guards, small IO helpers
  scripts/    convert_dataset_v21_to_v30.py — upstream's v2.1 → v3.0 converter
tests/        the fork's own suite: import isolation, v3.0 reads, video/workers,
              container cache, and upstream-parity tooling
```

## Install

```bash
uv sync --locked                      # runtime
uv sync --locked --extra test         # + pytest
uv sync --locked --extra torchcodec   # + the optional TorchCodec video backend
```

VLAForge installs this instead as an editable path dependency; see its `pixi.toml`.

## Use

```python
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata

meta = LeRobotDatasetMetadata(repo_id="any/name", root="/path/to/dataset")
dataset = LeRobotDataset(repo_id="any/name", root="/path/to/dataset")
sample = dataset[0]
```

Convert a local v2.1 tree to v3.0 in place (keeps a `*_old` copy of the source):

```bash
uv run python -m lerobot.scripts.convert_dataset_v21_to_v30 \
  --repo-id=any/name --root=/path/to/dataset --push-to-hub=false
```

## Tests

The suite asserts against a generated LeRobot v3.0 fixture, so build one first:

```bash
uv run python tests/create_v30_fixture.py --root /tmp/v30-fixture
LEROBOT_V30_FIXTURE=/tmp/v30-fixture uv run pytest -q
```

Without the variable everything except `test_import_contract.py` fails fast rather than
skipping quietly. Two standalone scripts read more than the suite does:

- `read_real_dataset.py` — bounded read + throughput report over any real v3.0 dataset.
- `dump_samples.py` + `compare_parity.py` — serialise samples under two environments and
  diff them, to prove this fork returns what official LeRobot on Python 3.12 returns.

## Relationship to upstream

The branch is 11 commits on top of the `v0.6.1` tag. The compatibility patch may only:

1. lower the Python floor from 3.12 to 3.11 and backport 3.12-only typing syntax;
2. drop eager imports of non-data modules from the dataset import path;
3. remove non-data dependencies from the installed distribution;
4. relax the `datasets` / PyArrow floors enough to coexist with PyArrow 18;
5. optimise decoding without changing what a read returns.

It **must not** introduce a replacement metadata parser, Parquet reader, video decoder,
window implementation, or any VLAForge-specific sample adapter. `tests/test_import_contract.py`
enforces the isolation and dependency half of that; keep it green.

To pull upstream fixes, rebase this branch onto a newer upstream tag and re-run the suite.

## License

Apache 2.0, unchanged from upstream — see [`LICENSE`](./LICENSE). This is a derivative work
of LeRobot by The HuggingFace Inc. team; the source files retain their original copyright
headers.
