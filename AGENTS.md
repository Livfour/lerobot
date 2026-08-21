# AGENTS.md

Guidance for AI agents working in this repository.

## What this repo is

A data-only fork of [huggingface/lerobot](https://github.com/huggingface/lerobot) `v0.6.1`,
trimmed to the LeRobotDataset read/write path and backported to Python 3.11. It is consumed
by [VLAForge](https://github.com/Livfour/VLAForge) as `submodules/lerobot`, pinned by gitlink
to one tested commit. Read [`README.md`](./README.md) first — it states the layout and the
upstream relationship.

## The one rule

**This fork carries a compatibility patch, not a reimplementation.** Do not add a
replacement metadata parser, Parquet reader, video decoder, window implementation, or any
VLAForge-specific sample adapter here — that code belongs in VLAForge. Allowed changes are
listed under "Relationship to upstream" in the README; anything else makes the next upstream
rebase unreviewable.

Corollary: do not re-add policies, robots, motors, cameras, teleoperators, envs, processor,
RL, or training code. They were deleted deliberately.

## Layout

- `src/lerobot/datasets/` — the read/write path. `lerobot_dataset.py` (`LeRobotDataset`),
  `dataset_metadata.py` (`LeRobotDatasetMetadata`), `dataset_reader.py` (delta-timestamp
  windows, padding masks), `dataset_writer.py`, `compute_stats.py`, `video_utils.py` +
  `pyav_utils.py` + `pyav_container_cache.py` (decoding).
- `src/lerobot/configs/` — feature types and video-encoder configs the dataset path needs.
- `src/lerobot/utils/` — constants, import guards, IO helpers.
- `src/lerobot/scripts/convert_dataset_v21_to_v30.py` — upstream's converter, kept because
  VLAForge's export flow ends with it.
- `tests/` — the fork's own suite. Flat, no `__init__.py`; helpers import as
  `from fixture_contract import ...`, which works because pytest puts the test dir on
  `sys.path`.
- `pyproject.toml` — single source of truth for deps, packaging, and ruff config.

## Commands

```bash
uv sync --locked --extra test                              # set up
uv run python tests/create_v30_fixture.py --root /tmp/fix  # build the test fixture
LEROBOT_V30_FIXTURE=/tmp/fix uv run pytest -q              # test
pre-commit run --all-files                                 # lint + format (ruff)
```

Every test but `test_import_contract.py` needs `LEROBOT_V30_FIXTURE`; without it they fail
rather than skip, so an "everything is broken" run usually just means you forgot the fixture.
Prefer `uv run` over bare `python` / `pip`.

## Conventions

- Python 3.11 floor. No 3.12-only syntax (`type X = ...`, PEP 695 generics).
- Imports: relative (`from .sibling import X`) within a module, absolute
  (`from lerobot.module import X`) across modules.
- Guard optional dependencies with the `_foo_available` flags in `utils/import_utils.py`
  plus a `require_package(...)` at use time; don't call `is_package_available` directly.
- Ruff, line length 110, double quotes. Run it before committing.

## Before you commit

Run the suite as shown above. `tests/test_import_contract.py` is the load-bearing one: it
asserts that importing `lerobot.datasets` pulls in no policy/robot/env module, that the
installed distribution declares no non-data dependency, and that the v2.1 → v3.0 converter
stays importable. If a change makes it fail, the change is out of charter.

Any commit here is a submodule bump in VLAForge. Say what moved and why in the message; the
VLAForge side has to justify the gitlink change.
