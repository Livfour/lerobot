import argparse
import json
from pathlib import Path

import numpy as np


def load_dump(path: Path) -> tuple[dict, np.lib.npyio.NpzFile]:
    manifest = json.loads(path.with_suffix(".json").read_text())
    arrays = np.load(path.with_suffix(".npz"), allow_pickle=False)
    return manifest, arrays


def compare_dumps(reference_path: Path, candidate_path: Path) -> int:
    reference_manifest, reference_arrays = load_dump(reference_path)
    candidate_manifest, candidate_arrays = load_dump(candidate_path)

    assert reference_manifest["package_version"] == candidate_manifest["package_version"] == "0.6.1"
    assert reference_manifest["dataset_length"] == candidate_manifest["dataset_length"]
    assert len(reference_manifest["samples"]) == len(candidate_manifest["samples"])

    compared_fields = 0
    for reference_sample, candidate_sample in zip(
        reference_manifest["samples"], candidate_manifest["samples"], strict=True
    ):
        assert reference_sample["requested_index"] == candidate_sample["requested_index"]
        assert reference_sample["task"] == candidate_sample["task"]
        assert reference_sample["fields"].keys() == candidate_sample["fields"].keys()

        for key in reference_sample["fields"]:
            reference_field = reference_sample["fields"][key]
            candidate_field = candidate_sample["fields"][key]
            assert reference_field["shape"] == candidate_field["shape"]
            assert reference_field["dtype"] == candidate_field["dtype"]

            reference_value = reference_arrays[reference_field["archive_key"]]
            candidate_value = candidate_arrays[candidate_field["archive_key"]]
            if reference_value.dtype.kind in "biu":
                np.testing.assert_array_equal(reference_value, candidate_value)
            else:
                np.testing.assert_allclose(reference_value, candidate_value, rtol=1e-6, atol=1e-7)
            compared_fields += 1

    return compared_fields


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    compared_fields = compare_dumps(args.reference, args.candidate)
    print(
        json.dumps(
            {
                "samples": len(json.loads(args.reference.with_suffix(".json").read_text())["samples"]),
                "fields": compared_fields,
                "parity": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
