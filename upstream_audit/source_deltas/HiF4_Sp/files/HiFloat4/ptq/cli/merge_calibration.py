from __future__ import annotations

import argparse

from ..calibration.artifacts import merge_calibration_artifacts, save_calibration_artifact


def main() -> None:
    parser = argparse.ArgumentParser("Merge HiF4 PTQ calibration shards")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    metadata, modules = merge_calibration_artifacts(args.inputs)
    save_calibration_artifact(args.output, modules, metadata=metadata)
    print(f"MERGE_CALIBRATION_OK modules={len(modules)} output={args.output}")


if __name__ == "__main__":
    main()
