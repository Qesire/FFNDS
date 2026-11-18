"""Prepare synchronized playback inputs from individually completed P2 clips."""

import argparse
import json
from pathlib import Path


SOURCES = ("vbi_0392", "vbi_1013", "vbi_0800")
BASE = (("Dense", "dense"), ("Q1r2", "Q1r2"), ("Global-P64", "Global-P64"))
CANDIDATES = (("L0", "l0"), ("L+", "lplus"), ("Q32", "q32"), ("N32", "n32"))


def read(path):
    return json.loads(path.read_bytes())


def videos(root, source, scope, expected_noise):
    identity = scope["source_identity"][source]
    records = [dict(arm=arm, **identity[key]["video"]) for arm, key in BASE]
    for arm, slug in CANDIDATES:
        output = root / slug / "output"
        row = read(output / f"{source}.json")
        clip = row["video"]
        if (row["source"], row["arm"], row["initial_noise"]) != (source, arm, expected_noise[source]):
            raise ValueError("P2 source condition changed")
        if (row["ffn_calls_by_step"] != [40] * 4 or len(row["sites"]) != 160 or len(row["gate_witness"]) != 12
                or not all(item["full_write_bitwise_equal"] for item in row["gate_witness"])):
            raise ValueError("P2 live gate and FFN witness is incomplete")
        if (clip["frames"], clip["width"], clip["height"], clip["fps"]) != (81, 832, 464, 16):
            raise ValueError("P2 clip geometry differs")
        records.append({"arm": arm, "path": str(output / f"{source}.mp4"), "sha256": clip["mp4_sha256"]})
    return records


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("p2", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("sources", nargs="+", choices=SOURCES)
    args = cli.parse_args()
    if args.sources != [source for source in SOURCES if source in args.sources]:
        raise ValueError("Playback must preserve the frozen source order")
    scope = read(args.p2 / "l0" / "scope.json")
    noise = read(Path(scope["dense_initial_noise"]["path"]))
    prepared = [{"source": source, "videos": videos(args.p2, source, scope, noise)} for source in args.sources]
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump({"frames": 81, "fps": 16, "sources": prepared}, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()

