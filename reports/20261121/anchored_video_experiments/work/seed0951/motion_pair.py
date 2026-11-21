"""Compare the two videos from a single seed at the same four-frame lag."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "improve0951"))
import soft_motion as sm

OUT = HERE.parents[1] / "outputs/FFNDS_0951_seeds"
HIST = HERE.parents[1] / "outputs/FFNDS_anchor_scale24/camera7"


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("seed", type=int)
    cli.add_argument("dense", type=Path)
    cli.add_argument("anchor", type=Path)
    args = cli.parse_args()
    sm.cv.setNumThreads(2)
    sm.EVAL = OUT / "motion"
    samples = []
    initial = []
    for path in (args.dense, args.anchor):
        note = json.loads(path.with_suffix(".json").read_bytes())
        assert hashlib.sha256(path.read_bytes()).hexdigest() == note["video"]["mp4_sha256"]
        assert note["video"]["frames"] == 81 and note["video"]["fps"] == 16
        if args.seed != 42:
            assert note["seed"] == args.seed
        initial.append(note["initial"])
        samples.append(sm.score(path))
    assert len(set(initial)) == 1
    dense, anchor = samples
    measures = sm.compare_order(anchor,dense)
    vectors = []
    for group in (dense,anchor):
        positions=np.array([f["world"] for f in group["frames"]],dtype=float)
        vectors.append(positions[4:]-positions[:-4])
    true, estimate = vectors
    energy = np.sum(true**2)
    gain = measures["world"]["gain"]
    assert np.isclose(gain,np.sum(true*estimate)/energy)
    measures["world"].update(orthogonal_relative_rms=float(np.sqrt(np.sum((estimate-gain*true)**2)/energy)),
                             vector_relative_rms=float(np.sqrt(np.sum((estimate-true)**2)/energy)),
                             four_frame_reference_rms_px=float(np.sqrt(energy/len(true))))
    record = dict(seed=args.seed, initial=initial[0], dense_sha256=dense["video"], anchor_sha256=anchor["video"],
                  estimator_sha256=dense["code"], points=81, displacements=77, metrics=measures,
                  median_background_inlier_fraction={k:float(np.median([f["background_support"] for f in g["frames"][1:]])) for k,g in (("dense",dense),("anchor",anchor))})
    sm.write(sm.EVAL / f"pair_s{args.seed}.json",record)
    print(args.seed, {k:round(measures["world"][k],4) for k in ("gain","cos","orthogonal_relative_rms","p90")})


if __name__ == "__main__":
    main()
