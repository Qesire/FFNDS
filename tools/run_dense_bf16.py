"""Run the unquantized distilled BF16 baseline; see run_distill for shared audit."""
# Retain the original entrypoint and helper imports used by existing run tools.
from run_distill import ROOT, atomic_json, digest, main

if __name__ == "__main__":
    main("D-BF16")
