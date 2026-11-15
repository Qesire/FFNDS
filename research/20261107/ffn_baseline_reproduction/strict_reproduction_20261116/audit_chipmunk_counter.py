"""Trace the pinned author counter without importing torch or running a model.

This checks a concrete host-interface boundary. It does not reproduce a paper
benchmark or validate any GPU adapter. Usage:
  python audit_chipmunk_counter.py --source /path/to/layer_counter.py --output trace.json
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path

PINNED_COMMIT = "b5e8d1c3da6bddc7ca9007c48e91b82087c11490"
PINNED_SOURCE_SHA256 = "cb9ffab57832ce69141ecd914fbb3f2c873c26bdca31eee1f7e2b41c01764288"


def trace(source: Path) -> dict:
    data = source.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != PINNED_SOURCE_SHA256:
        raise ValueError("Source bytes differ from the reviewed author commit")
    tree = ast.parse(data.decode("utf-8"))
    counter_class = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "LayerCounter"
    )
    cases = []
    for steps, invocations, registered_layers in [(4, 1, 40), (50, 2, 40), (4, 1, 80)]:
        config = {
            "mlp": {"full_step_every": 10},
            "attn": {"full_step_schedule": None, "full_step_every": 10},
            "num_model_invocations_per_inference_step": invocations,
            "steps": steps,
        }
        namespace = {"GLOBAL_CONFIG": config}
        module = ast.Module(body=[counter_class], type_ignores=[])
        exec(compile(module, str(source), "exec"), namespace)
        counter = namespace["LayerCounter"](registered_layers, 1)
        mismatches, last_calls = [], []
        actual_layers = 40
        for step in range(steps):
            for invocation in range(invocations):
                for layer in range(actual_layers):
                    expected = [step, invocation, layer]
                    observed = [counter.cur_inference_step,
                                counter.cur_model_invocation_per_step,
                                counter.cur_layer]
                    item = {"expected": expected, "observed": observed}
                    if expected != observed:
                        mismatches.append({
                            "call": (step * invocations + invocation) * actual_layers + layer,
                            **item,
                        })
                    if step == steps - 1 and invocation == invocations - 1 and layer >= 37:
                        last_calls.append(item)
                    counter.increment()
        cases.append({
            "steps": steps,
            "invocations_per_step": invocations,
            "registered_layers": registered_layers,
            "actual_layers_per_invocation": actual_layers,
            "mismatch_count": len(mismatches),
            "first_mismatches": mismatches[:3],
            "last_three_calls": last_calls,
        })
    return {
        "evidence_type": "NON_MODEL_PURE_PYTHON_INTERFACE_TEST",
        "upstream_commit": PINNED_COMMIT,
        "source": "src/chipmunk/util/layer_counter.py",
        "source_sha256": digest,
        "method": "Extract the unchanged author class; inject stated config; trace scheduler calls; no torch/CUDA/model",
        "cases": cases,
        "interpretation": "Only the stated interface call sequences were checked. Published paper runs and proposed GPU adapters were not executed.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = trace(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"mismatch_counts": [c["mismatch_count"] for c in result["cases"]],
                      "output": str(args.output)}))


if __name__ == "__main__":
    main()
