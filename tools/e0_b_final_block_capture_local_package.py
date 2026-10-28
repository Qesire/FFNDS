"""B0 startup fix: unchanged native capture with a byte-verified local package.

The original failed scope and driver remain immutable. This entry point changes
only the scope identity and Transformers transport/import provenance.
"""
import argparse
import hashlib
import importlib.machinery
import json
from pathlib import Path
import sys

import e0_b_final_block_capture as native
from local_python_package_mirror import canonical_sha, inventory, verify_origin

ID = "E0_B_FINAL_BLOCK_NATIVE_CAPTURE_LOCAL_PACKAGE_20261028"
TRANSPORT = "BYTE_IDENTICAL_NODE_LOCAL_TRANSFORMERS_NO_SOURCE_OR_MODEL_CHANGE"
PREVIOUS_ID = "E0_B_FINAL_BLOCK_NATIVE_CAPTURE_20261028"
PREVIOUS_PROTOCOL = "913cf90e0f812903104ee643fa417f2b16c879bfa124e4eac8ff398632dcf018"


def local_package(plan, roots, local_root):
    """Only read bytes and receipts; no Torch, CUDA or package imports."""
    contract = plan["local_transformers"]
    if (plan["id"] != ID or contract["transport"] != TRANSPORT
            or contract["version"] != "5.6.2"):
        raise ValueError("Exact local package contract differs")
    producer = json.loads(native.bound(contract["producer_RUN"], roots).read_text())
    launch = json.loads(native.bound(contract["producer_exit"], roots).read_text())
    if (producer["status"] != "PASS_LOCAL_MIRROR_IMPORT_NO_MODEL"
            or producer["protocol_sha256"] != contract["producer_protocol_sha256"]
            or producer["inventory_sha256"] != contract["inventory_sha256"]
            or producer["archive_sha256"] != contract["archive_sha256"]
            or producer.get("cuda_initialized") is not False
            or producer.get("cuda_device_count") != 0
            or producer.get("GPU_model_calls") != 0
            or launch["launcher_returncode"] != 0 or launch["process_exited"] is not True
            or launch["protocol_sha256"] != producer["protocol_sha256"]
            or launch["RUN_sha256"] != contract["producer_RUN"]["sha256"]):
        raise ValueError("Completed CPU mirror producer/actual exit identity differs")
    prior = json.loads(native.bound(plan["previous_failed_RUN"], roots).read_text())
    old_exit = json.loads(native.bound(plan["previous_failed_exit"], roots).read_text())
    if (prior["id"] != PREVIOUS_ID or old_exit["id"] != PREVIOUS_ID
            or prior["protocol_sha256"] != PREVIOUS_PROTOCOL
            or old_exit["protocol_sha256"] != PREVIOUS_PROTOCOL
            or prior["error_type"] != "TimeoutError"
            or prior["status"] != "FAILED_FIXED_CAPTURE_NO_AUTOMATIC_RETRY"
            or prior["cases"] or prior["predebited"] or prior["observed"]
            or old_exit["process_exited"] is not True or old_exit["launcher_returncode"] != 1):
        raise ValueError("Original startup failure must remain a closed zero-model scope")
    root = Path(local_root).resolve(strict=True)
    if hashlib.sha256(str(root).encode()).hexdigest() != contract["local_root_path_sha256"]:
        raise ValueError("Node-local root identity differs")
    receipt = json.loads(native.bound(contract["inventory"], roots).read_text())
    rows = receipt["files"]
    if (canonical_sha(rows) != contract["inventory_sha256"]
            or receipt["archive_sha256"] != contract["archive_sha256"]
            or inventory(root / "transformers") != rows):
        raise ValueError("Complete node-local package bytes differ")
    if any(name == "transformers" or name.startswith("transformers.") for name in sys.modules):
        raise ValueError("Transformers was imported before its checked local path")
    return root, rows


class CheckedPackageFinder:
    """Reject every Transformers Python module outside the frozen local copy."""
    def __init__(self, root, rows):
        self.root, self.rows, self.origins = Path(root), rows, {}

    def find_spec(self, fullname, path=None, target=None):
        if fullname != "transformers" and not fullname.startswith("transformers."):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is None or spec.origin is None:
            raise ImportError("Missing frozen Transformers module: " + fullname)
        relative = verify_origin(spec.origin, self.root / "transformers", self.rows)
        self.origins[fullname] = relative
        return spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--hf-root", type=Path, default=Path.home() / ".cache/huggingface")
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--execute-native", action="store_true")
    args = parser.parse_args()
    native.ID = ID
    native.REQUIRED_SOURCES = native.REQUIRED_SOURCES | {
        "tools/e0_b_final_block_capture_local_package.py",
        "tools/local_python_package_mirror.py",
        "tests/test_b_local_package_capture.py"}
    prepared = native.preflight(args.protocol, args.hf_root)
    plan, roots, readiness = prepared
    root, rows = local_package(plan, roots, args.local_root)
    readiness["local_transformers_inventory_sha256"] = plan["local_transformers"]["inventory_sha256"]
    readiness["local_transformers_files_checked"] = len(rows)
    if not args.execute_native:
        print(json.dumps(readiness, indent=2))
        return
    finder = CheckedPackageFinder(root, rows)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root))
    sys.meta_path.insert(0, finder)
    try:
        native.execute(args, prepared)
    finally:
        out = native.confined(roots["hf"], plan["output_HF_relative"])
        if (out / "ATTEMPT.json").exists():
            native.write(out / "LOCAL_PACKAGE_IMPORT.json", {
                "id": ID, "protocol_sha256": native.sha256(args.protocol),
                "transport": TRANSPORT,
                "inventory_sha256": plan["local_transformers"]["inventory_sha256"],
                "checked_import_origins": finder.origins,
                "source_or_model_changes": 0,
                "capture_calls_are_accounted_in": "RUN.json"})
        sys.meta_path.remove(finder)


if __name__ == "__main__":
    main()
