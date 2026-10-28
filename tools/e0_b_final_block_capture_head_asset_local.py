"""B0 V3: retain native offload; export startup head assets from the CPU view."""
import json
from pathlib import Path
import sys

import e0_b_final_block_capture_head_asset as native
import e0_b_final_block_capture_local_package as startup

ID = "E0_B_FINAL_BLOCK_NATIVE_CAPTURE_HEAD_ASSET_20261028"
PREVIOUS_ID = "E0_B_FINAL_BLOCK_NATIVE_CAPTURE_LOCAL_PACKAGE_20261028"
PREVIOUS_PROTOCOL = "e3b60df142b13ee108338cb934dc6d280cd5cf8e803022c07df9ec42fb0aff01"


def verify_previous_startup(plan, roots):
    failed = json.loads(native.bound(plan["previous_local_package_failed_RUN"], roots).read_text())
    exited = json.loads(native.bound(plan["previous_local_package_failed_exit"], roots).read_text())
    if (failed["id"] != PREVIOUS_ID or exited["id"] != PREVIOUS_ID
            or failed["protocol_sha256"] != PREVIOUS_PROTOCOL
            or exited["protocol_sha256"] != PREVIOUS_PROTOCOL
            or failed["status"] != "FAILED_FIXED_CAPTURE_NO_AUTOMATIC_RETRY"
            or failed["error_type"] != "AttributeError" or failed["cases"]
            or failed["observed"] != {"setup_decoded_weight_GPU_transfers": 800,
                                      "setup_decoded_weight_bytes": 56203673600}
            or exited["launcher_returncode"] != 1 or exited["process_exited"] is not True):
        raise ValueError("Exact closed V2 offload-head startup failure differs")


def main():
    original_preflight = native.preflight
    def preflight(protocol, hf_root, root=native.ROOT):
        prepared = original_preflight(protocol, hf_root, root)
        verify_previous_startup(prepared[0], prepared[1])
        return prepared
    native.preflight = preflight
    startup.native = native
    startup.ID = ID
    native.REQUIRED_SOURCES |= {"tests/test_b_head_asset_local.py"}
    startup.main()


if __name__ == "__main__":
    main()
