"""M0: inspect existing manifests, coordinates and tensor headers, never tensors.

This is a file inventory, not the A5 scientific experiment. Inherited tensor
hashes stay inherited; only small JSON/header bytes are read by this tool.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import struct
from collections import Counter
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def resolve(root, value):
    value = str(value)
    if "ffnds-derived/" in value:
        return root / value[value.index("ffnds-derived/"):]
    p = Path(value)
    return p if p.is_absolute() else root / p


def public_path(path):
    value = str(path)
    if "ffnds-derived/" in value:
        return value[value.index("ffnds-derived/"):]
    return value


class Inspector:
    def __init__(self, root):
        self.root = root
        self.bytes_read = 0
        self.file_checks = Counter()

    def json(self, ref):
        if isinstance(ref, dict):
            p = resolve(self.root, ref["path"])
        else:
            p = resolve(self.root, ref)
        b = p.read_bytes()
        self.bytes_read += len(b)
        if isinstance(ref, dict):
            assert len(b) == ref["bytes"], (public_path(p), "metadata length")
            assert digest(b) == ref["sha256"], (public_path(p), "metadata SHA")
            self.file_checks["metadata_full_sha_pass"] += 1
        return json.loads(b)

    def header(self, ref):
        p = resolve(self.root, ref["path"])
        assert p.stat().st_size == ref["bytes"], (public_path(p), "payload length")
        with p.open("rb") as f:
            prefix = f.read(8)
            n = struct.unpack("<Q", prefix)[0]
            assert 0 < n <= 2 * 1024**2
            b = f.read(n)
        self.bytes_read += n + 8
        h = json.loads(b)
        end = 0
        result = {}
        for k, v in h.items():
            if k == "__metadata__":
                continue
            start, stop = v["data_offsets"]
            assert 0 <= start <= stop
            end = max(end, stop)
            result[k] = {"shape": v["shape"], "dtype": v["dtype"]}
        assert 8 + n + end == p.stat().st_size
        self.file_checks["payload_size_and_header_pass_not_full_sha"] += 1
        return result


def pair_records(dataset, identity, scopes):
    by = {(int(s["step"]), int(s["block"])): s for s in scopes}
    assert set(by) == {(s, b) for s in range(4) for b in range(40)}
    out = []
    for old, new, expert in [(0, 1, "high_noise_model"), (2, 3, "low_noise_model")]:
        for block in range(40):
            a, b = by[old, block], by[new, block]
            ar, br = set(a["rows"]), set(b["rows"])
            intersection = ar & br
            row = dict(dataset=dataset, **identity, expert=expert, old_step=old,
                       current_step=new, block=block, old_rows=len(ar),
                       current_rows=len(br), matched_rows=len(intersection),
                       identical_row_coordinates=ar == br,
                       old_boundaries=",".join(sorted(a["header"])),
                       current_boundaries=",".join(sorted(b["header"])),
                       old_payload=a["path"], current_payload=b["path"])
            for m in [16, 32, 64]:
                # Physical GEMM M-tiles begin at multiples of M, not arbitrary
                # runs of sampled coordinates. No sampled-row concatenation.
                candidates = {r // m for r in intersection}
                row[f"complete_original_M{m}_tiles"] = sum(
                    all(r in intersection for r in range(t*m, min((t+1)*m, 31668)))
                    for t in candidates)
            out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", type=Path, default=Path.home()/".cache/huggingface")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    ins = Inspector(a.hf)
    rows, datasets = [], []
    config = json.loads(Path("config/e0_b_cal24_terminal_labels_20261028_v1.json").read_text())
    for ref in config["inputs"]:
        d = ins.json(ref)
        scopes = []
        for s in d["scopes"]:
            header = ins.header(s)
            scopes.append(dict(step=s["step"], block=s["block"], rows=s["selected_rows"],
                               header=header, path=public_path(s["path"])))
        ident = dict(source=str(d["input"].get("source_id", d["input"].get("native_source_ordinal", ref["index"]))),
                     case=str(ref["index"]), arm="dense", numeric_family="HIF4_FAKE_QDQ_BF16")
        rows.extend(pair_records("cal24", ident, scopes))
    datasets.append(dict(id="cal24", sources=24, sites_per_source=160,
                         boundaries="postQ1 V1, postQ2 V2, Y, gate, b2",
                         raw_x="MISSING", preQ2_h="MISSING", preGELU_a="MISSING",
                         identity="INPUT JSON SHA checked; operand sizes/headers checked; payload SHA inherited"))

    full = ins.json("ffnds-derived/e0_raw_bf16_full_depth_asset_audit_20261029/attempt1/worker/results/MANIFEST.json")
    for c in full["cases"]:
        scopes = []
        for packet in c["packets"]:
            d = ins.json(packet["metadata"])
            header = ins.header(d["payload"])
            scopes.append(dict(step=d["step"], block=d["block"], rows=d["selected_rows"],
                               header=header, path=public_path(d["payload"]["path"])))
        ident = dict(source=c["source_id"], case=str(c["native_source_ordinal"]),
                     arm="dense", numeric_family="BF16_ORIGINAL")
        rows.extend(pair_records("raw_bf16_full160", ident, scopes))
    datasets.append(dict(id="raw_bf16_full160", sources=8, sites_per_source=160,
                         boundaries="V1, h, Y, gate, r, u, rows; shift/scale",
                         HiF4_identity="INCOMPATIBLE_NUMERIC_FAMILY",
                         cross_step_pairing="Read actual selected_rows, not state_pair prose"))

    for part in range(2):
        run = ins.json(f"ffnds-derived/e0_s2_spatial_student_full160_portable_v2_20261029/part{part}/attempt1/worker/results/RUN.json")
        for phase in run["phases"]:
            scopes = []
            for packet in phase["packets"]:
                d = ins.json(packet["metadata"])
                header = ins.header(d["payload"])
                scopes.append(dict(step=d["step"], block=d["block"], rows=d["selected_rows"],
                                   header=header, path=public_path(d["payload"]["path"])))
            ident = dict(source=phase["source_id"], case=str(phase["native_source_ordinal"]),
                         arm=phase["mode"], numeric_family="BF16_ORIGINAL")
            rows.extend(pair_records("spatial52_full160", ident, scopes))
    datasets.append(dict(id="spatial52_full160", sources=2, arms=3, sites_per_source_arm=160,
                         HiF4_identity="INCOMPATIBLE_NUMERIC_FAMILY",
                         geometry="21 latent times x 52 spatial anchors; not full raster M"))

    a.out.parent.mkdir(parents=True, exist_ok=True)
    csv_path = a.out.with_suffix(".csv")
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    grouped = {}
    for name in sorted({r["dataset"] for r in rows}):
        sub = [r for r in rows if r["dataset"] == name]
        matched = [r["matched_rows"] for r in sub]
        grouped[name] = dict(pairs=len(sub), min_matched=min(matched), max_matched=max(matched),
                             mean_matched=sum(matched)/len(matched),
                             identical_coordinate_pairs=sum(r["identical_row_coordinates"] for r in sub),
                             complete_original_tiles={str(m):sum(r[f"complete_original_M{m}_tiles"] for r in sub)
                                                      for m in [16,32,64]})
    result = dict(schema="FFNDS_A5_HWG_M0_FILE_INVENTORY_V1", author_agent_id="/root",
                  status="COMPLETE_FILE_METADATA_INVENTORY_NOT_SCIENTIFIC_E0",
                  payload_values_read=False, scientific_GPU_calls=0,
                  checked_bytes=ins.bytes_read, checks=dict(ins.file_checks),
                  datasets=datasets, paired_coordinate_summary=grouped,
                  detailed_pairs=dict(path=str(csv_path), bytes=csv_path.stat().st_size,
                                       sha256=digest(csv_path.read_bytes())),
                  required_new_producer=["HiF4 same-expert old/current matched coordinates at all40 blocks both experts",
                    "preQ1 raw x and postQ1 V1; preGELU a, preQ2 h, postQ2 V2, old/current Y and current gate",
                    "full-raster or genuinely contiguous original M16/32/64 with fullM union statistics",
                    "native no-action neutrality and original Q1/Q2 scale/smooth identity"],
                  limitations=["Inherited operand SHA not rehashed in M0; numerical values not loaded",
                    "Existing BF16 evidence remains BF16; it cannot validate HiF4 A5 numerical fidelity",
                    "Current file/coordinate statistics do not establish mask quality, video quality or physical speed"])
    a.out.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({"status":result["status"], "summary":grouped, "checked_bytes":ins.bytes_read}))


if __name__ == "__main__":
    main()
