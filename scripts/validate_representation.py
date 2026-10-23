#!/usr/bin/env python3
"""Validate an optional PRD representation ledger; does not alter research_state v2.1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from jsonschema import Draft202012Validator


class RepresentationLedgerError(ValueError):
    pass


def validate_ledger(data: dict, schema_path: Path | None = None) -> None:
    root = Path(__file__).resolve().parents[1]
    path = schema_path or root / "schemas" / "representation_ledger.schema.json"
    schema = json.loads(path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    errs = list(Draft202012Validator(schema).iter_errors(data))
    if errs:
        err = sorted(errs, key=lambda e: (list(map(str, e.path)), e.message))[0]
        raise RepresentationLedgerError(f"schema at {list(err.path)}: {err.message}")

    def unique_ids(records: list[dict], label: str) -> set[str]:
        ids = [r["id"] for r in records]
        if len(ids) != len(set(ids)):
            raise RepresentationLedgerError(f"duplicate {label} id")
        return set(ids)

    rep_ids = unique_ids(data["representations"], "representation")
    unique_ids(data["candidates"], "candidate")
    for rep in data["representations"]:
        # Explicit scope + evidence prevent an exact-claim label without an explanation.
        if rep["relation_to_source"] in {"EXACT", "CONDITIONAL"} and not rep["assumptions"]:
            raise RepresentationLedgerError(
                f"representation {rep['id']} claims exact/conditional relation without assumptions"
            )
        if rep["relation_to_source"] == "EXACT" and rep["provenance"]["status"] == "CONJECTURE":
            raise RepresentationLedgerError(
                f"representation {rep['id']} cannot assert EXACT while provenance is CONJECTURE"
            )

    seen = set()
    for link in data["connections"]:
        if link["from"] not in rep_ids or link["to"] not in rep_ids:
            raise RepresentationLedgerError("connection references unknown representation")
        key = (link["from"], link["to"], link["relation"])
        if key in seen:
            raise RepresentationLedgerError("duplicate representation connection")
        seen.add(key)
        if link["relation"] in {"EXACT_EQUIVALENCE", "CONDITIONAL_EQUIVALENCE"}:
            by_id = {rep["id"]: rep for rep in data["representations"]}
            for endpoint in (link["from"], link["to"]):
                if by_id[endpoint]["relation_to_source"] not in {"EXACT", "CONDITIONAL"}:
                    raise RepresentationLedgerError(
                        f"{link['relation']} incompatible with representation {endpoint}"
                    )

    for c in data["candidates"]:
        if set(c["representation_ids"]) - rep_ids:
            raise RepresentationLedgerError(f"candidate {c['id']} cites unknown representation")
        if c["status"] == "OBSERVED" and not c.get("artifact_locator"):
            raise RepresentationLedgerError(f"candidate {c['id']} claims OBSERVED without artifact")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+", type=Path)
    args = p.parse_args()
    for filepath in args.files:
        validate_ledger(json.loads(filepath.read_text(encoding="utf-8")))
        print(f"OK: {filepath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
