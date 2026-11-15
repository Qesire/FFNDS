#!/usr/bin/env python3
"""Verify pinned source bytes and code quotations; never imports a model or torch.

Example:
  python verify_code_citations.py --source icc=/path/to/icc \
      --source chipmunk=/path/to/chipmunk --source ffnds=/path/to/FFNDS

The source directories must contain the exact files described in CODE_CITATIONS.json.
A matching working tree is sufficient: the file Git blob hashes, rather than a
mutable branch name, are checked. No newline normalization is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_blob(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def verify(manifest: dict, roots: dict[str, Path]) -> dict:
    failures: list[dict] = []
    checked_sources = []
    checked_citations = []
    source_text: dict[str, str] = {}
    sources = {s["id"]: s for s in manifest["sources"]}
    citations = {c["id"]: c for c in manifest["citations"]}
    if len(sources) != len(manifest["sources"]):
        failures.append({"id": "manifest", "error": "duplicate source ID"})
    if len(citations) != len(manifest["citations"]):
        failures.append({"id": "manifest", "error": "duplicate citation ID"})
    for sid, source in sources.items():
        try:
            rel = Path(source["path"])
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError("source path must be repository relative")
            data = (roots[source["repository_key"]] / rel).read_bytes()
            actual = {"bytes": len(data), "sha256": digest(data), "git_blob_sha1": git_blob(data)}
            for field, value in actual.items():
                if value != source[field]:
                    raise ValueError(f"{field}: expected {source[field]}, got {value}")
            # splitlines preserves line numbering across LF/CRLF; bytes were already checked.
            source_text[sid] = data.decode("utf-8")
            checked_sources.append({"id": sid, "path": source["path"], **actual})
        except (OSError, KeyError, UnicodeError, ValueError) as exc:
            failures.append({"id": sid, "error": str(exc)})
    for cid, citation in citations.items():
        try:
            source = sources[citation["source_id"]]
            text = source_text[citation["source_id"]]
            lines = text.splitlines()
            start, end = citation["context_start"], citation["context_end"]
            if not 1 <= start <= end <= len(lines):
                raise ValueError("invalid context line span")
            expected = (f"https://github.com/{source['repository']}/blob/"
                        f"{source['commit']}/{source['path']}#L{start}-L{end}")
            if citation["permalink"] != expected:
                raise ValueError("permalink is not the pinned commit/context span")
            quote = citation["quote"]
            # Quotes retain the upstream newline convention; compare exact bytes as text.
            quote_start, quote_end = citation["quote_char_start"], citation["quote_char_end"]
            if not quote or text[quote_start:quote_end] != quote:
                raise ValueError("quote does not match exact character offsets")
            actual_start = text[:quote_start].count("\n") + 1
            actual_end = text[:quote_end - 1].count("\n") + 1
            if (actual_start, actual_end) != (citation["quote_line_start"], citation["quote_line_end"]):
                raise ValueError("quote line coordinates disagree with exact offsets")
            if not start <= actual_start <= actual_end <= end:
                raise ValueError("quote lies outside cited context")
            for span in citation.get("supporting_spans", []):
                if not 1 <= span["start"] <= span["end"] <= len(lines):
                    raise ValueError("invalid supporting span")
            checked_citations.append(cid)
        except (KeyError, ValueError) as exc:
            failures.append({"id": cid, "error": str(exc)})
    seen_claims = set()
    for claim in manifest["claims"]:
        if claim["id"] in seen_claims:
            failures.append({"id": claim["id"], "error": "duplicate claim ID"})
        seen_claims.add(claim["id"])
        if not claim["citation_ids"]:
            failures.append({"id": claim["id"], "error": "claim has no citation"})
        for cid in claim["citation_ids"]:
            if cid not in citations:
                failures.append({"id": claim["id"], "error": f"missing citation {cid}"})
    return {
        "scope": "SOURCE_BYTES_QUOTES_AND_LINKS_ONLY_NOT_MODEL_CONFORMANCE",
        "all_checks_passed": not failures,
        "source_count": len(sources),
        "sources_passed": len(checked_sources),
        "citation_count": len(citations),
        "citations_passed": len(checked_citations),
        "claim_count": len(manifest["claims"]),
        "source_checks": checked_sources,
        "failures": failures,
        "new_model_runs": 0,
        "limitation": "Byte and quote integrity does not establish claim entailment, executable path, numerical correctness, or performance.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("CODE_CITATIONS.json"))
    parser.add_argument("--source", action="append", default=[], metavar="KEY=PATH")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    roots = {}
    for item in args.source:
        key, sep, path = item.partition("=")
        if not sep or not key or not path or key in roots:
            parser.error("--source must be a unique KEY=PATH mapping")
        roots[key] = Path(path)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    report = verify(manifest, roots)
    report["manifest_sha256"] = digest(args.manifest.read_bytes())
    report["verifier_sha256"] = digest(Path(__file__).read_bytes())
    if args.output:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "source_checks"}, ensure_ascii=False, indent=2))
    return 0 if report["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
