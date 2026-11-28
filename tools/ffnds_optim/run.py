"""Execute an explicitly selected profile through the actual school runner."""
import argparse
import json
from pathlib import Path

from .config import OptimizationConfig


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--runtime-dir", type=Path)
    p.add_argument("--fit-dir", type=Path)
    p.add_argument("--binding", type=Path)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--source", action="append")
    p.add_argument("--group")
    p.add_argument("--describe", action="store_true", help="Print choices and numerical scope; load no model")
    args = p.parse_args()
    config = OptimizationConfig.load(args.config)
    if args.describe:
        print(json.dumps(dict(profile=config.to_dict(), semantics=config.semantics(),
                             fingerprint=config.fingerprint()), ensure_ascii=False, indent=2))
        return
    if not all((args.runtime_dir, args.fit_dir, args.source, args.group)):
        p.error("generation requires runtime-dir, fit-dir, source and group")
    runtime_dir, fit_dir = args.runtime_dir.resolve(), args.fit_dir.resolve()
    binding = args.binding.resolve() if args.binding else None
    manifest = args.manifest.resolve() if args.manifest else None
    from .runtime import OptimizedRuntime
    runtime = OptimizedRuntime(runtime_dir, fit_dir, config, binding_path=binding)
    if manifest:
        runtime.register_manifest(manifest)
    for source in args.source:
        record = runtime.render(source, args.group)
        print(json.dumps(dict(source=source, candidate_id=record["candidate_id"],
                              attention=record["attention"]), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
