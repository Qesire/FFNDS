from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser("Inspect a HiF4 PTQ artifact")
    parser.add_argument("path")
    args = parser.parse_args()
    path = Path(args.path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    meta = payload.get("__meta__", {}) if isinstance(payload, dict) else {}
    modules = payload.get("modules", {}) if isinstance(payload, dict) else {}
    print("path=", path)
    print("meta=", meta)
    print("module_count=", len(modules))
    for name in list(modules)[:20]:
        record = modules[name]
        print(name, {k: v for k, v in record.items() if not torch.is_tensor(v)})


if __name__ == "__main__":
    main()
