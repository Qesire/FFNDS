"""Atomic, content-verified checkpoints for the AR2 offline diagnostic.

The run identity must include the protocol, source/environment and asset
identities. A completed part is reusable only under the same run identity and
after its payload checksum matches. An interrupted temporary file is ignored.
"""
import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    temp = Path(str(path) + ".tmp")
    with temp.open("w") as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


class CheckpointStore:
    def __init__(self, root, identity):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.fingerprint = hashlib.sha256(canonical_json(identity).encode()).hexdigest()
        path = self.root / "identity.json"
        if path.exists():
            previous = json.loads(path.read_text())
            if previous != {"fingerprint": self.fingerprint, "identity": identity}:
                raise ValueError("Checkpoint identity differs; select a new output directory")
        else:
            # The job-level lock in the runner protects this initial write.
            atomic_json(path, {"fingerprint": self.fingerprint, "identity": identity})

    def _paths(self, stage, key):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", stage) or not re.fullmatch(r"[A-Za-z0-9_-]+", key):
            raise ValueError("Checkpoint stage/key must be plain path components")
        parent = self.root / stage
        parent.mkdir(parents=True, exist_ok=True)
        return parent / f"{key}.npz", parent / f"{key}.json"

    def load(self, stage, key):
        payload, metadata = self._paths(stage, key)
        if not metadata.exists():
            return None
        meta = json.loads(metadata.read_text())
        if meta["status"] != "COMPLETE" or meta["fingerprint"] != self.fingerprint:
            raise ValueError(f"Invalid checkpoint metadata: {metadata}")
        if not payload.is_file() or digest(payload) != meta["payload_sha256"]:
            raise ValueError(f"Checkpoint payload is missing/corrupted: {payload}")
        with np.load(payload, allow_pickle=False) as arrays:
            return {"arrays": {name: arrays[name].copy() for name in arrays.files}, "metadata": meta["data"]}

    def save(self, stage, key, arrays, metadata):
        if self.load(stage, key) is not None:
            raise ValueError("Refusing to overwrite a completed checkpoint")
        payload, sidecar = self._paths(stage, key)
        temp = Path(str(payload) + ".tmp")
        with temp.open("wb") as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        temp.replace(payload)
        atomic_json(sidecar, {"status": "COMPLETE", "fingerprint": self.fingerprint,
                              "payload_sha256": digest(payload), "data": metadata})
