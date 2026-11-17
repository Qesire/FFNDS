"""School-side native BF16 latent observer for an already approved LightX2V Dense run.

This is a component for the actual school runner, not a standalone video
generator or authorization mechanism. Attach only after pipeline creation.
Unlike historical run_distill.py, it NEVER casts latents to FP32 before save.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

class NativeBF16PostStepObserver:
    def __init__(self, scheduler: Any, output_dir: Path):
        self.scheduler = scheduler
        self.output_dir = Path(output_dir)
        self._original = None
        self.captured: list[dict] = []

    def __enter__(self):
        import torch
        from safetensors.torch import save_file

        if self._original is not None:
            raise RuntimeError("Observer cannot be nested")
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self._original = self.scheduler.step_post
        expected = (16, 21, 58, 104)

        def observed_step_post(*args, **kwargs):
            result = self._original(*args, **kwargs)
            step = int(self.scheduler.step_index)
            z = self.scheduler.latents
            if z.dtype != torch.bfloat16 or tuple(z.shape) != expected:
                raise ValueError(f"Expected native BF16 {expected}, got {z.dtype}, {tuple(z.shape)}")
            if step != len(self.captured) or step not in (0, 1, 2, 3):
                raise ValueError("Unexpected or repeated native denoising step")
            # D2H copy changes timing, not the scheduler's original z tensor.
            frozen = z.detach().to("cpu", copy=True).contiguous()
            path = self.output_dir / f"latent_step{step}_bf16.safetensors"
            save_file({"latents": frozen}, str(path))
            self.captured.append({"step": step, "path": str(path.resolve()),
                                  "sha256": digest(path), "bytes": path.stat().st_size,
                                  "shape": list(z.shape), "dtype": "BF16"})
            return result

        self.scheduler.step_post = observed_step_post
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._original is not None:
            self.scheduler.step_post = self._original
            self._original = None
        return False

    def complete(self, *, video: Path, source_id: str, image: Path,
                 prompt: str, seed: int, high_sha256: str, low_sha256: str,
                 vae_sha256: str, source_commit: str, receipt_path: Path) -> dict:
        if [x["step"] for x in self.captured] != [0, 1, 2, 3]:
            raise ValueError("Incomplete native four-step latent capture")
        video, image = Path(video), Path(image)
        if not video.is_file() or not image.is_file():
            raise FileNotFoundError("School video or school input image missing")
        if receipt_path.exists():
            raise FileExistsError("Never overwrite a native capture receipt")
        if not source_id or not prompt.strip() or seed < 0:
            raise ValueError("Source/seed/prompt identity is required")
        if any(len(s) != 64 for s in (high_sha256, low_sha256, vae_sha256, source_commit)):
            raise ValueError("Require pinned model and source SHA")
        receipt = {
            "schema": "FFNDS_SCHOOL_FISHSCALE_NATIVE_CAPTURE_V1",
            "status": "CAPTURED_NEEDS_NEUTRAL_REVIEW",
            "source_id": source_id, "seed": int(seed),
            "image": {"path": str(image.resolve()), "sha256": digest(image)},
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "video": {"path": str(video.resolve()), "sha256": digest(video)},
            "final_latent": self.captured[-1],
            "all_native_latents": self.captured,
            "high_sha256": high_sha256, "low_sha256": low_sha256,
            "vae_sha256": vae_sha256, "source_commit": source_commit,
            "note": "Observed BF16 post-step state only. Must verify school input, FFV1/H264 and human review separately."
        }
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = receipt_path.with_name(receipt_path.name + ".tmp")
        tmp.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.replace(receipt_path)
        return receipt
