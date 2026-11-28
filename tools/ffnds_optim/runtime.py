"""Compose separately selected optimizations with the existing school runtime.

This module loads the actual archived Experiment and frozen CAL. It neither
submits jobs nor substitutes synthetic data when model assets are unavailable.
"""
from contextlib import contextmanager
from dataclasses import replace
from functools import wraps
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import os
import sys

import torch

from .attention_runtime import AttentionSession
from .config import OptimizationConfig


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class TensorDirectory:
    """Explicit per-site assets, read once on CPU and identified in receipts."""
    def __init__(self, directory, *, field=None):
        self.directory, self.field = Path(directory).resolve(), field
        if not self.directory.is_dir():
            raise FileNotFoundError(self.directory)
        self.cache, self.identities = {}, {}

    def __call__(self, expert, layer):
        key = f"{expert}{layer:02d}"
        if key not in self.cache:
            path = self.directory / (key + ".pt")
            value = torch.load(path, map_location="cpu", weights_only=True)
            self.identities[key] = dict(path=str(path), sha256=digest(path))
            if self.field is not None and isinstance(value, dict):
                value = value[self.field]
            self.cache[key] = value
        return self.cache[key]


@contextmanager
def bind_streamed_blocks(owner, attention, ffn, *, tile_rows=8192):
    """Only hit blocks use the row schedule; anchors retain their native graph."""
    from .wan_block import streamed_wan_hit_block
    original = owner.infer_block

    @wraps(original)
    def block(model, weights, x, pre):
        step, layer = int(model.scheduler.step_index), int(model.block_idx)
        if step not in (1, 3):
            return original(model, weights, x, pre)
        hit = attention.take_hit(step, layer)
        staged = ffn.take_hit(model, weights.compute_phases[2])
        row_plan = attention.row_plan(len(x), ffn.probes)
        result = streamed_wan_hit_block(model, weights, x, pre, hit, staged,
                                       probes=ffn.probes, tile_rows=tile_rows, row_plan=row_plan)
        ffn.counters["gate_native"] += 1
        ffn.counters["streamed_blocks"] = ffn.counters.get("streamed_blocks", 0) + 1
        return result

    owner.infer_block = block
    try:
        yield
    finally:
        owner.infer_block = original


class OptimizedRuntime:
    """One model/port reused across serialized requests and fixed HHLL steps."""
    def __init__(self, runtime_dir, fit_dir, config: OptimizationConfig, *, binding_path=None):
        config.assert_generation_backends()
        config = replace(config, **{key:str(Path(getattr(config, key)).resolve())
                                    for key in ("attention_basis_dir", "ffn_factor_dir", "ffn_gauge_dir")
                                    if getattr(config, key) is not None})
        self.config = config
        self.runtime, self.fit = Path(runtime_dir).resolve(), Path(fit_dir).resolve()
        if binding_path is None:
            binding_path = Path(__file__).resolve().parents[2]/"research/20261127/centered_medium_eval/FROZEN_METHOD_BINDING.json"
        self.binding_path = Path(binding_path).resolve()
        self.binding = json.loads(self.binding_path.read_text())
        self.binding_sha256 = digest(self.binding_path)
        for name, expected in self.binding["required_runtime_sources_sha256"].items():
            if digest(self.runtime/name) != expected:
                raise ValueError(f"Frozen runtime source differs: {name}")
        if digest(self.fit/"FIT.sha256") != self.binding["fit_manifest_sha256"]:
            raise ValueError("Frozen calibration manifest differs")
        os.chdir(self.runtime)
        sys.path.insert(0, str(self.runtime))
        self.generate = importlib.import_module("generate")
        centered = importlib.import_module("attention_mass_centered_low")
        online = importlib.import_module("attention_online_structured")
        for module in (self.generate, centered, online):
            if Path(inspect.getfile(module)).resolve().parent != self.runtime:
                raise ValueError("Selected runtime imported a module from another directory")
        from lightx2v.common.ops.attn.torch_sdpa import TorchSDPAWeight
        if digest(inspect.getfile(TorchSDPAWeight.apply)) != online.BACKEND:
            raise ValueError("Native attention backend differs from the frozen source binding")
        torch.backends.cuda.matmul.allow_tf32 = False
        self.experiment = self.generate.Experiment()
        self.legacy_attention = centered.MassCenteredLow(self.experiment, self.fit)
        self.legacy_attention.mode = "centered"
        if self.legacy_attention.calibration != self.binding["calibration"]:
            raise ValueError("Loaded attention CAL differs from the frozen recipe")
        if digest(inspect.getfile(self.experiment.owner.infer_block)) != self.experiment.base["native_transformer_sha256"]:
            raise ValueError("Native transformer source differs from the runner binding")
        self.basis_assets = TensorDirectory(config.attention_basis_dir) if config.attention_basis_dir else None
        self.factor_assets = TensorDirectory(config.ffn_factor_dir) if config.ffn_factor_dir else None
        self.gauge_assets = TensorDirectory(config.ffn_gauge_dir, field="gauge") if config.ffn_gauge_dir else None
        self.attention = AttentionSession(self.legacy_attention, config, basis_loader=self.basis_assets)
        if config.needs_custom_ffn:
            from exact_port import site_loader
            from .ffn_runtime import OptimizationFFNPort
            self.ffn = OptimizationFFNPort(site_loader(self.experiment.base),
                self.experiment.base["observer_ids"], tile_rows=config.tile_rows,
                backend=config.ffn_backend, factor_loader=self.factor_assets, gauge_loader=self.gauge_assets,
                projection_dtype=getattr(torch, config.ffn_projection_dtype),
                preserve_probe_u=config.preserve_ffn_probe_u, anchor_direct=config.anchor_direct)
        else:
            self.ffn = self.experiment.build("wide8192")
        self.source_manifest_sha256 = None

    def register_manifest(self, path):
        path = Path(path).resolve()
        if self.source_manifest_sha256 is not None:
            raise RuntimeError("One source manifest per resident runtime")
        manifest = json.loads(path.read_text())
        seen = set()
        for row in manifest["sources"]:
            sid = row["source_id"]
            if sid in seen or not Path(row["image_path"]).is_absolute():
                raise ValueError("Duplicate source ID or nonabsolute input path")
            seen.add(sid)
            if digest(row["image_path"]) != row["image_sha256"] or hashlib.sha256(row["prompt"].encode()).hexdigest() != row["prompt_sha256"]:
                raise ValueError("Input image/prompt identity differs")
            if row.get("seed", 42) != self.binding["seed"]:
                raise ValueError("The runner's frozen seed differs")
            if sid in self.experiment.rows:
                old = self.experiment.rows[sid]
                if any(old[key] != row[key] for key in ("image_sha256", "prompt_sha256")):
                    raise ValueError("An existing source ID denotes another image/prompt")
            self.experiment.rows[sid] = dict(row, record_id=sid, seed=self.binding["seed"])
        self.source_manifest_sha256 = digest(path)

    def render(self, source, group):
        if source not in self.experiment.rows:
            raise ValueError(f"Unknown source {source}; register its existing manifest")
        if Path(group).is_absolute() or ".." in Path(group).parts:
            raise ValueError("Output group must be relative to the existing runtime output")
        receipt = self.experiment.output/group/(source+".json")
        if receipt.exists():
            raise FileExistsError(f"Select a fresh output group: {receipt}")
        kwargs = {"reuse": False} if "reuse" in inspect.signature(self.experiment.render).parameters else {}
        @contextmanager
        def attention_binding():
            if self.config.uses_optimized_attention:
                with self.attention.bind() as counts:
                    yield counts
            else:
                # The reference and FFN-only optimization profiles execute the
                # archived attention adapter itself, with no new history copies.
                with self.legacy_attention.bind("OUT-D", audit=None, profile=False,
                                                archive_routes=False, high_step0=False) as (counts, _, _):
                    yield counts

        with torch.inference_mode(), attention_binding() as counts:
            if self.config.block_streaming:
                with bind_streamed_blocks(self.experiment.owner, self.attention, self.ffn,
                                          tile_rows=self.config.tile_rows):
                    result = self.experiment.render(source, group, self.ffn, **kwargs)
            else:
                result = self.experiment.render(source, group, self.ffn, **kwargs)
        if result["experts"] != [0, 0, 1, 1]:
            raise RuntimeError("Expected four-step HHLL execution")
        assets = {key: value.identities for key, value in (("attention_basis", self.basis_assets),
                  ("ffn_factor", self.factor_assets), ("ffn_gauge", self.gauge_assets)) if value is not None}
        wrapped = dict(candidate_id="wide8192_optim_"+self.config.fingerprint()[:16],
            profile=self.config.to_dict(), numerical_semantics=self.config.semantics(),
            frozen_reference_binding_sha256=self.binding_sha256,
            optimization_source_sha256={p.name:digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
            asset_identities=assets, source_manifest_sha256=self.source_manifest_sha256,
            record=result, attention=dict(counts), new_method_quality_status="this_run_unscored")
        self.generate.save(receipt.with_name(source+".optimization.json"), wrapped)
        return wrapped
