"""Nonmutating taps and exact state imports for a registered final-block capture.

Importing this module does not import Torch, model code, or initialize CUDA.
The native driver supplies the only four flows. No head/FFN shadow is evaluated.
"""
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import struct
import sys


SHAPE = (16, 21, 58, 104)
GRID = (21, 29, 52)
M, D, OUT = math.prod(GRID), 5120, 64
STEPS = (0, 3)


def raw_sha(value):
    import torch
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def tensor_record(value):
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "stride": list(value.stride()), "raw_tensor_sha256": raw_sha(value)}


def cpu_copy(value):
    return value.detach().to(device="cpu", copy=True).contiguous()


def exact_bf16_from_fp32_bytes(raw):
    """Recover only an exact BF16 widening; reject NaN/Inf and rounded FP32."""
    if len(raw) % 4:
        raise ValueError("Malformed FP32 bytes")
    recovered = bytearray(len(raw) // 2)
    for i, (bits,) in enumerate(struct.iter_unpack("<I", raw)):
        if bits & 0xFFFF or bits & 0x7F800000 == 0x7F800000:
            raise ValueError("Historical FP32 state is not a finite exact BF16 widening")
        struct.pack_into("<H", recovered, 2 * i, bits >> 16)
    return bytes(recovered)


def read_state_bytes(path, dtype):
    """Pure CPU safetensors header/raw validation; no Torch or CUDA import."""
    path = Path(path)
    with path.open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        if not 0 < size <= 1 << 20:
            raise ValueError("Unexpected state header size")
        header = json.loads(stream.read(size))
        if set(header) - {"__metadata__"} != {"latents"}:
            raise ValueError("One original latents tensor required")
        item = header["latents"]
        width = {"BF16": 2, "F32": 4}[dtype]
        expected = math.prod(SHAPE) * width
        if item != {"dtype": dtype, "shape": list(SHAPE), "data_offsets": [0, expected]}:
            raise ValueError("Original native state header differs")
        raw = stream.read()
        if len(raw) != expected:
            raise ValueError("Original native state byte count differs")
    if dtype == "F32":
        bf16 = exact_bf16_from_fp32_bytes(raw)
    else:
        if any(v & 0x7F80 == 0x7F80 for (v,) in struct.iter_unpack("<H", raw)):
            raise ValueError("Nonfinite historical BF16 state")
        bf16 = raw
    return {"dtype": dtype, "shape": list(SHAPE), "raw_tensor_sha256": hashlib.sha256(raw).hexdigest(),
            "exact_native_BF16_sha256": hashlib.sha256(bf16).hexdigest()}


def import_post_state(path, expected_raw_sha, device):
    """The saved post state is FP32 storage of exact native BF16 values."""
    import torch
    from safetensors.torch import load_file
    inspected = read_state_bytes(path, "F32")
    if inspected["raw_tensor_sha256"] != expected_raw_sha:
        raise ValueError("Historical post-state identity changed")
    value = load_file(str(path), device="cpu")["latents"].to(torch.bfloat16)
    if raw_sha(value) != inspected["exact_native_BF16_sha256"]:
        raise ValueError("CPU state import changed native BF16 bits")
    return value.to(device=device)


def tree_state(value, *, reference=False):
    """Bind all numeric conditioning/scheduler fields, with explicit references.

    Config string values are hashed, so private local paths do not enter RUN.
    Non-numeric live model references are identified, never silently serialized.
    """
    import torch
    from collections.abc import Mapping
    if torch.is_tensor(value):
        return tensor_record(value)
    if isinstance(value, torch.Generator):
        return {"generator_device": str(value.device), "initial_seed": value.initial_seed(),
                "state": tensor_record(value.get_state())}
    if value is None or isinstance(value, (bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Nonfinite scheduler scalar")
        return value
    if isinstance(value, str):
        return {"string_sha256": hashlib.sha256(value.encode()).hexdigest()}
    if isinstance(value, (torch.dtype, torch.device)):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): tree_state(v) for k, v in sorted(value.items(), key=lambda x: str(x[0]))}
    if isinstance(value, (tuple, list)):
        return [tree_state(v) for v in value]
    if reference:
        if callable(value) and hasattr(value, "__code__"):
            source = Path(value.__code__.co_filename)
            if not source.is_file():
                raise TypeError("Instrumented scheduler callable has no bound source")
            return {"non_numeric_instrumentation_callable": value.__module__ + "." + value.__qualname__,
                    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
        return {"non_numeric_reference": type(value).__module__ + "." + type(value).__qualname__}
    raise TypeError("Unbound state object: " + type(value).__qualname__)


def state_sha(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def scheduler_state(scheduler):
    fields = {}
    for name, value in sorted(vars(scheduler).items()):
        instrumentation = name in ("prepare", "prepare_latents")
        if instrumentation and not callable(value):
            raise TypeError("Unexpected non-callable scheduler preparation override")
        fields[name] = tree_state(value, reference=name == "transformer_infer" or instrumentation)
    return {"fields": fields, "sha256": state_sha(fields),
            "reference_scope": "numeric scheduler state; transformer code/state recreated by native pre/infer"}


@contextmanager
def patch_method(owner, name, value):
    had_own = name in vars(owner)
    previous = getattr(owner, name)
    setattr(owner, name, value)
    try:
        yield previous
    finally:
        if had_own:
            setattr(owner, name, previous)
        else:
            delattr(owner, name)


class FinalBlockTap:
    """Observe original runtime calls; every wrapper returns its original result.

    The norm call observes the *actual* native shift/scale local variable, after
    the original add/chunk and before LN. Its Python caller code is identity-
    checked. No duplicate modulation, normalization, QDQ or GEMM is computed.
    """
    def __init__(self, runtime, adapter, models, rows, save_tensor_pack, check=lambda: None):
        self.runtime, self.adapter, self.models = runtime, adapter, models
        self.rows, self.save_pack, self.check = rows, save_tensor_pack, check
        self.active = None
        self.pending = {}
        self.slot = None
        self.blocks, self.ffns, self.pre, self.heads, self.posts = [], [], [], [], []
        self.pre_mms, self.patch_convs, self.norms, self.linears, self.qdqs = [], [], [], [], []
        self._stack = None

    def begin(self, run_id, step):
        if self.active is not None or self.pending or step not in STEPS:
            raise RuntimeError("Incomplete capture or unregistered step")
        import torch
        self.active = {"run_id": run_id, "step": step, "expert_index": 0 if step == 0 else 1,
                       "blocks_start": len(self.blocks), "FFNs_start": len(self.ffns),
                       "starts": {k: len(getattr(self, k)) for k in
                           ("pre", "heads", "posts", "pre_mms", "patch_convs", "norms", "linears", "qdqs")}}
        self.cpu_rng, self.cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
        self.last_u = None
        self.head_output = None

    def selected(self, value):
        import torch
        if value.dtype != torch.bfloat16 or tuple(value.shape) != (M, D):
            raise ValueError("Actual BF16 full-M operand required")
        indices = torch.tensor(self.rows[self.active["step"]], device=value.device, dtype=torch.int64)
        return cpu_copy(value.index_select(0, indices))

    def add(self, name, value):
        if name in self.pending:
            raise RuntimeError("Duplicate actual native tap: " + name)
        self.pending[name] = cpu_copy(value)

    def __enter__(self):
        from contextlib import ExitStack
        self._stack = ExitStack()
        old_apply, old_qdq = self.runtime.apply, self.adapter.native_hif4_activation_qdq

        def qdq(value, *, force_fp32=True):
            result = old_qdq(value, force_fp32=force_fp32)
            if self.slot is not None:
                self.qdqs.append(self.slot)
            if self.slot is not None and self.slot[1] == 39 and self.slot[2].endswith(".ffn.0"):
                # Original runtime's following .to(x.dtype) is an identity here.
                if result.dtype != value.dtype:
                    raise ValueError("Original Q1 cast would change dtype: bind a different tap")
                self.add("V1", self.selected(result.reshape(-1, D)))
            return result

        def apply(mm, value, *, expert, step_index, block_index, module_name):
            selected = self.active is not None
            if selected:
                expected = "high_noise_model" if self.active["step"] == 0 else "low_noise_model"
                if expert != expected or int(step_index) != self.active["step"] or self.slot is not None:
                    raise ValueError("Native Q1 target identity changed")
                self.slot = (int(step_index), int(block_index), module_name)
            try:
                result = old_apply(mm, value, expert=expert, step_index=step_index,
                                   block_index=block_index, module_name=module_name)
                if selected:
                    self.linears.append(self.slot)
                return result
            finally:
                if selected:
                    self.slot = None
        self._stack.enter_context(patch_method(self.runtime, "apply", apply))
        self._stack.enter_context(patch_method(self.adapter, "native_hif4_activation_qdq", qdq))
        try:
            for index, model in enumerate(self.models):
                self._install_model(index, model)
        except BaseException:
            self._stack.close()
            raise
        return self

    def _install_model(self, index, model):
        infer, weights = model.transformer_infer, model.transformer_weights
        old_pre, old_block, old_ffn = model.pre_infer.infer, infer.infer_block, infer.infer_ffn
        old_non, old_norm, old_head, old_post = infer.infer_non_blocks, weights.norm.apply, weights.head.apply, model.post_infer.infer
        old_writeback = infer.post_process
        native_non_code = old_non.__func__.__code__

        def required():
            if self.active is None or index != self.active["expert_index"]:
                raise RuntimeError("Native call outside this registered source/step/expert")
            return self.active["step"]

        for name in ("time_embedding_0", "time_embedding_2", "time_projection_1", "text_embedding_0", "text_embedding_2", "patch_embedding"):
            operation = getattr(model.pre_weight, name)
            def counted_pre_op(value, _name=name, _original=operation.apply):
                step = required()
                result = _original(value)
                (self.patch_convs if _name == "patch_embedding" else self.pre_mms).append((step, _name))
                return result
            self._stack.enter_context(patch_method(operation, "apply", counted_pre_op))

        def pre(*args, **kwargs):
            step = required()
            result = old_pre(*args, **kwargs)
            if tuple(result.grid_sizes.tuple) != GRID or tuple(result.x.shape) != (M, D):
                raise ValueError("Real pre-infer geometry differs")
            self.pre.append(step)
            self.add("time_embed", result.embed)
            self.add("block_embed0", result.embed0)
            self.active["actual_pre_state"] = {name: tensor_record(getattr(result, name))
                for name in ("x", "embed", "embed0", "context", "cos_sin")}
            return result

        def block(current, x, pre_out):
            step = required()
            block_index = int(infer.block_idx)
            if block_index != len(self.blocks) - self.active["blocks_start"]:
                raise ValueError("All forty original blocks in order required")
            if ((hasattr(current.compute_phases[0], "before_proj") and current.compute_phases[0].before_proj.weight is not None)
                    or hasattr(current.compute_phases[2], "after_proj")):
                raise ValueError("Unsupported adapter projection")
            result = old_block(current, x, pre_out)
            self.blocks.append((step, block_index))
            if block_index == 39:
                self.last_u = result
                self.add("u_rows", self.selected(result))
            return result

        def ffn(phase, x, attn, shift, scale, gate=None):
            step, block_index = required(), int(infer.block_idx)
            if block_index != len(self.ffns) - self.active["FFNs_start"]:
                raise ValueError("Exactly one original FFN per original block required")
            if block_index == 39:
                if gate is None or gate.numel() != D:
                    raise ValueError("Actual final e5 must be a row-constant 5120 vector")
                self.add("gate", gate.reshape(D))
            result = old_ffn(phase, x, attn, shift, scale, gate)
            self.ffns.append((step, block_index))
            if block_index == 39:
                self.add("Y", self.selected(result))
            return result

        def writeback(x, y, gate, pre_out=None):
            required()
            if int(infer.block_idx) == 39:
                self.add("r_rows", self.selected(x))
            return old_writeback(x, y, gate, pre_out)

        def non(weights_arg, x, e):
            required()
            if weights_arg is not weights or x is not self.last_u or len(self.blocks) - self.active["blocks_start"] != 40:
                raise ValueError("Head must consume the actual final post-gate residual")
            if e.dim() != 2:
                raise ValueError("Ordinary single-step time modulation required")
            self.add("u", x)
            self.active["actual_pre_head_u"] = {"shape": list(x.shape), "dtype": str(x.dtype),
                "stride": list(x.stride()), "raw_tensor_sha256": raw_sha(self.pending["u"])}
            import torch
            if not bool(torch.isfinite(self.pending["u"]).all()):
                raise ValueError("Actual full final residual contains nonfinite values")
            self.add("head_modulation", weights.head_modulation.tensor)
            self.active["u_file"] = self.save_pack(self.active["run_id"], "u", {"u": self.pending.pop("u")})
            result = old_non(weights_arg, x, e)
            if result is not self.head_output:
                raise RuntimeError("Native head output object changed after the direct tap")
            return result

        def norm(value):
            required()
            caller = sys._getframe(1)
            if caller.f_code is not native_non_code or caller.f_locals.get("x") is not self.last_u or value is not self.last_u:
                raise ValueError("Actual native infer_non_blocks caller changed")
            e = caller.f_locals.get("e")
            if not isinstance(e, tuple) or len(e) != 2 or any(t.numel() != D for t in e):
                raise ValueError("Actual native head shift/scale identity changed")
            self.active["actual_combined_head_modulation"] = {
                "shift": tensor_record(e[0]), "scale": tensor_record(e[1])}
            self.add("head_shift", e[0])
            self.add("head_scale", e[1])
            self.norms.append(required())
            return old_norm(value)

        def head(value):
            step = required()
            if (weights.norm._get_actual_weight() is not None or weights.norm._get_actual_bias() is not None
                    or weights.norm.eps != 1e-6 or weights.head.has_lora_branch):
                raise ValueError("Original unaffined LN and Default unadapted head required")
            self.active["actual_head_operator"] = {
                "head_class": type(weights.head).__module__ + "." + type(weights.head).__qualname__,
                "head_has_lora_branch": weights.head.has_lora_branch,
                "weight": tensor_record(weights.head._get_actual_weight()),
                "bias": tensor_record(weights.head._get_actual_bias()),
                "norm_class": type(weights.norm).__module__ + "." + type(weights.norm).__qualname__,
                "norm_weight": None, "norm_bias": None, "norm_eps": weights.norm.eps,
                "infer_dtype": str(infer.infer_dtype), "sensitive_dtype": str(infer.sensitive_layer_dtype)}
            self.add("head_input_rows", self.selected(value))
            result = old_head(value)
            if tuple(result.shape) != (M, OUT):
                raise ValueError("Actual full head output shape differs")
            self.heads.append(step)
            self.add("head_output", result)
            self.head_output = result
            return result

        def post(value, pre_out):
            step = required()
            if value is not self.head_output:
                raise ValueError("Native unpatchify input changed")
            result = old_post(value, pre_out)
            self.posts.append(step)
            return result

        for owner, name, replacement in ((model.pre_infer, "infer", pre), (infer, "infer_block", block),
                (infer, "infer_ffn", ffn), (infer, "infer_non_blocks", non), (weights.norm, "apply", norm),
                (infer, "post_process", writeback), (weights.head, "apply", head), (model.post_infer, "infer", post)):
            self._stack.enter_context(patch_method(owner, name, replacement))

    def finish(self, flow, pre_latents, post_latents):
        import torch
        state = self.active
        if state is None:
            raise RuntimeError("No active flow")
        step = state["step"]
        deltas = {name: len(getattr(self, name)) - start for name, start in state["starts"].items()}
        if (self.blocks[state["blocks_start"]:] != [(step, b) for b in range(40)]
                or self.ffns[state["FFNs_start"]:] != [(step, b) for b in range(40)]
                or deltas != {"pre": 1, "heads": 1, "posts": 1, "pre_mms": 5, "patch_convs": 1,
                              "norms": 1, "linears": 400, "qdqs": 400}):
            raise ValueError("Actual native full-depth boundary ledger incomplete")
        calls = self.linears[state["starts"]["linears"]:]
        if (calls != self.qdqs[state["starts"]["qdqs"]:]
                or sum(name.endswith((".ffn.0", ".ffn.2")) for _, _, name in calls) != 80
                or {block: sum(b == block for _, b, _ in calls) for block in range(40)} != dict.fromkeys(range(40), 10)):
            raise ValueError("Actual module-level Dense linear/QDQ calls differ")
        state["actual_boundary_counts"] = deltas
        if not torch.equal(self.cpu_rng, torch.get_rng_state()) or not torch.equal(self.cuda_rng, torch.cuda.get_rng_state()):
            raise ValueError("Native observed flow changed global RNG state")
        self.add("flow", flow)
        self.add("pre_latents", pre_latents)
        self.add("post_latents", post_latents)
        self.add("rows", torch.tensor(self.rows[step], dtype=torch.int64))
        expected = {"V1", "Y", "gate", "time_embed", "block_embed0", "head_modulation", "head_shift", "head_scale",
                    "head_input_rows", "head_output", "flow", "pre_latents", "post_latents", "rows", "r_rows", "u_rows"}
        if set(self.pending) != expected or any(not bool(torch.isfinite(t).all()) for t in self.pending.values()):
            raise ValueError("Incomplete/nonfinite actual native capture")
        if (self.pending["flow"].dtype != torch.float32
                or any(t.dtype != torch.bfloat16 for name, t in self.pending.items() if name not in ("rows", "flow"))):
            raise ValueError("Actual captured native boundary is not BF16")
        # Four fixed 256x5120 CPU multiply/add checks across the whole scope;
        # no model, norm, head or QDQ call. Preserve native BF16 operation order.
        reconstructed = self.pending["r_rows"].clone()
        reconstructed.add_(self.pending["Y"] * self.pending["gate"])
        if not torch.equal(reconstructed.view(torch.uint8), self.pending["u_rows"].view(torch.uint8)):
            raise ValueError("Observed original BF16 gate/writeback does not reproduce captured u rows")
        state["capture_file"] = self.save_pack(state["run_id"], "capture", self.pending)
        state["native_noop"] = {"wrappers_return_original_objects": True,
            "extra_FFN_head_norm_QDQ_forwards": 0, "global_CPU_CUDA_RNG_unchanged": True,
            "CPU_BF16_writeback_replay_rows": len(self.rows[step]), "CPU_BF16_writeback_mul_add_byte_equal": True,
            "historical_raw_head_and_flow": "NOT_SAVED_BY_HISTORICAL_PRODUCER"}
        self.pending.clear()
        self.active, self.last_u, self.head_output = None, None, None
        return state

    def __exit__(self, *error):
        return self._stack.__exit__(*error)
