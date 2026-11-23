"""Joint fast INT8 y0 writer and native BF16 residual; one reusable video port."""
from contextlib import ExitStack, contextmanager
from functools import lru_cache

import torch

import anchor_variants
import fast_cache8
from fused_ops import HEADER


WIDTH = 5120
REFERENCE = "fast"
VARIANT = "joined"

JOINED = r"""
extern "C" __global__ void write_y0_and_gate(
    unsigned long long* x, const unsigned long long* y, const unsigned long long* gate,
    unsigned int* codes, B* scales, unsigned long long* exact, const long long* probes,
    int groups, int probe_groups) {
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp >= groups) {
        int g = warp - groups;
        if (g < probe_groups) {
            int row = (int)probes[g / (WIDTH / 128)];
            int quad = (g % (WIDTH / 128)) * 32 + lane;
            exact[g * 32 + lane] = y[row * (WIDTH / 4) + quad];
        }
        return;
    }
    int pos = warp * 32 + lane;
    unsigned long long yy = y[pos], xx = x[pos], gg = gate[pos % (WIDTH / 4)];
    unsigned long long residual = 0;
    float values[4], maximum = 0.0f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int shift = 16 * i;
        float v = f32((B)(yy >> shift));
        values[i] = v;
        maximum = fmaxf(maximum, fabsf(v));
        B gated = bf16(__fmul_rn(v, f32((B)(gg >> shift))));
        B updated = bf16(__fadd_rn(f32((B)(xx >> shift)), f32(gated)));
        residual |= (unsigned long long)updated << shift;
    }
    x[pos] = residual;
    #pragma unroll
    for (int offset = 16; offset; offset >>= 1) {
        maximum = fmaxf(maximum, __shfl_down_sync(0xffffffffu, maximum, offset));
    }
    float reciprocal = 0.0f;
    if (lane == 0) {
        B s = bf16(maximum == 0.0f ? 1.0f :
                   fmaxf(__fdiv_rn(maximum, 127.0f), 0x1p-126f));
        scales[warp] = s;
        reciprocal = __frcp_rn(f32(s));
    }
    reciprocal = __shfl_sync(0xffffffffu, reciprocal, 0);
    unsigned int packed = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        packed |= encode_value(values[i], reciprocal) << (8 * i);
    }
    codes[pos] = packed;
}
"""


@lru_cache(None)
def kernel(device):
    # Reuse the CPU- and GPU-verified BF16-ratio rounding helper from the writer.
    defines = f"#define WIDTH {WIDTH}\n#define INPUT_WIDTH {WIDTH}\n#define SELECTED 0\n"
    with torch.cuda.device(device):
        return torch.cuda._compile_kernel(defines + HEADER + fast_cache8.SOURCE + JOINED,
                                         "write_y0_and_gate")


def encode_gate(x, y, gate, field, layer):
    if (field.width != WIDTH or field.bits != 8 or not 0 <= layer < len(field.code)
            or x.shape != y.shape or y.shape != field.code.shape[1:]
            or gate.squeeze().shape != (WIDTH,) or not x.is_cuda
            or any(t.device != field.code.device or t.dtype != torch.bfloat16
                   or not t.is_contiguous() or t.data_ptr() % 8 for t in (x, y, gate))
            or torch._C._overlaps(x, y) or torch._C._overlaps(x, gate)):
        raise ValueError("Expected non-overlapping residual, native BF16 output and channel gate")
    scales = field.scale[layer]
    groups = scales.numel()
    probe_groups = field.probes.numel() * (WIDTH // 128)
    kernel(x.device.index)(grid=((groups + probe_groups + 7) // 8, 1, 1), block=(256, 1, 1),
                           args=[x, y, gate, field.code[layer], scales, field.exact[layer],
                                 field.probes, groups, probe_groups])
    field.stores += 1


class JoinedPort:
    def __init__(self, base):
        self.base = base
        self.inner = base.inner
        self.fields = self.inner.slots.fields
        self.variant = REFERENCE
        self.pending = None
        self.writes = dict.fromkeys(self.fields, 0)
        self.joined = 0

    def set_mode(self, mode):
        if self.inner.bound or self.inner.busy or self.pending is not None:
            raise RuntimeError("Choose the writer between trajectories")
        if mode not in (REFERENCE, VARIANT):
            raise ValueError(f"Unknown writer mode {mode}")
        self.variant = mode

    @property
    def counters(self):
        return dict(self.base.counters, variant=self.variant, fast_writes=dict(self.writes),
                    joined_y0=self.joined, joined_fallback=0)

    @contextmanager
    def trajectory(self):
        with self.base.trajectory():
            self.writes = dict.fromkeys(self.fields, 0)
            self.joined = 0
            try:
                yield self
                expected = 80 if self.variant == VARIANT else 0
                if (self.joined != expected or self.pending is not None or
                        self.writes != {"u0": 80, "z0k": 80, "y0": 80 - expected}):
                    raise RuntimeError("Incomplete or overlapping writer routes")
            finally:
                self.pending = None

    def _writer(self, name):
        def write(layer, value, selector=None):
            if fast_cache8.write(self.fields[name], layer, value, selector) is not True:
                raise RuntimeError(f"Fast {name} dispatch fell back")
            self.writes[name] += 1
        return write

    def _defer(self, layer, value, selector=None):
        if self.pending is not None or selector is not None:
            raise RuntimeError("Unexpected pending or selected W2 output")
        self.pending = (layer, value, self.inner.execution_stream)

    def _gated_post(self, original):
        def post(model, x, y, gate, pre_infer_out=None):
            if self.pending is None:
                return original(model, x, y, gate, pre_infer_out)
            layer, output, producer = self.pending
            if (not self.inner.busy or int(model.block_idx) != layer or y is not output
                    or gate is None or model.clean_cuda_cache
                    or model.sensitive_layer_dtype != torch.bfloat16 or model.infer_dtype != torch.bfloat16):
                raise RuntimeError("Unexpected native anchor gate; refusing an unmeasured fallback")
            stream = torch.cuda.current_stream(self.inner.device)
            if stream != producer:
                stream.wait_stream(producer)
                output.record_stream(stream)
                self.inner.execution_stream = stream
            encode_gate(x, y, gate, self.fields["y0"], layer)
            self.pending = None
            self.joined += 1
            self.inner.counters["gate_fused"] += 1
            return x
        return post

    @contextmanager
    def bind(self, owner, transformer_sha):
        with self.base.bind(owner, transformer_sha), ExitStack() as restore:
            for name, field in self.fields.items():
                if "write" in vars(field):
                    raise RuntimeError(f"{name} already has a writer hook")
                field.write = self._defer if name == "y0" and self.variant == VARIANT else self._writer(name)
                restore.callback(delattr, field, "write")
            if self.variant == VARIANT:
                original = owner.post_process
                owner.post_process = self._gated_post(original)
                restore.callback(setattr, owner, "post_process", original)
            yield self


def make_port(scope):
    anchor_variants.old._checked_source(anchor_variants,
        "18a343650060bed49fe6c6842db3a8c6f7885d92d19742ba110b8640c91d1d96")
    return JoinedPort(anchor_variants.make_port(scope, "wide8192_all_int8"))
