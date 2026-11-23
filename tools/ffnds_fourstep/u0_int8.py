"""Four-step wide8192 with INT8 u0 slots and exact BF16 probe sidecars.

Candidate for isolated, serialized school inference. The static packs, TF32
policy, native W2 direct write, gate, GLS solve and 8192-row hit GEMMs come
from the pinned wide8192 release.
"""
from contextlib import contextmanager
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from types import FunctionType

import torch


GROUP = 128
VARIANT = "wide8192_u0_int8"
ANCHOR_SHA = "ef91b5a4d41f23353358ee0cf49102acad9dc462566b987a8015aba88b7da87f"
HIT_SHA = "eee084fca85d49331625c67d9bf899df58a5e987cc3886454f8e0c37b2c31a2d"


_ENCODE = """
extern "C" __global__ void encode_u0(const B* u, signed char* codes, B* scales, int groups) {
    int group = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (group >= groups) return;
    int base = group * 128 + lane;
    float values[4];
    float mx = 0.0f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        values[i] = f32(u[base + 32*i]);
        mx = fmaxf(mx, fabsf(values[i]));
    }
    #pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1)
        mx = fmaxf(mx, __shfl_down_sync(0xffffffffu, mx, delta));
    mx = __shfl_sync(0xffffffffu, mx, 0);
    B scale_bits = bf16(mx == 0.0f ? 1.0f : fmaxf(__fdiv_rn(mx,127.0f), 0x1p-126f));
    float scale = f32(scale_bits);
    if (lane == 0) scales[group] = scale_bits;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int q = __float2int_rn(__fdiv_rn(values[i], scale));
        codes[base + 32*i] = (signed char)(q < -127 ? -127 : (q > 127 ? 127 : q));
    }
}
"""

_DIFFERENCE = """
extern "C" __global__ void u0_difference(const B* now, const signed char* codes,
    const B* scales, B* dub, float* du, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float a = __fmul_rn((float)codes[i], f32(scales[i >> 7]));
    float d = __fsub_rn(f32(now[i]), a);
    dub[i] = bf16(d);
    du[i] = d;
}
"""


@lru_cache(None)
def _kernels(device):
    from fused_ops import HEADER
    with torch.cuda.device(device):
        return (torch.cuda._compile_kernel(HEADER + _ENCODE, "encode_u0"),
                torch.cuda._compile_kernel(HEADER + _DIFFERENCE, "u0_difference"))


def encode_u0(u, codes, scales):
    """Contiguous BF16 [rows,width] -> INT8 codes and one BF16 scale per 128."""
    if (u.ndim != 2 or u.shape[1] % GROUP or codes.shape != u.shape
            or scales.shape != (u.shape[0], u.shape[1]//GROUP)
            or (u.dtype, codes.dtype, scales.dtype) != (torch.bfloat16, torch.int8, torch.bfloat16)
            or not u.is_cuda or codes.device != u.device or scales.device != u.device
            or not all(t.is_contiguous() for t in (u, codes, scales))):
        raise ValueError("Invalid u0 encoder operands")
    kernel, _ = _kernels(u.device.index)
    groups = scales.numel()
    kernel(grid=((groups+7)//8,1,1),block=(256,1,1),args=[u,codes,scales,groups])


def store_u0(u, probes, codes, scales, probe_u0):
    encode_u0(u,codes,scales)
    torch.index_select(u,0,probes,out=probe_u0)


def decoded_difference(now,codes,scales,dub,du):
    """Decode one contiguous tile directly into the two original difference operands."""
    if (now.ndim != 2 or now.shape[1] % GROUP or not now.is_cuda or
            codes.shape != now.shape or dub.shape != now.shape or du.shape != now.shape or
            scales.shape != (now.shape[0],now.shape[1]//GROUP) or
            (now.dtype,codes.dtype,scales.dtype,dub.dtype,du.dtype) !=
            (torch.bfloat16,torch.int8,torch.bfloat16,torch.bfloat16,torch.float32) or
            any(t.device != now.device or not t.is_contiguous() for t in (now,codes,scales,dub,du))):
        raise ValueError("Invalid compressed u0 difference operands")
    _, kernel = _kernels(now.device.index)
    n = now.numel()
    kernel(grid=((n+255)//256,1,1),block=(256,1,1),args=[now,codes,scales,dub,du,n])


class _CachedRows:
    """Row interface of the frozen hit: exact probes or an aligned full tile."""
    def __init__(self, slots, layer):
        self.slots = slots
        self.layer = layer

    def is_contiguous(self):
        return True

    def __getitem__(self, index):
        if index is self.slots.probes:
            self.slots.probe_reads += 1
            return self.slots.probe_u0[self.layer]
        if not isinstance(index,slice) or index.step not in (None,1):
            raise ValueError("Unsupported wide8192 row selection")
        return self.slots.u0[self.layer,index], self.slots.scales[self.layer,index]


def _prepare(phase, pack, device):
    import exact_port
    prepared = exact_port.prepare(phase,pack,device)
    native = prepared["difference"]
    _, compressed = _kernels(torch.device(device).index)

    def difference(*,grid,block,args):
        now, source, dub, du, n = args
        if isinstance(source,tuple):
            codes, scales = source
            compressed(grid=grid,block=block,args=[now,codes,scales,dub,du,n])
        else:
            native(grid=grid,block=block,args=args)

    prepared["difference"] = difference
    return prepared


class _Slots:
    def __init__(self, layers, rows, width, selected, device, probes):
        if (layers,rows,width,selected,probes.numel()) != (40,31668,5120,2560,64):
            raise RuntimeError("Unsupported frozen four-step geometry")
        self.probes = probes
        self.u0 = torch.empty((layers,rows,width),dtype=torch.int8,device=device)
        options = dict(dtype=torch.bfloat16,device=device)
        self.scales = torch.empty((layers,rows,width//GROUP),**options)
        self.probe_u0 = torch.empty((layers,len(probes),width),**options)
        self.y0 = torch.empty((layers,rows,width),**options)
        self.z0k = torch.empty((layers,rows,selected),**options)
        fields = dict(codes=self.u0,scales=self.scales,probes=self.probe_u0,y0=self.y0,z0k=self.z0k)
        self.storage = {k:v.numel()*v.element_size() for k,v in fields.items()}
        self.bytes = sum(self.storage.values())
        self.owner = [None]*layers
        self.encoded = self.probe_reads = 0

    def store_input(self, layer, u):
        store_u0(u,self.probes,self.u0[layer],self.scales[layer],self.probe_u0[layer])
        self.encoded += 1

    def anchor(self, *args):
        # The generic BF16 slot fallback cannot write a valid compressed anchor.
        raise RuntimeError("Compressed anchor requires the verified native W2 direct-write path")

    def take(self, expert, layer):
        if self.owner[layer] != expert:
            raise RuntimeError("Missing compressed anchor")
        self.owner[layer] = None
        return dict(u0=_CachedRows(self,layer),y0=self.y0[layer],z0k=self.z0k[layer])


def _checked_source(module, expected):
    data = Path(module.__file__).read_bytes().replace(b"\r\n",b"\n")
    if sha256(data).hexdigest() != expected:
        raise RuntimeError("Incompatible school module: "+module.__name__)
    return data.decode()


@lru_cache(None)
def _anchor_binder():
    """Specialize the sealed guard and the u0 store; retain native MM and alias guards."""
    import fourstep_anchor
    source = _checked_source(fourstep_anchor,ANCHOR_SHA)
    edits = (
        ("type(slots) is not resident.AnchorSlots", "type(slots) is not _Slots", 1),
        ("is not resident.AnchorSlots.anchor", "is not _Slots.anchor", 1),
        ("slots.u0.dtype != u.dtype", "slots.u0.dtype != torch.int8", 1),
        ("slots.u0[layer].copy_(u)", "slots.store_input(layer, u)", 3),
    )
    for old,new,count in edits:
        if source.count(old) != count:
            raise RuntimeError("Unexpected anchor implementation: "+old)
        source = source.replace(old,new)
    namespace = dict(__name__=__name__,_Slots=_Slots)
    exec(compile(source,"<school-u0-int8-anchor>","exec"),namespace)
    return namespace["bind_anchor"]


class _Port:
    variant = VARIANT

    def __init__(self,inner,binder):
        self.inner = inner
        self.binder = binder
        self.anchor_stats = {}

    @contextmanager
    def trajectory(self):
        slots = self.inner.slots
        with self.inner.trajectory():
            slots.encoded = slots.probe_reads = 0
            self.anchor_stats = {}
            with self.binder(self.inner,self.anchor_stats):
                if not self.anchor_stats.get("source_guard_ok"):
                    raise RuntimeError("Anchor guard: "+str(self.anchor_stats.get("source_guard_reason")))
                yield self
            if (slots.encoded,slots.probe_reads,self.anchor_stats["applied"],self.anchor_stats["fallback"]) != (80,80,80,0):
                raise RuntimeError("Compressed path did not execute for all anchor and hit sites")

    def bind(self,owner,transformer_sha):
        return self.inner.bind(owner,transformer_sha)

    @property
    def counters(self):
        slots = self.inner.slots
        return dict(self.inner.counters,variant=VARIANT,tile_rows=8192,group_size=GROUP,
                    anchor_direct=self.anchor_stats.get("applied",0),
                    anchor_fallback=self.anchor_stats.get("fallback",0),
                    encoded_anchors=slots.encoded,exact_probe_hits=slots.probe_reads,
                    anchor_storage_bytes=dict(slots.storage),anchor_gpu_capacity_bytes=slots.bytes)


def make_port(scope):
    """Initialize the same resident port using compressed storage from its first allocation."""
    import exact_port
    import fourstep_anchor
    import fourstep_hit
    import resident
    _checked_source(exact_port,fourstep_anchor.SOURCES["exact_port"])
    _checked_source(resident,fourstep_anchor.SOURCES["resident"])
    _checked_source(fourstep_hit,HIT_SHA)
    binder = _anchor_binder()
    if scope["residency"]["tile_rows"] != exact_port.TILE:
        raise ValueError("Unexpected historical resident scope")
    inner = exact_port.ResidentPort.__new__(exact_port.ResidentPort)

    def allocate(layers,rows,width,selected,device):
        return _Slots(layers,rows,width,selected,device,inner.probes)

    # Narrow factory injection keeps the production class, initialization,
    # lifecycle and conservative free-memory admission check unchanged.
    init = resident.ResidentJoint.__init__
    namespace = dict(init.__globals__,AnchorSlots=allocate)
    initialize = FunctionType(init.__code__,namespace,init.__name__,init.__defaults__,init.__closure__)
    initialize.__kwdefaults__ = init.__kwdefaults__
    initialize(inner,exact_port.site_loader(scope),scope["observer_ids"],_prepare,
               fourstep_hit.make_hit(8192,reuse_base=True))
    inner.mode = "fast512"
    return _Port(inner,binder)
