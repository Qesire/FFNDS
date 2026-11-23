"""Four-step anchor storage sweep. All GEMMs, probe solve and HHLL gates remain frozen."""
from contextlib import contextmanager
from functools import lru_cache
from types import FunctionType

import torch
import u0_int8 as old


BF16 = "wide8192"
U8 = "wide8192_u0_int8"
WIDTHS = {"u0":5120, "y0":5120, "z0k":2560}
FORMATS = {
    BF16: {},
    U8: {"u0":8},
    "wide8192_y0_int8": {"y0":8},
    "wide8192_u0_z0k_int8": {"u0":8,"z0k":8},
    "wide8192_all_int8": dict.fromkeys(WIDTHS,8),
    "wide8192_all_int4": dict.fromkeys(WIDTHS,4),
}
MODES = tuple(FORMATS)
GROUP = 128


def expected_bytes(mode, layers=40, rows=31668, probes=64):
    total = 0
    for field,width in WIDTHS.items():
        bits = FORMATS[mode].get(field)
        if bits:
            total += layers*(rows*width*bits//8 + rows*(width//GROUP)*2 + probes*width*2)
        else:
            total += layers*rows*width*2
    return total


# Adjacent channels are paired in each packed byte (even channel in the low nibble).
# Each warp encodes one aligned group of 128 channels. For z0k the selector is
# applied before the reduction, directly from native W1 output before its GELU.
CODEC = r"""
__device__ __forceinline__ float unpack(const unsigned char* bytes,const B* scales,int i) {
#if QBITS == 8
    int q = ((const signed char*)bytes)[i];
#else
    int q = (bytes[i>>1] >> ((i&1)*4)) & 15;
    if (q >= 8) q -= 16;
#endif
    return __fmul_rn((float)q,f32(scales[i>>7]));
}
extern "C" __global__ void encode(const B* src, const long long* columns,
    unsigned char* dst, B* scales, int width, int input_width, int selected, int groups) {
    int g = (blockIdx.x*blockDim.x+threadIdx.x)>>5;
    if (g >= groups) return;
    int lane = threadIdx.x & 31;
    int pos = g*128 + 2*lane;
    int row = pos/width;
    int col = pos%width;
    float values[4], a = 0.0f;
    #pragma unroll
    for (int j=0;j<4;++j){
        int off = 64*(j>>1) + (j&1);
        int channel = selected ? (int)columns[col+off] : col+off;
        values[j] = f32(src[row*input_width+channel]);
        a = fmaxf(a,fabsf(values[j]));
    }
    #pragma unroll
    for (int stride=16;stride;stride>>=1) a=fmaxf(a,__shfl_down_sync(0xffffffffu,a,stride));
    a=__shfl_sync(0xffffffffu,a,0);
    const int limit=(1<<(QBITS-1))-1;
    B sb=bf16(a==0.0f ? 1.0f : fmaxf(__fdiv_rn(a,(float)limit),0x1p-126f));
    if (lane==0) scales[g]=sb;
    int q[4];
    #pragma unroll
    for (int j=0;j<4;++j) {
        int n=__float2int_rn(__fdiv_rn(values[j],f32(sb)));
        q[j]=n < -limit ? -limit : (n>limit ? limit : n);
    }
    #pragma unroll
    for (int p=0;p<2;++p) {
        int i=pos+64*p;
#if QBITS == 8
        dst[i]=(unsigned char)(signed char)q[2*p];
        dst[i+1]=(unsigned char)(signed char)q[2*p+1];
#else
        dst[i>>1]=(unsigned char)((q[2*p]&15) | ((q[2*p+1]&15)<<4));
#endif
    }
}
extern "C" __global__ void diff(const B* now,const unsigned char* code,const B* scales,B* db,float* df,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n){float d=__fsub_rn(f32(now[i]),unpack(code,scales,i));db[i]=bf16(d);df[i]=d;}
}
extern "C" __global__ void base(const unsigned char* code,const B* scales,const B* linear,float* selected,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) selected[i]=__fadd_rn(__fadd_rn(unpack(code,scales,i),f32(linear[i])),selected[i]);
}
extern "C" __global__ void secant(const B* z,const unsigned char* code,const B* scales,const B* table,
    const float* beta,const float* scale,B* db,float* df,float* x,int n,int k){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n){int j=i%k;B z1=z[i],z0=bf16(unpack(code,scales,i));
        B b=bf16(f32(table[z1])-f32(table[z0]));float d=f32(b);
        db[i]=b;df[i]=d;x[i]=__fsub_rn(d,__fmul_rn(f32(z1)-f32(z0),beta[j]))/scale[j];}
}
"""


@lru_cache(None)
def kernels(device, bits):
    from fused_ops import HEADER
    if bits not in (4,8):
        raise ValueError("Invalid packed width")
    source = f"#define QBITS {bits}\n" + HEADER + CODEC
    with torch.cuda.device(device):
        return {name:torch.cuda._compile_kernel(source,name) for name in ("encode","diff","base","secant")}


class Quantized:
    def __init__(self, layers, rows, width, bits, probes):
        dev = probes.device
        self.width, self.bits, self.probes = width,bits,probes
        self.code = torch.empty((layers,rows,width*bits//8),device=dev,dtype=torch.uint8)
        self.scale = torch.empty((layers,rows,width//GROUP),device=dev,dtype=torch.bfloat16)
        self.exact = torch.empty((layers,len(probes),width),device=dev,dtype=torch.bfloat16)
        self.reads = self.stores = 0

    def write(self, layer, value, selector=None):
        if (value.dtype != torch.bfloat16 or value.device != self.code.device or not value.is_contiguous()
                or value.shape[0] != self.code.shape[1] or self.width % GROUP):
            raise ValueError("Unexpected source or layout for group quantization")
        if selector is None:
            if value.shape[1] != self.width:
                raise ValueError("Source width differs")
            torch.index_select(value,0,self.probes,out=self.exact[layer])
        else:
            if (selector.shape != (self.width,) or selector.dtype != torch.long or not selector.is_contiguous()
                    or selector.device != value.device):
                raise ValueError("Unexpected preactivation selector")
            probe_rows = value.index_select(0,self.probes)
            torch.index_select(probe_rows,1,selector,out=self.exact[layer])
        scale = self.scale[layer]
        kernel = kernels(value.device.index,self.bits)["encode"]
        kernel(grid=((scale.numel()+7)//8,1,1),block=(256,1,1),
               args=[value,self.probes if selector is None else selector,self.code[layer],scale,
                     self.width,value.shape[1],int(selector is not None),scale.numel()])
        self.stores += 1


class ReadRows:
    def __init__(self, field, layer):
        self.field, self.layer = field,layer

    def is_contiguous(self):
        return True

    def __getitem__(self,index):
        f,l=self.field,self.layer
        if index is f.probes:
            f.reads += 1
            return f.exact[l]
        if not isinstance(index,slice) or index.step not in (None,1):
            raise ValueError("Unsupported selection in sealed wide8192 reader")
        return f.code[l,index],f.scale[l,index]


def dispatch(original, fused, at):
    def launch(*,grid,block,args):
        source=args[at]
        if isinstance(source,tuple):
            fused(grid=grid,block=block,args=[*args[:at],*source,*args[at+1:]])
        else:
            original(grid=grid,block=block,args=args)
    return launch


def prepare_static(phase,pack,device,formats):
    import exact_port
    prepared = exact_port.prepare(phase,pack,device)
    device = torch.device(device).index
    for field,name,at in (("u0","difference",1),("y0","base",0)):
        if field in formats:
            kernel = kernels(device,formats[field])["diff" if field=="u0" else "base"]
            prepared[name]=dispatch(prepared[name],kernel,at)
    if "z0k" in formats:
        table,native=prepared["lut"]
        prepared["lut"]=(table,dispatch(native,kernels(device,formats["z0k"])["secant"],1))
    return prepared


class Slots:
    def __init__(self, port, mode, layers, rows, width, selected, device):
        if (layers,rows,width,selected,len(port.probes)) != (40,31668,5120,2560,64):
            raise ValueError("Different anchor geometry")
        self.port = port
        self.owner = [None]*layers
        self.fields = {}
        self.storage = {}
        self.formats = FORMATS[mode]
        for name,w in WIDTHS.items():
            if name in self.formats:
                field = Quantized(layers,rows,w,self.formats[name],port.probes)
                arrays = {name+"_codes":field.code,name+"_scales":field.scale,name+"_probes":field.exact}
            else:
                field = torch.empty((layers,rows,w),dtype=torch.bfloat16,device=device)
                arrays = {name:field}
            self.fields[name]=field
            self.storage.update({k:v.numel()*v.element_size() for k,v in arrays.items()})
        self.bytes=sum(self.storage.values())
        assert self.bytes==expected_bytes(mode)
        self.anchors = 0

    def write(self,name,layer,value,selector=None):
        field = self.fields[name]
        if name in self.formats:
            field.write(layer,value,selector)
        elif selector is None:
            field[layer].copy_(value)
        else:
            torch.index_select(value,1,selector,out=field[layer])

    def anchor(self,expert,layer,u,phase,selector):
        if self.owner[layer] is not None:
            raise RuntimeError("Unconsumed anchor")
        if (not self.port.busy or not self.port.bound or torch.is_grad_enabled()
                or torch.is_autocast_enabled() or not u.is_contiguous()):
            raise RuntimeError("Unexpected native anchor context")
        z=phase.ffn_0.apply(u)
        if (type(z) is not torch.Tensor or z.dtype!=u.dtype or z.device!=u.device
                or z.shape!=(31668,13824) or not z.is_contiguous()):
            raise RuntimeError("Unexpected native W1 layout")
        self.write("z0k",layer,z,selector)
        torch.ops.aten.gelu.out(z,approximate="tanh",out=z)
        if "y0" in self.formats:
            y=phase.ffn_2.apply(z)
            self.write("y0",layer,y)
        else:
            y=self.fields["y0"][layer]
            mm=phase.ffn_2
            if (type(mm) is not self.mm_type or getattr(mm.apply,"__func__",None) is not self.mm_type.apply
                    or getattr(mm,"has_lora_branch",False) or getattr(mm,"has_diff",False)
                    or hasattr(mm,"weight_diff") or hasattr(mm,"bias_diff")
                    or getattr(mm._get_actual_weight,"__func__",None) is not self.mm_template._get_actual_weight
                    or getattr(mm._get_actual_bias,"__func__",None) is not self.mm_template._get_actual_bias):
                raise RuntimeError("Direct W2 requires the audited native implementation")
            weight,bias=mm._get_actual_weight(),mm._get_actual_bias()
            if (weight is not mm.weight or bias is not mm.bias or weight.shape!=(13824,5120)
                    or bias.shape!=(5120,) or weight.dtype!=u.dtype or bias.dtype!=u.dtype
                    or weight.device!=u.device or bias.device!=u.device or y.data_ptr()%256
                    or tuple(weight.stride()) not in ((1,13824),(5120,1)) or not bias.is_contiguous()):
                raise RuntimeError("Unexpected direct W2 operands")
            if getattr(mm,"create_cuda_buffer",False) and (weight.data_ptr()!=mm.weight_cuda_buffer.data_ptr()
                                                          or bias.data_ptr()!=mm.bias_cuda_buffer.data_ptr()):
                raise RuntimeError("Invalid native W2 staging buffer")
            torch.addmm(bias,z,weight,out=y)
        self.write("u0",layer,u)
        self.owner[layer]=expert
        self.anchors+=1
        return y

    def take(self,expert,layer):
        if self.owner[layer]!=expert:
            raise RuntimeError("Wrong expert for anchor")
        self.owner[layer]=None
        return {k:(ReadRows(v,layer) if k in self.formats else v[layer]) for k,v in self.fields.items()}


class Port:
    def __init__(self,inner,mode):
        self.inner,self.variant=inner,mode

    def bind(self,owner,transformer_sha):
        return self.inner.bind(owner,transformer_sha)

    @contextmanager
    def trajectory(self):
        slots=self.inner.slots
        with self.inner.trajectory():
            slots.anchors=0
            for name in slots.formats:
                slots.fields[name].reads=slots.fields[name].stores=0
            yield self
            if slots.anchors!=80 or any((slots.fields[n].stores,slots.fields[n].reads)!=(80,80)
                                        for n in slots.formats):
                raise RuntimeError("Incomplete compressed anchor trajectory")

    @property
    def counters(self):
        s=self.inner.slots
        temporary="y0" in s.formats
        return dict(self.inner.counters,variant=self.variant,tile_rows=8192,group_size=GROUP,
                    anchor_gpu_capacity_bytes=s.bytes,anchor_storage_bytes=s.storage,
                    anchor_direct=0 if temporary else s.anchors,anchor_temporary=s.anchors if temporary else 0,
                    anchor_fallback=0,encoded_anchors=s.anchors,exact_probe_hits=min(s.fields[k].reads for k in s.formats),
                    exact_probe_reads_by_field={k:s.fields[k].reads for k in s.formats})


def make_port(scope,mode):
    import exact_port
    import fourstep_anchor
    import fourstep_backend
    import fourstep_hit
    import q1_fused_post
    import resident
    from lightx2v.common.ops.mm.mm_weight import MMWeight,MMWeightTemplate
    old._checked_source(old,"40c54474f33ac0f45e22395c69fd4c12c76611e598f1ac87e425c3edd3cb30a8")
    if mode==BF16:
        return fourstep_backend.make_port(scope,BF16)
    if mode==U8:
        return old.make_port(scope)
    formats=FORMATS[mode]
    old._checked_source(fourstep_anchor,old.ANCHOR_SHA)
    old._checked_source(fourstep_hit,old.HIT_SHA)
    old._checked_source(exact_port,fourstep_anchor.SOURCES["exact_port"])
    old._checked_source(resident,fourstep_anchor.SOURCES["resident"])
    old._checked_source(q1_fused_post,fourstep_anchor.SOURCES["q1_fused_post"])
    if fourstep_anchor._fingerprint(MMWeight)!=fourstep_anchor.SOURCES["mm_weight"]:
        raise RuntimeError("Unexpected LightX2V W2 implementation")
    if scope["residency"]["tile_rows"]!=exact_port.TILE:
        raise RuntimeError("Unexpected source configuration")
    inner=exact_port.ResidentPort.__new__(exact_port.ResidentPort)
    def allocate(layers,rows,width,selected,device):
        return Slots(inner,mode,layers,rows,width,selected,device)
    init=resident.ResidentJoint.__init__
    fn=FunctionType(init.__code__,dict(init.__globals__,AnchorSlots=allocate),init.__name__,
                    init.__defaults__,init.__closure__)
    fn.__kwdefaults__=init.__kwdefaults__
    fn(inner,exact_port.site_loader(scope),scope["observer_ids"],
       lambda phase,pack,device:prepare_static(phase,pack,device,formats),fourstep_hit.make_hit(8192,reuse_base=True))
    inner.mode="fast512"
    inner.slots.mm_type,inner.slots.mm_template=MMWeight,MMWeightTemplate
    return Port(inner,mode)
