"""Four-step FFNDS direct packed-input producer; import does not access CUDA.

prepare / prepare_from_tensors / hit mirror exact_port's prepared hit interface.
Two existing pointwise formulas write BF16 dub/dh into the left/right columns
of one [rows,d+k] allocation. Their FP32 du, dh32 and normalized secant remain
separate and unchanged. Only the BF16 destination indexing gains a row pitch.
No online torch.cat of dub/dh. Standard library GEMM consumes the packed input.

The 64-row native probe path and original FP64 GLS solve remain unchanged.
The direct base merges A0/W2k transport and removes A0's intermediate BF16
rounding; this is a numerical candidate, NOT an equivalent optimization.
mode='strict' supplies the original base arithmetic as an explicit fallback.
mode='cat' is the matched comparator: original producers, online concatenation,
and the identical merged GEMM/backend; it isolates the pitched-store change.
base_backend='auto' tries BF16-input/FP32-output addmm during preparation and
records a fallback to mm+FP32 add if that dtype/API is unsupported. No fallback
decision, kernel compilation, or weight packing occurs inside timed hit.

Packed weights are prepared once. A0/W2k views avoid retaining duplicate native
copies inside the returned dictionary. Callers retaining the original prepared
dictionary still hold its original allocations. A larger tile independently
changes GEMM shapes. All outputs/precision/latency require real GPU validation.
"""
from functools import lru_cache
from importlib import import_module

MODES=('strict','cat','direct')
BACKENDS=('auto','addmm','mm_add')
DIFFERENCE=r'''
extern "C" __global__ void difference_packed(const B* u,const B* a,B* db,float* df,int n,int d,int pitch){
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<n){float delta=f32(u[i])-f32(a[i]);db[(i/d)*pitch+i%d]=bf16(delta);df[i]=delta;}
}'''
LOOKUP=r'''
extern "C" __global__ void lookup_packed(const B* z,const B* a,const B* table,
 const float* beta,const float* scale,B* db,float* df,float* x,int n,int k,int offset,int pitch){
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<n){int j=i%k;B z1=z[i],z0=a[i];
  B b=bf16(f32(table[z1])-f32(table[z0]));float delta=f32(b);
  db[(i/k)*pitch+offset+j]=b;df[i]=delta;
  x[i]=__fsub_rn(delta,__fmul_rn(f32(z1)-f32(z0),beta[j]))/scale[j];}
}'''


def build_spec(*,mode='direct',tile_rows=8192,base_backend='auto'):
    if mode not in MODES or base_backend not in BACKENDS: raise ValueError('Unknown mode/backend')
    if not isinstance(tile_rows,int) or isinstance(tile_rows,bool) or tile_rows<1: raise ValueError('Positive integer tile_rows required')
    return dict(version='FFNDS_FOURSTEP_DIRECT_PRODUCER_V1',mode=mode,tile_rows=tile_rows,requested_backend=base_backend,
                probe_path='native64 unchanged; current g/C recomputed',packed_input='BF16 [dub | dh], row pitch d+k',
                fp32_features='original du,dh32,normalized secant',linear_intermediate_bf16_rounding=(mode=='strict'))


@lru_cache(None)
def _runtime():
    try:
        ep=import_module('exact_port')
    except ModuleNotFoundError as error:
        if error.name!='exact_port': raise
        # Offline validation alone configures its sealed source tree.
        helper=import_module('validate_int8_transport')
        return helper.v.ep,helper.torch
    return ep,import_module('torch')


@lru_cache(None)
def _kernels(device_index):
    ep,torch=_runtime()
    with torch.cuda.device(device_index):
        return (torch.cuda._compile_kernel(ep.HEADER+DIFFERENCE,'difference_packed'),
                torch.cuda._compile_kernel(ep.HEADER+LOOKUP,'lookup_packed'))


def prepare(phase,pack,device,*,mode='direct',tile_rows=8192,base_backend='auto'):
    build_spec(mode=mode,tile_rows=tile_rows,base_backend=base_backend); ep,_=_runtime()
    return prepare_from_tensors(ep.prepare(phase,pack,device),mode=mode,tile_rows=tile_rows,base_backend=base_backend)


def prepare_from_tensors(tensors,*,mode='direct',tile_rows=8192,base_backend='auto'):
    spec=build_spec(mode=mode,tile_rows=tile_rows,base_backend=base_backend); ep,torch=_runtime(); t=dict(tensors)
    state=dict(spec=spec,packed_weight=None,backend='strict',backend_note=None)
    if mode!='strict':
        a,w=t['A0'],t['w2k']; d=a.shape[0]
        if a.shape!=(d,d) or w.shape!=(t['zmap'].shape[0],d) or a.dtype!=torch.bfloat16 or w.dtype!=torch.bfloat16 or a.device!=w.device or not a.is_contiguous() or not w.is_contiguous():
            raise ValueError('Expected contiguous BF16 A0 and W2k on the same device')
        packed=torch.cat((a,w),dim=0).contiguous(); t['A0'],t['w2k']=packed[:d],packed[d:]
        assert t['A0'].stride()==a.stride() and t['w2k'].stride()==w.stride()
        state.update(packed_weight=packed,packed_weight_bytes=packed.numel()*packed.element_size(),kernels=_kernels(a.device.index),backend='mm_add')
        if base_backend!='mm_add':
            try:
                trial=torch.addmm(torch.zeros((1,d),device=a.device,dtype=a.dtype),torch.zeros((1,d+w.shape[0]),device=a.device,dtype=a.dtype),packed,out_dtype=torch.float32)
                if trial.dtype!=torch.float32: raise TypeError('addmm did not return FP32')
                state['backend']='addmm'; del trial
            except (TypeError,NotImplementedError,RuntimeError) as error:
                if isinstance(error,RuntimeError) and not any(s in str(error).lower() for s in ('dtype','out_dtype','not implemented','not supported','unsupported','same type')): raise
                if base_backend=='addmm': raise
                state['backend_note']='addmm unsupported; explicit mm_add fallback: '+str(error)
    t['_fourstep_direct']=state
    return t


def operands(now,u0,y0,z0,t,*,native_probe=False,audit=False):
    """Exposed for untimed producer bitwise audits, including final-tail shapes."""
    ep,torch=_runtime(); state=t['_fourstep_direct']; d=now.shape[1]; k=t['zmap'].shape[0]
    packed_path=state['spec']['mode']!='strict' and not native_probe
    direct=state['spec']['mode']=='direct' and not native_probe
    du=torch.empty_like(now,dtype=torch.float32)
    if direct:
        packed=torch.empty((len(now),d+k),device=now.device,dtype=now.dtype)
        ep.apply(state['kernels'][0],now.numel(),now,u0,packed,du,now.numel(),d,d+k)
    else:
        dub=torch.empty_like(now); ep.apply(t['difference'],du.numel(),now,u0,dub,du,du.numel())
        if not packed_path: linear=dub@t['A0']
    ju=du@t['jmap']; z=torch.nn.functional.linear(now,t['w1k'],t['b1k'])
    if direct:
        dh32=torch.empty_like(z,dtype=torch.float32); x=torch.empty_like(dh32)
        ep.apply(state['kernels'][1],z.numel(),z,z0,t['lut'][0],t['kernel_beta'],t['kernel_scale'],packed,dh32,x,z.numel(),k,d,d+k)
        dub,dh=packed[:,:d],packed[:,d:]
    else:
        dh,dh32,x=ep.lookup_secant(z,z0,t['kernel_beta'],t['kernel_scale'],*t['lut']); packed=None
    q=(dh32@t['zmap']).sub_(ju)
    if packed_path:
        if not direct: packed=torch.cat((dub,dh),dim=1)
        base=(torch.addmm(y0,packed,state['packed_weight'],out_dtype=torch.float32) if state['backend']=='addmm' else
              torch.mm(packed,state['packed_weight'],out_dtype=torch.float32).add_(y0.float()))
    else:
        base=torch.mm(dh,t['w2k'],out_dtype=torch.float32); ep.apply(t['base'],base.numel(),y0,linear,base,base.numel())
    result=dict(b=base,q=q,x=x)
    if audit: result.update(du=du,dub=dub,dh=dh,dh32=dh32,packed=packed)
    return result


def hit(u,phase,cached,t,probes,inv_metric):
    """Original interface: (BF16 output,None); caller owns inference_mode."""
    ep,torch=_runtime(); spec=t['_fourstep_direct']['spec']; k=t['zmap'].shape[0]
    if probes.ndim!=1 or probes.numel()!=64 or inv_metric.shape!=(64,64): raise ValueError('Expected fixed64 probes')
    if spec['mode']=='strict' and ep.TILE==spec['tile_rows']: return ep.fast_hit(u,phase,cached,t,probes,inv_metric)
    if not u.is_contiguous() or not all(x.is_contiguous() for x in cached.values()): raise ValueError('Contiguous input and anchors required')
    def at(index,probe=False):return operands(u[index],cached['u0'][index],cached['y0'][index],cached['z0k'][index],t,native_probe=probe)
    op=at(probes,True); base,qp,xp=op['b'],op['q'],op['x']
    observed=phase.ffn_2.apply(torch.nn.functional.gelu(phase.ffn_0.apply(u.index_select(0,probes)),approximate='tanh'))
    target,qd,v=(observed.float()-base).double(),qp.double(),t['vd']; system=((xp@xp.T)/k).double()+inv_metric
    chol,info=torch.linalg.cholesky_ex(system); torch._assert_async((info==0).all(),'Nonpositive probe covariance')
    iq=torch.cholesky_solve(qd,chol)
    gain,info=torch.linalg.solve_ex((qd.T@iq)*t['vv'],(iq*(target@v.T)).sum(0)); torch._assert_async((info==0).all(),'Singular probe gains')
    coeff=torch.cholesky_solve(target-(qd*gain)@v,chol)
    torch._assert_async(torch.isfinite(coeff).all() & torch.isfinite(gain).all(),'Nonfinite readout')
    readout=torch.cat((gain.float()[:,None]*t['v'],.5*coeff.float())); out=torch.empty_like(u)
    for start in range(0,len(u),spec['tile_rows']):
        sl=slice(start,start+spec['tile_rows']); op=at(sl); cross=(op['x']@xp.T)/k
        if spec.get('reuse_readout',False):
            torch.addmm(op['b'],torch.cat((op['q'],cross),dim=1),readout,out=op['b'])
            out[sl].copy_(op['b'])
        else:
            out[sl].copy_(torch.addmm(op['b'],torch.cat((op['q'],cross),dim=1),readout))
    out.index_copy_(0,probes,observed)
    return out,None


if __name__=='__main__':
    import argparse,json
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--mode',choices=MODES,default='direct'); parser.add_argument('--tile-rows',type=int,default=8192); parser.add_argument('--base-backend',choices=BACKENDS,default='auto'); args=parser.parse_args()
    print(json.dumps(build_spec(mode=args.mode,tile_rows=args.tile_rows,base_backend=args.base_backend),indent=2))
