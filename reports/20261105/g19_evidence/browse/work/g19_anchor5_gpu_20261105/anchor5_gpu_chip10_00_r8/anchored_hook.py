"""Original 40-step FFN anchored candidate. Must be independently reviewed."""
import hashlib
import math
import time
import torch
from anchor_schedule import AnchorSchedule
import cache_codec
from gpu_slab import Slab

class AnchoredHook:
    def __init__(self, actual_expert, actual_grid, load_matrix, emit, *, mode, cap_bytes, shadow=False, seed=43):
        if mode not in ('dense','gpu_fp32_anchor5','gpu_bf16_anchor5'): raise ValueError('mode')
        self.expert=actual_expert; self.grid=actual_grid; self.load_matrix=load_matrix; self.emit=emit; self.mode=mode; self.cap=cap_bytes; self.shadow=shadow; self.seed=seed
        self.cache_dtype=torch.bfloat16 if mode=='gpu_bf16_anchor5' else torch.float32; self.timetable=None; self.schedule=AnchorSchedule(); self.cache={}; self.slab=None; self.physical_slab_bytes=0; self.seen=set(); self.counts={'anchor':0,'hit':0}; self.live=0; self.peak=0; self.read=0; self.written=0; self.owner=None
    def install(self, owner):
        if self.owner is not None: raise RuntimeError('hook already installed')
        if self.mode!='dense':
            self.slab=Slab(torch,self.cache_dtype,torch.device('cuda',torch.cuda.current_device()))
            self.physical_slab_bytes=self.slab.physical_bytes
            if self.physical_slab_bytes>self.cap: raise MemoryError('physical slab cap')
        self.owner=owner; self.original=owner.infer_ffn; h=self; original=self.original
        def ffn(model, phase, x, attn_out, shift, scale, gate=None):
            step=int(model.scheduler.step_index); layer=int(model.block_idx); condition=model.scheduler.infer_condition
            if type(condition) is not bool: raise RuntimeError('invalid CFG marker')
            branch='conditional' if condition else 'unconditional'; expert=h.expert(); timestep=h.timetable[step] if h.timetable is not None else (_ for _ in ()).throw(RuntimeError('timetable not frozen'))
            if expert!=('high' if timestep>=900 else 'low') or not 0<=layer<40: raise RuntimeError('bad dispatch')
            if step!=h.schedule.step:
                previous=h.schedule.last_expert; h.schedule.begin_step(step,expert)
                if previous is not None and previous!=expert:
                    h.cache.clear(); h.live=0
                if h.slab is not None: h.slab.bind(expert)
            key=(expert,branch,layer); event=(step,expert,branch,layer)
            if event in h.seen: raise RuntimeError('duplicate site')
            h.seen.add(event)
            if h.mode=='dense':
                result=original(model,phase,x,attn_out,shift,scale,gate)
                h.emit(dict(event='DENSE',step=step,expert=expert,branch=branch,layer=layer,timestep=timestep)); return result
            if gate is None or hasattr(phase,'smooth_norm2_weight') or hasattr(phase,'smooth_norm2_bias') or model._use_mxfp8_quant_fuse(): raise RuntimeError('unsupported FFN')
            action=h.schedule.action(*key); h.counts[action]+=1; first=phase.ffn_0.apply; second=phase.ffn_2.apply; state={}; start=time.monotonic()
            def w1(u):
                if state or u.ndim!=2 or u.shape[1]!=5120 or u.dtype!=torch.bfloat16: raise RuntimeError('invalid W1 input')
                state['u']=u.detach().clone()
                return first(u) if action=='anchor' else u
            def w2(v):
                if 'u' not in state or state.get('done'): raise RuntimeError('invalid W2 sequence')
                state['done']=True; u=state['u']; A=h.load_matrix(expert,layer,u.device)
                if A.shape!=(5120,5120) or A.dtype!=torch.bfloat16: raise RuntimeError('bad affine matrix')
                pred=(u@A).float()
                if action=='anchor':
                    y=second(v); c=cache_codec.encode(y,pred,h.cache_dtype)
                    if key in h.cache:
                        h.cache.pop(key); h.live-=h.slab.slot_bytes
                    n=c.numel()*c.element_size()
                    if h.live+n>h.cap: raise MemoryError('cache cap')
                    h.slab.put(expert,branch,layer,c); h.cache[key]=True; h.live+=n; h.written+=n; h.peak=max(h.peak,h.live); h.schedule.record_anchor(*key)
                    result=y
                else:
                    if key not in h.cache: raise RuntimeError('missing anchor cache')
                    c=h.slab.get(expert,branch,layer); n=c.numel()*c.element_size()
                    if c.shape!=u.shape or c.device!=u.device or c.dtype!=h.cache_dtype: raise RuntimeError('cache geometry or identity changed')
                    h.read+=n; approx=cache_codec.decode(pred,c,h.cache_dtype,torch.bfloat16)
                    result=approx
                    if h.shadow:
                        digest=hashlib.sha256(f'{h.seed}|{expert}|{layer}|{u.shape[0]}'.encode()).digest()
                        gen=torch.Generator(device='cpu').manual_seed(int.from_bytes(digest[:8],'little') & ((1<<63)-1))
                        ids=torch.randperm(u.shape[0],generator=gen)[:64].sort().values.to(u.device)
                        if ids.numel()!=64 or ids.unique().numel()!=64: raise RuntimeError('shadow token selection')
                        sample=u.index_select(0,ids); ref=second(torch.nn.functional.gelu(first(sample),approximate='tanh'))
                        error=(approx.index_select(0,ids).float()-ref.float()); norms=sample.float().square().sum(1)
                        g=gate.squeeze().float()
                        if g.ndim!=1 or g.shape[0]!=5120 or not torch.isfinite(g).all(): raise RuntimeError('invalid gate')
                        gated_error=error*g; gated_ref=ref.float()*g
                        T,H,W=(int(x) for x in h.grid())
                        if T*H*W!=u.shape[0]: raise RuntimeError('token grid mismatch')
                        groups={f'{t},{y},{x}':{'count':0,'sum_mean':0.,'sum_norm2':0.} for t in range(3) for y in range(4) for x in range(4)}
                        for i,m,n in zip(ids.cpu().tolist(),sample.float().mean(1).cpu().tolist(),norms.cpu().tolist()):
                            t=i//(H*W); y=(i//W)%H; x=i%W; g=groups[f'{t*3//T},{y*4//H},{x*4//W}']; g['count']+=1; g['sum_mean']+=m; g['sum_norm2']+=n
                        h.emit(dict(groups=groups,token_grid=[T,H,W],event='SHADOW_DIAGNOSTIC',step=step,expert=expert,branch=branch,layer=layer,token_ids=ids.cpu().tolist(),input_mean=sample.float().mean(1).cpu().tolist(),input_norm2=norms.cpu().tolist(),input_variance=sample.float().var(1,unbiased=False).cpu().tolist(),age=step-h.schedule.anchors[key],gated_sse=gated_error.square().sum(1).cpu().tolist(),gated_energy=gated_ref.square().sum(1).cpu().tolist(),sse=error.square().sum(1).cpu().tolist(),energy=ref.float().square().sum(1).cpu().tolist(),diagnostic_only=True))
                h.emit(dict(event='ANCHOR' if action=='anchor' else 'HIT',step=step,expert=expert,branch=branch,layer=layer,timestep=timestep,anchor_step=h.schedule.anchors[key],cache_live_bytes=h.live,cache_peak_bytes=h.peak,seconds=time.monotonic()-start))
                return result
            phase.ffn_0.apply=w1; phase.ffn_2.apply=w2
            try: result=original(model,phase,x,attn_out,shift,scale,gate)
            finally: phase.ffn_0.apply=first; phase.ffn_2.apply=second
            if not state.get('done'): raise RuntimeError('FFN incomplete')
            return result
        owner.infer_ffn=ffn
    def finish(self, dispatch):
        if len(dispatch)!=40: raise RuntimeError('incomplete dispatch')
        expected={(step,dispatch[step],branch,layer) for step in range(40) for branch in ('conditional','unconditional') for layer in range(40)}
        if self.seen!=expected: raise RuntimeError('incomplete FFN coverage')
        anchor_steps=[i for i in range(40) if i%5==0 or (i>0 and dispatch[i]!=dispatch[i-1])]
        if anchor_steps!=list(range(0,40,5)): raise RuntimeError('unexpected actual anchor steps')
        if self.mode!='dense' and self.counts!={'anchor':len(anchor_steps)*80,'hit':(40-len(anchor_steps))*80}: raise RuntimeError('anchor/hit count mismatch')
        before=self.live; self.cache.clear(); self.live=0
        if self.slab is not None: self.slab.release(); self.slab=None
        return {'anchor_steps':anchor_steps,'action_counts':self.counts,'persistent_cache_peak_bytes':self.peak,'physical_slab_bytes':self.physical_slab_bytes,'physical_slab_live_after_release_bytes':0,'read_bytes':self.read,'written_bytes':self.written,'live_before_release_bytes':before,'live_after_release_bytes':self.live}
    def uninstall(self):
        self.cache.clear(); self.live=0
        if self.slab is not None: self.slab.release(); self.slab=None
        if self.owner is not None: self.owner.infer_ffn=self.original; self.owner=None
