"""Pending-review student P64 FFN tap. No execution authority or import-time GPU work.

The original infer_ffn remains responsible for normalization, residual and gate.
On candidate reuse only, a transaction-local GELU shim passes an identity-only
expanded sentinel from intercepted W1 dispatch to intercepted W2 dispatch.
Both logical dispatches are counted by NativeTap, while their original dense
GEMMs are explicitly absent; P64 + packed BF16 W1/W2 are separately charged.
"""

import threading,hashlib
_SHIM_LOCK=threading.Lock()

def need(v,msg):
    if not v: raise ValueError(msg)

def validate_active_compute_buffer(torch,infer,phase):
    manager=infer.offload_manager
    need(not infer.lazy_load and not manager.lazy_load and len(manager.cuda_buffers)==2 and phase is manager.cuda_buffers[0].compute_phases[2],'active FFN buffer or offload identity differs')
    need(torch.cuda.current_stream()==manager.compute_stream,'active FFN compute stream differs')

def validate_live_w2(torch,mm,mapping,expert,h,d):
    need(mapping.get(id(mm))==('block',expert,'ffn.2'),'W2 buffer mapping differs')
    need(type(mm).__name__=='MMWeight' and type(mm).__module__=='lightx2v.common.ops.mm.mm_weight' and not mm.has_lora_branch and not mm.has_diff and not hasattr(mm,'weight_diff') and not hasattr(mm,'bias_diff'),'W2 type/adapters differ')
    w,b=mm.weight,mm.bias
    need(hasattr(mm,'weight_cuda_buffer') and hasattr(mm,'bias_cuda_buffer') and w.data_ptr()==mm.weight_cuda_buffer.data_ptr() and b.data_ptr()==mm.bias_cuda_buffer.data_ptr(),'W2 is not active owned prefetch buffer')
    need(torch.is_tensor(w) and torch.is_tensor(b) and w.dtype==b.dtype==torch.bfloat16 and w.device.type==b.device.type=='cuda' and w.device.index in (None,0) and b.device.index in (None,0) and tuple(w.shape)==(h,d) and tuple(w.stride())==(1,h) and tuple(b.shape)==(d,) and tuple(b.stride())==(1,),'W2 layout/dtype/device/bias differs')

class CacheBook:
    def __init__(self,limit):self.limit=limit;self.items={};self.bytes=0;self.peak=0
    def put(self,expert,block,h,y):
        key=(expert,block);need(key not in self.items and expert in (0,1) and 0<=block<40,'duplicate cache')
        n=h.numel()*h.element_size()+y.numel()*y.element_size();need(self.bytes+n<=self.limit,'cache limit')
        self.items[key]=(h,y);self.bytes+=n;self.peak=max(self.peak,self.bytes)
    def get(self,expert,block):
        key=(expert,block);need(key in self.items,'missing same-expert cache');return self.items[key]
    def release(self,expert,block):
        h,y=self.items.pop((expert,block));self.bytes-=h.numel()*h.element_size()+y.numel()*y.element_size()
    def empty(self):need(not self.items and self.bytes==0,'unreleased cache')

def make_tap(vm,native,*,mode,rows,kernels,emit,geometry=(31668,5120,13824,5120)):
    m,d,h,k=geometry
    need(m>0 and d>0 and h>0 and 0<k<=h,'geometry')
    need(mode in ('passthrough','cache_dense','candidate'),'invalid mode')
    class P64Tap(vm.LightTap):
        def __init__(self,*a,**kw):
            super().__init__(*a,**kw);self.book=CacheBook(40*m*(h+d)*2);self.pending=None;self.cache_puts=0;self.cache_releases=0;self.sparse_sites=0;self.thread_id=threading.get_ident()
        def charged(self,cost,fn):
            def run():
                result=fn();self.guard.torch.cuda.synchronize();return result
            return self.guard.ledger.call(self.rid,cost,run)
        def mm_call(self,mm,value,meta,original):
            slot=self.mapping.get(id(mm));suffix=slot[2] if slot is not None and slot[0]=='block' else None
            p=self.pending
            if p is None or suffix not in ('ffn.0','ffn.2'):return super().mm_call(mm,value,meta,original)
            need(self.block==(p['expert'],p['block']) and self.step(p['expert'])==p['step'],'FFN escaped transaction')
            torch=self.guard.torch;step=p['step'];expert=p['expert'];block=p['block']
            if mode=='candidate' and step%2 and suffix=='ffn.0':
                need(mm is p['phase'].ffn_0 and p.get('sentinel') is None,'wrong W1')
                old_h,old_y=self.book.get(expert,block);w2m=p['phase'].ffn_2
                validate_live_w2(torch,w2m,self.mapping,expert,h,d)
                need(tuple(value.shape)==(m,d) and tuple(mm.weight.shape)==(d,h) and tuple(w2m.weight.shape)==(h,d) and tuple(mm.bias.shape)==(h,),'weight/input identity')
                def invoke(_mm,x):
                    from chipmunk_fullm_proxy_math import rms_score,selection
                    from chipmunk_wan_packed_pregate import packed_pregate
                    import torch.nn.functional as F
                    def weights():return mm.weight.T.contiguous(),w2m.weight.T.contiguous()
                    w1,w2=self.charged({'weight_materializations':2,'weight_materialization_logical_bytes':2*d*h*2},weights)
                    emit({'event':'weight_views','w1_alias':w1.data_ptr()==mm.weight.data_ptr(),'w2_alias':w2.data_ptr()==w2m.weight.data_ptr(),'logical_bytes':w1.numel()*w1.element_size()+w2.numel()*w2.element_size()})
                    need(w1.is_contiguous() and w2.is_contiguous(),'weight layout')
                    scores=self.charged({'p64_score':1,'p64_gather_input':1,'p64_gather_old_h':1,'p64_W1_gemm':1,'p64_gelu':1,'p64_rms':1,'p64_input_gather_logical_bytes':len(rows)*d*2,'p64_old_h_gather_logical_bytes':len(rows)*h*2},lambda:rms_score(F.linear(x.index_select(0,rows),w1,mm.bias),old_h.index_select(0,rows)))
                    indices=self.charged({'p64_select':1},lambda:selection(scores,k))
                    indices_sha=self.charged({'mask_sha_calls':1,'mask_sha_logical_bytes':k*8},lambda:hashlib.sha256(indices.to('cpu').numpy().tobytes()).hexdigest())
                    cost={'packed_action':1,'packed_gather_w1':1,'packed_gather_w2':1,'packed_gather_bias':1,'packed_clone_h':1,'packed_clone_y':1,'packed_gather_h':1,'packed_W1_gemm':1,'packed_selected_gelu':1,'packed_delta_round':1,'packed_update_h':1,'packed_W2_gemm_fp32':1,'packed_accumulate_y':1,'packed_cache_clone_logical_bytes':m*(h+d)*2,'packed_delta_logical_bytes':m*k*2,'packed_weight_gather_logical_bytes':(k*d+d*k+k)*2}
                    out=self.charged(cost,lambda:packed_pregate(x,w1,mm.bias,w2,old_h,old_y,indices))
                    if block==0 and step in (1,3):
                        from chipmunk_packed_native_math import packed_action
                        ones=self.charged({'observer_ones':1,'observer_ones_logical_bytes':d*2},lambda:torch.ones((d,),dtype=torch.bfloat16,device=x.device))
                        obs_cost={'observer_'+name:value for name,value in cost.items()};obs_cost.update(observer_gate_product=1,observer_gate_logical_bytes=m*d*2)
                        obs=self.charged(obs_cost,lambda:packed_action(x,w1,mm.bias,w2,old_h,old_y,ones,indices))
                        def compare():
                            v=out.float()-obs.float();need(bool(torch.isfinite(v).all()) and bool(torch.isfinite(obs).all()),'nonfinite packed observer')
                            absmax=float(v.abs().max());norm=float(torch.linalg.vector_norm(obs.float()));dnorm=float(torch.linalg.vector_norm(v));rel=dnorm/norm if norm else (0.0 if dnorm==0 else None)
                            return {'relative_l2':rel,'max_absolute':absmax,'reference_norm':norm}
                        metrics=self.charged({'observer_compare':1,'observer_compare_elements':m*d},compare)
                        emit({'event':'packed_pregate_control','step':step,'block':block,'expert':expert,'mask_indices_sha256':indices_sha,**metrics,'relative_l2_limit':1e-5,'max_absolute_limit':0.001})
                        need(metrics['relative_l2'] is not None and metrics['relative_l2']<=1e-5 and metrics['max_absolute']<=0.001,'packed pregate control mismatch')
                    need(tuple(out.shape)==(m,d),'candidate output shape')
                    p['out']=out;p['sentinel']=out[:,0:1].expand(-1,h);self.sparse_sites+=1
                    emit({'event':'candidate_packed_action','step':step,'block':block,'expert':expert,'original_dense_ffn_GEMMs':0,'logical_native_dispatches':2,'mask_indices_sha256':indices_sha})
                    return p['sentinel']
                return super().mm_call(mm,value,meta,invoke)
            if mode=='candidate' and step%2 and suffix=='ffn.2':
                need(mm is p['phase'].ffn_2 and value is p.get('sentinel') and p.get('out') is not None,'sentinel identity')
                out=super().mm_call(mm,value,meta,lambda _mm,_x:p['out']);self.charged({'cache_release_sync':1},lambda:None);self.book.release(expert,block);self.cache_releases+=1;return out
            if mode in ('cache_dense','candidate') and step%2==0 and suffix=='ffn.2':
                need(mm is p['phase'].ffn_2 and tuple(value.shape)==(m,h) and value.dtype==torch.bfloat16 and (geometry!=(31668,5120,13824,5120) or value.device.type=='cuda'),'cache h boundary')
                need(self.book.bytes+2*m*(h+d)<=self.book.limit,'cache budget before cloning')
                hc=self.charged({'cache_h_clone':1,'cache_h_clone_bytes':value.numel()*value.element_size()},lambda:value.detach().clone())
                out=super().mm_call(mm,value,meta,original)
                need(tuple(out.shape)==(m,d) and out.dtype==torch.bfloat16 and (geometry!=(31668,5120,13824,5120) or out.device.type=='cuda'),'cache Y boundary')
                yc=self.charged({'cache_y_clone':1,'cache_y_clone_bytes':out.numel()*out.element_size()},lambda:out.detach().clone())
                self.book.put(expert,block,hc,yc);self.cache_puts+=1;return out
            if mode=='cache_dense' and step%2 and suffix=='ffn.2':
                out=super().mm_call(mm,value,meta,original);self.charged({'cache_release_sync':1},lambda:None);self.book.release(expert,block);self.cache_releases+=1;return out
            return super().mm_call(mm,value,meta,original)
        def install(self):
            super().install()
            for expert,model in enumerate(self.runner.model.model):
                infer=model.transformer_infer;old=infer.infer_ffn
                def wrapped(phase,x,attn,shift,scale,gate=None,*,_e=expert,_i=infer,_old=old):
                    step=self.step(_e);block=int(_i.block_idx)
                    if mode=='candidate' and step%2:validate_active_compute_buffer(self.guard.torch,_i,phase)
                    need(threading.get_ident()==self.thread_id,'FFN Python thread changed');need(self.pending is None and self.block==(_e,block),'nested/wrong FFN transaction')
                    p={'phase':phase,'expert':_e,'step':step,'block':block};self.pending=p
                    F=self.guard.torch.nn.functional;old_gelu=F.gelu
                    def gelu(x,*args,**kwargs):
                        if x is p.get('sentinel') and p.get('sentinel') is not None:
                            need(mode=='candidate' and step%2 and kwargs.get('approximate')=='tanh','sentinel GELU mismatch');return x
                        return old_gelu(x,*args,**kwargs)
                    locked=False
                    try:
                        if mode=='candidate' and step%2:
                            need(_SHIM_LOCK.acquire(blocking=False),'concurrent GELU transaction');locked=True;F.gelu=gelu
                        out=_old(phase,x,attn,shift,scale,gate)
                        if mode=='candidate' and step%2:need(out is p.get('out'),'candidate postgate identity mismatch')
                        return out
                    finally:
                        if locked:F.gelu=old_gelu;_SHIM_LOCK.release()
                        self.pending=None
                self.stack.enter_context(native.patch_method(infer,'infer_ffn',wrapped))
        def finish(self):
            super().finish();self.book.empty()
            if mode in ('cache_dense','candidate'):need(self.cache_puts==self.cache_releases==80,'cache coverage')
            if mode=='candidate':need(self.sparse_sites==80,'sparse coverage')
    return P64Tap
