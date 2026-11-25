"""One-model, serialized HHLL inference with GPU-resident anchor slots.

The caller binds a frozen checkpoint and its site packs for the lifetime of
ResidentJoint. Each request must use trajectory(), including on failure.
"""
from contextlib import contextmanager
import inspect

import numpy as np
import torch


class AnchorSlots:
    """One reusable GPU allocation per tensor. Expert lifetimes do not overlap."""
    def __init__(self, layers, rows, width, selected, device):
        self.device=torch.device(device)
        kw=dict(dtype=torch.bfloat16,device=self.device)
        self.u0=torch.empty((layers,rows,width),**kw)
        self.y0=torch.empty_like(self.u0)
        self.z0k=torch.empty((layers,rows,selected),**kw)
        self.bytes=sum(v.numel()*v.element_size() for v in (self.u0,self.y0,self.z0k))
        self.owner=[None]*layers

    def anchor(self, expert, layer, u, phase, selector):
        if self.owner[layer] is not None:
            raise RuntimeError("Anchor slot has an unconsumed value")
        z=phase.ffn_0.apply(u)
        # Launch on the native stream before the in-place GELU overwrites z.
        torch.index_select(z,1,selector,out=self.z0k[layer])
        torch.ops.aten.gelu.out(z,approximate="tanh",out=z)
        y=phase.ffn_2.apply(z)
        self.u0[layer].copy_(u)
        self.y0[layer].copy_(y)
        self.owner[layer]=expert
        return y

    def take(self, expert, layer):
        if self.owner[layer]!=expert:
            raise RuntimeError("Hit has no matching expert's native anchor")
        self.owner[layer]=None
        return dict(u0=self.u0[layer],y0=self.y0[layer],z0k=self.z0k[layer])


class ResidentJoint:
    """Shares the immutable selected kernels and calibration across requests."""
    def __init__(self, site_loader, observer_ids, prepare_static, hit_prepared, *,
                 device="cuda:0", grid=(21,29,52), width=5120, selected=2560,
                 layers=40, reserve_bytes=10*2**30, prepared_budget=None):
        from native_joint import probe_metric_inverse
        dev=torch.device(device)
        self.device=torch.device("cuda",torch.cuda.current_device()) if dev==torch.device("cuda") else dev
        self.rows=int(np.prod(grid))
        self.width,self.layers=width,layers
        self.selected=selected
        self.site_loader=site_loader
        self.prepare_static,self.hit_prepared=prepare_static,hit_prepared
        self.weights={}
        self.selectors={}
        self.bound=False
        self.busy=False
        self.completion=None
        self.execution_stream=None
        self.counters={}
        ids=np.asarray(observer_ids,dtype=np.int64)
        if ids.shape!=(64,) or len(np.unique(ids))!=64 or np.any((ids<0)|(ids>=self.rows)):
            raise ValueError("64 unique in-grid native observation indices required")
        self.probes=torch.as_tensor(ids,dtype=torch.long,device=self.device)
        coords=np.column_stack(np.unravel_index(ids,grid))
        self.inv_metric,_=probe_metric_inverse(ids,coords,self.device)
        anchor_bytes=2*layers*self.rows*(2*width+selected)
        # Includes A0, selected BF16 weights, FP32 maps, FP64 V/Gram,
        # the selected bias, selector and slopes.
        # Pass the backend's own bound if it stores additional large constants.
        if prepared_budget is None:
            prepared_budget=2*layers*(2*width*width+4*width*selected+
                                      4*16*(2*width+selected)+8*16*width+
                                      8*16**2+18*selected)
        self.prepared_budget=prepared_budget
        free,_=torch.cuda.mem_get_info(self.device)
        if free<anchor_bytes+prepared_budget+reserve_bytes:
            raise MemoryError(f"Free {free} bytes; anchors {anchor_bytes}, static {prepared_budget}, workspace reserve {reserve_bytes}")
        self.slots=AnchorSlots(layers,self.rows,width,selected,self.device)
        self.completion=torch.cuda.Event()
        self.completion.record(torch.cuda.current_stream(self.device))

    @contextmanager
    def trajectory(self):
        if self.busy:
            raise RuntimeError("Resident slots already assigned to a request")
        if torch.backends.cuda.matmul.allow_tf32:
            raise RuntimeError("Joint calibration requires the registered FP32 matmul policy")
        stream=torch.cuda.current_stream(self.device)
        stream.wait_event(self.completion)
        self.execution_stream=stream
        self.busy=True
        self.counters=dict(sites=0,anchors=0,hits=0,new_packed_sites=0,gate_fused=0,gate_native=0)
        try:
            yield self
            if self.counters["sites"]!=4*self.layers or any(v is not None for v in self.slots.owner):
                raise RuntimeError("Incomplete four-step resident trajectory")
        finally:
            self.slots.owner[:]=[None]*self.layers
            if self.execution_stream!=stream:
                stream.wait_stream(self.execution_stream)
            self.completion=torch.cuda.Event()
            self.completion.record(stream)
            self.execution_stream=None
            self.busy=False

    @torch.inference_mode()
    def __call__(self, model, phase, u):
        if not self.busy or u.device!=self.device or u.dtype!=torch.bfloat16 or u.shape!=(self.rows,self.width):
            raise RuntimeError("Unexpected native input or inactive trajectory")
        step,layer=int(model.scheduler.step_index),int(model.block_idx)
        if (step,layer)!=divmod(self.counters["sites"],self.layers) or step>=4:
            raise RuntimeError("Out-of-order native FFN")
        current=torch.cuda.current_stream(self.device)
        if current!=self.execution_stream:
            current.wait_stream(self.execution_stream)
            self.execution_stream=current
        expert="high" if step<2 else "low"
        key=(expert,layer)
        if step in (0,2):
            if key not in self.weights:
                pack=self.site_loader(expert,layer)
                selector=pack["sel"].to(self.device)
                if selector.shape!=(self.selected,) or selector.dtype!=torch.long:
                    raise ValueError("Invalid frozen channel selection")
                params=self.prepare_static(phase,pack,self.device)
                # Packed tensors must own their selected data. Native staged
                # weight buffers can be overwritten at the next layer.
                self.weights[key]=params
                self.selectors[key]=selector
                self.counters["new_packed_sites"]+=1
            y=self.slots.anchor(expert,layer,u,phase,self.selectors[key])
            self.counters["anchors"]+=1
        else:
            cached=self.slots.take(expert,layer)
            y,_=self.hit_prepared(u,phase,cached,self.weights[key],self.probes,self.inv_metric)
            self.counters["hits"]+=1
        self.counters["sites"]+=1
        return y

    @contextmanager
    def bind(self, owner, transformer_sha):
        """Use the installed FFN prefix and fuse only its BF16 native post."""
        from wan_ffn_port import bind_native_ffn,source_digest
        from q1_fused_post import run as fused_post
        original=owner.post_process
        if self.bound or source_digest(inspect.getfile(original))!=transformer_sha:
            raise RuntimeError("Unexpected transformer post implementation")

        def post(model,x,y,gate,pre_infer_out=None):
            if (self.busy and y is not None and gate is not None and
                    model.sensitive_layer_dtype==model.infer_dtype==torch.bfloat16 and
                    not model.clean_cuda_cache and self.width==5120 and
                    x.dtype==y.dtype==gate.dtype==torch.bfloat16 and
                    x.device==y.device==gate.device==self.device and x.shape==y.shape and
                    gate.squeeze().shape==(self.width,) and
                    x.is_contiguous() and y.is_contiguous() and gate.is_contiguous()):
                fused_post(x,y,None,None,gate,action="post")
                self.counters["gate_fused"]+=1
                return x
            if self.busy:
                self.counters["gate_native"]+=1
            return original(model,x,y,gate,pre_infer_out)

        self.bound=True
        try:
            with bind_native_ffn(owner,self,exact_source_sha=transformer_sha):
                owner.post_process=post
                try:
                    yield self
                finally:
                    owner.post_process=original
        finally:
            self.bound=False
