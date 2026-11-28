"""CPU integration witnesses for the actual staged readers and Wan row adapter."""
from types import SimpleNamespace
from collections import Counter

import pytest
import torch
import torch.nn.functional as F

from ffnds_optim.attention import AttentionReadout, LowHistory, selected_attention, value_moments, fold_high_output
from ffnds_optim.block import BlockScheduleStats, DeferredLayerNormLinear, RowPlan, make_row_plan, stream_hit_block
from ffnds_optim.ffn import FFNStaged, encode_anchor
from ffnds_optim.wan_block import WanBlockOperators, ProjectedAttentionRows, native_rope_rows, select_modulation


class Linear:
    def __init__(self, weight, bias=None):
        self.weight, self.bias, self.calls = weight, bias, []

    def apply(self, x):
        self.calls.append(len(x))
        y = x @ self.weight
        return y if self.bias is None else y + self.bias

    def _get_actual_weight(self):
        return self.weight

    def _get_actual_bias(self):
        return self.bias


class Norm:
    def apply(self, x):
        return F.layer_norm(x, (x.shape[-1],), eps=1e-6)


class Identity:
    def apply(self, x):
        return x


def dense_attention(q, k, v):
    scores = torch.einsum("qhd,khd->hqk", q, k) * q.shape[-1] ** -.5
    return torch.einsum("hqk,khd->qhd", scores.softmax(-1), v)


class CrossAttention:
    def apply(self, *, q, k, v, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv):
        assert cu_seqlens_q.tolist() == [0, len(q)]
        assert cu_seqlens_kv.tolist() == [0, len(k)]
        assert max_seqlen_q == len(q) and max_seqlen_kv == len(k)
        return dense_attention(q, k, v).flatten(1)


def rotate_inplace(q, k, cache):
    # Deliberately mutate both inputs to detect a shared dummy Q/K buffer.
    c, s = cache.chunk(2, dim=-1)
    c, s = c[:, None], s[:, None]
    def rotate(x):
        pairs = x.reshape(*x.shape[:-1], -1, 2)
        even, odd = pairs[..., 0].clone(), pairs[..., 1].clone()
        pairs[..., 0] = even*c - odd*s
        pairs[..., 1] = even*s + odd*c
        return x
    return rotate(q), rotate(k)


class AttentionSession:
    def __init__(self, reader, route, cache=None):
        self.reader, self.route, self.cache = reader, route, cache

    def prepare(self, q, k, v, probes):
        selected, _ = selected_attention(q, k, v, self.route, probes, block_size=getattr(self,"block_size",16))
        truth = dense_attention(q, k, v)
        self.mean, self.scale, _ = value_moments(v)
        fitted = self.reader.fit(selected, truth.transpose(0, 1), cache=self.cache,
                                 mean=self.mean, scale=self.scale)
        return fitted, truth

    def read(self, q, k, v, ids, fitted):
        selected, _ = selected_attention(q, k, v, self.route, ids, block_size=getattr(self,"block_size",16))
        return self.reader.read(selected, ids, fitted, cache=self.cache,
                                mean=self.mean, scale=self.scale).transpose(0, 1).contiguous()


class FoldedAttentionSession(AttentionSession):
    def bind_output_projection(self, self_phase):
        self.output = self_phase.self_attn_o

    def prepare(self, q, k, v, probes):
        fitted, native = super().prepare(q, k, v, probes)
        self.folded = fold_high_output(self.reader, fitted, self.output.weight,
                                       bias=self.output.bias)
        return fitted, native  # P64 deliberately still uses the original W_o

    def read(self, q, k, v, ids, fitted):
        selected, _ = selected_attention(q, k, v, self.route, ids, block_size=getattr(self,"block_size",16))
        return ProjectedAttentionRows(self.folded.apply(selected, ids, output_dtype=q.dtype))


def make_fixture(low=False):
    torch.manual_seed(4819)
    n, width, heads, dim, hidden, selected, modes = 137, 16, 2, 8, 40, 20, 4
    dtype = torch.float64
    random = lambda *shape: torch.randn(*shape, dtype=dtype)
    linear = lambda a, b: Linear(random(a, b) / a**.5)
    probe = torch.randperm(n)[:64]  # deliberately unsorted original production order
    x = random(n, width)
    phase0 = SimpleNamespace(norm1=Norm(), modulation=SimpleNamespace(tensor=torch.zeros(1)),
        self_attn_q=linear(width, width), self_attn_k=linear(width, width),
        self_attn_v=linear(width, width), self_attn_o=linear(width, width),
        self_attn_norm_q=Identity(), self_attn_norm_k=Identity())
    phase1 = SimpleNamespace(norm3=Norm(), cross_attn_q=linear(width, width),
        cross_attn_k=linear(width, width), cross_attn_v=linear(width, width),
        cross_attn_o=linear(width, width), cross_attn_norm_q=Identity(),
        cross_attn_norm_k=Identity(), cross_attn_1=CrossAttention())
    phase2 = SimpleNamespace(norm2=Norm(), ffn_0=linear(width, hidden), ffn_2=linear(hidden, width))
    fields = [random(n, width)*.06 for _ in range(6)]  # requires original modulation row IDs
    fields[2] += .2
    fields[5] += .2
    angles = torch.arange(n, dtype=dtype)[:, None] * torch.arange(1, dim//2+1, dtype=dtype)[None] * .07
    model = SimpleNamespace(config={"seq_parallel": False, "use_image_encoder": False},
        scheduler=SimpleNamespace(step_index=3 if low else 1), has_post_adapter=False,
        num_heads=heads, head_dim=dim, infer_dtype=dtype, sensitive_layer_dtype=dtype,
        pre_process=lambda modulation, embed: fields,
        modulate_func=lambda z, scale, shift: z*(1+scale.squeeze())+shift.squeeze(),
        cos_sin=torch.cat((angles.cos(), angles.sin()), -1), apply_rope_func=rotate_inplace,
        cross_attn_cu_seqlens_q=torch.tensor([0,n], dtype=torch.int32),
        cross_attn_cu_seqlens_kv=torch.tensor([0,23], dtype=torch.int32))
    block = SimpleNamespace(compute_phases=[phase0, phase1, phase2])
    pre = SimpleNamespace(context=random(23, width), embed0=torch.zeros(1))
    nblocks = (n+15)//16
    route = torch.stack([torch.stack([torch.randperm(nblocks)[:3] for _ in range(nblocks)]) for _ in range(heads)])
    reader = AttentionReadout(random(dim, 5), random(5, 3), probe, feature_rounding=False)
    cache = None
    if low:
        cache = LowHistory.from_fields(random(heads,n,dim)*.1, random(heads,n,3)*.1,
                                       torch.rand(heads,n,1,dtype=dtype)*.7, probe)
    attention = AttentionSession(reader, route, cache)
    u0 = random(n, width)
    z0 = phase2.ffn_0.apply(u0)[:, :selected]
    y0 = phase2.ffn_2.apply(F.gelu(phase2.ffn_0.apply(u0), approximate="tanh"))
    v = random(modes,width)*.1
    prepared = {"A0":random(width,width)*.01, "w1k":phase2.ffn_0.weight[:, :selected].T.contiguous(),
        "w2k":phase2.ffn_2.weight[:selected].contiguous(), "b1k":torch.zeros(selected,dtype=dtype),
        "zmap":random(selected,modes)*.1, "jmap":random(width,modes)*.1,
        "v":v, "vd":v, "vv":v@v.T, "kernel_beta":torch.full((selected,),.4,dtype=dtype),
        "kernel_scale":torch.ones(selected,dtype=dtype)}
    ffn_cache = encode_anchor(u0,y0,z0,prepared,probe)
    ffn = FFNStaged(phase2,prepared,ffn_cache,probe,torch.eye(len(probe),dtype=dtype),backend="torch")
    return x, probe, model, block, pre, attention, ffn, fields


def full_field_reference(x, probe, model, block, pre, attention, ffn, fields):
    """Conventional all-row graph, independent of streamed Wan adapter methods."""
    p0,p1,p2=block.compute_phases
    n, heads, dim = len(x),model.num_heads,model.head_dim
    shift,scale,gate,shift2,scale2,gate2=fields
    u = p0.norm1.apply(x)*(1+scale)+shift
    q = p0.self_attn_norm_q.apply(p0.self_attn_q.apply(u)).view(n,heads,dim)
    k = p0.self_attn_norm_k.apply(p0.self_attn_k.apply(u)).view(n,heads,dim)
    v = p0.self_attn_v.apply(u).view(n,heads,dim)
    q,k=model.apply_rope_func(q.clone(),k.clone(),model.cos_sin)
    all_ids=torch.arange(n)
    selected,_=selected_attention(q,k,v,attention.route,all_ids,block_size=getattr(attention,"block_size",16))
    truth=dense_attention(q[probe],k,v)
    mean,sd,_=value_moments(v)
    fit=attention.reader.fit(selected[:,probe],truth.transpose(0,1),cache=attention.cache,mean=mean,scale=sd)
    a=attention.reader.read(selected,all_ids,fit,cache=attention.cache,mean=mean,scale=sd).transpose(0,1).reshape(n,-1)
    residual=x+p0.self_attn_o.apply(a)*gate
    q=p1.cross_attn_norm_q.apply(p1.cross_attn_q.apply(p1.norm3.apply(residual))).view(n,heads,dim)
    k=p1.cross_attn_norm_k.apply(p1.cross_attn_k.apply(pre.context)).view(-1,heads,dim)
    v=p1.cross_attn_v.apply(pre.context).view(-1,heads,dim)
    residual=residual+p1.cross_attn_o.apply(dense_attention(q,k,v).reshape(n,-1))
    ffn_input=p2.norm2.apply(residual)*(1+scale2)+shift2
    y,_=ffn.full(ffn_input,tile_rows=31)
    return residual+y*gate2


@pytest.mark.parametrize("low",[False,True])
def test_whole_wan_graph_with_real_staged_readers(low):
    x,probe,model,block,pre,attention,ffn,fields=make_fixture(low)
    expected=full_field_reference(x,probe,model,block,pre,attention,ffn,fields)
    original=x.clone()
    for phase in block.compute_phases:
        for item in vars(phase).values():
            if isinstance(item,Linear):item.calls.clear()
    ops=WanBlockOperators(model,block,pre,attention,ffn,tile_rows=31)
    stats=BlockScheduleStats()
    actual=stream_hit_block(x,probe,ops,tile_rows=31,stats=stats)
    torch.testing.assert_close(actual,expected,rtol=2e-12,atol=2e-12)
    assert torch.equal(x,original)
    assert stats.probe_attention_chains == stats.probe_ffn_fits == 1
    assert stats.nonprobe_tiles == 3 and stats.maximum_nonprobe_tile_rows == 31
    assert ops.stats["self_kv_passes"] == ops.stats["text_kv_passes"] == 1
    assert ops.stats["query_rows"] == len(x)
    assert max(block.compute_phases[0].self_attn_q.calls) == len(probe) < len(x)
    assert not ops.stats["all_row_query_tensor"]
    assert block.compute_phases[1].cross_attn_k.calls == [23]
    assert block.compute_phases[1].cross_attn_v.calls == [23]


def test_rope_uses_global_unsorted_positions_and_does_not_mutate_input():
    x,probe,model,*_=make_fixture()
    q=torch.randn(len(probe),model.num_heads,model.head_dim,dtype=x.dtype)
    original=q.clone()
    expected,_=rotate_inplace(q.clone(),q.clone(),model.cos_sin[probe])
    actual=native_rope_rows(model,q,probe)
    assert torch.equal(actual,expected) and torch.equal(q,original)
    wrong,_=rotate_inplace(q.clone(),q.clone(),model.cos_sin[:len(probe)])
    assert not torch.allclose(actual,wrong)


def test_modulation_rows_and_shared_fields():
    ids=torch.tensor([6,1,4])
    rows=torch.arange(24).reshape(8,3)
    assert torch.equal(select_modulation(rows[None],ids,8,3),rows[ids])
    shared=torch.tensor([1.,2.,3.])
    assert torch.equal(select_modulation(shared[None,None],ids,8,3),shared)
    with pytest.raises(ValueError):select_modulation(torch.zeros(2,3),ids,8,3)


def test_schedule_rejects_input_alias_before_running_operators():
    x=torch.randn(7,3)
    with pytest.raises(ValueError,match="alias"):
        stream_hit_block(x,torch.tensor([5,1]),None,out=x.view_as(x))


@pytest.mark.parametrize("bad_branch",["seq_parallel","use_image_encoder","anchor","adapter","quant"])
def test_unsupported_graph_rejected_explicitly(bad_branch):
    x,p,model,block,pre,attention,ffn,_=make_fixture()
    if bad_branch in model.config:model.config[bad_branch]=True
    elif bad_branch=="anchor":model.scheduler.step_index=0
    elif bad_branch=="adapter":model.has_post_adapter=True
    else:model._use_mxfp8_quant_fuse=lambda:True
    with pytest.raises(ValueError):WanBlockOperators(model,block,pre,attention,ffn)


@pytest.mark.parametrize("rank",[1,5,13])
def test_deferred_layernorm_consumer_matches_materialized_reference(rank):
    torch.manual_seed(78+rank)
    n,d,o=29,32,17
    b=torch.randn(n,d,dtype=torch.float64)+17
    f=torch.randn(n,rank,dtype=torch.float64)
    u=torch.randn(rank,d,dtype=torch.float64)
    w=torch.randn(d,o,dtype=torch.float64)
    gamma,beta=torch.randn(2,d,dtype=torch.float64)
    consumer=DeferredLayerNormLinear(u,w,gamma=gamma,beta=beta)
    actual=consumer(b,f)
    expected=(F.layer_norm(b+f@u,(d,),eps=1e-6)*gamma+beta)@w
    torch.testing.assert_close(actual,expected,rtol=2e-12,atol=2e-12)


def test_one_row_final_tile_schedule():
    x,p,model,block,pre,attention,ffn,fields=make_fixture()
    # A one-row final nonprobe tile exercises native .squeeze edge handling.
    expected=full_field_reference(x,p,model,block,pre,attention,ffn,fields)
    ops=WanBlockOperators(model,block,pre,attention,ffn,tile_rows=24)
    actual=stream_hit_block(x,p,ops,tile_rows=24)
    torch.testing.assert_close(actual,expected,rtol=2e-12,atol=2e-12)


def test_high_fold_skips_nonprobe_projection_and_preserves_native_probe_projection():
    x,p,model,block,pre,attention,ffn,fields=make_fixture()
    expected=full_field_reference(x,p,model,block,pre,attention,ffn,fields)
    output=block.compute_phases[0].self_attn_o
    output.calls.clear()
    folded=FoldedAttentionSession(attention.reader,attention.route)
    ops=WanBlockOperators(model,block,pre,folded,ffn,tile_rows=31)
    actual=stream_hit_block(x,p,ops,tile_rows=31)
    torch.testing.assert_close(actual,expected,rtol=2e-12,atol=2e-12)
    assert output.calls == [len(p)]  # native P only; no duplicate non-P W_o


def test_projected_attention_requires_explicit_feature_matrix():
    with pytest.raises(ValueError,match="shape"):
        ProjectedAttentionRows(torch.zeros(3,2,8))


def test_row_plan_reuses_exact_tile_objects_across_blocks():
    x,p,model,block,pre,attention,ffn,fields=make_fixture()
    plan=make_row_plan(len(x),p,31)
    assert torch.equal(torch.cat((p,*plan.tiles)).sort().values,torch.arange(len(x)))
    identities=[id(ids) for ids in plan.tiles]
    seen=[]
    original_read=attention.read
    def observed_read(q,k,v,ids,fitted):
        seen.append(id(ids))
        return original_read(q,k,v,ids,fitted)
    attention.read=observed_read
    outputs=[]
    for _ in range(2):
        ops=WanBlockOperators(model,block,pre,attention,ffn,tile_rows=31)
        outputs.append(stream_hit_block(x,p,ops,tile_rows=31,row_plan=plan))
    assert seen==identities*2
    assert torch.equal(outputs[0],outputs[1])


def test_row_plan_rejects_changed_probe_order_and_missing_coverage():
    p=torch.tensor([5,1])
    plan=make_row_plan(7,p,3)
    with pytest.raises(ValueError,match="original probe"):
        plan.check_reuse(torch.empty(7,2),p.clone(),3)
    p[0]=4
    with pytest.raises(ValueError,match="modified"):
        plan.check_reuse(torch.empty(7,2),p,3)
    with pytest.raises(RuntimeError,match="cover"):
        RowPlan(7,torch.tensor([5,1]),3,(torch.tensor([0,2,3]),torch.tensor([3,6])))


def test_all_probe_row_plan_and_inference_mode_ids():
    with torch.inference_mode():
        p=torch.tensor([2,0,1])
        plan=make_row_plan(3,p,2)
        plan.check_reuse(torch.empty(3,4),p,2)
        assert plan.tiles==()


@pytest.mark.parametrize("low,fold",[(False,False),(True,False),(False,True)])
def test_root_hit_attention_and_streamed_binding_integration(low,fold):
    """Close the runtime session -> bound block -> actual readers interface."""
    from ffnds_optim.attention_runtime import AttentionSession as RuntimeAttentionSession, AttentionState
    from ffnds_optim.config import OptimizationConfig
    from ffnds_optim.runtime import bind_streamed_blocks
    x,p,model,block,pre,attention,ffn,fields=make_fixture(low)
    model.block_idx=0
    # Runtime uses the production 128-row route grid, including its short tail.
    attention.block_size=128
    attention.route=torch.tensor([[[0],[1]],[[1],[0]]])
    attention.reader.feature_rounding=not fold
    expected=full_field_reference(x,p,model,block,pre,attention,ffn,fields)
    basis=attention.reader.basis
    legacy=SimpleNamespace(probes=p,pv=attention.reader.pv,
        fits={"high":({"out_current":{"basis":basis}},{"OUT-ABS":.1},1.,1.),
              "low":({"out_delta":{"basis":basis}},{"OUT-D":.1},1.,1.)})
    config=OptimizationConfig(name="cpu_seam",block_streaming=True,tile_rows=31,
                               selected_backend="torch",ffn_backend="torch",high_output_fold=fold)
    session=RuntimeAttentionSession(legacy,config,layers=1)
    # Float64 fold witness isolates real reassociation from the documented
    # source BF16 feature-quantization defect; default production keeps it on.
    session.readers["high" if not low else "low"].feature_rounding=not fold
    session.slots[0]=AttentionState(None,attention.route,attention.cache)
    take_calls=[]
    port=SimpleNamespace(probes=p,counters=Counter())
    def take_hit(model_arg,phase):
        assert model_arg is model and phase is block.compute_phases[2]
        take_calls.append((model.scheduler.step_index,model.block_idx))
        port.counters["hits"]+=1
        return ffn
    port.take_hit=take_hit
    original_calls=[]
    def original(model_arg,weights_arg,hidden,pre_arg):
        original_calls.append(int(model_arg.scheduler.step_index))
        return hidden+7
    owner=SimpleNamespace(infer_block=original)
    output=block.compute_phases[0].self_attn_o
    output.calls.clear()
    with bind_streamed_blocks(owner,session,port,tile_rows=31):
        actual=owner.infer_block(model,block,x,pre)
        model.scheduler.step_index=0
        fallback=owner.infer_block(model,block,x,pre)
    torch.testing.assert_close(actual,expected,rtol=3e-12,atol=3e-12)
    assert torch.equal(fallback,x+7) and original_calls==[0]
    assert owner.infer_block is original
    assert take_calls==[(3 if low else 1,0)]
    assert port.counters["hits"]==port.counters["gate_native"]==port.counters["streamed_blocks"]==1
    assert session.counter["hits"]==session.counter["attention_sites"]==1
    assert session.counter["native_probe_queries"]==64 and not session.slots
    assert len(session._row_plans)==1
    if fold:assert output.calls==[64]
