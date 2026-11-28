"""CPU semantic checks; native CUDA kernels and Wan quality require their runtime."""
import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from ffnds_optim.ffn import (FFNCache, FFNStaged, apply_gauge,
    compile_factor_asset, encode_anchor, factor_cycle_budget, prepare_runtime_tensors,
    recovery_aware_factor_loss)
from ffnds_optim.ffn_runtime import OptimizationFFNPort, ProjectedAnchorSlots


class Linear:
    def __init__(self, weight, bias):
        self.weight, self.bias = weight, bias
    def apply(self, value):
        return torch.addmm(self.bias, value, self.weight)
    def _get_actual_weight(self):
        return self.weight
    def _get_actual_bias(self):
        return self.bias


def fixture(dtype=torch.float64, rows=79):
    gen = torch.Generator().manual_seed(20261128)
    def rand(shape, scale=.2):
        return (torch.randn(shape, generator=gen, dtype=torch.float64)*scale).to(dtype)
    d, m, k, modes, p = 24, 48, 12, 3, 10
    phase = SimpleNamespace(ffn_0=Linear(rand((d, m)),rand((m,))),
                            ffn_2=Linear(rand((m, d)),rand((d,))))
    maptype = torch.float64 if dtype == torch.float64 else torch.float32
    pack = dict(A0=rand((d,d)), sel=torch.arange(k),
                zmap=rand((k,modes)).to(maptype),jmap=rand((d,modes)).to(maptype),
                v=rand((modes,d)).to(maptype),kernel_beta=torch.full((k,),.4,dtype=maptype),
                kernel_scale=torch.ones(k,dtype=maptype))
    t = prepare_runtime_tensors(phase,pack,"cpu",backend="torch")
    u0, u = rand((rows,d),.7), rand((rows,d),.7)
    probes = torch.randperm(rows,generator=gen)[:p]
    z0 = phase.ffn_0.apply(u0)
    y0 = phase.ffn_2.apply(F.gelu(z0,approximate="tanh"))
    cache=encode_anchor(u0,y0,z0[:,:k],t,probes)
    inv=torch.eye(p,dtype=torch.float64)
    return phase,pack,t,u0,u,probes,cache,inv


def reference(u, phase, cache, t, probes, inv, tile=17):
    work = torch.float64 if u.dtype == torch.float64 else torch.float32
    k=t["zmap"].shape[0]
    def op(ix):
        du=u[ix].to(work)-cache["u0"][ix].to(work)
        z=F.linear(u[ix],t["w1k"],t["b1k"])
        z0=cache["z0k"][ix]
        dh=(F.gelu(z,approximate="tanh").to(work)-F.gelu(z0,approximate="tanh").to(work)).to(u.dtype)
        x=(dh.to(work)-(z.to(work)-z0.to(work))*t["kernel_beta"])/t["kernel_scale"]
        q=dh.to(work)@t["zmap"]-du@t["jmap"]
        linear=(du.to(u.dtype)@t["A0"]).to(work)
        b=(cache["y0"][ix].to(work)+linear)+dh.to(work)@t["w2k"].to(work)
        return b,q,x
    b,q,xp=op(probes)
    observed=phase.ffn_2.apply(F.gelu(phase.ffn_0.apply(u[probes]),approximate="tanh"))
    target=(observed.to(work)-b).double();qd=q.double();v=t["v"].double()
    s=((xp@xp.T)/k).double()+inv
    chol=torch.linalg.cholesky(s);iq=torch.cholesky_solve(qd,chol)
    gain=torch.linalg.solve((qd.T@iq)*(v@v.T),(iq*(target@v.T)).sum(0))
    c=torch.cholesky_solve(target-(qd*gain)@v,chol)
    read=torch.cat((gain.to(work)[:,None]*t["v"],.5*c.to(work)))
    out=torch.empty_like(u)
    for start in range(0,len(u),tile):
        sl=slice(start,start+tile);b,q,x=op(sl)
        out[sl]=torch.addmm(b,torch.cat((q,x@xp.T/k),dim=1),read).to(u.dtype)
    out[probes]=observed
    return out


@pytest.mark.parametrize("dtype",[torch.float64,torch.bfloat16])
def test_staged_matches_independent_full_formula(dtype):
    phase,pack,t,u0,u,p,cache,inv=fixture(dtype)
    staged=FFNStaged(phase,t,cache,p,inv,backend="torch")
    actual,_=staged.full(u,tile_rows=17)
    expected=reference(u,phase,cache,t,p,inv)
    assert torch.equal(actual,expected)
    assert torch.equal(actual[p],phase.ffn_2.apply(F.gelu(phase.ffn_0.apply(u[p]),approximate="tanh")))


def test_arbitrary_global_ids_and_exact_probe_order():
    phase,pack,t,u0,u,p,cache,inv=fixture()
    staged=FFNStaged(phase,t,cache,p,inv,backend="torch")
    fit=staged.fit(u[p]);ids=torch.tensor([78,0,52,17,3,22,51,9])
    actual=staged.apply_rows(ids,u[ids],fit,overwrite_probes=True)
    expected=reference(u,phase,cache,t,p,inv)
    torch.testing.assert_close(actual,expected[ids],rtol=1e-12,atol=1e-12)
    p_rows=staged.apply_rows(p,u[p],fit,overwrite_probes=True)
    assert torch.equal(p_rows,fit.observed)
    with pytest.raises(ValueError,match="probe order"):
        FFNStaged(phase,t,cache,p.flip(0),inv,backend="torch")


def test_gauge_and_gls_output_equivalence_real():
    phase,pack,t,u0,u,p,cache,inv=fixture()
    teacher_weight=phase.ffn_2.weight.clone()
    c=torch.tensor([.25,-.1,.3],dtype=torch.float64)
    shifted=apply_gauge(t,c)
    old=FFNStaged(phase,t,cache,p,inv,backend="torch")
    new=FFNStaged(phase,shifted,cache,p,inv,backend="torch")
    f0,f1=old.fit(u[p]),new.fit(u[p])
    torch.testing.assert_close(f1.gain,f0.gain+c,rtol=1e-11,atol=1e-11)
    torch.testing.assert_close(f1.coefficient,f0.coefficient,rtol=1e-11,atol=1e-11)
    torch.testing.assert_close(new.full(u,tile_rows=17)[0],old.full(u,tile_rows=17)[0],rtol=1e-11,atol=1e-11)
    assert torch.equal(phase.ffn_2.weight,teacher_weight)


def test_projected_cache_has_no_full_input_and_preserves_qp():
    phase,pack,t,u0,u,p,cache,inv=fixture(torch.bfloat16)
    asset=compile_factor_asset(pack,6)
    t2=prepare_runtime_tensors(phase,pack,"cpu",factor_asset=asset,backend="torch")
    assert "A0" not in t2
    cache2=encode_anchor(u0,cache["y0"],cache["z0k"],t2,p,tile_rows=13)
    assert set(cache2.fields)=={"y0","z0k","a0","j0","u0p"}
    assert cache2["a0"].shape==(len(u),6)
    assert cache2["u0p"].shape==(len(p),u.shape[1])
    assert all(v.untyped_storage().data_ptr()!=u0.untyped_storage().data_ptr() for v in cache2.fields.values())
    orig=FFNStaged(phase,t,cache,p,inv,backend="torch")
    fact=FFNStaged(phase,t2,cache2,p,inv,backend="torch")
    _,q0,x0=orig.operands(p,u[p],probe_fit=True)
    _,q1,x1=fact.operands(p,u[p],probe_fit=True)
    assert torch.equal(q0,q1) and torch.equal(x0,x1)
    result,_=fact.full(u,tile_rows=19)
    assert torch.equal(result[p],fact.fit(u[p]).observed)


def test_factor_error_is_filtered_by_existing_recovery_real():
    phase,pack,t,u0,u,p,cache,inv=fixture()
    asset=compile_factor_asset(pack,7,dtype=torch.float64)
    t2=prepare_runtime_tensors(phase,pack,"cpu",factor_asset=asset,backend="torch")
    cache2=encode_anchor(u0,cache["y0"],cache["z0k"],t2,p,projection_dtype=torch.float64)
    old=FFNStaged(phase,t,cache,p,inv,backend="torch")
    new=FFNStaged(phase,t2,cache2,p,inv,backend="torch")
    e=(u-u0)@(t2["factor_left"]@t2["factor_right"]-t["A0"])
    _,q,x=old.operands(slice(None),u)
    qp,xp=q[p],x[p];s=xp@xp.T/old.k+inv
    iq=torch.linalg.solve(s,qp)
    gain=torch.linalg.solve((qp.T@iq)*(t["v"]@t["v"].T),(iq*(e[p]@t["v"].T)).sum(0))
    coef=torch.linalg.solve(s,e[p]-(qp*gain)@t["v"])
    expected=e-(q*gain)@t["v"]-.5*(x@xp.T/old.k)@coef
    expected[p]=0
    actual=new.full(u,tile_rows=17)[0]-old.full(u,tile_rows=17)[0]
    torch.testing.assert_close(actual,expected,rtol=1e-10,atol=1e-10)


def test_factor_identity_binding_and_paired_gauge():
    phase,pack,t,u0,u,p,cache,inv=fixture()
    c=torch.tensor([.2,.1,-.4])
    asset=compile_factor_asset(pack,6,gauge=c,dtype=torch.float64)
    actual=prepare_runtime_tensors(phase,pack,"cpu",factor_asset=asset,backend="torch")
    expected=apply_gauge(t,c)
    torch.testing.assert_close(actual["w2k"],expected["w2k"])
    with pytest.raises(ValueError,match="gauge"):
        prepare_runtime_tensors(phase,pack,"cpu",factor_asset=asset,gauge=-c,backend="torch")
    changed=dict(pack,A0=pack["A0"]+.01)
    with pytest.raises(ValueError,match="another A0"):
        prepare_runtime_tensors(phase,changed,"cpu",factor_asset=asset,backend="torch")


def test_real_projected_slots_and_full_hhll_port_reuse():
    phase,pack,t,u0,u,p,cache,inv=fixture(torch.bfloat16,rows=80)
    asset=compile_factor_asset(pack,6)
    port=OptimizationFFNPort(lambda e,l:pack,p.tolist(),device="cpu",grid=(1,8,10),
        width=24,selected=12,layers=2,tile_rows=17,backend="torch",
        factor_loader=lambda e,l:asset,inv_metric=inv)
    model=SimpleNamespace(scheduler=SimpleNamespace(step_index=0),block_idx=0)
    for repeat in range(2):
        with port.trajectory():
            for step in range(4):
                model.scheduler.step_index=step
                for layer in range(2):
                    model.block_idx=layer
                    now=u0 if step%2==0 else u
                    if step==3:
                        staged=port.take_hit(model,phase)
                        out=staged.full(now,tile_rows=17)[0]
                    else:
                        out=port(model,phase,now)
                    assert out.shape==now.shape and out.dtype==torch.bfloat16
            assert port.counters["anchors"]==4 and port.counters["hits"]==4
            assert "u0" not in port.slots.fields
            assert all("A0" not in t for t in port.weights.values())
            assert not any(owner is not None for owner in port.slots.owner)
    expected_bytes=sum(v.numel()*v.element_size() for v in port.slots.fields.values())
    assert port.counters["anchor_gpu_capacity_bytes"]==expected_bytes
    assert port.counters["new_packed_sites"]==0


def test_exception_cleans_slot_ownership_and_rejects_out_of_order():
    phase,pack,t,u0,u,p,cache,inv=fixture(torch.bfloat16)
    port=OptimizationFFNPort(lambda e,l:pack,p.tolist(),device="cpu",grid=(1,1,len(u)),
        width=24,selected=12,layers=1,backend="torch",inv_metric=inv)
    model=SimpleNamespace(scheduler=SimpleNamespace(step_index=0),block_idx=0)
    with pytest.raises(RuntimeError,match="out-of-order"):
        with port.trajectory():
            port(model,phase,u0)
            port(model,phase,u0)
    assert not port.busy and port.slots.owner==[None]
    with pytest.raises(RuntimeError,match="matching"):
        port.slots.take("high",0)


def test_budget_includes_anchor_and_exact_probe_sidecar():
    low=factor_cycle_budget(31668,5120,2560,512)
    high=factor_cycle_budget(31668,5120,2560,2048)
    assert low["whole_video_net_saved_flops"]>0>high["whole_video_net_saved_flops"]
    assert low["compute_break_even_rank"]==pytest.approx(1701.3333333333333)
    assert low["resident_history_bytes"]/2**30==pytest.approx(19.428529739379883)


def test_recovery_aware_loss_matches_actual_error_and_differentiates():
    phase,pack,t,u0,u,p,cache,inv=fixture()
    asset=compile_factor_asset(pack,7,dtype=torch.float64)
    t2=prepare_runtime_tensors(phase,pack,"cpu",factor_asset=asset,backend="torch")
    cache2=encode_anchor(u0,cache["y0"],cache["z0k"],t2,p,projection_dtype=torch.float64)
    old=FFNStaged(phase,t,cache,p,inv,backend="torch")
    new=FFNStaged(phase,t2,cache2,p,inv,backend="torch")
    _,q,x=old.operands(slice(None),u)
    left=asset["left"].clone().requires_grad_();right=asset["right"].clone().requires_grad_()
    args=(t["A0"],u-u0,q,x,t["v"],p,inv)
    field=recovery_aware_factor_loss(left,right,*args,reduction="none")
    actual=new.full(u,tile_rows=17)[0]-old.full(u,tile_rows=17)[0]
    torch.testing.assert_close(field,actual,rtol=1e-10,atol=1e-10)
    loss=recovery_aware_factor_loss(left,right,*args);loss.backward()
    assert torch.isfinite(left.grad).all() and torch.isfinite(right.grad).all()
    eps=1e-6;plus=left.detach().clone();minus=left.detach().clone()
    plus[0,0]+=eps;minus[0,0]-=eps
    fd=(recovery_aware_factor_loss(plus,right.detach(),*args)-recovery_aware_factor_loss(minus,right.detach(),*args))/(2*eps)
    assert left.grad[0,0].item()==pytest.approx(fd.item(),rel=1e-6,abs=1e-8)


def test_offline_factor_cli_export_and_no_overwrite(tmp_path):
    from ffnds_optim.compile_ffn import main
    phase,pack,t,u0,u,p,cache,inv=fixture(torch.bfloat16)
    source=tmp_path/"frozen.pt";torch.save(pack,source)
    gauge=tmp_path/"gauge.json";gauge.write_text(json.dumps([.25,-.1,.3]))
    output=tmp_path/"factors"
    argv=["--pack",str(source),"--site","high00","--rank","6","--output-dir",str(output),"--gauge",str(gauge)]
    receipt=main(argv)
    assert receipt["status"]=="COMPLETE" and receipt["new_model_forwards"]==0
    assert receipt["recovery_aware_fitting_performed"] is False
    saved=torch.load(output/"high00.pt",map_location="cpu",weights_only=True)
    assert saved["method"]=="weight_frobenius_svd_initializer" and saved["task_quality"]=="unmeasured"
    prepared=prepare_runtime_tensors(phase,pack,"cpu",factor_asset=saved,backend="torch")
    assert "factor_left" in prepared and "A0" not in prepared
    bad=dict(saved);del bad["source_jmap_sha256"]
    with pytest.raises(ValueError,match="missing source identity"):
        prepare_runtime_tensors(phase,pack,"cpu",factor_asset=bad,backend="torch")
    with pytest.raises(FileExistsError):
        main(argv)
