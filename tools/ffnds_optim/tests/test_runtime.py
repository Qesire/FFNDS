"""Exercise the actual integration adapter's binding and state lifecycle."""
from types import SimpleNamespace

import pytest
import torch

from ffnds_optim.attention import AttentionReadout, LowHistory, value_moments, centered_history
from ffnds_optim.attention_runtime import AttentionSession
from ffnds_optim.config import OptimizationConfig


def dense(q, k, v):
    logits = q.transpose(0, 1) @ k.permute(1, 2, 0) * q.shape[-1]**-.5
    return logits.softmax(-1) @ v.transpose(0, 1), torch.logsumexp(logits, -1)


class Op:
    def apply(self, q, k, v, attn_mask=None, causal=False, drop_rate=0.):
        return dense(q, k, v)[0].transpose(0, 1).reshape(len(q), -1)


class Owner:
    def __init__(self):
        self.scheduler, self.block_idx = SimpleNamespace(step_index=0), 0

    def infer_self_attn(self, phase, q, k, v):
        return phase.self_attn_1.apply(q=q, k=k, v=v)


class Legacy:
    def __init__(self, n=137):
        g = torch.Generator().manual_seed(911)
        self.probes = torch.randperm(n, generator=g)[:64]
        self.pv = torch.randn(8, 5, generator=g, dtype=torch.float64)*.2
        matrices = {key: {"basis": torch.randn(5, 3, generator=g, dtype=torch.float64)*.2}
                    for key in ("out_current", "out_delta")}
        self.fits = {expert: (matrices, {"OUT-ABS": .1, "OUT-D": .1}, 2.1, 1.)
                     for expert in ("high", "low")}
        self.mode = "centered"
        self.exp = SimpleNamespace(owner=Owner)

    def route(self, q, k):
        n, h, _ = q.shape
        blocks = (n+127)//128
        route = torch.arange(blocks).reshape(1, blocks, 1).expand(h, -1, -1).clone().to(torch.int32)
        return route, route

    def selected(self, q, k, v, route, finite):
        logits = q.transpose(0, 1) @ k.permute(1, 2, 0) * q.shape[-1]**-.5
        qi = torch.arange(len(q))//128
        ki = torch.arange(len(k))//128
        support = (route[:, qi, :, None] == ki[None, None, None]).any(2)
        logits = logits.masked_fill(~support, -torch.inf)
        return logits.softmax(-1) @ v.transpose(0, 1), torch.logsumexp(logits, -1), None

    def low_cache(self, q, k, v, route, native, lse, arm):
        selected, ls, _ = self.selected(q, k, v, route, False)
        rho = (-torch.expm1(ls-lse)).unsqueeze(-1)
        mean, _, inv = value_moments(v)
        c = centered_history(native, selected, rho, mean, inv)
        a = ((selected @ self.pv) @ self.fits["low"][0]["out_delta"]["basis"]).bfloat16().to(selected.dtype)
        return (route, rho, c, a, None), 0

    def approximate(self, q, k, v, cache, arm):
        route = cache if arm == "OUT-ABS" else cache[0]
        selected, _, _ = self.selected(q, k, v, route, False)
        truth = dense(q[self.probes], k, v)[0]
        expert, key = ("high", "out_current") if arm == "OUT-ABS" else ("low", "out_delta")
        m, ls, sr, _ = self.fits[expert]
        reader = AttentionReadout(self.pv, m[key]["basis"], self.probes, ridge_lambda=ls[arm], scale_r=sr)
        low = None if arm == "OUT-ABS" else LowHistory.from_fields(cache[2], cache[3], cache[1], self.probes)
        mean, scale, _ = value_moments(v)
        fit = reader.fit(selected[:, self.probes], truth, cache=low, mean=mean, scale=scale)
        result = reader.read(selected, torch.arange(len(q)), fit, cache=low, mean=mean, scale=scale)
        return result.transpose(0, 1).contiguous(), truth, selected


def flash(q, k, v):
    o, lse = dense(q, k, v)
    return o[None], lse[None]


@pytest.mark.parametrize("changes", [{}, {"anchor_partition": "torch"},
    {"attention_streaming": True, "selected_backend": "torch"}])
def test_binding_four_steps_and_original_probe_teacher(changes):
    torch.manual_seed(117)
    legacy = Legacy()
    cfg = OptimizationConfig(name="test", tile_rows=31, **changes)
    session = AttentionSession(legacy, cfg, layers=2, teacher=lambda q,k,v:dense(q,k,v)[0], native_flash=flash)
    original_self, original_apply = Owner.infer_self_attn, Op.apply
    model, phase = Owner(), SimpleNamespace(self_attn_1=Op())
    with session.bind(owner=Owner, op_type=Op) as counts:
        for step in range(4):
            model.scheduler.step_index = step
            for layer in range(2):
                model.block_idx = layer
                q,k,v = [torch.randn(137,2,8,dtype=torch.float64) for _ in range(3)]
                output = model.infer_self_attn(phase,q,k,v).reshape(137,2,8)
                assert output.isfinite().all()
                if step % 2:
                    expected = dense(q[legacy.probes],k,v)[0].transpose(0,1)
                    torch.testing.assert_close(output[legacy.probes],expected,atol=0,rtol=0)
        assert counts["attention_sites"] == 8 and counts["native_probe_queries"] == 256
        assert not session.slots
    assert Owner.infer_self_attn is original_self and Op.apply is original_apply


def test_joint_cache_releases_full_fields_and_probes_remain_exact():
    torch.manual_seed(8)
    legacy = Legacy()
    basis = torch.linalg.qr(torch.randn(11,5,dtype=torch.float64)).Q
    cfg = OptimizationConfig(name="joint", attention_basis_dir="explicit_test_asset")
    session = AttentionSession(legacy, cfg, layers=1, basis_loader=lambda e,l:{"basis":basis},
                               teacher=lambda q,k,v:dense(q,k,v)[0],native_flash=flash)
    model,phase = Owner(),SimpleNamespace(self_attn_1=Op())
    with session.bind(owner=Owner,op_type=Op):
        for step in range(4):
            model.scheduler.step_index=step
            q,k,v=[torch.randn(137,2,8,dtype=torch.float64) for _ in range(3)]
            out=model.infer_self_attn(phase,q,k,v).reshape(137,2,8)
            if step==2:
                state=session.slots[0]
                assert state.original_cache is None
                assert state.history.history is state.history.anchor_design is None
                assert state.history.code.shape == (2,73,5)
            if step%2:
                torch.testing.assert_close(out[legacy.probes],dense(q[legacy.probes],k,v)[0].transpose(0,1),atol=0,rtol=0)


def test_exception_restores_hooks_and_consumed_slots():
    legacy=Legacy()
    session=AttentionSession(legacy,OptimizationConfig(),layers=1,native_flash=flash)
    before=(Owner.infer_self_attn,Op.apply)
    with pytest.raises(RuntimeError,match="injected failure"):
        with session.bind(owner=Owner,op_type=Op):
            session.slots[0]="pending"
            raise RuntimeError("injected failure")
    assert (Owner.infer_self_attn,Op.apply)==before and not session.slots and not session._bound


def test_config_declares_each_numerical_class_and_implied_ffn():
    cfg=OptimizationConfig(name="all",ffn_staged=True,block_streaming=True,
        anchor_partition="triton",ffn_factor_dir="factors",attention_basis_dir="bases")
    assert cfg.needs_custom_ffn
    assert {x["kind"] for x in cfg.semantics()} == {
        "preserved_operation_order","real_arithmetic_equivalent","controlled_approximation"}
    with pytest.raises(ValueError,match="block_streaming"):
        OptimizationConfig(high_output_fold=True)
    with pytest.raises(ValueError,match="reference"):
        OptimizationConfig(anchor_partition="torch").assert_generation_backends()
