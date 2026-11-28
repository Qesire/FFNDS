"""Independent numerical oracles for attention graph/state transformations."""
import base64
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from ffnds_optim.attention import (
    AttentionReadout, LowHistory, centered_history, fold_high_output,
    partition_attention, prepare_selected_layout, selected_attention, value_moments,
)


def tensors(n=137, h=2, d=16, block=32, top=2, dtype=torch.float64):
    gen = torch.Generator().manual_seed(20261128)
    q,k,v = [torch.randn(n,h,d,generator=gen,dtype=dtype) for _ in range(3)]
    nb = (n+block-1)//block
    route=torch.stack([torch.stack([torch.randperm(nb,generator=gen)[:top] for _ in range(nb)]) for _ in range(h)]).to(torch.int32)
    return q,k,v,route


def dense_oracle(q,k,v,route,ids,block):
    work=torch.float64 if q.dtype == torch.float64 else torch.float32
    logits=q.transpose(0,1).to(work)@k.permute(1,2,0).to(work)*q.shape[-1]**-.5
    keyblocks=torch.arange(len(k))//block
    selected=(route[:,ids//block,:,None] == keyblocks[None,None,None,:]).any(2)
    native=logits.softmax(-1)@v.transpose(0,1).to(work)
    small=logits.masked_fill(~selected,-torch.inf).softmax(-1)@v.transpose(0,1).to(work)
    rho=(logits.softmax(-1)*(~selected)).sum(-1,keepdim=True)
    return native,small,rho


def test_single_partition_tail_and_complement_encoding():
    q,k,v,route=tensors()
    ids=torch.arange(len(q))
    truth,small,rho=dense_oracle(q,k,v,route,ids,32)
    mean,scale,inverse=value_moments(v)
    got=partition_attention(q,k,v,route,block_size=32,mean=mean,inverse=inverse)
    torch.testing.assert_close(got.full,truth,atol=2e-14,rtol=2e-14)
    torch.testing.assert_close(got.selected,small,atol=2e-14,rtol=2e-14)
    torch.testing.assert_close(got.rho,rho,atol=2e-14,rtol=2e-14)
    torch.testing.assert_close(got.centered,(truth-(1-rho)*small-rho*mean)*inverse,atol=2e-14,rtol=2e-14)
    torch.testing.assert_close(-torch.expm1(got.selected_lse-got.full_lse),rho[...,0],atol=2e-14,rtol=2e-14)


def test_empty_complement_has_zero_mass_and_finite_centered():
    q,k,v,route=tensors(n=45,block=32,top=2)
    mean,_,inverse=value_moments(v)
    got=partition_attention(q,k,v,route,block_size=32,mean=mean,inverse=inverse)
    assert torch.count_nonzero(got.rho)==0
    assert torch.count_nonzero(got.centered)==0
    torch.testing.assert_close(got.full,got.selected,rtol=0,atol=0)


def test_arbitrary_original_query_ids_and_tail_support():
    q,k,v,route=tensors()
    ids=torch.tensor([136,0,95,35,128,63,64,2,31])
    _,expected,_=dense_oracle(q[ids],k,v,route,ids,32)
    actual,_=selected_attention(q[ids],k,v,route,ids,block_size=32)
    torch.testing.assert_close(actual,expected,atol=2e-14,rtol=2e-14)
    layout=prepare_selected_layout(ids,len(k),block_size=32,query_tile=16)
    torch.testing.assert_close(layout.row_map[ids],torch.arange(len(ids),dtype=torch.int32))
    with pytest.raises(ValueError,match='Repeated'):
        prepare_selected_layout(torch.tensor([2,2]),len(k),block_size=32,query_tile=16)


def setup_reader(dtype=torch.float64,rounding=False):
    gen=torch.Generator().manual_seed(42)
    h,n,d,f,r,p=2,257,16,12,7,16
    pv=torch.randn(d,f,generator=gen,dtype=dtype)/d**.5
    basis=torch.randn(f,r,generator=gen,dtype=dtype)/f**.5
    ids=torch.randperm(n,generator=gen)[:p]
    selected=torch.randn(h,n,d,generator=gen,dtype=dtype)
    truth=torch.randn(h,p,d,generator=gen,dtype=dtype)
    old_c=torch.randn(h,n,d,generator=gen,dtype=dtype)
    old_a=torch.randn(h,n,r,generator=gen,dtype=dtype)
    rho=torch.rand(h,n,1,generator=gen,dtype=dtype)
    mean=torch.randn(h,1,d,generator=gen,dtype=dtype)
    scale=.5+torch.rand(h,1,d,generator=gen,dtype=dtype)
    reader=AttentionReadout(pv,basis,ids,scale_r=2.7,feature_rounding=rounding)
    cache=LowHistory.from_fields(old_c,old_a,rho,ids)
    return reader,selected,truth,cache,mean,scale,gen


def source_reference(reader,selected,truth,cache,mean,scale):
    mapped=((selected@reader.pv)@reader.basis)
    if reader.feature_rounding:mapped=mapped.bfloat16().to(selected.dtype)
    diff=mapped-cache.anchor_design
    x=torch.cat((torch.ones_like(diff[...,:1]),diff),-1)
    base=((1-cache.rho)*selected+cache.rho*mean)+cache.history*scale
    xp=x[:,reader.probes]
    xt=xp.transpose(-1,-2)
    chol=torch.linalg.cholesky(xt@xp+len(reader.probes)*reader.ridge_lambda*torch.eye(x.shape[-1],dtype=x.dtype))
    coeff=torch.cholesky_solve(xt,chol)@((truth-base[:,reader.probes])/reader.scale_r)
    expected=base+(x@coeff)*reader.scale_r
    expected[:,reader.probes]=truth
    return expected,coeff


@pytest.mark.parametrize('dtype,rounding',[(torch.float64,False),(torch.float32,True)])
def test_probe_first_reader_matches_source_order(dtype,rounding):
    reader,selected,truth,cache,mean,scale,_=setup_reader(dtype,rounding)
    expected,coefficient=source_reference(reader,selected,truth,cache,mean,scale)
    prepared=reader.fit(selected[:,reader.probes],truth,cache=cache,mean=mean,scale=scale)
    torch.testing.assert_close(prepared.coefficient,coefficient,rtol=0,atol=0)
    ids=torch.cat((torch.arange(100,257),torch.arange(100)))
    got=reader.read(selected[:,ids],ids,prepared,cache=cache,mean=mean,scale=scale)
    torch.testing.assert_close(got,expected[:,ids],atol=2e-6 if dtype==torch.float32 else 2e-14,rtol=2e-6 if dtype==torch.float32 else 2e-14)
    for row,probe in enumerate(reader.probes.tolist()):
        loc=torch.nonzero(ids==probe).item()
        assert torch.equal(got[:,loc],truth[:,row])


def test_joint_code_preserves_fit_and_current_error_identity():
    reader,selected,truth,cache,mean,scale,gen=setup_reader()
    h,n,d=selected.shape
    basis=torch.linalg.qr(torch.randn(h,d+reader.basis.shape[-1],9,generator=gen,dtype=torch.float64)).Q
    compressed=cache.encode_joint(basis,tile_rows=43,code_dtype=torch.float64)
    assert compressed.history is None and compressed.anchor_design is None
    assert compressed.code.shape[1] == n-len(reader.probes)
    assert compressed.storage_bytes() < cache.storage_bytes()
    f0=reader.fit(selected[:,reader.probes],truth,cache=cache,mean=mean,scale=scale)
    fc=reader.fit(selected[:,reader.probes],truth,cache=compressed,mean=mean,scale=scale)
    assert torch.equal(f0.coefficient,fc.coefficient)
    ids=torch.randperm(n,generator=gen)
    original=reader.read(selected[:,ids],ids,f0,cache=cache,mean=mean,scale=scale)
    direct=reader.read(selected[:,ids],ids,fc,cache=compressed,mean=mean,scale=scale)
    source=reader.read(selected[:,ids],ids,fc,cache=compressed,mean=mean,scale=scale,joint_mode='source_order')
    torch.testing.assert_close(direct,source,atol=2e-14,rtol=2e-14)
    c,a=compressed.fields(ids)
    defect=(c-cache.history[:,ids])*scale-(a-cache.anchor_design[:,ids])@f0.gamma[:,1:]
    torch.testing.assert_close(direct-original,defect,atol=2e-14,rtol=2e-14)
    cp,ap=compressed.fields(reader.probes)
    assert torch.equal(cp,cache.probe_history) and torch.equal(ap,cache.probe_design)


def test_high_projection_fold_and_unrounded_projection_precompose():
    reader,selected,truth,_,_,_,gen=setup_reader()
    prepared=reader.fit(selected[:,reader.probes],truth)
    ids=torch.randperm(selected.shape[1],generator=gen)
    weight=torch.randn(selected.shape[0]*selected.shape[2],19,generator=gen,dtype=torch.float64)
    bias=torch.randn(19,generator=gen,dtype=torch.float64)
    explicit=reader.read(selected[:,ids],ids,prepared).transpose(0,1).reshape(len(ids),-1)@weight+bias
    folded=fold_high_output(reader,prepared,weight,bias=bias).apply(selected[:,ids],ids)
    torch.testing.assert_close(folded,explicit,atol=3e-14,rtol=3e-14)
    nonprobe=ids[~torch.isin(ids,reader.probes)]
    only_nonprobe=fold_high_output(reader,prepared,weight,bias=bias).apply(selected[:,nonprobe],nonprobe,overwrite_probes=False)
    expected_nonprobe=reader.read(selected[:,nonprobe],nonprobe,prepared).transpose(0,1).reshape(len(nonprobe),-1)@weight+bias
    torch.testing.assert_close(only_nonprobe,expected_nonprobe,atol=3e-14,rtol=3e-14)
    fused=AttentionReadout(reader.pv,reader.basis,reader.probes,feature_rounding=False,precompose=True)
    torch.testing.assert_close(fused.features(selected),reader.features(selected),atol=3e-14,rtol=3e-14)


def test_high_projection_storage_dtype_keeps_bf16_main_gemm():
    reader,selected,truth,_,_,_,gen=setup_reader(torch.float32,True)
    truth=truth.bfloat16()
    prepared=reader.fit(selected[:,reader.probes],truth)
    weight=torch.randn(32,21,generator=gen).bfloat16()
    bias=torch.randn(21,generator=gen).bfloat16()
    folded=fold_high_output(reader,prepared,weight,bias=bias,storage_dtype=torch.bfloat16)
    assert folded.weight.dtype==folded.bias.dtype==torch.bfloat16
    out=folded.apply(selected.bfloat16(),torch.arange(selected.shape[1]))
    assert out.dtype==torch.bfloat16
    native=torch.addmm(bias,truth.transpose(0,1).reshape(len(reader.probes),-1),weight)
    assert torch.equal(out[reader.probes],native)


def test_actual_frozen_cal_projection_and_probe_fit():
    candidates=[Path('research/20261128/graph_state_theory/CAL_ASSETS.json'),Path('round20_theory/CAL_ASSETS.json')]
    archive=next((p for p in candidates if p.exists()),None)
    if archive is None:pytest.skip('Frozen CAL archive is not present in this isolated checkout')
    files={e['path']:e for e in json.loads(archive.read_text())['files']}
    def asset(path):
        row=files[path];raw=base64.b64decode(row['base64'])
        assert hashlib.sha256(raw).hexdigest()==row['sha256']
        return np.load(io.BytesIO(raw),allow_pickle=False)
    pv=torch.from_numpy(asset('plan_assets/bases/P_v.npy')[:,:64].copy()).float()
    with asset('calibration/self_low_transform.npz') as data:
        basis=torch.from_numpy(data['out_delta_basis'].copy()).float()
    g=torch.Generator().manual_seed(20261128)
    ids=torch.randperm(133,generator=g)[:64]
    reader=AttentionReadout(pv,basis,ids,scale_r=1.75)
    selected=torch.randn(2,133,128,generator=g).bfloat16().float()
    truth=torch.randn(2,64,128,generator=g).bfloat16()
    fit=reader.fit(selected[:,ids],truth)
    got=reader.read(selected,torch.arange(133),fit,output_dtype=torch.bfloat16)
    assert torch.equal(got[:,ids],truth)
    assert torch.isfinite(got).all()
    torch.testing.assert_close(reader.features(selected),((selected@pv)@basis).bfloat16().float(),rtol=0,atol=0)


@pytest.mark.skipif(os.environ.get('TRITON_INTERPRET') != '1',reason='Run separately with TRITON_INTERPRET=1 for CPU kernel interpretation')
def test_triton_interpreted_partition_and_global_selected():
    q,k,v,route=tensors(n=75,h=2,d=16,block=32,top=2,dtype=torch.float32)
    mean,_,inverse=value_moments(v)
    ref=partition_attention(q,k,v,route,block_size=32,mean=mean,inverse=inverse)
    got=partition_attention(q,k,v,route,block_size=32,query_tile=16,key_tile=16,mean=mean,inverse=inverse,backend='triton')
    for key in ('full','selected','rho','centered'):
        torch.testing.assert_close(getattr(got,key),getattr(ref,key),atol=2e-6,rtol=2e-5)
    ids=torch.tensor([74,0,63,64,31,32,2,47])
    selected,_=selected_attention(q[ids],k,v,route,ids,block_size=32,query_tile=16,key_tile=16,backend='triton')
    expected,_=selected_attention(q[ids],k,v,route,ids,block_size=32)
    torch.testing.assert_close(selected,expected,atol=2e-6,rtol=2e-5)


def test_paired_basis_objective_gradient_and_fit(tmp_path):
    from ffnds_optim.compile_attention import paired_action_loss,fit_joint_basis,compile_asset,SCHEMA
    gen=torch.Generator().manual_seed(314)
    samples=[dict(Z=torch.randn(2,17,7,generator=gen,dtype=torch.float64),
                  G=torch.randn(2,7,3,generator=gen,dtype=torch.float64)) for _ in range(2)]
    output=torch.randn(6,5,generator=gen,dtype=torch.float64)
    initial=torch.linalg.qr(torch.randn(2,7,2,generator=gen,dtype=torch.float64)).Q
    numer=denom=0.
    for sample in samples:
        z,g=sample['Z'],sample['G']
        target=(z@g).transpose(0,1).reshape(17,-1)@output
        defect=(z@(torch.eye(7)-initial@initial.transpose(-2,-1))@g).transpose(0,1).reshape(17,-1)@output
        numer=numer+defect.square().sum();denom=denom+target.square().sum()
    actual=paired_action_loss(initial,samples,output_weight=output,tile_rows=6)
    torch.testing.assert_close(actual,numer/denom,atol=1e-13,rtol=1e-13)
    assert torch.autograd.gradcheck(lambda e:paired_action_loss(e,samples,output_weight=output,tile_rows=8),
                                   (initial.detach().requires_grad_(),),atol=1e-5,rtol=1e-4)
    paired=actual
    swapped=[dict(Z=samples[0]['Z'],G=samples[1]['G']),dict(Z=samples[1]['Z'],G=samples[0]['G'])]
    assert abs(float(paired-paired_action_loss(initial,swapped,output_weight=output))) > 1e-4
    basis,record=fit_joint_basis(samples,2,output_weight=output,initial_basis=initial,steps=6,tile_rows=8)
    assert record['final_loss'] < record['initial_loss']
    assert all(a>=b-1e-14 for a,b in zip(record['loss_history'],record['loss_history'][1:]))
    assert record['orthogonality_max_abs'] < 1e-12
    raw=tmp_path/'cal.pt';out=tmp_path/'low17.pt'
    torch.save(dict(schema=SCHEMA,expert='low',layer=17,
        calibration=dict(split='CAL',source_ids=['synthetic_unit_test'],rows='nonprobe',kind='SYNTHETIC_TEST'),
        samples=samples,output_weight=output),raw)
    metadata=compile_asset(raw,out,rank=2,steps=2,tile_rows=8)
    asset=torch.load(out,weights_only=True)
    assert asset['basis'].shape==(2,7,2)
    assert asset['basis'].dtype==torch.bfloat16
    assert metadata['optimization']['quality_status']=='UNASSESSED'
    assert metadata['calibration']['kind']=='SYNTHETIC_TEST'
    assert out.with_suffix('.json').exists()
