"""Fit a fixed-rank joint C/A basis to supplied paired CAL actions.

Input .pt schema:
  schema='ffnds_attention_joint_cal_v1', expert='low', layer=<0..39>,
  calibration={split:'CAL', source_ids:[...], rows:'nonprobe', ...},
  samples=[{Z: tensor[H,M,159], G: tensor[H,159,128]}, ...],
  output_weight: optional tensor[H*128,Dout] (or [H,128,Dout]).
Z and G within each sample must come from the same anchor->hit trajectory.
The optional output projection retains cross-head error terms. The program
never samples a model, infers a quality-safe rank, or treats synthetic data as
CAL evidence. Rank and the calibration file are explicit caller inputs.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch


SCHEMA = 'ffnds_attention_joint_cal_v1'


def _validate_samples(samples):
    if not samples:
        raise ValueError('At least one paired calibration sample is required')
    h,w,d = samples[0]['G'].shape
    for sample in samples:
        z,g=sample['Z'],sample['G']
        if z.ndim != 3 or z.shape[0] != h or z.shape[-1] != w or g.shape != (h,w,d) or z.shape[1] < 1:
            raise ValueError('Paired Z[H,M,W] and G[H,W,D] axes must match')
        if not z.is_floating_point() or not g.is_floating_point():
            raise ValueError('Calibration operands must be floating point')
        if not torch.isfinite(z).all() or not torch.isfinite(g).all():
            raise ValueError('Nonfinite calibration operand')
    return h,w,d


def paired_action_loss(basis, samples, *, output_weight=None, tile_rows=1024):
    """Normalized squared *paired* action error; differentiable in basis.

    Computes ZG - (ZE)(E^T G) for each pair before averaging. If W_o is
    supplied, joins head channels before W_o, preserving all head cross terms.
    It does not factor E[Z^T Z] and E[GG^T] or use an independent-reader proxy.
    """
    if tile_rows < 1:
        raise ValueError('tile_rows must be positive')
    h,w,rank=basis.shape
    device,dtype=basis.device,basis.dtype
    numerator=torch.zeros((),device=device,dtype=dtype)
    denominator=torch.zeros_like(numerator)
    weight=None
    if output_weight is not None:
        weight=output_weight.to(device=device,dtype=dtype)
        if weight.ndim == 3:
            weight=weight.reshape(weight.shape[0]*weight.shape[1],weight.shape[2])
    for sample in samples:
        g=sample['G'].to(device=device,dtype=dtype)
        if g.shape[:2] != (h,w):
            raise ValueError('Basis and sample axes disagree')
        reduced=basis.transpose(-2,-1)@g
        for start in range(0,sample['Z'].shape[1],tile_rows):
            z=sample['Z'][:,start:start+tile_rows].to(device=device,dtype=dtype)
            target=z@g
            delta=target-(z@basis)@reduced
            if weight is not None:
                rows=z.shape[1]
                target=target.transpose(0,1).reshape(rows,-1)@weight
                delta=delta.transpose(0,1).reshape(rows,-1)@weight
            numerator=numerator+delta.square().sum()
            denominator=denominator+target.square().sum()
    return numerator/denominator.detach().clamp_min(torch.finfo(dtype).tiny)


def _retract(matrix):
    q,r=torch.linalg.qr(matrix,mode='reduced')
    diagonal=r.diagonal(dim1=-2,dim2=-1)
    signs=torch.where(diagonal >= 0,1.,-1.)
    return q*signs[:,None,:]


def fit_joint_basis(samples, rank, *, output_weight=None, initial_basis=None,
                    steps=40, learning_rate=.25, tile_rows=1024,
                    device='cpu', dtype=torch.float64):
    """PCA initialization followed by paired-action Stiefel descent.

    The step is the exact autograd derivative of the paired objective,
    projected to the orthogonal-basis tangent space, then QR-retracted.
    A short backtracking search keeps accepted calibration loss nonincreasing.
    Initialization uses state covariance only; every optimization decision uses
    the full paired objective above.
    """
    h,w,d=_validate_samples(samples)
    if not 1 <= rank <= w or steps < 0 or learning_rate <= 0:
        raise ValueError('Explicit rank in [1,W], nonnegative steps and positive learning_rate required')
    device=torch.device(device)
    if output_weight is not None and output_weight.numel()//output_weight.shape[-1] != h*d:
        raise ValueError('Output weight input axis must be H*D')
    if initial_basis is None:
        covariance=torch.zeros((h,w,w),device=device,dtype=dtype)
        for sample in samples:
            for start in range(0,sample['Z'].shape[1],tile_rows):
                z=sample['Z'][:,start:start+tile_rows].to(device=device,dtype=dtype)
                covariance.add_(z.transpose(-2,-1)@z)
        _,vectors=torch.linalg.eigh(covariance)
        basis=vectors[:,:,-rank:].flip(-1).contiguous()
        initialization='state_PCA_then_paired_objective'
    else:
        if initial_basis.shape != (h,w,rank):
            raise ValueError('Initial basis must match [H,W,rank]')
        basis=_retract(initial_basis.to(device=device,dtype=dtype))
        initialization='caller_basis_QR'
    history=[]
    stop='requested_steps_completed'
    for iteration in range(steps+1):
        basis=basis.detach().requires_grad_(iteration < steps)
        loss=paired_action_loss(basis,samples,output_weight=output_weight,tile_rows=tile_rows)
        value=float(loss.detach())
        history.append(value)
        if not torch.isfinite(loss):
            raise ValueError('Paired calibration loss became nonfinite')
        if iteration == steps:
            break
        grad,=torch.autograd.grad(loss,basis)
        normal=basis.transpose(-2,-1)@grad
        tangent=grad-basis@((normal+normal.transpose(-2,-1))*.5)
        if float(tangent.detach().square().sum()) < 1e-24:
            stop='stationary_paired_objective'
            break
        rate=learning_rate
        accepted=None
        with torch.no_grad():
            for _ in range(10):
                proposal=_retract(basis-rate*tangent)
                proposed=paired_action_loss(proposal,samples,output_weight=output_weight,tile_rows=tile_rows)
                if float(proposed) <= value:
                    accepted=proposal
                    break
                rate*=.5
        if accepted is None:
            stop='backtracking_no_decrease'
            break
        basis=accepted
    basis=basis.detach()
    gram=basis.transpose(-2,-1)@basis
    orthogonality=float((gram-torch.eye(rank,device=device,dtype=dtype)).abs().max())
    return basis,dict(initialization=initialization,rank=rank,requested_steps=steps,
                      accepted_steps=len(history)-1,loss_history=history,
                      initial_loss=history[0],final_loss=history[-1],
                      orthogonality_max_abs=orthogonality,stop=stop,
                      objective='paired_joint_after_output_projection' if output_weight is not None else 'paired_joint_sum_head_energy',
                      quality_status='UNASSESSED',model_forwards=0)


def compile_asset(input_path, output_path, *, rank, steps=40, learning_rate=.25,
                  tile_rows=1024, device='cpu', storage_dtype=torch.bfloat16):
    input_path,output_path=Path(input_path),Path(output_path)
    data=torch.load(input_path,map_location='cpu',weights_only=True)
    if data.get('schema') != SCHEMA or data.get('expert') != 'low' or type(data.get('layer')) is not int or not 0 <= data['layer'] < 40:
        raise ValueError('Expected a schema-bound low-expert CAL site')
    calibration=data.get('calibration',{})
    if calibration.get('split') != 'CAL' or calibration.get('rows') != 'nonprobe' or not calibration.get('source_ids'):
        raise ValueError('Explicit CAL source IDs and nonprobe-row metadata are required')
    expected=f"low{data['layer']:02d}.pt"
    if output_path.name != expected:
        raise ValueError(f'TensorDirectory expects this site asset to be named {expected}')
    if output_path.exists():
        raise FileExistsError(output_path)
    basis,record=fit_joint_basis(data['samples'],rank,output_weight=data.get('output_weight'),
        initial_basis=data.get('initial_basis'),steps=steps,learning_rate=learning_rate,
        tile_rows=tile_rows,device=device)
    stored=basis.to(device='cpu',dtype=storage_dtype)
    # This records basis-storage error only. Online BF16 code quantization is
    # separately accounted for by LowHistory and is not claimed by this fit.
    stored_loss=float(paired_action_loss(stored.double(),data['samples'],
        output_weight=data.get('output_weight'),tile_rows=tile_rows))
    metadata=dict(schema='ffnds_attention_joint_basis_v1',expert='low',layer=data['layer'],
                  calibration=calibration,input_sha256=hashlib.sha256(input_path.read_bytes()).hexdigest(),
                  basis_storage_dtype=str(storage_dtype),basis_storage_loss=stored_loss,
                  code_quantization_in_calibration_loss=False,optimization=record)
    output_path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(basis=stored,metadata=metadata),output_path)
    output_path.with_suffix('.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
    return metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--rank',type=int,required=True)
    parser.add_argument('--steps',type=int,default=40)
    parser.add_argument('--learning-rate',type=float,default=.25)
    parser.add_argument('--tile-rows',type=int,default=1024)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--storage-dtype',choices=('float32','bfloat16'),default='bfloat16')
    args=parser.parse_args()
    record=compile_asset(args.input,args.output,rank=args.rank,steps=args.steps,
                         learning_rate=args.learning_rate,tile_rows=args.tile_rows,
                         device=args.device,storage_dtype=getattr(torch,args.storage_dtype))
    print(json.dumps(record,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
