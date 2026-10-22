"""Original HiF4 CPU counterexample to a row-local activation kernel."""
import argparse
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=ROOT/'research/20261023/R15_joint_kernel_CPU_protocol.json'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    from worker_identity import require_worker
    from ar2_checkpoint import atomic_json,digest
    require_worker()
    if args.output.exists():raise RuntimeError('Preserve first joint-kernel counterexample')
    p=json.loads(PROTOCOL.read_text())
    if digest(__file__)!=p['source_sha256']:raise RuntimeError('Frozen CPU source changed')
    for name,value in p['input_sha256'].items():
        if digest(ROOT/name)!=value:raise RuntimeError('Original HiF4 source changed')
    import torch
    from audit_hif4_group_separability import reference
    torch.set_num_threads(8)
    qtype,original=reference()
    x=torch.zeros((2,64),dtype=torch.bfloat16)
    x[:,0]=p['target_post_GELU']
    x[:,63]=torch.tensor(p['other_channel_anchors'],dtype=torch.bfloat16)
    out=original(x.float(),qtype,-1).bfloat16()
    target=out[:,0].float().tolist()
    if target!=[.25,0.] or torch.cuda.is_initialized():
        raise RuntimeError('Fixed original CPU reference did not discriminate')
    result={'id':p['id'],'status':'PASS_ORIGINAL_CPU_JOINT_QUANTIZER_COUNTEREXAMPLE',
            'protocol_sha256':digest(PROTOCOL),'source_sha256':digest(__file__),
            'input_sha256':p['input_sha256'],'target_channel_input':x[:,0].float().tolist(),
            'other_channel_input':x[:,63].float().tolist(),
            'target_channel_original_HiF4_output':target,
            'unchanged_target_activations_different_quantized_outputs':True,
            'original_Python_reference_calls':1,'synthetic_rows':2,
            'CUDA_initialized':False,'new_model_generations':0,'new_W1_W2_calls':0,'new_fitted_parameters':0,
            'conclusion':'A universal scalar activation depending only on the target hidden preactivation cannot reproduce this group64 HiF4 Q2: other hidden channels change the target code. This does not forbid joint-group feature maps or an empirically trained approximation.',
            'native_registration':None,'model_execution_eligibility_change':False}
    atomic_json(args.output,result)
    print(json.dumps({'target_input':result['target_channel_input'],'other_channel_input':result['other_channel_input'],'target_HiF4_output':target}),flush=True)


if __name__=='__main__':main()
