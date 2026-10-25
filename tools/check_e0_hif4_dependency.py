"""Execute two CPU calls to the pinned original Python HiF4 reference.

AST-extract only quant_hifx; do not import extension-loading package initializers.
This is a synthetic source counterexample, not CUDA-kernel validation.
"""
import argparse
import ast
from pathlib import Path
from types import SimpleNamespace

from ffnds_observation_common import sha256,write_json


def check(root):
    import torch
    source=root/'.runtime/code/HiF4_Sp/HiFloat4/hif4_gpu/quant_cy/base/QFuncs/hifx.py'
    tree=ast.parse(source.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='quant_hifx')
    namespace={'torch':torch,'Tensor':torch.Tensor,'QType':object}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),namespace)
    q=SimpleNamespace(desc='hifx4',exp_bits=0,man_bits=3)
    old=torch.full((1,64),1.25,dtype=torch.float32);old[0,0]=7
    current=old.clone();current[0,0]=14
    qo=namespace['quant_hifx'](old,q,-1);qc=namespace['quant_hifx'](current,q,-1)
    changed=torch.nonzero(qo!=qc,as_tuple=False)[:,1].tolist()
    if changed!=list(range(64)) or float(qo[0,8])!=1.25 or float(qc[0,8])!=1.5:
        raise RuntimeError('Pinned original reference counterexample failed')
    return {'status':'PASS_SYNTHETIC_ORIGINAL_PYTHON_REFERENCE','source_sha256':sha256(source),
            'source':'HiFloat4/hif4_gpu/quant_cy/base/QFuncs/hifx.py#quant_hifx',
            'CPU_reference_calls':2,'native_CUDA_QDQ_calls':0,'model_calls':0,
            'input_changed_indices':[0],'input_change':[7,14],'all_other_inputs':1.25,
            'output_changed_indices':changed,'unchanged_input_index8_output':[1.25,1.5],
            'scope':'Original CPU reference plus inspected CUDA equations; no empirical CUDA equivalence claim'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise FileExistsError('Preserve first counterexample')
    result=check(Path(__file__).resolve().parents[1]);write_json(a.out,result);print(result['status'])
