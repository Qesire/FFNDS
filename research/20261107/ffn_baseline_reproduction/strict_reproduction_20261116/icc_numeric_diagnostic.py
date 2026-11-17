"""Fixed ICC BF16-operands / FP32-increment diagnostic, at most 16 cells.

No new native anchor, factorization, calibration, or full-model forward. TF32
controls are recorded; whether two settings use distinct kernels is unresolved.
"""
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from icc_local_replay import gated_risk
from icc_stage3_assets import checked_path,load_pair,sha
from icc_stage3_grid import exclusive_json,validate_scope

SOURCES=('vbi_1013','vbi_0800')
SITES=(('high',19),('low',19))


def predict_fp32(c,fl1,fr1,fl2,fr2,mode):
    """Lift fixed BF16 values losslessly, do all hit/cache arithmetic in F32."""
    if mode not in ('4S','2R'): raise ValueError('bad cache mode')
    if any(c[k].dtype!=torch.bfloat16 for k in ('u0','u1','z0','h0','y0')) or any(f.dtype!=torch.bfloat16 for f in (fl1,fr1,fl2,fr2)):
        raise ValueError('numeric diagnostic requires fixed BF16 values')
    u0,u1,z0,h0,y0=(c[k].float() for k in ('u0','u1','z0','h0','y0'))
    def first(x): return F.linear(F.linear(x,fr1.float()),fl1.float())
    def second(x): return F.linear(F.linear(x,fr2.float()),fl2.float())
    if mode=='4S':
        z=first(u1-u0)+z0
        h=F.gelu(z,approximate='tanh')
        y=second(h-h0)+y0
    else:
        qz=z0-first(u0); qy=y0-second(h0)
        h=F.gelu(first(u1)+qz,approximate='tanh')
        y=second(h)+qy
    return y.bfloat16()


def run(scope,grid_dir,*,device,authorize,check_resources):
    pairs,factors=validate_scope(scope); authorize(scope)
    grid=Path(grid_dir)
    selection_path=checked_path(scope['selection']); family=json.loads(selection_path.read_text(encoding='utf8'))['family']
    if family not in ('SVD','CA','CD'): raise ValueError('invalid selected family')
    cache_path=checked_path(scope['cache_manifest']); cache=json.loads(cache_path.read_text(encoding='utf8'))
    receipts={r['file']:r for r in cache['files']}
    if len(receipts)!=len(cache['files']): raise ValueError('duplicate cached tensors')
    out=Path(scope['numeric_output']); out.mkdir(parents=True,exist_ok=False)
    before=torch.backends.cuda.matmul.allow_tf32
    rows=[]
    try:
        with torch.no_grad():
            for expert,layer in SITES:
                factor=factors[(family,expert,layer)]
                f=torch.load(checked_path(factor),map_location=device,weights_only=True)
                if f['identity']!=factor['identity']: raise ValueError('factor identity mismatch')
                for source in SOURCES:
                    check_resources(); pair=load_pair(pairs[(source,expert,layer)],device)
                    name=f'native__{source}__{expert}__{layer:02d}.pt'
                    if name not in receipts or sha(grid/'cache'/name)!=receipts[name]['sha256']:
                        raise ValueError('native anchor identity mismatch')
                    c=torch.load(grid/'cache'/name,map_location=device,weights_only=True)
                    if 'z0' not in c: raise ValueError('FP32 diagnostic requires saved native z0')
                    c['u0'],c['u1']=pair['u0'],pair['u1']
                    for mode in ('4S','2R'):
                        for tf32 in (False,True):
                            torch.backends.cuda.matmul.allow_tf32=tf32
                            y=predict_fp32(c,f['first_left'],f['first_right'],f['second_left'],f['second_right'],mode)
                            rows.append({'source':source,'expert':expert,'layer':layer,'mode':mode,'tf32_allowed':tf32,
                                         'risk':gated_risk(y,c['y1'],pair['gate']),
                                         'float32_matmul_precision':torch.get_float32_matmul_precision(),
                                         'factor_sha256':factor['sha256'],
                                         'native_anchor_sha256':receipts[name]['sha256']})
                    del c,pair
                del f
    finally:
        torch.backends.cuda.matmul.allow_tf32=before
    if len(rows)!=16: raise RuntimeError('numeric diagnostic must contain exactly 16 cells')
    exclusive_json(out/'numeric_diagnostic.json',{'selection_sha256':sha(selection_path),'cache_manifest_sha256':sha(cache_path),
                   'rows':rows,'count':16,'tf32_kernel_distinction':'NOT_MEASURED','status':'NUMERIC_CELLS_COMPLETE'})
    return {'cells':len(rows),'tf32_kernel_distinction':'NOT_MEASURED'}

