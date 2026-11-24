"""Check native LSE and grouped selected attention on the school A100."""
import ast
from hashlib import sha256
import json
from pathlib import Path

import torch
from torch.nn.attention.flex_attention import AuxRequest, BlockMask, flex_attention
from torch.nn.functional import pad
from torch.nn.functional import scaled_dot_product_attention as sdpa


SCALE = 128**-0.5


def relative(a, b):
    return float((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-12))


@torch.inference_mode()
def main():
    base=json.loads(Path("base.json").read_bytes())
    scope=json.loads((Path(base["v2_reference_dir"])/"scope.json").read_bytes())
    backend="lightx2v/common/ops/attn/torch_sdpa.py"
    backend_path=next(p for entry in scope["python_import_paths"] if (p:=Path(entry)/backend).is_file())
    backend_code=backend_path.read_bytes()
    assert sha256(backend_code).hexdigest()=="ae87ba65009c454f62e0e558cf821ee2bada1420928f408e5a7e26760984cf7d"
    klass=next(s for s in ast.parse(backend_code).body if isinstance(s,ast.ClassDef) and s.name=="TorchSDPAWeight")
    wrapper=next(s for s in klass.body if isinstance(s,ast.FunctionDef) and s.name=="apply")
    import triton

    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    n, h, d = 31668, 4, 128
    q, k, v = [torch.randn((1,h,n,d),device="cuda",dtype=torch.bfloat16) for _ in range(3)]
    native = sdpa(q,k,v,scale=SCALE)
    fused = torch.ops.aten._scaled_dot_product_flash_attention(q,k,v,0.0,False,False,scale=SCALE)
    out, lse = fused[:2]
    assert out.shape == q.shape and lse.shape[-1] >= n
    ids = torch.tensor([0,64,126,127,128,256,31667],device="cuda")
    logits = (q[:,:,ids].float() @ k.float().transpose(-2,-1))*SCALE
    lse_ref = logits.logsumexp(-1)
    lse_error = float((lse[:,:,ids]-lse_ref).abs().max())
    assert lse_error < 1e-4
    assert torch.equal(native,out), relative(native,out)

    tile = (n+127)//128
    blocks = torch.arange(31,device="cuda",dtype=torch.int32).expand(1,h,tile,-1).clone()
    blocks[...,-1] = tile-1
    blocks[...,:-1] = (blocks[...,:-1]+torch.arange(tile,device="cuda",dtype=torch.int32)[None,None,:,None])%(tile-1)
    sel = BlockMask.from_kv_blocks(torch.full((1,h,tile),31,device="cuda",dtype=torch.int32),pad(blocks,(0,tile-31)),BLOCK_SIZE=128,seq_lengths=(n,n))
    call = torch.compile(flex_attention,fullgraph=True,dynamic=False)
    v16 = torch.randn((1,h,n,16),device="cuda",dtype=v.dtype)
    va = torch.cat((v,v16),dim=-1)
    status = {}
    for width, value in ((128,v),(144,va)):
        try:
            selected, aux = call(q,k,value,block_mask=sel,scale=SCALE,return_aux=AuxRequest(lse=True))
            selected_lse=aux.lse
            assert selected.shape==(1,h,n,width) and selected_lse.shape==(1,h,n)
            keep = (blocks[0,0,ids//128,:,None].long()*128+torch.arange(128,device="cuda")).flatten(1)
            safe = keep.clamp_max(n-1)
            scores = logits[0].gather(-1,safe[None].expand(h,-1,-1)).masked_fill(keep[None]>=n,float("-inf"))
            gathered = value[0].index_select(1,safe.flatten()).reshape(h,len(ids),-1,width)
            reference = torch.einsum("hqk,hqkd->hqd",scores.softmax(-1),gathered.float())
            err = relative(selected[0,:,ids],reference)
            err_lse = float((selected_lse[0,:,ids]-scores.logsumexp(-1)).abs().max())
            assert err < 0.008 and err_lse < 1e-4,(err,err_lse)
            status[str(width)] = {"relative_l2":err,"maximum_lse_error":err_lse}
        except Exception as exc:
            status[str(width)] = {"error":type(exc).__name__,"detail":str(exc)[:1800]}
    result = dict(status="complete",torch=str(torch.__version__),triton=triton.__version__,gpu=torch.cuda.get_device_name(),
        native_wrapper=ast.get_source_segment(backend_code.decode(),wrapper),dense_matches_sdpa_bitwise=True,
        dense_lse_maximum_absolute_error=lse_error,selected=status)
    Path("BACKEND.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf8")
    print(json.dumps({k:v for k,v in result.items() if k!="native_wrapper"}),flush=True)


if __name__ == "__main__":
    main()
