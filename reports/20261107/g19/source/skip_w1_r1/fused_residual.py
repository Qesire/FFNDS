"""BF16 boundary preserving residual arithmetic; CUDA only."""
import torch
import triton
import triton.language as tl


@triton.jit
def _residual(left, right, output, n: tl.constexpr, SUB: tl.constexpr, BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(left + idx, idx < n, other=0).to(tl.float32)
    y = tl.load(right + idx, idx < n, other=0).to(tl.float32)
    if SUB:
        value = x - y
    else:
        value = x + y
    tl.store(output + idx, value, idx < n)


def run(left, right, output, *, subtract):
    if any(t.dtype != torch.bfloat16 or t.device.type != 'cuda' or not t.is_contiguous() for t in (left,right,output)):
        raise RuntimeError('residual requires contiguous CUDA BF16 tensors')
    if left.shape != right.shape or left.shape != output.shape or left.device != right.device or left.device != output.device:
        raise RuntimeError('residual operands mismatch')
    n = left.numel(); block = 1024
    _residual[(triton.cdiv(n,block),)](left,right,output,n,SUB=subtract,BLOCK=block)


def validate_synthetic():
    """Small exact-bit CUDA unit check, run only inside reviewed allocation."""
    generator = torch.Generator(device='cpu').manual_seed(7)
    a = torch.randn(1027, generator=generator).to(torch.bfloat16)
    b = torch.randn(1027, generator=generator).to(torch.bfloat16)
    a[:4] = torch.tensor([0.0,-0.0,1.0,-1.0],dtype=torch.bfloat16)
    b[:4] = torch.tensor([-0.0,0.0,1.0,-1.0],dtype=torch.bfloat16)
    a, b = a.cuda(), b.cuda()
    alternatives = {}
    for sub in (True, False):
        output = torch.empty_like(a)
        run(a,b,output,subtract=sub)
        reference = (a.float()-b.float() if sub else a.float()+b.float()).to(torch.bfloat16)
        if not torch.equal(output.view(torch.int16),reference.view(torch.int16)):
            raise RuntimeError('fused BF16 residual synthetic bitwise mismatch')
        torch.sub(a,b,out=output) if sub else torch.add(a,b,out=output)
        mismatches = int(torch.count_nonzero(output.view(torch.int16) != reference.view(torch.int16)).item())
        alternatives['torch_sub_out' if sub else 'torch_add_out'] = {'mismatches':mismatches,'status':'PASS' if mismatches == 0 else 'FAIL'}
    torch.cuda.synchronize()
    return {'status':'PASS','elements':1027,'identity':'BF16 bit patterns, add and subtract','torch_out_alternatives':alternatives}
