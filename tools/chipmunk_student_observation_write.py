"""BF16 same-dtype Wan post_process shadow; never mutates live residual."""
def native_bf16_write(torch,value,gate,start,end,residual,attention_unused):
 g=gate.squeeze()
 if value.dtype!=torch.bfloat16 or residual.dtype!=torch.bfloat16 or g.dtype!=torch.bfloat16 or g.ndim!=1 or g.numel()!=value.shape[-1] or residual.ndim!=2 or residual.shape[0]<end:raise ValueError('BF16 postprocess identity differs')
 return residual[start:end].clone().add_(value*g)
