"""Shared GPU cache operations; requires caller-provided torch tensors."""
def encode(F,pred,dtype):
 return (F.float()-pred.float()).to(dtype)
def decode(pred,cache,dtype,output_dtype):
 if cache.dtype!=dtype or cache.device!=pred.device or cache.shape!=pred.shape:raise RuntimeError('cache identity')
 return (pred.float()+cache.float()).to(output_dtype)
