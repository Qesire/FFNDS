"""Four causal single-transition actions; measured callback owns predebit/sync."""
import torch
import torch.nn.functional as F
from chipmunk_fullm_proxy_math import rms_score,selection
from chipmunk_input_state_math import encode,selected,readonly_action

def exercise(x,old_x,w1,b1,w2,old_h,old_y,gate,rows,k,*,measured,mm_fp32,check_mask):
 if rows.numel()!=64 or rows.unique().numel()!=64:raise ValueError('64 distinct shadow rows required')
 rows=rows.to(x.device);masks={};outputs={};encoding={};old_selected={}
 def make_shadow():return old_h.index_select(0,rows).clone()
 def make_oldx():return old_x.clone(),make_shadow()
 for arm in 'ABCD':
  if arm=='A':cache,shadow,scale=old_h,measured('cache_A',{'cache_A':1},make_shadow),None
  elif arm=='B':cache,shadow,scale=measured('cache_B',{'cache_B':1},lambda:encode(old_h,rows,'fp8'))
  else:cache,shadow=measured('cache_'+arm,{'cache_'+arm:1},make_oldx);scale=None
  encoding[arm]={'payload_bytes':cache.numel()*cache.element_size(),'shadow_bytes':shadow.numel()*shadow.element_size(),'scale_bytes':0 if scale is None else scale.numel()*scale.element_size(),'payload_dtype':str(cache.dtype),'shadow_dtype':str(shadow.dtype)}
  def action():
   mask=selection(rms_score(F.linear(x.index_select(0,rows),w1,b1),shadow),k)
   if arm=='A':old=cache.index_select(1,mask)
   elif arm=='B':old=selected(cache,mask,scale)
   elif arm=='C':old=F.gelu(F.linear(cache,w1.index_select(0,mask),b1.index_select(0,mask)),approximate='tanh')
   else:old=F.gelu(F.linear(cache,w1,b1),approximate='tanh').index_select(1,mask)
   y=(readonly_action(x,w1,b1,w2,old_y,mask,old,mm_fp32=mm_fp32)*gate).to(torch.bfloat16)
   if not bool(torch.isfinite(old).all()) or not bool(torch.isfinite(y).all()):raise ValueError('nonfinite action')
   return mask,y,old
  result=None
  for repeat in range(5):result=measured(f'{arm}_{repeat}',{'action_'+arm:1},action)
  masks[arm],outputs[arm],selected_old=result
  if arm in ('C','D'):old_selected[arm]=selected_old
  measured('mask_sha_'+arm,{'mask_sha':1},lambda:check_mask(arm,masks[arm]))
  if arm!='A':
   equal=measured('mask_gate_'+arm,{'mask_gate_'+arm:1},lambda:torch.equal(masks['A'],masks[arm]))
   if not equal:raise ValueError('mask differs')
 return masks,outputs,encoding,old_selected
