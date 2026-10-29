"""Finite single-transition work, called only by an authorized durable owner."""
import torch
import torch.nn.functional as F
from chipmunk_fullm_proxy_math import rms_score, selection
from chipmunk_packed_native_math import packed_action
from chipmunk_history_encoding_math import encode, selected, readonly_action

def exercise(x,w1,b1,w2,old_h,old_y,gate,rows,k,*,measured,mm_fp32,check_mask):
    """Each measured callback must predebit, persist, synchronize and complete.

    One warm, three timed and one comparison per arm; encoding done once per
    compressed arm and charged independently. Reference inputs never enter here.
    """
    if rows.numel()!=64 or rows.unique().numel()!=64:raise ValueError('64 distinct shadow rows required')
    rows=rows.to(x.device)
    def choose(hsample):
        return selection(rms_score(F.linear(x.index_select(0,rows),w1,b1),hsample),k)
    shadow=measured('shadow',{'shadow':1},lambda:old_h.index_select(0,rows).clone())
    enc={}
    masks={};outputs={};timings={}
    for arm in ('A','B','C','D'):
        if arm in ('C','D'):
            kind='fp8' if arm=='C' else 'int8'
            enc[kind]=measured('encode_'+kind,{'encode_'+kind:1},lambda:encode(old_h,rows,kind))
        def action(arm=arm):
            sample=shadow if arm in ('A','B') else enc['fp8' if arm=='C' else 'int8'][1]
            mask=choose(sample)
            if arm=='A': y=packed_action(x,w1,b1,w2,old_h,old_y,gate,mask,mm_fp32=mm_fp32)
            else:
                h=old_h.index_select(1,mask) if arm=='B' else selected(enc['fp8' if arm=='C' else 'int8'][0],mask,enc['fp8' if arm=='C' else 'int8'][2])
                y=(readonly_action(x,w1,b1,w2,old_y,mask,h,mm_fp32=mm_fp32)*gate).to(torch.bfloat16)
            if not bool(torch.isfinite(y).all()):raise ValueError("nonfinite arm output")
            return mask,y
        values=[]
        for repeat in range(5):
            values.append(measured(f'{arm}_{repeat}',{'action_'+arm:1},action))
        masks[arm],outputs[arm]=values[-1]
        measured('mask_sha_'+arm,{'mask_sha':1},lambda:check_mask(arm,masks[arm]))
        if arm=='B':
            ok=measured('gate_AB',{'gate_AB':1},lambda:torch.equal(masks['A'],masks['B']) and torch.equal(outputs['A'].view(torch.uint8),outputs['B'].view(torch.uint8)))
            if not ok:raise ValueError('A/B bitwise gate failed')
        if arm in ('C','D'):
            ok=measured('gate_'+arm,{'gate_'+arm:1},lambda:torch.equal(masks['A'],masks[arm]))
            if not ok:raise ValueError('compressed shadow mask differs')
    return masks,outputs,{kind:{'payload_bytes':q.numel()*q.element_size(),'shadow_bytes':shadow.numel()*shadow.element_size(),'scale_bytes':0 if scale is None else scale.numel()*scale.element_size(),'payload_dtype':str(q.dtype),'shadow_dtype':str(shadow.dtype)} for kind,(q,shadow,scale) in enc.items()}
