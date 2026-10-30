"""Observer math only; explicit Torch object and no model dispatch at import."""
def paired_errors(torch,candidate,dense,gate,write_fn):
 if candidate.shape!=dense.shape:raise ValueError('output shape differs')
 if not torch.isfinite(candidate).all() or not torch.isfinite(dense).all() or not torch.isfinite(gate).all():raise ValueError('nonfinite observer input')
 pre=candidate.float()-dense.float();gated=pre*gate.float()
 # write_fn must faithfully apply the original residual/gate dtype order to
 # each independent input; it is never allowed to return into model forward.
 cw=write_fn(candidate);dw=write_fn(dense)
 return {'pre_error':pre,'ideal_gated_error':gated,'actual_write_error':cw.float()-dw.float(),'dense_pre':dense.float(),'dense_gated':dense.float()*gate.float(),'dense_write':dw.float()}

def prepare_omission(torch,w2,mask,gate):
 """One fixed W2 conversion and diagonal weight per site; D-only gate."""
 if w2.ndim!=2 or gate.squeeze().ndim!=1 or gate.squeeze().numel()!=w2.shape[0]:raise ValueError('gate must broadcast as output vector')
 hidden=w2.shape[1];indices=mask.to(dtype=torch.int64)
 if indices.ndim!=1 or indices.numel()!=torch.unique(indices).numel() or (indices<0).any() or (indices>=hidden).any():raise ValueError('mask identity')
 omitted=torch.ones(hidden,device=w2.device,dtype=torch.bool);omitted[indices]=False
 w=w2.float()[:,omitted];g=gate.squeeze().float();q=(w*w*(g*g)[:,None]).sum(dim=0)
 return {'weight':w,'gate':g,'diagonal_weights':q,'omitted':omitted,'omitted_count':int(hidden-indices.numel())}

def omission_energy(torch,delta,prepared):
 if delta.ndim!=2 or delta.shape[1]!=prepared['omitted'].numel():raise ValueError('projection dimensions')
 d=delta.float()[:,prepared['omitted']];projection=(d@prepared['weight'].T)*prepared['gate'];total=(projection*projection).sum(dim=1);diagonal=(d*d)@prepared['diagonal_weights']
 return {'projection':projection,'total':total,'diagonal':diagonal,'cross':total-diagonal,'omitted_count':prepared['omitted_count']}

def partition_sums(torch,start,end,pre,gated,write,ideal,dense_pre=None,dense_gated=None,dense_write=None,*,h=29,w=52):
 """Per tau x quadrant exact additive energy summaries; full-row chunks."""
 if pre.ndim!=2 or pre.shape!=gated.shape or pre.shape!=write.shape or pre.shape[0]!=end-start:raise ValueError('partition geometry')
 rows=torch.arange(start,end,device=pre.device,dtype=torch.int64);tau=rows//(h*w);y=(rows%(h*w))//w;x=rows%w
 strata=tau*4+(y>=h//2).to(torch.int64)*2+(x>=w//2).to(torch.int64)
 values={'pre_squared':(pre*pre).sum(1),'gated_squared':(gated*gated).sum(1),'write_squared':(write*write).sum(1),'ideal_total':ideal['total'],'ideal_diagonal':ideal['diagonal'],'ideal_cross':ideal['cross'],'ideal_vs_actual_gated_squared':((gated+ideal['projection'])**2).sum(1)}
 if any(v is None or v.shape!=pre.shape for v in (dense_pre,dense_gated,dense_write)):raise ValueError('reference energy required')
 values.update(dense_pre_squared=(dense_pre*dense_pre).sum(1),dense_gated_squared=(dense_gated*dense_gated).sum(1),dense_write_squared=(dense_write*dense_write).sum(1),signed_pre_sum=pre.sum(1),signed_gated_sum=gated.sum(1),signed_write_sum=write.sum(1))
 return {'count':torch.bincount(strata,minlength=84),**{name:torch.bincount(strata,weights=value.double(),minlength=84) for name,value in values.items()}}
