"""Batched FP64 Torch statistics; no data, model, GPU calls or host sync until materialize."""
import math
from dataclasses import dataclass
import torch

EPS=1e-30
BAND_MIN_AC_FRACTION=1e-12
BANDS=(('low_1_2',1,2),('mid_3_6',3,6),('high_7_plus',7,None))


def need(ok,msg):
    if not ok: raise ValueError(msg)


@dataclass(frozen=True)
class Slot:
    tensor: torch.Tensor
    index: tuple
    is_bool: bool=False


class Session:
    def __init__(self, channel_chunk=1024):
        need(type(channel_chunk) is int and channel_chunk>0,'Invalid channel chunk')
        self.channel_chunk=channel_chunk; self.checks=[]
    def check(self,x,msg): self.checks.append((x.reshape(()),msg))
    def masked(self,x,valid):
        self.check((~valid | torch.isfinite(x)).all(),'Valid result nonfinite')
        return torch.where(valid,x,torch.full_like(x,float('nan')))

    @staticmethod
    def mask_matrix(masks,anchors,device):
        need(bool(masks),'Empty ROI mapping')
        result=torch.zeros((len(masks),anchors),dtype=torch.float64,device=device)
        for i, mask in enumerate(masks.values()):
            need(mask and len(set(mask))==len(mask) and all(type(x) is int and 0<=x<anchors for x in mask),'Invalid spatial mask')
            result[i,mask]=1.
        return result

    @staticmethod
    def per_anchor(x, frequency):
        return torch.sum(x,dim=2) if frequency else torch.sum(x,dim=(0,2)).unsqueeze(0)

    def basis_stats(self, operands, frequency=False, bridge=False):
        sums=close=scale=None
        for start in range(0,operands[0].shape[-1],self.channel_chunk):
            chunk=tuple(x[...,start:start+self.channel_chunk] for x in operands)
            s,c,m=self._basis_stats_chunk(chunk,frequency=frequency,bridge=bridge)
            sums=s if sums is None else sums+s
            close=c if close is None else torch.maximum(close,c)
            scale=m if scale is None else torch.maximum(scale,m)
        return sums,close,scale

    def _basis_stats_chunk(self, operands, frequency=False, bridge=False):
        """Return direct small-difference reductions [metric, frequency-or-1, anchor]."""
        a,b,c,d=operands
        for x in operands: self.check(torch.isfinite(x).all(),'Nonfinite operands')
        energy=lambda x: self.per_anchor(torch.abs(x).square() if x.is_complex() else x.square(),frequency)
        cross=lambda x,y: self.per_anchor(torch.conj(x)*y,frequency)
        if not bridge:
            t,s,cs,ct=operands
            L=ct-t; P=s-t; student=cs-s; I=student-L; total=cs-t
            # All small differences are formed before any squared norm or cross term.
            lp=cross(L,P); li=cross(L,I); pi=cross(P,I); sp=cross(student,P); tc=cross(t,cs)
            values=[energy(t),energy(L),energy(P),energy(I),energy(student),energy(total),energy(cs),energy(ct),
                    2*lp.real,2*li.real,2*pi.real,2*sp.real,tc.real,
                    tc.imag if tc.is_complex() else torch.zeros_like(tc)]
            closure=torch.abs(total-student-P)
        else:
            nt,nc,rt,rc=operands
            error=nc-nt; replay=rc-rt; tb=nt-rt; cb=nc-rc; delta=cb-tb
            signed=2*cross(replay,delta).real
            values=[energy(nt),energy(error),energy(replay),energy(tb),energy(cb),energy(delta),signed]
            closure=torch.abs(error-replay-delta)
        reduce_max=lambda x: x.amax(dim=2) if frequency else x.amax(dim=(0,2)).unsqueeze(0)
        scale=torch.stack([reduce_max(torch.abs(x)) for x in operands]).amax(dim=0)
        return torch.stack(values),reduce_max(closure),scale

    def aggregate(self, stats, closure, scale, roi, start=None, end=None, weights=None, global_scale=False):
        """Summation over frequencies where requested; ROI applied after per-anchor reduction."""
        if weights is not None:
            stats=stats*weights[None,:,None]
        if start is not None:
            stats=stats[:,start:end].sum(dim=1,keepdim=True)
            closure=closure[start:end].amax(dim=0,keepdim=True)
            scale=scale[start:end].amax(dim=0,keepdim=True)
        sums=torch.matmul(stats,roi.T)  # [metric, group, ROI]
        mask=roi.bool()[None,:,:]
        maxclosure=torch.where(mask,closure[:,None,:],0.).amax(dim=2)
        maxscale=(scale.amax(dim=1,keepdim=True).expand(-1,roi.shape[0]) if global_scale
                  else torch.where(mask,scale[:,None,:],0.).amax(dim=2))
        return sums,maxclosure,maxscale

    def component(self, stats, closure, scale, threshold):
        t,l,p,i,student,total,cs,ct,lp,li,pi,sp,tc_re,tc_im=stats.unbind(0)
        self.check((closure <= 1e-10*torch.clamp(scale,min=1.)).all(),'Local-plus-prefix identity failed')
        available=t>threshold
        ratio=lambda v:self.masked(v/t,available)
        norm=torch.sqrt(t*cs); coherent=available & (cs>threshold)
        inner_abs=torch.sqrt(tc_re.square()+tc_im.square())
        phase_ok=coherent & (inner_abs>torch.maximum(torch.as_tensor(threshold,dtype=t.dtype,device=t.device),1e-12*norm))
        return {'reference_energy':t,'reference_available':available,
                'teacher_local_error_fraction':ratio(l),'student_local_error_fraction':ratio(student),
                'prefix_error_fraction':ratio(p),'interaction_error_fraction':ratio(i),
                'twice_teacher_local_prefix_cross_fraction':ratio(lp),
                'twice_teacher_local_interaction_cross_fraction':ratio(li),
                'twice_prefix_interaction_cross_fraction':ratio(pi),
                'L_P_I_energy_signed_remainder_fraction':ratio(total-l-p-i-lp-li-pi),
                'twice_local_prefix_cross_fraction':ratio(sp),'total_error_fraction':ratio(total),
                'energy_identity_signed_remainder_fraction':ratio(total-student-p-sp),
                'operand_identity_max_absolute':closure,'candidate_energy_ratio':ratio(cs),
                'teacher_candidate_energy_ratio':ratio(ct),'candidate_signed_aligned_gain':ratio(tc_re),
                'candidate_complex_coherence_real':self.masked(tc_re/norm,coherent),
                'candidate_complex_coherence_imag':self.masked(tc_im/norm,coherent),
                'candidate_phase_radians':self.masked(torch.atan2(tc_im,tc_re),phase_ok)}

    def bridge_component(self,stats,closure,scale,threshold):
        t,total,replay,tb,cb,delta,signed=stats.unbind(0)
        self.check((closure <= 1e-10*torch.clamp(scale,min=1.)).all(),'Native bridge identity failed')
        valid=t>threshold; ratio=lambda x:self.masked(x/t,valid)
        return {'native_reference_energy':t,'native_total_error_fraction':ratio(total),
            'replay_total_error_fraction':ratio(replay),'teacher_shape_bridge_fraction':ratio(tb),
            'candidate_shape_bridge_fraction':ratio(cb),'combined_shape_bridge_fraction':ratio(delta),
            'twice_replay_bridge_cross_fraction':ratio(signed),
            'energy_signed_remainder_fraction':ratio(total-replay-delta-signed),
            'operand_identity_max_absolute':closure}

    @staticmethod
    def slots(metrics,group,roi):
        return {name:Slot(t,(group,roi),t.dtype==torch.bool) for name,t in metrics.items()}

    def _views(self, inputs, masks, *, spectrum=True, bridge=False):
        arrays=[x.to(dtype=torch.float64) for x in inputs]
        need(len({x.shape for x in arrays})==1 and arrays[0].ndim==3 and arrays[0].shape[0]==21
             and all(x.device==arrays[0].device for x in arrays),'Four matched 21-time operands required')
        if bridge: need(arrays[0].shape[1]==52,'Matched native bridge needs 52 anchors')
        roi=self.mask_matrix(masks,arrays[0].shape[1],arrays[0].device); result={}
        for view,sl in (('all21',slice(None)),('tau1_to20',slice(1,None)),('tau0',slice(0,1))):
            selected=[x[sl] for x in arrays]; length=selected[0].shape[0]
            dc=[x.mean(dim=0,keepdim=True) for x in selected]; ac=[x-m for x,m in zip(selected,dc)]
            metrics={}
            for label, values, factor in (('total',selected,1.),('DC',dc,float(length)),('AC',ac,1.)):
                stats,close,scale=self.basis_stats(values,bridge=bridge)
                sums,cl,sc=self.aggregate(stats,close,scale,roi,global_scale=bridge)
                if factor!=1.: sums=sums*factor
                metrics[label]=self.bridge_component(sums,cl,sc,EPS) if bridge else self.component(sums,cl,sc,EPS)
            bands={}; bins=None
            if spectrum and length>1:
                f=[torch.fft.rfft(x,dim=0) for x in ac]
                w=torch.full((f[0].shape[0],),2./length,dtype=torch.float64,device=arrays[0].device)
                w[0]=1./length
                if length%2==0: w[-1]=1./length
                if not bridge:
                    lhs=torch.sum(w[:,None,None]*torch.abs(f[0]).square()); rhs=torch.sum(ac[0].square())
                    tolerance=torch.maximum(1e-10*torch.maximum(torch.abs(lhs),torch.abs(rhs)),torch.full_like(lhs,1e-10))
                    self.check(torch.abs(lhs-rhs)<=tolerance,'Parseval failed')
                stats,close,scale=self.basis_stats(f,frequency=True,bridge=bridge)
                reference=metrics['AC']['native_reference_energy' if bridge else 'reference_energy']
                threshold=torch.clamp(reference*BAND_MIN_AC_FRACTION,min=EPS)
                if not bridge:
                    sums,cl,sc=self.aggregate(stats,close,scale,roi,weights=w)
                    bins=self.component(sums,cl,sc,threshold)
                for label,lo,hi in BANDS:
                    end=min(f[0].shape[0],hi+1) if hi is not None else f[0].shape[0]
                    sums,cl,sc=self.aggregate(stats,close,scale,roi,start=lo,end=end,weights=w,global_scale=bridge)
                    bands[label]=self.bridge_component(sums,cl,sc,threshold) if bridge else self.component(sums,cl,sc,threshold)
            rois={}
            for j,name in enumerate(masks):
                item={key:self.slots(m,0,j) for key,m in metrics.items()}
                if spectrum and length>1:
                    if bridge:
                        for label in bands: item[label]=self.slots(bands[label],0,j)
                    else:
                        item['frequency']={label:self.slots(m,0,j) for label,m in bands.items()}
                        item['frequency_bins']={str(k):self.slots(bins,k,j) for k in range(1,f[0].shape[0])}
                rois[name]=item
            result[view]=rois
        return result

    def temporal_views(self,teacher,student_raw,student_candidate,teacher_candidate,masks,*,spectrum=True):
        return self._views((teacher,student_raw,student_candidate,teacher_candidate),masks,spectrum=spectrum)

    def native_rounding_bridge(self,native_teacher,native_candidate,replay_teacher,replay_candidate,masks,*,spectrum=True):
        return self._views((native_teacher,native_candidate,replay_teacher,replay_candidate),masks,spectrum=spectrum,bridge=True)

    def materialize(self,tree):
        tensors=[]; seen={}
        def flatten(v):
            if isinstance(v,dict): return {k:flatten(x) for k,x in v.items()}
            if isinstance(v,list): return [flatten(x) for x in v]
            need(isinstance(v,Slot),'Expected Slot leaf'); key=id(v.tensor)
            if key not in seen: seen[key]=len(tensors); tensors.append(v.tensor)
            offset=0
            for i,n in zip(v.index,v.tensor.shape): offset=offset*n+i
            return (seen[key],offset,v.is_bool)
        structure=flatten(tree); check_start=len(tensors)
        tensors.extend(x for x,_ in self.checks)
        need(tensors and len({x.device for x in tensors})==1,'One device required')
        lengths=[x.numel() for x in tensors]; starts=[]; pos=0
        for n in lengths: starts.append(pos); pos+=n
        values=torch.cat([x.reshape(-1).to(dtype=torch.float64) for x in tensors]).detach().to('cpu').tolist()
        for idx,(_,message) in enumerate(self.checks): need(values[starts[check_start+idx]]==1.,message)
        def rebuild(v):
            if isinstance(v,dict):return {k:rebuild(x) for k,x in v.items()}
            if isinstance(v,list):return [rebuild(x) for x in v]
            index,offset,is_bool=v; x=values[starts[index]+offset]
            if is_bool:return bool(x)
            if math.isnan(x):return None
            need(math.isfinite(x),'Unexpected nonfinite result');return x
        return rebuild(structure)
