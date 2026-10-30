"""Pure CPU matched-output decomposition; no I/O or model dispatch."""
import numpy as np

NAMES=('L2','P2','T2','cross2','dense2','shadow2','student2',
       'gate_L2','gate_P2','gate_T2','gate_cross2','gate_dense2','gate_shadow2','gate_student2')

def bf16_values(raw, rows, width):
    words=np.frombuffer(raw,dtype='<u2')
    if words.size!=rows*width:raise ValueError('BF16 payload shape differs')
    return (words.astype(np.uint32)<<16).view(np.float32).reshape(rows,width)

def ratios(row):
    result=dict(row)
    for prefix in ('','gate_'):
        reference=row[prefix+'dense2']
        for name in ('L2','P2','T2','cross2'):
            result[prefix+name+'_over_dense2']=row[prefix+name]/reference if reference>0 else None
        for denominator in ('shadow2','student2'):
            den=row[prefix+denominator]
            for name in ('L2','P2','T2','cross2'):
                result[prefix+name+'_over_'+denominator]=row[prefix+name]/den if den>0 else None
        residual=row[prefix+'T2']-(row[prefix+'L2']+row[prefix+'P2']+row[prefix+'cross2'])
        result[prefix+'identity_residual']=residual
        scale=sum(abs(row[prefix+name]) for name in ('L2','P2','T2','cross2'))
        bound=64*np.finfo(np.float64).eps*(max(1,row['count'])*5120+1)*scale
        result[prefix+'identity_tolerance']=float(bound)
        result[prefix+'identity_within_tolerance']=bool(abs(residual)<=bound)
        if not result[prefix+'identity_within_tolerance']:raise ValueError('FP64 energy identity exceeds tolerance')
    if row['vector_tolerance_excess_max']>0:raise ValueError('FP64 vector identity exceeds tolerance')
    return result

def aggregate(dense,shadow,student,gate,groups,weights,group_count=84):
    if dense.shape!=shadow.shape or dense.shape!=student.shape or dense.ndim!=2:raise ValueError('shape mismatch')
    if gate.shape!=(dense.shape[1],) or len(groups)!=len(dense) or len(weights)!=len(dense):raise ValueError('coordinate mismatch')
    if not all(np.isfinite(x).all() for x in (dense,shadow,student,gate,weights)):raise ValueError('nonfinite')
    d=dense.astype(np.float64);s=shadow.astype(np.float64);a=student.astype(np.float64)
    g=gate.astype(np.float64);l=a-s;p=s-d;t=a-d
    vector_residual=np.abs(t-(l+p));vector_bound=16*np.finfo(np.float64).eps*np.maximum.reduce((np.abs(a),np.abs(s),np.abs(d)))
    per={}
    for prefix,m in (('',1.0),('gate_',g)):
        lv=l*m;pv=p*m;tv=t*m;dv=d*m;sv=s*m;av=a*m
        for name,value in (('L2',lv*lv),('P2',pv*pv),('T2',tv*tv),('cross2',2*lv*pv),
                           ('dense2',dv*dv),('shadow2',sv*sv),('student2',av*av)):
            per[prefix+name]=np.sum(value,axis=1,dtype=np.float64)
    out=[]
    for i in range(group_count):
        take=groups==i
        vmax=float(np.max(vector_residual[take],initial=0));excess=float(np.max(vector_residual[take]-vector_bound[take],initial=0))
        out.append({'count':int(take.sum()),'weight_sum':float(weights[take].sum(dtype=np.float64)),
                    'vector_residual_max_abs':vmax,'vector_tolerance_excess_max':excess,
                    **{name:float(np.dot(values[take],weights[take])) for name,values in per.items()}})
    return out

def add_groups(total,partial):
    if len(total)!=len(partial):raise ValueError('group count differs')
    for dst,src in zip(total,partial):
        for key,value in src.items():
            dst[key]=max(dst.get(key,0),value) if key in ('vector_residual_max_abs','vector_tolerance_excess_max') else dst.get(key,0)+value
    return total
