"""Stream original post-Q2 innovations without caching old activations on GPU."""
import math

from e0_intrinsic_math import top_support


def innovation_support(current, previous_cpu, fractions, row_tiles):
    import torch
    if current.shape!=previous_cpu.shape or current.ndim!=2 or current.shape[1]%64:
        raise ValueError('Same full-M/H coordinate geometry required')
    m,h=current.shape;groups=h//64
    out={str(f):{'elements':0,'hot_columns':torch.zeros(h,dtype=torch.bool,device=current.device),
                 'row_tiles':{str(t):{'hot_group_tiles':0,'total_group_tiles':0,'work':0} for t in row_tiles}}
         for f in fractions}
    for start in range(0,m,128):
        # Native Q2 input to W2 is BF16. Promote both before subtraction.
        x=current[start:start+128].float()-previous_cpu[start:start+128].to(current.device).float()
        for fraction in fractions:
            item=out[str(fraction)]
            mask=top_support(x.abs(),int(round(h*fraction))) & (x!=0)
            item['elements']+=int(mask.sum());item['hot_columns']|=mask.any(0)
            row_groups=mask.reshape(len(x),groups,64).any(-1)
            for tile in row_tiles:
                number=math.ceil(len(x)/tile)
                padded=torch.zeros((number*tile,groups),dtype=torch.bool,device=x.device)
                padded[:len(x)]=row_groups
                hot=padded.reshape(number,tile,groups).any(1).sum(1)
                real_rows=(len(x)-tile*torch.arange(number,device=x.device)).clamp(max=tile)
                counts=item['row_tiles'][str(tile)]
                counts['hot_group_tiles']+=int(hot.sum());counts['total_group_tiles']+=number*groups
                counts['work']+=int((real_rows*hot).sum())
    for item in out.values():
        item['retained_nonzero_element_fraction']=item.pop('elements')/(m*h)
        item['union_column_fraction_full_M']=float(item.pop('hot_columns').double().mean())
        for counts in item['row_tiles'].values():
            counts['hot_group_tile_fraction']=counts['hot_group_tiles']/counts['total_group_tiles']
            counts['retained_dense_tile_work_fraction']=counts.pop('work')/(m*groups)
    return {'M':m,'H':h,'K_group':64,'coefficient_semantics':'current BF16 post-Q2 minus same-expert previous BF16 post-Q2, promoted before subtraction','fractions':out}
