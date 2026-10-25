"""All-block passive operands and full-token old GELU derivative means."""
import math

import ffnds_observation_capture as original
from e0_contiguous_capture import ContiguousCapture
from e0_gelu_tangent import derivative
from ffnds_observation_common import row_plan,sha256,write_json


def one_tile_plan(grid,source_id,uniform_count=32):
    p=row_plan(grid,source_id,uniform_count);m=math.prod(grid)
    start=128*((m//128)//2);tile=list(range(start,start+128))
    rows=sorted(set(p['uniform_rows'])|set(tile))
    p.update(rows=rows,coordinates=[[r//(grid[1]*grid[2]),r//grid[2]%grid[1],r%grid[2]] for r in rows],
             neighbor_pairs=[],contiguous_tiles=[tile],tile_sampling='One original center M128 tile plus original random32 controls')
    return p


class AllBlockTangentCapture(ContiguousCapture):
    def __enter__(self):
        self.original_plan=original.row_plan;original.row_plan=one_tile_plan
        try:return original.DiscoveryCapture.__enter__(self)
        except BaseException:
            original.row_plan=self.original_plan
            raise

    def put(self,state,name,value):
        super().put(state,name,value)
        if name=='z' and state['step'] in [0,2]:
            import torch
            if tuple(value.shape)!=(31668,13824):raise ValueError('Full-M native W1 layout changed')
            total=torch.zeros(value.shape[1],device=value.device,dtype=torch.float64)
            for start in range(0,len(value),512):
                total.add_(derivative(value[start:start+512].float()).sum(0,dtype=torch.float64))
            state['samples']['full_M_old_derivative_mean']=(total/len(value)).reshape(1,-1).cpu().contiguous()

    def weight_binding(self,expert,block):
        return {'expert':expert,'block':block,'cache_identity':self.cache_identity,
                'scope':'Full original shard hashes in frozen cache manifest; no duplicate weight panels saved'}

    def finish(self,state):
        expected=self.fields|{'gate','b1','b2'}
        if state['step'] in [0,2]:expected=expected|{'full_M_old_derivative_mean'}
        if set(state['samples'])!=expected or state['seen']!=[0,2] or state['qseen']!=[0,2]:
            raise RuntimeError('Incomplete all-block tangent capture')
        if any(not bool(self.torch.isfinite(v).all()) for v in state['samples'].values()):
            raise RuntimeError('Nonfinite passive operand')
        path=self.out/f"step{state['step']}_block{state['block']:02d}.safetensors"
        if path.exists():raise FileExistsError(path)
        self.save_file(state['samples'],str(path))
        self.scopes.append({'step':state['step'],'block':state['block'],'expert':state['expert'],
            'row_plan':state['plan'],'path':path.name,'sha256':sha256(path),
            'full_M_derivative_rows':31668 if state['step'] in [0,2] else 0,
            'weight_binding':self.weight_binding(state['expert'],state['block'])})
        write_json(self.out/'SCOPES.json',self.scopes);state['samples'].clear()
