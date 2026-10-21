"""One source-derived layout repair; retain the failed default2of4 preflight."""
from torch.sparse import SparseSemiStructuredTensor

from probe_weight_2of4_shape import ROOT, main

if __name__=='__main__':
    SparseSemiStructuredTensor._FUSE_TRANSPOSE=True
    main(ROOT/'config/weight_2of4_fused_layout_preflight.json')
