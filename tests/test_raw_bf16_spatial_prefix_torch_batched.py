import math
import unittest
import numpy as np
import torch
from tools import raw_bf16_spatial_prefix_diagnostics as old
from tools import raw_bf16_spatial_prefix_torch_batched as new


def compare(a,b,path='root'):
    if isinstance(a,dict):
        assert set(a)==set(b),(path,set(a)^set(b))
        for k in a: compare(a[k],b[k],path+'/'+k)
    elif a is None or isinstance(a,bool): assert a is b,(path,a,b)
    else:
        err=abs(math.atan2(math.sin(a-b),math.cos(a-b))) if path.endswith('phase_radians') else abs(a-b)
        assert err <= 1e-9 + 1e-8*abs(a),(path,a,b)


def tensor(x):return torch.from_numpy(np.array(x,dtype=np.float64,copy=True))


class BatchedControls(unittest.TestCase):
    def test_temporal_signed_phase_tau0_and_null(self):
        t=np.arange(21.)[:,None,None]; a=np.broadcast_to(1+np.sin(2*np.pi*t/21)+np.sin(4*np.pi*t/21),(21,4,3)).copy()
        b=np.broadcast_to(.3+np.cos(2*np.pi*t/21)-np.cos(4*np.pi*t/21),(21,4,3)).copy()
        changed=b.copy();changed[0]+=2; masks={'all':[0,1,2,3],'part':[1,2]}
        cases=[(a,a,b,b),(a,a,-a,-a),(a,.7*a,changed,1.1*a),(np.ones_like(a),)*4]
        session=new.Session(); output=[session.temporal_views(*(tensor(x) for x in case),masks) for case in cases]
        got=session.materialize(output)
        for case,item in zip(cases,got): compare(old.temporal_views(*case,masks,np),item)
        self.assertIsNone(got[0]['all21']['all']['frequency']['low_1_2']['candidate_phase_radians'])
        self.assertLess(got[0]['all21']['all']['frequency_bins']['1']['candidate_phase_radians']*got[0]['all21']['all']['frequency_bins']['2']['candidate_phase_radians'],0)
        self.assertIsNone(got[3]['tau1_to20']['all']['AC']['total_error_fraction'])

    def test_bridge_small_difference_and_frequency(self):
        rng=np.random.default_rng(99); rt=1000+rng.normal(size=(21,52,3)); rc=rt+1e-7*rng.normal(size=rt.shape)
        nt=rt+1e-8*rng.normal(size=rt.shape);nc=rc+1e-8*rng.normal(size=rt.shape)
        masks={'all':list(range(52)),'center':list(range(25)),'far':[50,51]}
        session=new.Session(); data=session.native_rounding_bridge(*(tensor(x) for x in (nt,nc,rt,rc)),masks)
        result=session.materialize(data);compare(old.native_rounding_bridge(nt,nc,rt,rc,masks,np),result)
        self.assertGreater(result['all21']['all']['total']['combined_shape_bridge_fraction'],0)

    def test_finite_and_duplicate_guards(self):
        x=torch.ones((21,3,2),dtype=torch.float64);y=x.clone();y[0,0,0]=float('nan')
        session=new.Session(); out=session.temporal_views(x,x,y,x,{'all':[0,1,2]},spectrum=False)
        with self.assertRaises(ValueError):session.materialize(out)
        with self.assertRaises(ValueError):new.Session().temporal_views(x,x,x,x,{'bad':[1,1]},spectrum=False)

    def test_channel_chunk_matches_original_at_nontrivial_boundary(self):
        rng=np.random.default_rng(120); t=rng.normal(size=(21,3,130))+2
        s=t+.05*rng.normal(size=t.shape); c=s+.03*rng.normal(size=t.shape); d=t+.02*rng.normal(size=t.shape)
        masks={'all':[0,1,2],'one':[1]}
        session=new.Session(channel_chunk=17)
        got=session.materialize(session.temporal_views(*(tensor(x) for x in (t,s,c,d)),masks))
        compare(old.temporal_views(t,s,c,d,masks,np),got)


if __name__=='__main__':unittest.main()
