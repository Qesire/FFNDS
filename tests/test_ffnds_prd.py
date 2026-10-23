"""AutoResearcher PRD applied to FFNDS: CPU algebra only, never a native Wan result."""
import json,math,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from validate_representation import validate_ledger
def LN(xs,eps=1e-6):
    u=sum(xs)/len(xs)
    s=math.sqrt(sum((x-u)**2 for x in xs)/len(xs)+eps)
    return [(x-u)/s for x in xs]
def dot(a,b):return sum(x*y for x,y in zip(a,b))
def norm(a,b):return math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))
class FfnPrdTests(unittest.TestCase):
    def test_ledger_and_experimental_evidence_boundary(self):
        j=json.loads((ROOT/"research/20261023/FFNDS_PRD_representation_ledger.json").read_text())
        validate_ledger(j)
        self.assertEqual(len(j["candidates"]),3)
        self.assertTrue(all(c["status"]=="SEED" for c in j["candidates"]))
        d=json.loads((ROOT/"research/CURRENT_DECISION.json").read_text())
        self.assertEqual(d["eligible_model_method_ids"],[])
        r=json.loads((ROOT/"research/20261023/R17E0_execution_summary.json").read_text())
        self.assertEqual(r["actual_debits"]["metric_evaluator_calls"],0)
    def test_ideal_group_sum_and_metric_interaction(self):
        gelu=lambda t:t/2*(1+math.tanh(math.sqrt(2/math.pi)*(t+.044715*t**3)))
        x=[.7,-1.5];w=[[1.1,-.3],[-.4,.8],[.9,.2]]
        out=[[.5,-1.],[1.,.4],[-.3,.8]]
        hs=[gelu(dot(x,wi)) for wi in w]
        pieces=[[hs[i]*v for v in out[i]] for i in range(3)]
        total=[sum(p[k] for p in pieces) for k in range(2)]
        self.assertLess(norm(total,[sum(p[k] for p in pieces[:1])+sum(p[k] for p in pieces[1:]) for k in range(2)]),1e-12)
        e=[1.,2.];g=[.5,-.1];h=[-.3,.2]
        gain=lambda v:dot(e,e)-dot([a-b for a,b in zip(e,v)],[a-b for a,b in zip(e,v)])
        self.assertAlmostEqual(gain([a+b for a,b in zip(g,h)])-gain(g)-gain(h),-2*dot(g,h),places=12)
    def test_mask_dependent_compensation_breaks_additivity(self):
        # Y(S)=sum(D_g for g in S)-C(S), with nonlinear mask-dependent correction.
        D={"g":3.,"h":2.}; C={"":0.,"g":1.,"h":0.,"gh":4.}
        y=lambda ss:sum(D[x] for x in ss)-C[ss]
        self.assertEqual(y("gh")-y("g")-y("h")+y(""),-3.)
    def test_norm_hidden_scale_and_sensitivity(self):
        r=[2.,-1.,0.,3.];d=[.1,-.2,.05,.3];scaled=[10*x for x in r]
        self.assertLess(norm(LN(r),LN([x+19 for x in r])),1e-12)
        self.assertLess(norm(LN(r),LN(scaled)),1e-5)
        eps=1e-4
        J=lambda v:norm(LN([x+eps*y for x,y in zip(v,d)]),LN(v))/eps
        self.assertTrue(8<J(r)/J(scaled)<12)
    def test_conditional_action_value_not_best_fixed(self):
        two_states=[{"A":2.,"B":-1.},{"A":-1.,"B":2.}]
        fixed=max(sum(s[a] for s in two_states)/2 for a in ("A","B"))
        conditioned=sum(max(s.values()) for s in two_states)/2
        self.assertEqual((fixed,conditioned),(.5,2.))
if __name__=="__main__":unittest.main()
