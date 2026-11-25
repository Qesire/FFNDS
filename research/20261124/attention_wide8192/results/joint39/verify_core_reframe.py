"""Exact algebra and real-shape budget checks; no model inference."""
from pathlib import Path
import json
import sympy as s

n,d,k,r=31668,128,64,31
B=s.MatrixSymbol("B",n,d); P=s.MatrixSymbol("P",d,k)
T=s.MatrixSymbol("T",k,r); W=s.MatrixSymbol("W",r,d)
U=s.MatrixSymbol("one",n,1); b=s.MatrixSymbol("b",1,d)
expr=B+B*P*T*W+U*b
affine=B*(s.Identity(d)+P*T*W)+U*b
affine_zero=s.simplify(s.expand(expr-affine))==s.ZeroMatrix(n,d)

Fs,Fa,Gs,Ga,R=s.symbols("F_s F_a G_s G_a R",real=True)
innovation=(Fs-Gs)-(Fa-Ga)
innovation_zero=s.expand(Fs-(Fa+(Gs-Ga)+innovation))==0
recovery_error_zero=s.expand((Fa+(Gs-Ga)+R-Fs)-(R-innovation))==0

Zs,Ns,z1,z2,a1,a2,v1,v2,c1,c2=s.symbols("Zs Ns z1 z2 a1 a2 v1 v2 c1 c2", real=True)
Zbar=Zs+z1+z2
Nbar=Ns+z1*v1+z2*v2
Obar=Nbar/Zbar
Z=Zbar+z1*a1+z2*a2
N=Nbar+z1*(a1*v1+c1)+z2*(a2*v2+c2)
rhs=(z1*(a1*(v1-Obar)+c1)+z2*(a2*(v2-Obar)+c2))/Z
coarse_zero=s.simplify(N/Z-Obar-rhs)==0

A1,A2,m=s.symbols("A1 A2 m",real=True)
pair_excess=s.expand((m-A1)**2+(m-A2)**2
                    -s.Rational(1,2)*(A1-A2)**2
                    -2*(m-(A1+A2)/2)**2)
assert affine_zero and coarse_zero and pair_excess==0 and innovation_zero and recovery_error_zero
H,block,top=40,128,31
tiles=(n+block-1)//block
means_bytes=2*tiles*H*d*4
result={
  "scope":"Symbolic identities and static arithmetic at actual model dimensions. No synthetic numerical activations, GPU run or measured speed.",
  "finite_defect_innovation_identity":innovation_zero,
  "finite_defect_recovery_error_identity":recovery_error_zero,
  "outabs_shared_affine_identity":affine_zero,
  "same_selected_pair_error_lower_bound_identity":str(pair_excess),
  "coarse_omitted_mass_value_exact_error_identity":coarse_zero,
  "real_shape":{"queries":n,"heads":H,"channels_per_head":d,"block_size":block,
                "blocks":tiles,"selected_blocks":top,"omitted_blocks":tiles-top},
  "hypothetical_coarse_core_pair_ratio_vs_full":(tiles-top)/n,
  "hypothetical_all_block_core_pair_ratio_vs_full":tiles/n,
  "current_layer_two_means_FP32_MiB":means_bytes/2**20,
  "current_layer_coarse_output_BF16_MiB":n*H*d*2/2**20,
  "current_layer_coarse_lse_FP32_MiB":n*H*4/2**20,
  "all_coarse_score_materialization_FP32_GiB":n*H*tiles*4/2**30,
  "implementation_conditions":["Use actual valid key counts; last block has 52 keys",
     "Do not double-count exact selected blocks","Stream coarse normalization rather than materializing all scores",
     "Include reductions, support masking, fusion, output and workspace in actual timing"]
}
p=Path(__file__).with_name("core_reframe_math.json")
p.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
print(json.dumps(result,ensure_ascii=False,indent=2))
