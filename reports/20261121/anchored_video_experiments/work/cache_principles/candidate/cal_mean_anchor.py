"""Zero cross-step state with a complete current 64-probe joint GLS readout."""
import json
import hashlib
import argparse
from pathlib import Path
import torch

from replay import HERE,WORK,setup,budget,fixed,anchor,prep_hit,probe_metric_inverse,hit
from compact import capture,readout


def capture_actual(root,row,phase,m,t):
    a=fixed.panel(root,row["anchor"])
    u=a["U"].cuda()
    y=a["Y_pre_gate"].cuda().bfloat16()
    _,zk=anchor(u,phase,m["sel"])
    return capture(u,y,zk,t,"intercept_features")


def cal_constants(root,rows,m,t,phase):
    # Use equal source weights and equal token weights within each source.
    # Average encoded endpoint features, not raw U/Z before a GELU.
    source_means=[]
    for src in fixed.CAL:
        encoded=capture_actual(root,rows[src],phase,m,t)
        source_means.append({k:v.double().mean(0) for k,v in encoded.items()})
    constants={}
    for key,dtype in (("c",torch.bfloat16),("f",torch.bfloat16),("q",torch.float32)):
        constants[key]=torch.stack([s[key] for s in source_means]).mean(0).to(dtype)
    return constants


EXTRA=tuple("vbi_"+x for x in ("0056","0176","0886","0947","0950","1040","1084","1109"))


def tensor_sha(value):
    data=value.detach().contiguous().cpu().view(torch.uint8).numpy()
    return hashlib.sha256(data.tobytes()).hexdigest()


def main(extra=False):
    rows_out=[]
    sources=EXTRA if extra else tuple(fixed.DEV)
    assert set(EXTRA).isdisjoint((*fixed.CAL,*fixed.DEV))
    binding=hashlib.sha256((WORK/"native-inputs/BINDINGS.json").read_bytes()).hexdigest()
    exported=dict(schema="joint_ffn_cal_mean_v1",cal=list(fixed.CAL),layer=19,inputs_sha256=binding,
                  axis=dict(c="BF16, output/model width d=5120",f="BF16, selected normalized kernel coordinates k=2560 in sel order",
                            q="FP32, rank-16 site-specific modal coordinates"),sites={})
    for expert in ("high","low"):
        root,rows,m,pack,phase=setup(expert)
        t=prep_hit(phase,pack,"cuda",5120,64)
        static=cal_constants(root,rows,m,t,phase)
        exported["sites"][expert+"19"]=dict(means={k:v.cpu().contiguous() for k,v in static.items()},
            site_pack_sha256={k:tensor_sha(v) for k,v in pack.items()},
            means_sha256={k:tensor_sha(v) for k,v in static.items()},
            selected_ids=pack["sel"].cpu().contiguous(),
            bound_input_panels=[rows[c]["anchor"]["sha256"] for c in fixed.CAL])
        for src in sources:
            ids,coords,_,_,y0,truth,observe=budget.panel(root,rows[src],m)
            a,h=[fixed.panel(root,rows[src][key]) for key in ("anchor","hit")]
            u0,u1=a["U"].cuda(),h["U"].cuda()
            _,zk=anchor(u0,phase,m["sel"])
            genuine=capture(u0,y0.bfloat16(),zk,t,"intercept_features")
            avg={k:v.expand(len(u1),-1) for k,v in static.items()}
            gate=h["gate"].reshape(-1).cuda().float()
            for fold,(pool,test) in enumerate(budget.split_panel(ids)):
                p=budget.layout(pool,ids,coords,1120,"uniform")[:64]
                pp=torch.tensor(p,device="cuda")
                inv,ne=probe_metric_inverse(ids[p],coords[p],u1.device)
                ref,_=hit(u1,phase,dict(u0=u0,y0=y0.bfloat16(),z0k=zk),pack,pp,inv,tile_rows=512)
                # Only this arm ignores the held-out row's true anchor at inference.
                zero,_=readout(u1,phase,avg,t,"intercept_features",pp,inv)
                paired,_=readout(u1,phase,genuine,t,"intercept_features",pp,inv)
                obs=observe(p).bfloat16()
                assert torch.equal(ref[pp],obs) and torch.equal(paired[pp],obs) and torch.equal(zero[pp],obs)
                state=budget.eval_cache(truth[test],budget.edge_indices(test,coords,ids),gate)
                s={k:budget.metrics(v[test].float()-y0[test],state) for k,v in
                   (("original",ref),("actual_endpoint",paired),("cal_mean",zero))}
                rows_out.append(dict(expert=expert,source=src,fold=fold,probe_rows=len(p),test_rows=len(test),
                                     probe_edges=ne,probe_bitwise=True,scores=s,
                                     J_ratio_original=s["cal_mean"]["J"]/s["original"]["J"],
                                     J_ratio_actual_endpoint=s["cal_mean"]["J"]/s["actual_endpoint"]["J"],
                                     delta_gated_angle=s["cal_mean"]["gated_angle"]-s["original"]["gated_angle"],
                                     delta_gated_signed_scale=s["cal_mean"]["gated_signed_scale"]-s["original"]["gated_signed_scale"],
                                     delta_gated_amplitude=s["cal_mean"]["gated_amplitude"]-s["original"]["gated_amplitude"],
                                     static_extra_bytes=sum(v.numel()*v.element_size() for v in static.values()),
                                     request_state_bytes_across_steps=0))
            print(expert,src,flush=True)
    report=dict(scope="Current U1, W1k, full native FFN on the same 64 paid rows and the same 16-gain / 64 kernel GLS. Two complementary held-out test folds with common geometry and target. Ground-truth held-out Y0 used only in scoring. Building real anchor caches for the two control arms is outside the zero-state arm.",
                definition="CAL-only equal source/row means of the stored BF16 intercept C=Y0-B(U0), BF16 standardized endpoint f0=(GELU(z0)-beta*z0)/scale and FP32 endpoint q0=GELU(z0)Zmap-U0J. Static means are broadcast to all runtime token positions and cast back to their stored dtype. Like the full-state endpoint construction, BF16 round placement differs from the original deployed differences.",
                cal=list(fixed.CAL),eval_sources=list(sources),original_dev=list(fixed.DEV),sha256_inputs=binding,
                results=rows_out)
    (HERE/("EXTRA_CAL_MEAN.json" if extra else "CAL_MEAN_ANCHOR.json")).write_text(json.dumps(report,indent=1,allow_nan=False),encoding="utf8")
    artifact=HERE/"CAL_MEANS.pt"
    torch.save(exported,artifact)
    loaded=torch.load(artifact,weights_only=True)
    for site,data in loaded["sites"].items():
        assert data["site_pack_sha256"]["sel"]==tensor_sha(data["selected_ids"])
        assert all(data["means_sha256"][k]==tensor_sha(v) for k,v in data["means"].items())


if __name__=="__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("--extra",action="store_true")
    args=cli.parse_args()
    torch.set_num_threads(3)
    torch.backends.cuda.matmul.allow_tf32=False
    with torch.inference_mode():
        main(args.extra)
