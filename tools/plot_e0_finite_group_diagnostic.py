"""Standalone scientific figure for the six-source fixed native E0 study."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--outdir",type=Path,required=True)
    args=p.parse_args()
    data=json.loads(args.input.read_text())
    if data["status"]!="COMPLETE_FIXED_E0_NATIVE_FINITE_DIAGNOSTIC" or data["source_clusters"]!=6:
        raise ValueError("Require the complete six-source native result")
    args.outdir.mkdir(parents=True,exist_ok=True)
    outputs=[args.outdir/f"native_finite_precision.{suffix}" for suffix in ["png","pdf"]]
    if any(p.exists() for p in outputs):
        raise FileExistsError("Preserve the first figure")
    fig,axes=plt.subplots(1,3,figsize=(15,4.3),layout="constrained")
    indices=data["indices"]
    x=np.arange(6)
    for key,label,color in [("local_post_A_over_B","Residual writeback","#3b6fb6"),
                            ("nextnorm_A_over_B","Next norm1","#45a478"),
                            ("flow_A_over_B_by_step","Final-step flow","#cf653a")]:
        values=[r[key][-1] if isinstance(r[key],list) else r[key] for r in data["rankings_A_vs_B"]]
        axes[0].plot(x,values,"o-",label=label,color=color)
    axes[0].axhline(1,color="black",lw=.8,ls="--")
    axes[0].set(xticks=x,xticklabels=[str(i) for i in indices],xlabel="Exposed source index",
                ylabel="A / B native error SSE",title="Local and downstream rankings differ")
    axes[0].legend(fontsize=8)
    colors=plt.cm.tab10(np.linspace(0,.6,6))
    shares=[]
    for source,color in zip(data["sources"],colors):
        values=[]
        for key in ["ffn_output","post","nextnorm"]:
            r=source["native_local_interaction"][key]
            values.append(r["AB_SSE"]/(r["A_SSE"]+r["B_SSE"]))
        values.extend(source["steps"][s]["interaction"]["flow"]["AB_over_A_plus_B_SSE"] for s in [0,3])
        axes[1].plot(range(5),values,"o-",label=str(source["index"]),color=color)
        shares.append(np.mean([r["numerical_boundary_signed_share_of_total_prediction_error"]
                               for r in source["sample_direction_check"]["arms"].values()],axis=0))
    axes[1].axhline(1,color="black",lw=.8,ls="--")
    axes[1].set(xticks=range(5),xticklabels=["FFN2 Y","Residual","Norm1","Flow0","Flow3"],
                ylabel="AB SSE / (A SSE + B SSE)",title="Finite joint response is nonadditive")
    axes[1].legend(title="Source",fontsize=7,ncol=2)
    shares=np.array(shares)
    for j,(row,color) in enumerate(zip(shares,colors)):
        axes[2].scatter(np.arange(3)+(j-2.5)*.04,row,color=color,s=25)
    axes[2].plot(range(3),np.median(shares,axis=0),"_",color="black",ms=20,mew=2)
    axes[2].axhline(0,color="black",lw=.8)
    axes[2].set(xticks=range(3),xticklabels=["Native FFN2","Gate multiply","Residual add"],
                ylabel="Signed share of decoded-to-native error energy",
                title="Writeback rounding dominates this mismatch")
    for ax in axes:
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y",alpha=.2)
    fig.suptitle("Wan2.2 native D-Q / HiF4: fixed groups0 and53 at step0 / block0",fontsize=12)
    for path in outputs:
        fig.savefig(path,dpi=180)
    plt.close(fig)


if __name__=="__main__":
    main()
