"""Measure texture and motion-compensated instability on the 0951 toy and background."""
import argparse
import hashlib
import json
from pathlib import Path
import cv2 as cv
import numpy as np

ROOT=Path(__file__).resolve().parents[2]/"outputs/FFNDS_anchor_scale24"
OUT=ROOT/"camera7/diagnosis"
W,H=832,464
XX,YY=np.meshgrid(np.arange(W,dtype=np.float32),np.arange(H,dtype=np.float32))


def region(frame):
    lab=cv.cvtColor(cv.GaussianBlur(frame,(0,0),2.5),cv.COLOR_BGR2Lab)
    chroma=(lab[:,:,1]>=128)&(lab[:,:,2]>=141)&(YY<.90*H)&(XX>.03*W)&(XX<.82*W)
    closed=cv.morphologyEx(chroma.astype(np.uint8),cv.MORPH_CLOSE,np.ones((13,13),np.uint8))
    curves,_=cv.findContours(closed,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    fg=np.zeros_like(closed)
    cv.drawContours(fg,[max(curves,key=cv.contourArea)],-1,1,cv.FILLED)
    core=cv.erode(fg,np.ones((11,11),np.uint8)).astype(bool)
    edge=cv.morphologyEx(fg,cv.MORPH_GRADIENT,np.ones((5,5),np.uint8)).astype(bool)
    return fg.astype(bool),core,edge


def warp(a,flow):
    return cv.remap(a,XX+flow[:,:,0],YY+flow[:,:,1],cv.INTER_LINEAR,borderMode=cv.BORDER_REFLECT101)


def flow_pair(a,b,dis):
    small=[cv.resize(x,(W//2,H//2),interpolation=cv.INTER_AREA) for x in (a,b)]
    return [2*cv.resize(dis.calc(small[i],small[1-i],None),(W,H)) for i in (0,1)]


def fence_motion(flow,mask):
    points=np.stack([XX,YY],axis=2)[::8,::8][mask[::8,::8]]
    end=points+flow[::8,::8][mask[::8,::8]]
    transform,inliers=cv.estimateAffinePartial2D(points,end,method=cv.RANSAC,ransacReprojThreshold=1.25)
    if transform is None: raise ValueError("Cannot fit fence displacement")
    return np.dstack(((transform[0,0]-1)*XX+transform[0,1]*YY+transform[0,2],
                      transform[1,0]*XX+(transform[1,1]-1)*YY+transform[1,2])),float(inliers.mean())


def run(path):
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest==json.loads(path.with_suffix(".json").read_bytes())["video"]["mp4_sha256"]
    cap=cv.VideoCapture(str(path))
    frames=[]
    while True:
        ok,img=cap.read()
        if not ok:break
        frames.append(img)
    cap.release()
    assert len(frames)==81 and frames[0].shape==(H,W,3)
    gray=[cv.cvtColor(x,cv.COLOR_BGR2GRAY) for x in frames]
    dis=cv.DISOpticalFlow_create(cv.DISOPTICAL_FLOW_PRESET_MEDIUM)
    measurements=[]
    left,back=flow_pair(gray[0],gray[1],dis)
    preview=[]
    for t in range(1,80):
        fwd,rev=flow_pair(gray[t],gray[t+1],dis)
        g=gray[t].astype(np.float32)
        dx=cv.Sobel(g,cv.CV_32F,1,0,ksize=3,scale=.125)
        dy=cv.Sobel(g,cv.CV_32F,0,1,ksize=3,scale=.125)
        grad=np.hypot(dx,dy)
        fg,core,edge=region(frames[t])
        if t in (1,20,40,60,79):
            rgb=frames[t].copy()
            rgb[edge]=(0,0,255)
            preview.append(cv.resize(rgb,(208,116)))
        fence=(XX>.62*W)&(XX<.98*W)&(YY>.03*H)&(YY<.86*H)&(~fg)&(grad>4)
        fence_f,inf=fence_motion(fwd,fence)
        fence_b,inb=fence_motion(back,fence)
        closure=np.maximum(np.linalg.norm(fwd+warp(rev,fwd),axis=2),np.linalg.norm(back+warp(left,back),axis=2))
        valid=closure<1
        for flow in (fwd,back):
            valid&=(XX+flow[:,:,0]>2)&(XX+flow[:,:,0]<W-3)&(YY+flow[:,:,1]>2)&(YY+flow[:,:,1]<H-3)
        residual=fwd+back-fence_f-fence_b
        normal=np.abs(residual[:,:,0]*dx+residual[:,:,1]*dy)/np.maximum(grad,1e-4)
        err=np.abs(g-0.5*(warp(gray[t-1].astype(np.float32),back)+warp(gray[t+1].astype(np.float32),fwd)))
        def band(im):
            raw=im.astype(np.float32)
            return raw-cv.GaussianBlur(raw,(0,0),1.4)
        h=band(g)
        neigh=.5*(warp(band(gray[t-1]),back)+warp(band(gray[t+1]),fwd))
        mask=core&valid
        coherence=float(np.corrcoef(h[mask],neigh[mask])[0,1])
        texture=np.abs(h)
        rim=edge&(grad>4)
        if min((core&valid).sum(),(rim&valid).sum(),fence.sum())<100: raise ValueError((path,t))
        measurements.append(dict(frame=t,body_area_pct=100*fg.mean(),fence_fit_inlier_pct=50*(inf+inb),
            body_texture=float(texture[core].mean()),fence_texture=float(texture[fence].mean()),
            body_compensated_flicker=float(err[mask].mean()),body_band_correlation=coherence,
            body_compensated_flicker_tight=float(err[core&valid&(closure<.5)].mean()),
            coherent_band_response=float(np.mean(h[mask]*neigh[mask])),
            fence_compensated_flicker=float(err[fence&valid].mean()),edge_normal_accel_p75=float(np.percentile(normal[rim&valid],75)),
            body_valid_pct=100*(core&valid).sum()/core.sum(),edge_valid_pct=100*(rim&valid).sum()/rim.sum()))
        left,back=fwd,rev
    def median(rows):
        return {key:float(np.median([r[key] for r in rows])) for key in measurements[0] if key!="frame"}
    return dict(version=2,sha256=digest,all=median(measurements),early=median(measurements[:26]),late=median(measurements[52:]),by_frame=measurements),np.hstack(preview)


def main():
    cli=argparse.ArgumentParser()
    cli.add_argument("conditions",nargs="+")
    args=cli.parse_args()
    OUT.mkdir(exist_ok=True)
    cv.setNumThreads(2)
    cv.setRNGSeed(42)
    for cond in args.conditions:
        for arm in ("candidate","dense"):
            base=ROOT if cond=="base" else ROOT/"camera7"
            name="vbi_0951"+("" if cond=="base" else "_"+cond)
            out=OUT/(cond+"_"+arm+".json")
            if out.exists() and json.loads(out.read_bytes()).get("version")==2: continue
            result,preview=run(base/arm/(name+".mp4"))
            out.write_text(json.dumps(result,indent=1,allow_nan=False),encoding="utf8")
            cv.imwrite(str(OUT/(cond+"_"+arm+"_mask.jpg")),preview)
            print(cond,arm,{k:round(v,3) for k,v in result["all"].items()},flush=True)


if __name__=="__main__":
    main()
