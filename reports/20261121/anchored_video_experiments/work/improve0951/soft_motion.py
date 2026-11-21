"""Continuous warm-pixel moments and separate background registration on 0951."""
import argparse
import hashlib
from pathlib import Path
import sys

import cv2 as cv
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"improve_0951"))
from measure import ARMS, CONDS, EVAL, OUT, HISTORY, diag, files, read, write

W,H = diag.W,diag.H
Y,X = np.mgrid[0:H:2,0:W:2].astype(np.float64)
X += .5
Y += .5


def soft_box(coord, left, right, margin):
    return np.clip((coord-left)/margin,0,1)*np.clip((right-coord)/margin,0,1)


# The historical warm selector can merge the fence and bear in one component.
# Fixed inward tapers and continuous weights avoid component-count discontinuities.
ROI = soft_box(X/W,.03,.82,.05)*soft_box(Y/H,.015,.90,.05)


def sigmoid(v):
    return 1./(1.+np.exp(-np.clip(v,-30,30)))


def measure(frame):
    small=cv.resize(frame,(W//2,H//2),interpolation=cv.INTER_AREA)
    lab=cv.cvtColor(cv.GaussianBlur(small,(0,0),1.25).astype(np.float32)/255.,cv.COLOR_BGR2Lab)
    warm=sigmoid((lab[:,:,1]-0)/2.5)*sigmoid((lab[:,:,2]-17)/3.)
    # Squaring suppresses broad nearly-neutral fence/background pixels.
    mass=ROI*warm**2
    total=mass.sum()
    xx,yy=np.sum(mass*X)/total,np.sum(mass*Y)/total
    center=np.array([xx,yy])
    # Relative coordinates disconnect central shape statistics from pure translation.
    cov=np.array([np.sum(mass*(X-xx)**2),np.sum(mass*(Y-yy)**2),np.sum(mass*(X-xx)*(Y-yy))])/total
    return center, [np.sqrt(cov[0]),np.sqrt(cov[1]),cov[2]/np.sqrt(cov[0]*cov[1])], 4.*total/(W*H)


def solve_background(old,now,mask):
    points=cv.goodFeaturesToTrack(old,400,.01,8,mask=mask)
    q,ok,_=cv.calcOpticalFlowPyrLK(old,now,points,None)
    back,rev,_=cv.calcOpticalFlowPyrLK(now,old,q,None)
    good=ok.ravel().astype(bool)&rev.ravel().astype(bool)&(np.linalg.norm(back-points,axis=2).ravel()<1)
    a,inliers=cv.estimateAffinePartial2D(points[good,0],q[good,0],method=cv.RANSAC,ransacReprojThreshold=2.)
    assert a is not None and inliers.sum()>=12
    return np.vstack((a,[0,0,1.])), float(inliers.mean())


def score(path):
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    assert read(path.with_suffix(".json"))["video"]["mp4_sha256"]==digest
    cache=EVAL/"soft_motion"/(digest+".json")
    source_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if cache.is_file():
        cached=read(cache)
        if cached["code"]==source_sha:return cached
    cv.setRNGSeed(42)
    cap=cv.VideoCapture(str(path))
    mask=np.full((H,W),255,np.uint8)
    mask[H//4:3*H//4,W//4:3*W//4]=0
    previous=None
    bg=np.eye(3)
    rows=[]
    while True:
        ok,img=cap.read()
        if not ok:break
        gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        center,shape,area=measure(img)
        support=None
        if previous is not None:
            a,support=solve_background(previous,gray,mask)
            bg=a@bg
        world=np.linalg.solve(bg,[*center,1.])[:2]
        rows.append(dict(screen=center.tolist(),world=world.tolist(),shape=shape,area=area,
                         background_support=support))
        previous=gray
    cap.release()
    assert len(rows)==81
    result=dict(code=source_sha,video=digest,frames=rows)
    write(cache,result)
    return result


def compare_order(first,reference):
    result={}
    for field in ("screen","world"):
        a,b=[np.array([row[field] for row in group["frames"]]) for group in (first,reference)]
        four_a,four_b=a[4:]-a[:-4],b[4:]-b[:-4]
        d=np.linalg.norm(a-b,axis=1)
        diff=np.linalg.norm(four_a-four_b,axis=1)
        directional=np.sum(four_a*four_b)
        result[field]=dict(p50=float(np.median(d)),p90=float(np.quantile(d,.9)),four_p90=float(np.quantile(diff,.9)),
                           gain=float(directional/np.sum(four_b**2)),cos=float(directional/(np.linalg.norm(four_a)*np.linalg.norm(four_b))),
                           segments_p90=[float(np.quantile(v,.9)) for v in np.array_split(d,3)])
    for k in ("shape","area"):
        a,b=[np.array([r[k] for r in z["frames"]]) for z in (first,reference)]
        result[k]=np.quantile(np.abs(a-b),.9,axis=0).tolist()
    return result


def control():
    # Entire synthetic object stays well within the flat measurement window;
    # a neutral, uniform background is used for a known-direction check.
    states=[]
    for sign in (1,-1):
        xy=[]
        for t in range(41):
            a=np.full((H,W,3),130,np.uint8)
            cv.ellipse(a,(240+sign*t,190+sign*t),(35,50),30,0,360,(53,135,211),-1)
            xy.append(measure(a)[0])
        states.append(np.diff(xy,n=1,axis=0))
    step=states[0].mean(0)
    cos=lambda a,b:float(np.sum(a*b)/(np.linalg.norm(a)*np.linalg.norm(b)))
    return dict(measured_positive_step=step.tolist(), directional_cosine_opposite=cos(*states))


def main():
    cli=argparse.ArgumentParser()
    cli.add_argument("mode",choices=("controls","candidates"))
    mode=cli.parse_args().mode
    cv.setNumThreads(2)
    name="vbi_0951_pan_left.mp4"
    old=diag.ROOT/"camera7/candidate"/name
    alt=OUT/"attempt1/warm_old"/name
    aa,bb=[read(p.with_suffix(".json")) for p in (old,alt)]
    assert all(aa[k]==bb[k] for k in ("initial","flows","latents"))
    reference=HISTORY/"runtime/low_only"/name
    dense=diag.ROOT/"camera7/dense"/name
    frozen=dict(alternate_encoding=compare_order(score(alt),score(old)),
                dense_against_low_only=compare_order(score(dense),score(reference)),
                old_against_low_only=compare_order(score(old),score(reference)),
                synthetic=control())
    write(EVAL/"连续暖色动作对照.json",frozen)
    print(frozen,flush=True)
    if mode=="candidates":
        result={}
        for cond in CONDS:
            entries={arm:score(path) for arm in ARMS if (path:=files(cond,arm)) is not None and path.is_file()}
            key="low_only" if "low_only" in entries else "dense"
            if key in entries:
                result[cond]=dict(reference=key,cases={k:compare_order(v,entries[key]) for k,v in entries.items()})
        write(EVAL/"连续暖色动作实测.json",result)


if __name__=="__main__":main()
