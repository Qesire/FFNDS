"""Map the frozen FFN probe rows to image-space proxies for source 0951."""
import hashlib
import json
import sys
import tarfile
from pathlib import Path

import cv2 as cv
import numpy as np

HOME = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HOME / "work/vbench_scale24"))
from diagnose0951 import region

SOURCE = HOME / "outputs/FFNDS_anchor_scale24"
OUT = HOME / "outputs/FFNDS_0951_mechanism/geometry"
GRID = (21, 29, 52)
N = int(np.prod(GRID))
CASES = ("base", "static", "pan_left", "pan_right")
ARMS = ("candidate", "dense")
LAGS = (-3, -1, 0, 2)


def cell_mean(img):
    return img.reshape(29,16,52,16).mean(axis=(1,3))


def classes(frame):
    fg, core, _ = region(frame)
    mass = cell_mean(fg)
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    dx = cv.Sobel(gray,cv.CV_32F,1,0,ksize=3,scale=.125)
    dy = cv.Sobel(gray,cv.CV_32F,0,1,ksize=3,scale=.125)
    yy,xx = np.indices(fg.shape)
    wood = (xx>.62*832)&(xx<.98*832)&(yy>.03*464)&(yy<.86*464)&(~fg)&(np.hypot(dx,dy)>4)
    return {"core":cell_mean(core)>.5, "rim":(mass>0)&(mass<1),
            "rim_central":(mass>=.2)&(mass<=.8),"fence":cell_mean(wood)>.45}


def image_fields(path):
    cap = cv.VideoCapture(str(path))
    labs=[]
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        assert frame.shape==(464,832,3)
        labs.append(classes(frame))
    cap.release()
    assert len(labs)==81
    return {name:np.stack([f[name] for f in labs]) for name in labs[0]}


def historical():
    with tarfile.open(HOME/"work/anchored_scale/stage.tar.gz") as tar:
        base = tar.extractfile("base.json").read()
    seal=(SOURCE/"camera7/MANIFEST.sha256").read_text().splitlines()
    assert any(line.split()==[hashlib.sha256(base).hexdigest(),"base.json"] for line in seal)
    p=np.array(json.loads(base)["observer_ids"])
    assert p.shape==(64,) and len(set(p))==64 and p.min()>=0 and p.max()<N
    return p


def masks_and_hashes():
    result={}
    hashes={}
    for cond in CASES:
        root= SOURCE if cond=="base" else SOURCE/"camera7"
        name="vbi_0951"+("" if cond=="base" else "_"+cond)
        for arm in ARMS:
            p=root/arm/(name+".mp4")
            digest=hashlib.sha256(p.read_bytes()).hexdigest()
            assert json.loads(p.with_suffix(".json").read_bytes())["video"]["mp4_sha256"]==digest
            result[cond+"_"+arm]=image_fields(p)
            hashes[cond+"_"+arm]=digest
    return result,hashes


def temporal_view(fields,lag):
    frames=np.clip(np.arange(21)*4+lag,0,80)
    return {k:v[frames] for k,v in fields.items()}


def low_discrepancy(n,b):
    x,s=0.,1.
    while n:
        n,d=divmod(n,b)
        s/=b
        x+=s*d
    return x


def alternative_panels():
    image=SOURCE/"inputs/vbi_0951.jpg"
    decoded=cv.imdecode(np.frombuffer(image.read_bytes(),np.uint8),cv.IMREAD_COLOR)
    masks=classes(cv.resize(decoded,(832,464),interpolation=cv.INTER_AREA))
    height,width=GRID[1:]
    rng=np.random.default_rng(951)
    def spread(candidates):
        pts=np.argwhere(candidates)
        order=[int(rng.integers(len(pts)))]
        dist=np.sum((pts-pts[order[0]])**2,axis=1)
        for _ in range(20):
            order.append(int(np.argmax(dist)))
            dist=np.minimum(dist,np.sum((pts-pts[order[-1]])**2,axis=1))
        return pts[order]
    # Freeze the two targeted spatial sequences from the original input only.
    center,edge=spread(masks["core"]),spread(masks["rim_central"]&~masks["core"])
    blind,guide,paired=[],[],[]
    for t in range(GRID[0]):
        count=4 if t==10 else 3
        frame_blind=[]
        for j in range(count):
            n=1+t*3+j+(t>10)
            frame_blind.append((int(low_discrepancy(n,2)*height),int(low_discrepancy(n,3)*width)))
        frame_guide=[tuple(center[t]),tuple(edge[t])]
        frame_guide.extend(pt for pt in frame_blind if pt not in frame_guide)
        # Adjacent times spend two of their observations at the same one-cell
        # spatial edge. Edge locations rotate every two latent times.
        y,x=(int(v) for v in edge[t//2])
        nb=[(y+dy,x+dx) for dy,dx in ((-1,0),(1,0),(0,-1),(0,1))
            if 0<=y+dy<height and 0<=x+dx<width]
        neighbor=next((pt for pt in nb if masks["core"][pt]),nb[0])
        frame_pair=[(y,x),neighbor]
        frame_pair.extend(pt for pt in frame_blind if pt not in frame_pair)
        for ids,points in ((blind,frame_blind),(guide,frame_guide[:count]),(paired,frame_pair[:count])):
            ids.extend(np.ravel_multi_index((t,y,x),GRID) for y,x in points)
    assert all(len(p)==len(set(p))==64 for p in (blind,guide,paired))
    return {"balanced64":np.array(blind),"image64":np.array(guide),"edge64":np.array(paired)}


def edge_observability(rows):
    pts=np.stack(np.unravel_index(rows,GRID),axis=1)
    i,j=np.triu_indices(len(rows),1)
    dt,dy,dx=np.abs(pts[i]-pts[j]).T
    edges=set()
    for a,r in enumerate(pts):
        same=np.flatnonzero((pts[:,0]==r[0]) & (np.arange(len(rows))!=a))
        if len(same):
            ds=((pts[same,1:]-r[1:])**2).sum(1)
            b=int(same[np.lexsort((rows[same],ds))[0]])
            edges.add(tuple(sorted((a,b))))
    e=np.array(sorted(edges))
    dist=np.linalg.norm(pts[e[:,0],1:]-pts[e[:,1],1:],axis=1)
    return {"adjacent_xy_edges":int(((dt==0)&(dy+dx==1)).sum()),
            "adjacent_t_same_xy":int(((dt==1)&(dy+dx==0)).sum()),
            "adjacent_t_a_maximum_of_one_cell":int(((dt==1)&(dy+dx<=1)).sum()),
            "gls_edges":len(e),"gls_median_spatial_length_cells":float(np.median(dist)),
            "gls_min_max_spatial_length_cells":[float(dist.min()),float(dist.max())]}


def coverage(rows,field):
    p=np.stack(np.unravel_index(rows,GRID),axis=1)
    hit=field[tuple(p.T)]
    nt=field.reshape(21,-1).sum(axis=1)
    w=np.bincount(p[:,0],minlength=21)
    touch=float(0)
    for t in range(21):
        pts=p[p[:,0]==t]
        if len(pts):
            z=np.ones(GRID[1:],np.uint8)
            z[pts[:,1],pts[:,2]]=0
            touch+=np.count_nonzero(field[t]&(cv.distanceTransform(z,cv.DIST_L2,cv.DIST_MASK_PRECISE)<=2.00001))
    # Preserve all time counts and within-plane configuration, then enumerate
    # every common toroidal spatial shift as an exact geometric reference.
    shifts=sum(np.roll(field[t],(-y,-x),axis=(0,1)).astype(np.int16) for t,y,x in p)
    expect=float(w@nt/(29*52))
    assert np.isclose(shifts.mean(),expect) and int(shifts[0,0])==int(sum(hit))
    return {"hits":int(hit.sum()),"time_hit":int(len(set(p[hit,0]))),"cells":int(nt.sum()),
            "uniform_xy_expected":expect,"translation_q10_q90":np.quantile(shifts,[.1,.9]).tolist(),
            "cover_within_2_cells":float(touch/nt.sum()),"probe_rows_in_region":rows[hit].tolist()}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    cv.setNumThreads(2)
    cfg=json.loads((SOURCE/"camera7/EFFECTIVE_CONFIG.json").read_bytes())
    assert cfg["vae_stride"]==[4,8,8] and cfg["patch_size"]==[1,2,2]
    probes=historical()
    fields,hashes=masks_and_hashes()
    nominal={s+"_"+r:v for s,f in fields.items() for r,v in temporal_view(f,0).items()}
    np.savez_compressed(OUT/"labels.npz",**nominal)
    panels={"actual64":probes,**alternative_panels()}
    results={}
    for s,f in fields.items():
        results[s]={}
        for lag in LAGS:
            masks=temporal_view(f,lag)
            results[s][str(lag)]={name:{r:coverage(ids,m) for r,m in masks.items()} for name,ids in panels.items()}
    data={"grid":GRID,"probes":{k:v.tolist() for k,v in panels.items()},"actual_per_latent_time":np.bincount(probes//(29*52),minlength=21).tolist(),
          "edge_observability":{k:edge_observability(p) for k,p in panels.items()},
          "hashes":hashes,"roi_code_sha256":hashlib.sha256((HOME/"work/vbench_scale24/diagnose0951.py").read_bytes()).hexdigest(),
          "labels_sha256":hashlib.sha256((OUT/"labels.npz").read_bytes()).hexdigest(),"results":results}
    (OUT/"coverage.json").write_text(json.dumps(data,ensure_ascii=False,indent=1,allow_nan=False),encoding="utf8")
    for s in fields:
        print(s,[(k,[r[m]["hits"] for m in ("core","rim","rim_central","fence")],[round(r[m]["cover_within_2_cells"],3) for m in ("core","rim")]) for k,r in results[s]["0"].items()],flush=True)


if __name__=="__main__":
    main()
