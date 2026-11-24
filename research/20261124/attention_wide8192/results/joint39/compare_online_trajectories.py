"""Compare the four matched raw flow and latent tensors per DEV source."""
from functools import lru_cache
from hashlib import file_digest
import json
from pathlib import Path
import sys

from safetensors.torch import load_file


def read(path):
    return json.loads(path.read_bytes())


@lru_cache(maxsize=12)
def checked(path, checksum):
    with path.open("rb") as stream:
        assert file_digest(stream,"sha256").hexdigest()==checksum,path.name
    return load_file(str(path),device="cpu")


def relative(x,y):
    value=x.double()-y.double()
    return float(value.norm()/y.double().norm().clamp_min(1e-12))


def main(stage, capture):
    quality=read(stage/"output/ONLINE_RESULT.json")
    assert quality["status"]=="complete" and len(quality["renders"])==4
    by_id={(r["source"],r["arm"]):r for r in quality["renders"]}
    records=[]
    for source,i in (("vbi_0392",3),("vbi_1013",5)):
        folder=capture/f"tasks/{i}/output/capture"
        info=read(folder/"INDEX.json")
        assert info["status"]=="complete" and info["source"]["record_id"]==source
        raw=info["backgrounds"]["wide8192"]
        assert raw["record"]["initial"]==quality["control"][source]["wide8192"]["initial"]

        def get(arm,name):
            if arm=="A":
                key=f"trajectory/{name}"
                return checked(folder/"wide8192"/(key+".safetensors"),raw["assets"][key]["sha256"])
            p=stage/"output"/f"candidate_{arm}"/(source+"_trajectory")/(name+".safetensors")
            return checked(p,by_id[source,arm]["trace_tensors"][name]["sha256"])

        assert all((get(a,"initial")["noise"]==get("A","initial")["noise"]).all() for a in ("B","C"))
        for step in range(4):
            x={a:get(a,f"step{step}") for a in ("A","B","C")}
            for a,b in (("B","A"),("C","A"),("B","C")):
                values={k:relative(x[a][k],x[b][k]) for k in ("flow","latent")}
                if step==0 or (b=="C" and step<3):
                    assert all(v==0 for v in values.values())
                records.append(dict(source=source,step=step,candidate=a,reference=b,relative_l2=values))
    assert len(records)==24
    print(json.dumps(dict(status="complete",quality_job=quality["job"],records=records),allow_nan=False))


if __name__=="__main__":
    main(*(Path(s).resolve() for s in sys.argv[1:]))
