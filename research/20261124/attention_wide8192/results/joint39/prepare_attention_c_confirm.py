"""Build and freeze independent C/A inputs with the repository's selector."""
from collections import defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sys

import freeze_confirmation as selector


def check(path, expected):
    data=path.read_bytes()
    assert sha256(data).hexdigest()==expected, path.name
    return json.loads(data)


def write(path, value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf8")


def main(root, cohort):
    frozen=root/"work/g20_v0_binding_v2_20261113"
    historical=frozen/"output/VB_BOUND_INPUTS.json"
    hist=check(historical,selector.HISTORICAL_BOUND_SHA256)
    clusters=check(frozen/"g0a_clusters.json","b882383262a6b795c58370a5d0962d289539e6f20de352f1a512bb595692465e")["clusters"]
    official=check(frozen/"vbench_metadata.json","8cf4d34dc11779a336c216a1a71d482927cfa24a1bf736ab5c92425609f129f8")
    assert len(clusters)==833 and len(official)==1118
    assert all(r["record_id"]==f"vbi_{r['official_record_index_zero_based']:04d}" for r in hist["records"])
    cropped=root/"data/VBench/vbench2_beta_i2v/crop/16-9"
    by_name={}
    for cluster in clusters:
        for member in cluster["members"]:
            p=Path(member["path"])
            if p.parent!=cropped or member["variant"]!="vbench_crop169" or not p.is_file():
                continue
            value=(cluster["cluster_id"],member["sha256"],str(p))
            assert by_name.setdefault(p.name,value)==value
    prompt_rows=[]
    name_to_clusters=defaultdict(set)
    for i,row in enumerate(official):
        name=row["image_name"]
        if name not in by_name:
            continue
        cluster,sha256_image,path=by_name[name]
        name_to_clusters[Path(name).stem].add(cluster)
        prompt_rows.append(dict(source_id=f"vbi_{i:04d}",source_cluster_id=cluster,category=row["image_type"],
                 image_sha256=sha256_image,image_path=path,prompt=row["prompt_en"],
                 prompt_sha256=sha256(row["prompt_en"].encode()).hexdigest()))
    by_id={r["source_id"]:r["source_cluster_id"] for r in prompt_rows}
    cohort_rows=json.loads(cohort.read_bytes())
    for row in cohort_rows:
        assert (row["source_cluster_id"],row["image_sha256"])==by_name[Path(row["image_path"]).name][:2]

    # Search run artifacts, not the catalogue's name-reference flags. Names
    # in the catalogue alone do not show that a model output was observed.
    observed=defaultdict(set)
    registered=json.loads(Path("sources_locked.json").read_bytes())["sources"]
    for row in registered:
        observed[row["source_cluster_id"]].add("attention CAL/DEV registry")
    evidence_counts=defaultdict(int)
    record_id=re.compile(r"vbi_[0-9]{4}")
    excluded_directories={"__pycache__","node_modules","repo_snapshot","_guard","_fixharness","_review_throwaway",
                          "triton_cache","torchinductor_cache",".git","weights","plan_assets"}
    for scan_root in (root/"work",root/"evaluation_runs",root/"evaluation_results"):
        for dirname,dirs,files in os.walk(scan_root):
            dirs[:]=[d for d in dirs if d not in excluded_directories and not d.startswith("FFNDS-")]
            for file in files:
                path=Path(dirname)/file
                stem=path.stem
                if path.suffix.lower()==".mp4":
                    ids=set(name_to_clusters.get(stem,()))
                    if record_id.fullmatch(stem) and stem in by_id:
                        ids.add(by_id[stem])
                    for cid in ids:
                        observed[cid].add("archived output video")
                        evidence_counts["archived output video"]+=1
                elif path.suffix==".json" and record_id.fullmatch(stem) and stem in by_id:
                    try:
                        rec=json.loads(path.read_bytes())
                    except (OSError,ValueError):
                        continue
                    if isinstance(rec,dict) and "video" in rec and ("flows" in rec or "latents" in rec):
                        observed[by_id[stem]].add("generation receipt")
                        evidence_counts["generation receipt"]+=1
    write(Path("CANDIDATES.json"),prompt_rows)
    write(Path("OBSERVED.json"),dict(observed_clusters=sorted(observed),basis={k:sorted(v) for k,v in observed.items()},
                                    evidence_counts=dict(evidence_counts)))
    status=selector.main(["--historical-bound",str(historical),"--cohort24",str(cohort),
           "--candidates","CANDIDATES.json","--observed-clusters","OBSERVED.json", "--recipe","SELECTION_RECIPE.json",
           "--method-recipe","CONFIRM_METHOD.json","--output","CONFIRM_SOURCES.json"])
    if status:
        raise SystemExit(status)
    selected=json.loads(Path("CONFIRM_SOURCES.json").read_bytes())
    for row in selected["sources"]:
        assert sha256(Path(row["image_path"]).read_bytes()).hexdigest()==row["image_sha256"]


if __name__=="__main__":
    main(*(Path(p).resolve() for p in sys.argv[1:]))
