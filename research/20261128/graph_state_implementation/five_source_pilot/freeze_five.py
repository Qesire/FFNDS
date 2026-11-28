"""Select the preregistered stress and median sources from the sealed 50-source run."""
import hashlib
import json
from pathlib import Path
import sys


def read(path):
    raw=Path(path).read_bytes()
    return json.loads(raw),hashlib.sha256(raw).hexdigest()


def run(prior, plan):
    config,_=read(plan)
    cohort,cohort_sha=read(prior/"cohort/cohort.json")
    archive,archive_sha=read(prior/"runs.json")
    assert archive["status"]=="complete" and archive["job"]=="206224" and archive["cohort_sha256"]==cohort_sha==config["cohort_sha256"]
    people={row["source_id"]:row for row in cohort["sources"]}
    selections=[people[source] for source in config["sources"]]
    assert len(selections)==len({x["source_cluster_id"] for x in selections})==5 and all(x["seed"]==42 for x in selections)
    config.update(sources=selections,generation_206224_sha256=archive_sha)
    Path("SOURCES.json").write_text(json.dumps(config,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")


if __name__=="__main__":
    run(*(Path(s).resolve() for s in sys.argv[1:]))
