"""One deterministic, exposure-labelled image-cluster cohort; no model imports."""
import hashlib
import json
import random
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BANKS={'consumer20':'.runtime/inputs/disjoint20/canonical.json','formal20':'.runtime/inputs/formal20/canonical.json','confirmation20':'.runtime/inputs/diverse_confirmation20/canonical.json'}
REV='577ba45c1653ede26e29996737f42841f2ec9eea'


def sha(b): return hashlib.sha256(b).hexdigest()


def need(ok,msg):
    if not ok: raise ValueError(msg)


def make_cohort(hf):
    hf=Path(hf).resolve(); banks={k:json.loads((ROOT/p).read_bytes()) for k,p in BANKS.items()}
    blocked={x['input_provenance']['image_sha256'] for name in ('formal20','confirmation20') for x in banks[name]}
    rows=banks['consumer20']; dev=[rows[i] for i in (0,12)]; blocked.update(x['input_provenance']['image_sha256'] for x in dev)
    rng=random.Random(42); selected=[]; pools={}
    for dim in ('i2v_subject','i2v_background'):
        pool=sorted((x for x in rows if x['dimension']==dim and x['input_provenance']['image_sha256'] not in blocked),key=lambda x:x['input_provenance']['image_sha256'])
        need(len({x['input_provenance']['image_sha256'] for x in pool})==len(pool) and len(pool)>=3,'insufficient unique source pool')
        rng.shuffle(pool); pools[dim]=[x['source_group'] for x in pool];selected.extend(pool[:3])
    def convert(x,role,part):
        path=Path(x['image_path']).resolve();need(path.is_relative_to(hf),'image outside HF')
        return {'source_id':f"lxl-158/vbench-i2v@{REV}:{x['source_group']}",'native_source_ordinal':x['source_index'],
            'consumer20_index':rows.index(x),'source_group':x['source_group'],'dimension':x['dimension'],'prompt':x['prompt'],
            'prompt_sha256':x['prompt_sha256'],'image':{'root':'hf','path':str(path.relative_to(hf)),'bytes':path.stat().st_size,'sha256':x['input_provenance']['image_sha256']},
            'role':role,'part':part,'seed':42,'history':'HISTORICALLY_EXPOSED_CONSUMER20_DEVELOPMENT'}
    # Preserve one development source and one balanced 3-source calibration assignment per part.
    records=[convert(dev[0],'development',0),convert(dev[1],'development',1)]
    records += [convert(x,'calibration',i%2) for i,x in enumerate(selected)]
    obj={'id':'E0_ORIGINAL_BF16_FULL_DEPTH_SOURCES_20261028','status':'DRAFT_NOT_AUTHORIZED','selection':'consumer20 fallback; image SHA sorted, one Python Random(42), sequential subject then background shuffle, first 3 per stratum; excluded dev/formal20/confirmation20 by image SHA',
         'history':'All eight sources historically exposed; calibration role is for this new raw BF16 scope only.',
         'bank_sha256':{k:sha((ROOT/p).read_bytes()) for k,p in BANKS.items()},'ordered_candidate_pools':pools,'sources':records}
    validate_sources(obj,hf,verify_bytes=True);return obj


def validate_sources(cohort,hf,verify_bytes=False):
    hf=Path(hf).resolve(); records=cohort['sources']; need(cohort['id']=='E0_ORIGINAL_BF16_FULL_DEPTH_SOURCES_20261028' and len(records)==8,'cohort identity')
    need(len({x['image']['sha256'] for x in records})==len({x['source_id'] for x in records})==len({x['native_source_ordinal'] for x in records})==8,'duplicate image/source/ordinal')
    need(sum(x['role']=='development' for x in records)==2 and sum(x['role']=='calibration' for x in records)==6,'role count')
    need(all(sum(x['part']==p and x['role']==role for x in records)==n for p in (0,1) for role,n in (('development',1),('calibration',3))),'part count')
    need(all(sum(x['dimension']==d and x['role']=='calibration' for x in records)==3 for d in ('i2v_subject','i2v_background')),'stratum count')
    banks={k:json.loads((ROOT/p).read_bytes()) for k,p in BANKS.items()}
    need(cohort['bank_sha256']=={k:sha((ROOT/p).read_bytes()) for k,p in BANKS.items()},'source bank changed')
    rows=banks['consumer20']; blocked={x['input_provenance']['image_sha256'] for k in ('formal20','confirmation20') for x in banks[k]}
    blocked.update(rows[i]['input_provenance']['image_sha256'] for i in (0,12)); rng=random.Random(42); chosen=[]; pools={}
    for dim in ('i2v_subject','i2v_background'):
        pool=sorted((x for x in rows if x['dimension']==dim and x['input_provenance']['image_sha256'] not in blocked),key=lambda x:x['input_provenance']['image_sha256']); rng.shuffle(pool)
        pools[dim]=[x['source_group'] for x in pool];chosen.extend(pool[:3])
    need(cohort['ordered_candidate_pools']==pools and [x['source_group'] for x in records]==[rows[0]['source_group'],rows[12]['source_group']]+[x['source_group'] for x in chosen],'fixed selection differs')
    for i,x in enumerate(records):
        origin=rows[x['consumer20_index']]; need(x['source_group']==origin['source_group'] and x['native_source_ordinal']==origin['source_index'] and x['source_id']==f"lxl-158/vbench-i2v@{REV}:{origin['source_group']}" and x['dimension']==origin['dimension'] and x['prompt']==origin['prompt'] and x['image']['sha256']==origin['input_provenance']['image_sha256'] and x['role']==('development' if i<2 else 'calibration') and x['part']==(i if i<2 else (i-2)%2),'source bank identity differs')
        need((hf/x['image']['path']).resolve()==Path(origin['image_path']).resolve(),'source bank locator differs')
    for x in records:
        b=x['image']; p=Path(b['path']); need(b['root']=='hf' and not p.is_absolute() and '..' not in p.parts and (hf/p).resolve().is_relative_to(hf),'image path')
        need(x['seed']==42 and x['history']=='HISTORICALLY_EXPOSED_CONSUMER20_DEVELOPMENT' and sha(x['prompt'].encode())==x['prompt_sha256'],'source identity')
        if verify_bytes:
            data=(hf/p).read_bytes();need(len(data)==b['bytes'] and sha(data)==b['sha256'],'image bytes differ')
    return True
