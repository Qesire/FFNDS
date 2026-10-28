"""Read-only source inventory; no image extraction, model or GPU calls."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REV='577ba45c1653ede26e29996737f42841f2ec9eea'
HF=Path('/home/xiang/.cache/huggingface')
BANKS=('calibration12','calibration24','formal20','disjoint20','diverse_confirmation20','r17_fresh23','e0_fresh_official4_20261026')


def sha(data):return hashlib.sha256(data).hexdigest()


def scan(obj,field,out):
    if isinstance(obj,dict):
        for key,value in obj.items():
            if key in ('image_sha256','legacy_image_fingerprint') and isinstance(value,str) and len(value)==64:
                out[value].add(field)
            scan(value,field,out)
    elif isinstance(obj,list):
        for value in obj:scan(value,field,out)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--hf-root',type=Path,default=HF);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    hf=args.hf_root.resolve();directory=hf/f'hub/models--lxl-158--vbench-i2v/snapshots/{REV}/vbench2_beta_i2v';meta_path=directory/'vbench2_i2v_full_info.json';metadata=meta_path.read_bytes();entries=json.loads(metadata)
    exposed=defaultdict(set);names=set();indices=set();banks={}
    for bank in BANKS:
        path=ROOT/'.runtime/inputs'/bank/'canonical.json';raw=path.read_bytes();rows=json.loads(raw);banks[bank]={'sha256':sha(raw),'records':len(rows)};scan(rows,bank,exposed)
        for row in rows:
            if row.get('image_name'):names.add(row['image_name'])
            if row.get('image_path'):names.add(Path(row['image_path']).name)
            names.update(Path(p).name for p in row.get('img_paths',[]))
            if row.get('source_group','').startswith(('i2v_subject:','i2v_background:')):indices.add(row['source_index'])
    all_rows=[];eligible=[]
    for i,row in enumerate(entries):
        item={'source_index':i,'image_name':row['image_name'],'prompt':row['prompt_en'],'image_type':row.get('image_type'),
              'dimensions':row['dimension'],'known_name_or_ordinal_exposure':i in indices or row['image_name'] in names}
        all_rows.append(item)
        if any(d in ('i2v_subject','i2v_background') for d in row['dimension']):eligible.append(item)
    byname=defaultdict(list)
    for row in all_rows:byname[row['image_name']].append(row['source_index'])
    cached=[];directory=hf/f'ffnds-derived/vbench-i2v/{REV}/crop/16-9'
    for path in sorted(directory.iterdir()):
        if not path.is_file():continue
        raw=path.read_bytes();h=sha(raw)
        cached.append({'path':str(path.relative_to(hf)),'bytes':len(raw),'sha256':h,'metadata_source_indices':byname[path.name],
                       'known_exposure_banks':sorted(exposed.get(h,()))})
    preliminary=[r for r in eligible if not r['known_name_or_ordinal_exposure']]
    result={'status':'READ_ONLY_INVENTORY_NOT_NEW_COHORT','dataset_revision':REV,'metadata_sha256':sha(metadata),'metadata_rows':len(all_rows),
        'eligible_i2v_unique_images':len({x['image_name'] for x in eligible}),'eligible_i2v_rows':len(eligible),
        'eligible_i2v_dimensions':dict(Counter('subject' if 'i2v_subject' in x['dimensions'] else 'background' for x in eligible)),
        'canonical_banks_scanned':banks,'known_image_sha256_count':len(exposed),'cached_images':cached,
        'cached_not_in_known_hashes':sum(not x['known_exposure_banks'] for x in cached),
        'preliminary_name_ordinal_excluded_candidates':len(preliminary),
        'preliminary_by_dimension':dict(Counter('subject' if 'i2v_subject' in x['dimensions'] else 'background' for x in preliminary)),
        'preliminary_by_image_type':dict(Counter(x['image_type'] for x in preliminary)),
        'all_metadata_rows':all_rows,'preliminary_candidates':preliminary,
        'boundary':'NAME_AND_ORDINAL_FILTER_ONLY; NOT A CLAIM OF SHA-DISJOINT OR NEAR-DUPLICATE-DISJOINT; DO NOT ALTER RESERVED CONFIRMATION20',
        'future_selection':'Freeze stratified image-type/subject-background-camera rule before source outputs; recover exact original images, verify SHA vs all historic exposures including reserved confirmation, then near-duplicate/manual semantic audit; select 6-12 distinct image clusters without adaptive redraw.'}
    if args.out.exists():raise ValueError('Fresh output required')
    args.out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')


if __name__=='__main__':main()
