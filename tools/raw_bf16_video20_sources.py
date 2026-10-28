"""Fixed exposed development image clusters for future raw BF16 video comparison."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BANKS={'consumer20':'.runtime/inputs/disjoint20/canonical.json','formal20':'.runtime/inputs/formal20/canonical.json','confirmation20':'.runtime/inputs/diverse_confirmation20/canonical.json'}
FIT={240,288,304,896,897,899}
FORMAL_SUBJECT=(882,885,888)
CAMERA={96:'Slowly pan the camera to the right.',128:'Slowly dolly the camera forward.',144:'Slowly pan the camera to the left.'}
REV='577ba45c1653ede26e29996737f42841f2ec9eea'
ID='E0_RAW_BF16_FULL160_VIDEO20_SOURCES_20261029'


def sha(data):return hashlib.sha256(data).hexdigest()
def need(value,message):
    if not value:raise ValueError(message)


def make_cohort(hf):
    hf=Path(hf).resolve();banks={k:json.loads((ROOT/v).read_bytes()) for k,v in BANKS.items()};consumer=banks['consumer20'];formal=banks['formal20']
    by=lambda rows,i:next(x for x in rows if x['source_index']==i)
    fit={by(consumer,i)['input_provenance']['image_sha256'] for i in FIT}
    confirmation={x['input_provenance']['image_sha256'] for x in banks['confirmation20']}
    selected=[(x,'consumer20',None) for x in consumer if x['source_index'] not in FIT]+[(by(formal,i),'formal20',None) for i in FORMAL_SUBJECT]+[(by(formal,i),'formal20',CAMERA[i]) for i in CAMERA]
    need(len(selected)==20,'fixed source count')
    entries=[]
    for i,(x,origin,suffix) in enumerate(selected):
        img=Path(x['image_path']).resolve();need(img.is_relative_to(hf),'image outside HF');data=img.read_bytes();image_sha=sha(data)
        need(image_sha==x['input_provenance']['image_sha256'] and image_sha not in fit and image_sha not in confirmation,'image identity or excluded cluster')
        prompt=x['prompt'];need(sha(prompt.encode())==x['prompt_sha256'],'original prompt SHA')
        actual=prompt if suffix is None else prompt.rstrip()+' '+suffix
        entries.append({'source_id':f"lxl-158/vbench-i2v@{REV}:{x['source_group']}",'native_source_ordinal':x['source_index'],'source_group':x['source_group'],
            'stratum':'camera' if suffix else ('subject' if x['dimension']=='i2v_subject' else 'background'),
            'source_origin':origin,'prior_history':'HISTORICALLY_EXPOSED_DEVELOPMENT_NOT_BLIND',
            'original_prompt':prompt,'original_prompt_sha256':x['prompt_sha256'],'prompt':actual,'prompt_sha256':sha(actual.encode()),
            'camera_instruction':suffix,'conditioning_identity':'NEW_CAMERA_PROMPT_NOT_HISTORICAL_CONDITIONING' if suffix else 'ORIGINAL_PROMPT',
            'image':{'root':'hf','path':str(img.relative_to(hf)),'bytes':len(data),'sha256':image_sha},'seed':42,'part':i%2,'index':i})
    obj={'id':ID,'status':'DRAFT_NOT_AUTHORIZED','numeric_path':'BF16_ORIGINAL','sampling':'Fixed explicit image-cluster selection before candidate outputs; no result-based redraw.',
         'source_history':'All sources historically exposed; formal20 images are reused for development, not independent confirmation.',
         'bank_sha256':{k:sha((ROOT/v).read_bytes()) for k,v in BANKS.items()},'excluded_fit_ordinals':sorted(FIT),'sources':entries,
         'video_scope':'Future paired Dense/shared-factor/step-factor full160, one seed42 per source; no execution or fit authorized by cohort.'}
    validate_cohort(obj,hf,verify_bytes=False);return obj


def validate_cohort(cohort,hf,verify_bytes=False):
    hf=Path(hf).resolve();banks={k:json.loads((ROOT/v).read_bytes()) for k,v in BANKS.items()}; rows=banks['consumer20'];formal=banks['formal20'];
    need(cohort['id']==ID and cohort['numeric_path']=='BF16_ORIGINAL' and cohort['bank_sha256']=={k:sha((ROOT/v).read_bytes()) for k,v in BANKS.items()} and cohort['excluded_fit_ordinals']==sorted(FIT),'cohort bank identity')
    expected=[(x,'consumer20',None) for x in rows if x['source_index'] not in FIT]+[(next(x for x in formal if x['source_index']==i),'formal20',None) for i in FORMAL_SUBJECT]+[(next(x for x in formal if x['source_index']==i),'formal20',CAMERA[i]) for i in CAMERA]
    entries=cohort['sources'];need(len(entries)==len(expected)==20 and len({x['image']['sha256'] for x in entries})==len({x['native_source_ordinal'] for x in entries})==20,'unique twenty source clusters')
    excluded={x['input_provenance']['image_sha256'] for x in rows if x['source_index'] in FIT}|{x['input_provenance']['image_sha256'] for x in banks['confirmation20']}
    for i,(item,(orig,origin,suffix)) in enumerate(zip(entries,expected)):
        prompt=orig['prompt'] if suffix is None else orig['prompt'].rstrip()+' '+suffix;image=item['image']; path=Path(image['path'])
        need(item['index']==i and item['part']==i%2 and item['seed']==42 and item['source_origin']==origin and item['native_source_ordinal']==orig['source_index'] and item['source_group']==orig['source_group'] and item['source_id']==f"lxl-158/vbench-i2v@{REV}:{orig['source_group']}" and item['prior_history']=='HISTORICALLY_EXPOSED_DEVELOPMENT_NOT_BLIND','source/part identity')
        need(item['stratum']==('camera' if suffix else ('subject' if orig['dimension']=='i2v_subject' else 'background')) and item['camera_instruction']==suffix and item['conditioning_identity']==('NEW_CAMERA_PROMPT_NOT_HISTORICAL_CONDITIONING' if suffix else 'ORIGINAL_PROMPT'),'conditioning role')
        need(item['original_prompt']==orig['prompt'] and item['original_prompt_sha256']==orig['prompt_sha256']==sha(orig['prompt'].encode()) and item['prompt']==prompt and item['prompt_sha256']==sha(prompt.encode()),'prompt identity')
        need(image['root']=='hf' and not path.is_absolute() and '..' not in path.parts and (hf/path).resolve().is_relative_to(hf) and (hf/path).resolve()==Path(orig['image_path']).resolve() and image['sha256']==orig['input_provenance']['image_sha256'] and image['sha256'] not in excluded,'image locator or excluded identity')
        if verify_bytes:
            data=(hf/path).read_bytes();need(len(data)==image['bytes'] and sha(data)==image['sha256'],'image bytes differ')
    need({s:sum(x['stratum']==s for x in entries) for s in ('subject','background','camera')}=={'subject':10,'background':7,'camera':3},'stratum count')
    return True
