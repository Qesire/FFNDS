"""Freeze24 additional source-disjoint calibration inputs without viewing test images."""
import hashlib
import io
import json
from pathlib import Path,PurePosixPath
import tarfile

from huggingface_hub import hf_hub_download
from PIL import Image

from ar2_checkpoint import atomic_json,digest
from fetch_formal_inputs import ROOT,REPOSITORY,REVISION,ARCHIVE_FILE,ARCHIVE_SHA


def main():
    protocol_path=ROOT/'config/calibration24_ffn_operands.json';protocol=json.loads(protocol_path.read_text())
    metadata_path=Path(hf_hub_download(REPOSITORY,protocol['metadata_file'],revision=REVISION))
    if digest(metadata_path)!=protocol['metadata_sha256']:raise RuntimeError('Pinned source metadata changed')
    metadata=json.loads(metadata_path.read_text())
    formal_path=ROOT/'.runtime/inputs/formal20/canonical.json';formal=json.loads(formal_path.read_text())
    formal_names={r['image_name'] for r in formal};formal_prompts={r['prompt'] for r in formal}
    formal_sources={r['source_group'] for r in formal}
    formal_images={r['input_provenance']['image_sha256'] for r in formal}
    selected=[]
    for dimension,kind in [('i2v_subject','animal'),('i2v_background','architecture')]:
        rows=[(i,r) for i,r in enumerate(metadata) if dimension in r['dimension']]
        for reference in [r for r in formal if r['dimension']==dimension]:
            source_index,row=rows[reference['filtered_index']]
            if source_index!=reference['source_index'] or row['prompt_en']!=reference['prompt'] or row['image_name']!=reference['image_name']:
                raise RuntimeError('Full source metadata does not reproduce original formal indices')
        candidates=[]
        for filtered_index,(source_index,row) in enumerate(rows):
            group=f'{dimension}:{filtered_index}'
            if (row['image_type']==kind and group not in formal_sources and row['image_name'] not in formal_names
                    and row['prompt_en'] not in formal_prompts):
                candidates.append({**row,'dimension':dimension,'filtered_index':filtered_index,'source_index':source_index,
                    'source_group':group})
        if len(candidates)<12:raise RuntimeError('Insufficient eligible disjoint calibration sources')
        selected.extend(candidates[:12])
    names={r['image_name'] for r in selected}
    if len(names)!=24 or len({r['source_group'] for r in selected})!=24:
        raise RuntimeError('Selected names or underlying source IDs overlap')
    archive=Path(hf_hub_download(REPOSITORY,ARCHIVE_FILE,revision=REVISION))
    if digest(archive)!=ARCHIVE_SHA:raise RuntimeError('Pinned image archive changed')
    root=Path.home()/'.cache/huggingface/ffnds-derived/vbench-i2v'/REVISION/'crop/16-9'
    root.mkdir(parents=True,exist_ok=True);found={}
    with tarfile.open(archive,'r|gz') as handle:
        for member in handle:
            parts=PurePosixPath(member.name).parts
            if len(parts)<3 or parts[-3:-1]!=('crop','16-9') or parts[-1] not in names:continue
            name=parts[-1]
            if name in found or not member.isfile() or not 0<member.size<64<<20:
                raise RuntimeError('Duplicate/nonregular/oversized calibration archive member')
            content=handle.extractfile(member).read((64<<20)+1)
            if len(content)!=member.size:raise RuntimeError('Truncated calibration image')
            with Image.open(io.BytesIO(content)) as image:
                size=list(image.size);image.verify()
            if abs(size[0]/size[1]-16/9)>.02:raise RuntimeError('Incorrect crop ratio')
            image_sha=hashlib.sha256(content).hexdigest();path=root/name
            if path.exists() and digest(path)!=image_sha:raise RuntimeError('Existing cached image differs')
            if not path.exists():
                temp=Path(str(path)+'.tmp');temp.write_bytes(content);temp.replace(path)
            found[name]={'path':str(path),'image_sha256':image_sha,'size':size,'archive_member':member.name}
    if set(found)!=names or len({r['image_sha256'] for r in found.values()})!=24:
        raise RuntimeError('Missing or duplicate source image bytes; stop without replacement')
    old_path=ROOT/'.runtime/inputs/calibration12/recovery_report.json';old=json.loads(old_path.read_text())
    old_images={r['image_sha256'] for r in old['records']}
    research_images=set()
    for task in [0,6]:
        source=json.loads((ROOT/f'.runtime/inputs/task_{task:03d}_seed_20261001.json').read_text())[0]
        research_images.add(digest(source['image_path']))
    new_images={r['image_sha256'] for r in found.values()}
    if new_images&(formal_images|old_images|research_images):raise RuntimeError('New calibration image overlaps formal20, original12 or research sources')
    out=ROOT/'.runtime/inputs/calibration24';out.mkdir(parents=True,exist_ok=True)
    canonical=[]
    for index,row in enumerate(selected):
        image=found[row['image_name']]
        canonical.append({'videoid':f'vbench2_i2v_cal24:{row["source_group"]}','prompt':row['prompt_en'],
            'img_paths':[image['path']],'source_index':row['source_index'],'calibration_seed':42+index,
            'source_group':row['source_group'],'dimension':row['dimension'],'image_type':row['image_type'],
            'input_provenance':{'repository':REPOSITORY,'revision':REVISION,'metadata_sha256':digest(metadata_path),
                'archive_sha256':ARCHIVE_SHA,'source_sample_id':row['source_group'],'image_name':row['image_name'],
                'image_sha256':image['image_sha256'],'prompt_sha256':hashlib.sha256(row['prompt_en'].encode()).hexdigest(),
                'archive_member':image['archive_member'],'filtered_index':row['filtered_index']}})
    target=out/'canonical.json'
    if target.exists() and json.loads(target.read_text())!=canonical:raise RuntimeError('Calibration24 is already frozen with a different identity')
    atomic_json(target,canonical)
    result={'status':'PASS_SOURCE_DISJOINT_CALIBRATION24_INPUT_RECOVERY','protocol_sha256':digest(protocol_path),
        'metadata_path':str(metadata_path),'metadata_sha256':digest(metadata_path),'archive':str(archive),
        'archive_sha256':ARCHIVE_SHA,'canonical_sha256':digest(target),'input_count':24,
        'formal20_sha256':digest(formal_path),'original12_report_sha256':digest(old_path),
        'formal_source_name_prompt_image_overlap':[],'original12_image_overlap':[],'research_image_overlap':[],
        'records':[{'index':i,'source_group':r['source_group'],'image_type':r['image_type'],
            'image_sha256':r['input_provenance']['image_sha256'],'filtered_index':r['input_provenance']['filtered_index']} for i,r in enumerate(canonical)],
        'formal_benchmark_will_be_reused':True,'quality_or_speed_acceptance':False}
    atomic_json(out/'recovery_report.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
