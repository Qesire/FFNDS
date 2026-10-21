"""Recover complete pinned original-O shards for local FFN work, without CUDA."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import time

from huggingface_hub import HfApi,hf_hub_download
from safetensors import safe_open

from ar2_checkpoint import atomic_json,digest

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,default=ROOT/'.runtime/original_o_block20_assets.json')
    ap.add_argument('--cache-dir',type=Path,default=Path.home()/'.cache/huggingface/hub')
    args = ap.parse_args()
    protocol_path = ROOT/'config/original_o_block20_assets.json'
    protocol = json.loads(protocol_path.read_text())
    if (not args.cache_dir.resolve().is_relative_to((Path.home()/'.cache/huggingface').resolve())
            or shutil.disk_usage(Path.home()).free < 240*1024**3):
        raise RuntimeError('Use the shared standard HF cache and retain200GiB plus40GiB staging allowance')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    lock = args.output.with_suffix('.lock').open('w')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    identity = {'protocol_sha256':digest(protocol_path),'source_sha256':digest(Path(__file__)),
                'repo':protocol['repo'],'revision':protocol['revision'],'cache_dir':str(args.cache_dir.resolve())}
    if args.output.exists() and json.loads(args.output.read_text())['identity'] != identity:
        raise RuntimeError('Asset recovery identity changed; preserve the old record and choose a fresh output')
    report = {'status':'RUNNING_ORIGINAL_O_PARTIAL_ASSET_RECOVERY','identity':identity,'files':[],
              'selected_shards':{},'tensor_headers':[],'complete_original_model':False,'O_D_experiment_executed':False}
    atomic_json(args.output,report)
    started = time.perf_counter()
    try:
        info = HfApi().model_info(protocol['repo'],revision=protocol['revision'],files_metadata=True)
        if info.sha != protocol['revision']:
            raise RuntimeError('Pinned original-O revision changed')
        metadata = {v.rfilename:v for v in info.siblings}

        def fetch(name):
            row = metadata[name]
            path = Path(hf_hub_download(protocol['repo'],name,revision=protocol['revision'],cache_dir=args.cache_dir))
            if path.stat().st_size != row.size:
                raise RuntimeError('Original-O file size mismatch')
            actual = digest(path)
            if row.lfs:
                if actual != row.lfs.sha256:
                    raise RuntimeError('Original-O full-shard SHA256 mismatch')
                expected = {'LFS_SHA256':row.lfs.sha256}
            else:
                raw = path.read_bytes()
                gitsha = hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest()
                if gitsha != row.blob_id:
                    raise RuntimeError('Original-O small-file git object identity mismatch')
                expected = {'git_blob_sha1':row.blob_id}
            report['files'].append({'filename':name,'path':str(path),'blob':str(path.resolve()),
                                    'bytes':row.size,'sha256':actual,'expected':expected})
            atomic_json(args.output,report)
            print(f'ORIGINAL_O verified {name}: {actual}',flush=True)
            return path

        indices = {}
        for expert in protocol['experts']:
            fetch(f'{expert}/config.json')
            path = fetch(f'{expert}/diffusion_pytorch_model.safetensors.index.json')
            index = json.loads(path.read_text())['weight_map']
            if any(key not in index for key in protocol['keys']):
                raise RuntimeError('Declared block20 tensors absent from pinned original-O index')
            indices[expert] = index
            report['selected_shards'][expert] = sorted({f'{expert}/{index[key]}' for key in protocol['keys']})
        selected = sorted({name for values in report['selected_shards'].values() for name in values})
        total = sum(metadata[name].size for name in selected)
        if total > 40*1024**3 or shutil.disk_usage(Path.home()).free < total+200*1024**3:
            raise RuntimeError('Unexpected shard selection or shared HF reserve insufficient')
        report['selected_shard_bytes'] = total
        atomic_json(args.output,report)
        paths = {name:fetch(name) for name in selected}
        shapes = {'blocks.20.ffn.0.weight':[13824,5120],'blocks.20.ffn.0.bias':[13824],
                  'blocks.20.ffn.2.weight':[5120,13824],'blocks.20.ffn.2.bias':[5120]}
        for expert in protocol['experts']:
            for key in protocol['keys']:
                name = f'{expert}/{indices[expert][key]}'
                with safe_open(paths[name],framework='pt',device='cpu') as handle:
                    value = handle.get_slice(key)
                    shape,dtype = value.get_shape(),value.get_dtype()
                if shape != shapes[key] or dtype not in ['F32','BF16','F16']:
                    raise RuntimeError('Unexpected original-O FFN geometry or storage precision')
                report['tensor_headers'].append({'expert':expert,'key':key,'shape':shape,'source_dtype':dtype,'filename':name})
        report.update(status='PASS_ORIGINAL_O_BLOCK20_SHARDS_COMPLETE_FILE_HASHES_AND_HEADERS',
                      elapsed_seconds=time.perf_counter()-started,
                      scope='partial original-O assets only; no original-O video model or fixed-input numerical result')
    except BaseException as error:
        report.update(status='FAILED_RESUMABLE',error=str(error))
        raise
    finally:
        atomic_json(args.output,report)


if __name__ == '__main__':
    main()
