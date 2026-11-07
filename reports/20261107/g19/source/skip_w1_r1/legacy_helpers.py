"""G19 native trajectory runner, pending separate independent review and ACTIVE.
All data/model operations require an exact reviewed finite Slurm scope.
"""
import argparse,hashlib,importlib,json,os,re,subprocess,sys,time
from pathlib import Path
from runtime_guard import require_compute_allocation,verify_frozen_scope,RunReceipt,sha256_file


def native_latent_geometry(width,height):
    import math
    if width <= 0 or height <= 0: raise ValueError('invalid image size')
    ratio=height/width
    return (int(math.sqrt(480*832*ratio)//16)*2, int(math.sqrt(480*832/ratio)//16)*2)


def verify_inputs(scope,root):
    rows=json.loads((root/scope['input_manifest_file']).read_text(encoding='utf8'))
    if not isinstance(rows,list) or len(rows) != 1: raise RuntimeError('requires 6 or 7 source-seed pairs')
    ids=[(r['run_id'],r['seed']) for r in rows]
    if len(set(ids))!=len(ids) or [list(x) for x in ids] != scope['pair_keys'] or scope['mode']!='SKIP_W1_PILOT': raise RuntimeError('unregistered pairs')
    from PIL import Image
    source_identity={}
    for r in rows:
        if r.get('exposure_status') not in ('UNVERIFIED','HISTORICALLY_EXPOSED','VERIFIED_UNEXPOSED'):raise RuntimeError('missing exposure identity')
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,96}',r['run_id']) or not isinstance(r['seed'],int): raise RuntimeError('bad source metadata')
        if hashlib.sha256(r['prompt'].encode('utf8')).hexdigest()!=r['prompt_sha256']: raise RuntimeError('prompt hash mismatch')
        if r['seed'] not in (43,44) or not re.fullmatch(r'[0-9a-f]{64}',r['image_sha256']): raise RuntimeError('bad seed or image SHA')
        identity=(r['image_path'],r['image_sha256'],r['prompt'],r['prompt_sha256'],r.get('negative_prompt',''))
        if r['run_id'] in source_identity and source_identity[r['run_id']] != identity: raise RuntimeError('source identity changed across seeds')
        source_identity[r['run_id']]=identity
        if sha256_file(r['image_path'])!=r['image_sha256']: raise RuntimeError('source image changed')
        with Image.open(r['image_path']) as image: width,height=image.size
        if native_latent_geometry(width,height)!=(58,104): raise RuntimeError('source native output geometry differs from 832x464')
    if len({v[1] for v in source_identity.values()})!=len(source_identity): raise RuntimeError('duplicate source images')
    for name in ('config.json','vae/config.json'):
        if (Path(scope['base_model_dir'])/name).exists():raise RuntimeError('undeclared priority config '+name)
    assets=scope.get('base_assets_sha256',{})
    if not assets: raise RuntimeError('base pipeline assets must be content bound')
    base=Path(scope['base_model_dir']).resolve()
    required={'Wan2.1_VAE.pth','models_t5_umt5-xxl-enc-bf16.pth','configuration.json','high_noise_model/config.json','low_noise_model/config.json','google/umt5-xxl/tokenizer.json','google/umt5-xxl/tokenizer_config.json','google/umt5-xxl/special_tokens_map.json','google/umt5-xxl/spiece.model'}
    if set(assets)!=required:raise RuntimeError('base asset inventory incomplete or unexpected')
    identities={}
    for relative,spec in assets.items():
        path=(base/relative).resolve()
        if not path.is_relative_to(base) or not path.is_file() or path.stat().st_size!=spec['size']:raise RuntimeError('base asset path/size mismatch '+relative)
        digest=sha256_file(path);expected=spec.get('sha256')
        if expected is None and scope['mode']!='PILOT_PAIR':raise RuntimeError('formal source requires frozen base SHA')
        if expected is not None and (not re.fullmatch(r'[0-9a-f]{64}',expected) or digest!=expected):raise RuntimeError('base asset SHA mismatch '+relative)
        identities[relative]={'size':spec['size'],'sha256':digest,'expected_known':expected is not None}
    for model in ('high','low'):
        spec=scope['checkpoints'][model]
        if sha256_file(spec['path'])!=spec['sha256']:raise RuntimeError('checkpoint changed '+model)
    # Hash every Python file under each frozen source tree, not only entrypoints.
    expected=scope['external_sources_sha256']
    for tree in scope['python_source_roots']:
        paths={str(p.resolve()) for p in Path(tree).rglob('*.py') if p.is_file()}
        expected_paths={str(Path(p).resolve()) for p in expected if Path(p).resolve().is_relative_to(Path(tree).resolve()) and p.endswith('.py')}
        if not paths or paths!=expected_paths:raise RuntimeError('source tree inventory mismatch '+tree)
    return rows,identities


def validate_effective_config(config,scope,find_model_path):
    base=Path(scope['base_model_dir']).resolve()
    for name in ('config.json','vae/config.json'):
        if (base/name).exists():raise RuntimeError('undeclared priority configuration exists: '+name)
    for key,expected in scope['required_runtime_config'].items():
        if config.get(key)!=expected:raise RuntimeError('preload config mismatch: '+key)
    if Path(config.get('model_path','')).resolve()!=base:raise RuntimeError('model path differs')
    for model in ('high','low'):
        val=config.get(model+'_noise_original_ckpt')
        if not val or Path(val).resolve()!=Path(scope['checkpoints'][model]['path']).resolve():raise RuntimeError('checkpoint path mismatch '+model)
    for key in ('dit_original_ckpt','adapter_model_path','dit_quantized_ckpt','high_noise_quantized_ckpt','low_noise_quantized_ckpt','t5_quantized_ckpt','clip_quantized_ckpt','lora_configs','lora_path','config_json'):
        if config.get(key) not in (None,False,'',[]):raise RuntimeError('forbidden alternate model path: '+key)
    for key in ('dit_quantized','t5_quantized','clip_quantized','use_lightvae','dummy_model','lazy_load','unload_modules','compile','use_compile','seq_parallel','parallel','mxfp8_fuse_enable','video_frame_interpolation','video_super_resolution','use_image_encoder'):
        if config.get(key) not in (None,False,''):raise RuntimeError('forbidden model mode: '+key)
    if config.get('quant_method') not in (None,''):raise RuntimeError('quant_method forbidden')
    if config.get('feature_caching') not in (None,'NoCaching'):raise RuntimeError('feature cache forbidden')
    for key,filename in (('t5_original_ckpt','models_t5_umt5-xxl-enc-bf16.pth'),('vae_path','Wan2.1_VAE.pth')):
        if Path(find_model_path(config,key,filename)).resolve()!=(base/filename).resolve():raise RuntimeError('actual asset selection mismatch: '+key)
    if config.get('vae_name') not in (None,'Wan2.1_VAE.pth'):raise RuntimeError('alternate VAE name forbidden')
    return config

def finite_tensor_sha(t,torch):
    t=t.detach().contiguous()
    if not torch.isfinite(t).all().item():raise RuntimeError('nonfinite initial noise or scheduler latent')
    return {'sha256':hashlib.sha256(t.view(torch.uint8).cpu().numpy().tobytes()).hexdigest(),'shape':list(t.shape),'dtype':str(t.dtype)}

def reset_scheduler_generator(scheduler,model):
    if getattr(model,'scheduler',None) is not scheduler:raise RuntimeError('MODEL_SCHEDULER_IDENTITY_MISMATCH')
    scheduler.generator=None

def append_progress(receipt,event):
    with (receipt.path/'PROGRESS.jsonl').open('a',encoding='utf8') as f:
        f.write(json.dumps(event,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
    receipt.result.setdefault('events',[]).append(event)


def require_matching_tap_noise(mode,tag,noise,results):
    if mode=='PILOT_PAIR' and tag=='tap':
        if len(results)!=1 or noise!=results[0]['initial_noise']:
            raise RuntimeError('INITIAL_NOISE_MISMATCH_BEFORE_TAP_FORWARD')


def exclusive_json(path,data):
    with path.open('x',encoding='utf8') as f:
        json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())

def video_digest(path):
    import imageio_ffmpeg
    reader=imageio_ffmpeg.read_frames(str(path),pix_fmt='rgb24')
    meta=next(reader);w,h=meta['size'];fps=meta.get('fps')
    digest=hashlib.sha256();count=0
    try:
        if (w,h)!=(832,464) or fps is None or abs(float(fps)-16.0)>0.01:raise RuntimeError('video dimensions/FPS differ from scope')
        for frame in reader:
            if len(frame)!=w*h*3:raise RuntimeError('decoded frame geometry mismatch')
            digest.update(frame);count+=1
    finally:reader.close()
    if count!=81:raise RuntimeError('video is not complete 81 frames')
    return {'frames':count,'width':w,'height':h,'fps':float(fps),'decoded_rgb24_sha256':digest.hexdigest(),'mp4_sha256':sha256_file(path)}


