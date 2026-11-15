"""Draft normal Wan2.2 pipeline builder. Not a dispatch or approval mechanism."""
import copy

def build_normal_pipeline(base_model_dir, overlay):
    from lightx2v.pipeline import LightX2VPipeline
    from lightx2v.utils.set_config import get_default_config, auto_calc_config
    from lightx2v.utils.utils import validate_config_paths
    required = {'infer_steps':40, 'target_height':480, 'target_width':832,
                'target_video_length':81, 'sample_guide_scale':[3.5,3.5],
                'sample_shift':5.0, 'boundary':0.9, 'enable_cfg':True,
                'parallel':False, 'seq_parallel':False}
    if any(overlay.get(k) != v for k,v in required.items()):
        raise ValueError('original40 frozen overlay mismatch')
    pipe = LightX2VPipeline(task='i2v', model_path=base_model_dir, model_cls='wan2.2_moe')
    pipe.set_infer_config(attn_mode=overlay['self_attn_1_type'], rope_type=overlay['rope_type'],
        infer_steps=40,num_frames=81,height=480,width=832,guidance_scale=[3.5,3.5],sample_shift=5.0,
        fps=overlay['fps'],aspect_ratio=overlay.get('aspect_ratio'),boundary=0.9,
        boundary_step_index=None,denoising_step_list=None,audio_fps=overlay['audio_fps'],
        double_precision_rope=overlay['double_precision_rope'],
        norm_modulate_backend=overlay['norm_modulate_backend'],distilled_sigma_values=None)
    # Equivalent configuration preservation fix used in local LightX2V distill adapter;
    # the upstream argument filter drops target_video_length and other input keys.
    config = get_default_config()
    config.update({k:copy.deepcopy(v) for k,v in vars(pipe).items() if v is not None})
    config.update(copy.deepcopy(overlay))
    config = auto_calc_config(config)
    config.update(copy.deepcopy(overlay))
    if any(config.get(k) != v for k,v in overlay.items()):
        raise RuntimeError('effective configuration mismatch')
    if config.get('model_cls')!='wan2.2_moe' or config.get('task')!='i2v' or config.get('model_path')!=base_model_dir: raise RuntimeError('model identity mismatch')
    for k in ('dit_quantized','t5_quantized','clip_quantized','use_lightvae','dummy_model','lazy_load','unload_modules','compile','use_compile','mxfp8_fuse_enable','video_frame_interpolation','video_super_resolution','use_image_encoder','cfg_parallel'):
        if config.get(k) not in (None,False,''): raise RuntimeError('unsupported execution mode '+k)
    if config.get('feature_caching') not in (None,'NoCaching'): raise RuntimeError('feature cache forbidden')
    if config.get('scheduler_type') not in (None,'','WanScheduler') or config.get('warmup',False) or config.get('changing_resolution',False): raise RuntimeError('unsupported scheduler or warmup')
    for key in ('dit_original_ckpt','low_noise_original_ckpt','high_noise_original_ckpt','dit_quantized_ckpt','adapter_model_path','quant_method','lora_path','lora_configs','denoising_step_list','boundary_step_index'):
        if config.get(key) not in (None,False,'',[]): raise RuntimeError('unexpected model override '+key)
    validate_config_paths(config)
    pipe.runner = pipe._init_runner(config)
    from lightx2v.models.schedulers.wan.scheduler import WanScheduler
    if type(pipe.runner.scheduler) is not WanScheduler: raise RuntimeError('unexpected scheduler class')
    if any(pipe.runner.config.get(k) != v for k,v in overlay.items()):
        raise RuntimeError('runner configuration mismatch')
    return pipe, config
