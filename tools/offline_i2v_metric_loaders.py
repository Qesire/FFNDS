"""Bind official evaluator loaders to verified local HF assets only.

Scoring equations/frame transforms/aggregation are the original vbench0.1.5
functions. Adapters change checkpoint/cache/source resolution only.
"""
from contextlib import contextmanager
import importlib
import importlib.metadata
import os
from pathlib import Path

from rollout_contract import sha


def verify_source_view(assets, hf):
    """Verify bytes and environment before importing official model code."""
    out = hf / assets['HF_relative_root']
    for name, version in assets['versions'].items():
        if importlib.metadata.version(name) != version:
            raise RuntimeError('Original metric environment version changed: ' + name)
    for item in assets['installed_source_inventory']:
        if sha(out / 'installed_source_view' / item['path']) != item['sha256']:
            raise RuntimeError('Exact evaluator package source view changed')


def install_source_supplement(supplement, hf):
    root = hf / supplement['HF_relative_root']
    for row in supplement['files']:
        if sha(root / row['path']) != row['sha256']:
            raise RuntimeError('Original optional-visualizer source supplement changed')
    import vbench2_beta_i2v
    path = str(root / 'vbench2_beta_i2v')
    if path not in vbench2_beta_i2v.__path__:
        vbench2_beta_i2v.__path__.append(path)


@contextmanager
def offline_loaders(assets, hf, device):
    import torch
    out = hf / assets['HF_relative_root']
    by_name = {row['name']: row for row in assets['assets']}
    verified = set()

    def verify(name):
        record = by_name[name]
        path = hf / record['HF_relative']
        if name not in verified:
            if sha(path) != record['sha256'] or path.stat().st_size != record['bytes']:
                raise RuntimeError('Fixed official evaluator checkpoint changed: ' + name)
            verified.add(name)
        return path

    repositories = {'dino': out / 'source/dino', 'cotracker': out / 'source/cotracker'}
    for key, directory in repositories.items():
        for item in assets['source_repositories'][key]['files']:
            if sha(directory / item['path']) != item['sha256']:
                raise RuntimeError('Pinned evaluator source repository changed')
    verify_source_view(assets, hf)
    old_hub_load = torch.hub.load
    old_state_load = torch.hub.load_state_dict_from_url
    old_download = torch.hub.download_url_to_file
    old_hub_dir = torch.hub.get_dir()

    def no_download(*args, **kwargs):
        raise RuntimeError('Missing staged checkpoint; implicit metric-time download forbidden')

    def hub_load(repo, model, *args, **kwargs):
        if (repo in ('facebookresearch/dino:main', str(repositories['dino'])) and model == 'dino_vitb16'):
            directory = repositories['dino']
        elif repo in ('facebookresearch/co-tracker', str(repositories['cotracker'])) and model == 'cotracker2':
            directory = repositories['cotracker']
        else:
            raise RuntimeError('Unregistered pretrained evaluator source/model')
        for key in ('source', 'trust_repo', 'force_reload'):
            kwargs.pop(key, None)
        return old_hub_load(str(directory), model, *args, source='local', **kwargs)

    def state_load(url, model_dir=None, map_location=None, progress=True, check_hash=False,
                   file_name=None, weights_only=False):
        if url == 'https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth':
            path = verify('DINO_base')
        elif url in ('https://huggingface.co/facebook/cotracker/resolve/main/cotracker2.pth',
                     'https://huggingface.co/facebook/cotracker/resolve/60495999ebd33af25fc51d89f83d1d6de3bc18a5/cotracker2.pth'):
            path = verify('CoTracker2')
        else:
            raise RuntimeError('Unregistered evaluator checkpoint URL')
        return torch.load(path, map_location=map_location, weights_only=weights_only)

    torch.hub.set_dir(str(out / 'torch'))
    torch.hub.load = hub_load
    torch.hub.load_state_dict_from_url = state_load
    torch.hub.download_url_to_file = no_download
    background = importlib.import_module('vbench2_beta_i2v.i2v_background')
    camera = importlib.import_module('vbench2_beta_i2v.camera_motion')
    old_dream = background.dreamsim
    old_camera_init = camera.CameraPredict.__init__

    def dream(*args, **kwargs):
        for row in assets['assets']:
            if row['name'].startswith('dreamsim:'):
                verify(row['name'])
        kwargs.update(cache_dir=str(out / 'dreamsim'), device=device)
        return old_dream(*args, **kwargs)

    def camera_init(self, target_device, submodules_list):
        # Same original grid/model settings; omit only the upstream remote
        # download/TLS-fallback loader. No tracking/scoring code changes.
        self.device = target_device
        self.grid_size = 10
        self.number_points = 1
        self.model = hub_load(submodules_list['repo'], submodules_list['model']).to(self.device)

    background.dreamsim = dream
    camera.CameraPredict.__init__ = camera_init
    try:
        yield verify
    finally:
        background.dreamsim = old_dream
        camera.CameraPredict.__init__ = old_camera_init
        torch.hub.load = old_hub_load
        torch.hub.load_state_dict_from_url = old_state_load
        torch.hub.download_url_to_file = old_download
        torch.hub.set_dir(old_hub_dir)


def submodules(dimension, assets, hf):
    out = hf / assets['HF_relative_root']
    if dimension in ('i2v_subject', 'subject_consistency'):
        result = {'repo_or_dir': str(out / 'source/dino'),
                  'path': str(out / 'dino_model/dino_vitbase16_pretrain.pth'),
                  'model': 'dino_vitb16', 'source': 'local'}
        result.update(resolution='16-9') if dimension == 'i2v_subject' else result.update(read_frame=False)
        return result
    if dimension == 'i2v_background':
        return {'resolution': '16-9'}
    if dimension == 'camera_motion':
        return {'repo': str(out / 'source/cotracker'), 'model': 'cotracker2'}
    if dimension == 'background_consistency':
        return [str(out / 'clip_model/ViT-B-32.pt'), False]
    if dimension == 'motion_smoothness':
        return {'config': str(out / 'installed_source_view/vbench/third_party/amt/cfgs/AMT-S.yaml'),
                'ckpt': str(out / 'amt_model/amt-s.pth')}
    if dimension == 'dynamic_degree':
        return {'model': str(out / 'raft_model/models/raft-things.pth')}
    if dimension == 'aesthetic_quality':
        return [str(out / 'clip_model/ViT-L-14.pt'), str(out / 'aesthetic_model/emb_reader')]
    if dimension == 'imaging_quality':
        return {'model_path': str(out / 'pyiqa_model/musiq_spaq_ckpt-358bb6af.pth')}
    raise ValueError('Only the registered nine official dimensions are allowed')
