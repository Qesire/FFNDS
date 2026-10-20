"""Demonstrate native cached-generator behavior and fresh-request seed reset."""
import argparse
from pathlib import Path
from types import SimpleNamespace

import torch

from ar2_checkpoint import atomic_json, digest
from compare_dense_trajectory import same_tensor_bits


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    from lightx2v.models.schedulers.wan.scheduler import WanScheduler
    import lightx2v.models.schedulers.wan.scheduler as module
    instance = SimpleNamespace(generator=None, _uses_conditioned_latent_prefix=lambda: False)
    shape = (1, 1, 2, 2)
    WanScheduler.prepare_latents(instance, 42, shape)
    first = instance.latents.clone()
    WanScheduler.prepare_latents(instance, 42, shape)
    continued = instance.latents.clone()
    instance.generator = None
    WanScheduler.prepare_latents(instance, 42, shape)
    reset = instance.latents.clone()
    assert not same_tensor_bits(first, continued)
    assert same_tensor_bits(first, reset)
    instance.generator = None
    WanScheduler.prepare_latents(instance, 43, shape)
    other = instance.latents.clone()
    assert not same_tensor_bits(first, other)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, {'status': 'PASS_NATIVE_RNG_CACHE_CAUSE_AND_RESET',
        'scheduler_source': module.__file__, 'scheduler_source_sha256': digest(module.__file__),
        'probe_source_sha256': digest(Path(__file__)), 'device': str(first.device),
        'same_seed_without_reset_differs': True, 'reset_then_same_seed_bitwise': True,
        'reset_then_other_seed_differs': True, 'scope': 'native prepare_latents, not full video no-op'})


if __name__ == '__main__':
    main()
