"""Portable C2N2 numerical, budget, resource and media checks; no models."""
from c2_contract import ROOT, read
from check_c2_native_results import main as check


def main():
    check(prefix='C2N2', registration_key='registered_c2_environment_smoke',
          complete_status='COMPLETE_C2N2_NATIVE_VIDEOS_PENDING_FULL_REVIEW')
    run = read(ROOT / 'research/20261023/C2N2_native_results.json')
    if run['status'] == 'COMPLETE_C2N2_NATIVE_VIDEOS_PENDING_FULL_REVIEW':
        lease = run['early_CUDA_reservation']
        assert all(lease[k] for k in ('before_CPU_asset_loading', 'same_model_process',
                    'NVML_UUID_matches_locked_GPU', 'own_process_visible_to_NVML'))


if __name__ == '__main__':
    main()
