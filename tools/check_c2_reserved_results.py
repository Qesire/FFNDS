"""Portable C2N1 result check; reads saved evidence, imports no model."""
from check_c2_native_results import main


if __name__ == '__main__':
    main(prefix='C2N1', registration_key='registered_c2_reserved_smoke',
         complete_status='COMPLETE_C2N1_NATIVE_VIDEOS_PENDING_FULL_REVIEW')
