"""Build a compact evidence index; local/proxy PASS never becomes video acceptance."""
import argparse
import json
from pathlib import Path
from ar2_checkpoint import atomic_json,digest

ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,default=ROOT/'reports/worker_20261020/EXPERIMENT_INDEX.json')
    args=ap.parse_args()
    d=ROOT/'reports/worker_20261020'
    entries=[]
    definitions=[
        ('RR1','video','formal20_RR1_COMPARISON.json','original20 observed'),
        ('C24','video','formal20_C24_comparison.json','original20 observed'),
        ('VM1','video','formal20_VM1_comparison.json','original20 observed'),
        ('UB1','video','formal20_UB1_comparison.json','original20 observed'),
        ('UB1','video','disjoint20_UB1_comparison.json','supplement20, now observed'),
        ('LR2','video','LR2_formal20_COMPARISON.json','original20 observed'),
        ('LR2','video','LR2_disjoint20_COMPARISON.json','supplement20 observed'),
        ('WR1','native CV','wr1_native_requantization_CV_RUN.json','local frozen calibration states'),
        ('JR1','native CV','joint_reconstruction_jr1_CV_RUN.json','local frozen calibration states'),
        ('OP1','native CV','OP1_residual_CV_REJECTED_RUN.json','C24 student states; source-split heads only'),
        ('LA1','native CV','LA1_reference_fix_CV_REJECTED_RUN.json','C24 student states, after signed-zero fix'),
        ('FA1','native CV','FA1_CV_REJECTED_RUN.json','C24 student states;8epochs'),
        ('FA2','native CV','FA2_development_CV_RUN.json','C24 student states;64epochs'),
        ('LR1','native CV','LR1_native_risk_RUN.json','C24 student states; both frozen and matched controls'),
        ('RB1','C proxy','RB1_proxy_RUN.json','stopped before training on cluster-size rule'),
        ('RB2','C proxy','RB2_proxy_RUN.json','balanced banks; not native output risk'),
        ('HP1','C proxy','HP1_proxy_RUN.json','quadratic versus linear reused residual features'),
        ('HP2','C proxy','HP2_proxy_RUN.json','MLP versus linear reused residual features'),
        ('TF2','native flow diagnostic','TF2_controller_RUN.json','four supplemental sources, forced D-Q states; forced videos excluded'),
        ('OD1','fixed-input function diagnostic','OD1_local_function_RUN.json','two images, two seeds, two steps at block20; D states/gate, no O trajectory'),
    ]
    for name,scope in [('UB2_formal20_COMPARISON.json','original20 observed'),('UB2_disjoint20_COMPARISON.json','supplement20 observed')]:
        if (d/name).exists():definitions.append(('UB2','video',name,scope))
    if (d/'NF1_native_RUN.json').exists():definitions.append(('NF1','native CV','NF1_native_RUN.json','four fixed omitted-group nonlinear readout folds on C24 student states'))
    for trial,stage,name,scope in definitions:
        path=d/name;value=json.loads(path.read_text())
        row={'trial':trial,'stage':stage,'scope':scope,'status':value['status'],
            'evidence':str(path.relative_to(ROOT)),'sha256':digest(path),
            'official_video_quality_pass':None,'full_FFN_speed':None,'joint_acceptance':False}
        if stage=='video':
            row['official_video_quality_pass']=value['formal20_quality_pass']
            row['relative_drop_percent']={k:100*v['relative_drop'] for k,v in value['metrics'].items()}
            row['candidate_policy_sha256']=value['candidate_policy_sha256']
        entries.append(row)
    speedpath=d/'UB1_complete_FFN_comparison.json';speed=json.loads(speedpath.read_text())
    for row in entries:
        if row['trial']=='UB1' and row['stage']=='video':
            if row['candidate_policy_sha256']!=speed['candidate_policy_sha256']:
                raise RuntimeError('Quality and timing policies differ')
            row['full_FFN_speed']=speed['full_FFN_wall_speed']
            row['speed_scope']='isolated research006, complete160 FFNs; original source and same frozen policy, not end-to-end'
            row['speed_evidence']={'path':str(speedpath.relative_to(ROOT)),'sha256':digest(speedpath)}
            row['joint_acceptance']=bool(row['official_video_quality_pass'] and speed['full_FFN_speed_pass'])
    active=[]
    for name in ['terminal_dense_ub2_identity','terminal_dense_ub2_flow','terminal_dense_ub2_controller','terminal_dense_ub2_acceptance']:
        path=ROOT/'runs/worker_restart_20261020'/name/'RUN.json'
        if path.exists():
            v=json.loads(path.read_text())
            active.append({'trial':'UB2','stage':name,'status':v['status'],'current_step':v.get('stage'),
                'evidence':str(path.relative_to(ROOT)),'sha256':digest(path),
                'scope':'live checkpoint snapshot, no implicit quality or speed acceptance'})
    atomic_json(args.output,{'status':'BUILT_SCOPED_EXPERIMENT_EVIDENCE_INDEX','source_sha256':digest(Path(__file__)),
        'closed_entries':entries,'active_snapshots':active,
        'scope':'all formal/supplement sources have now informed development; local/proxy controls are separate from full video and engineering gates'})
    print(f'Indexed {len(entries)} closed evidence records and {len(active)} live snapshots')


if __name__=='__main__': main()
