"""CPU verification of the archived joint replay's numerical and source semantics."""
from pathlib import Path
import argparse, gzip, hashlib, json, math
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parent.parent/'joint_action')
    parser.add_argument('--out', type=Path, default=Path(__file__).with_name('runtime_semantics_verified.json'))
    args = parser.parse_args()
    source = args.source
    raw = gzip.decompress((source/'JOINT_ACTION_201276.json.gz').read_bytes())
    record = json.loads(raw)
    assert record['status'] == 'complete'
    assert hashlib.sha256(raw).hexdigest() == '73f5d6f446f2a14d307170df3851ad186857849956f0b9c778a9da6cad1daf66'
    assert record['source_sha256'] == sha(source/'attention_joint_action.py')
    assert record['method_sha256'] == '52fd025d3b82eb7be12002e38b98ddffa7e5417932dbc3ee93cb86d209c8c752'
    assert set(record['sources']) == {'vbi_0392','vbi_1013'}
    stages = {'attention_head','output_projection','attention_gate_response','self_residual','cross_response','cross_residual','ffn_input','ffn_response','ffn_gate_response','block_exit'}
    expected_sites = {f'{step}:{layer}' for step in (1,3) for layer in (0,13,26,39)}
    details = []
    eigenmins = []
    max_probe = max_live = 0.
    stage_count = dense_count = 0
    for sid, source_row in record['sources'].items():
        assert source_row['trace_exact']
        assert source_row['counters'] == {'sites':16,'anchors':8,'peak_slots':4,'hits':8,'native_probe_queries':512}
        assert set(source_row['sites']) == expected_sites
        for site, row in source_row['sites'].items():
            tail = row['wan_wide8192_tail']
            assert tail['shared_probe_ffn_input_exact'] and tail['reused_live_ffn_probe_teacher']
            assert tail['dense_native_ffn_extra_held_rows'] == 448
            assert tail['axes'] == [448,5120]
            assert tail['shared_probe_vs_live_relative_l2'] == 0.
            max_probe = max(max_probe,tail['shared_probe_vs_live_relative_l2'])
            max_live = max(max_live,tail['live_relative_l2'])
            grams = tail['stage_gram_raw_transport']
            assert set(grams) == stages
            for g in grams.values():
                g=np.array(g); assert g.shape==(2,2)
                assert np.array_equal(g,g.T)
                eig=np.linalg.eigvalsh(g);assert eig[0] >= -1e-10*max(g.diagonal().max(),1e-10)
                eigenmins.append(float(eig[0]));stage_count+=1
            backgrounds=tail['stage_gram_raw_transport_background']
            assert set(backgrounds) == {'ffn_response','ffn_gate_response','block_exit'}
            for name,g in backgrounds.items():
                g=np.array(g); assert g.shape==(3,3) and np.array_equal(g,g.T)
                assert np.allclose(g[:2,:2],np.array(grams[name]),rtol=1e-10,atol=1e-12)
                assert np.linalg.eigvalsh(g)[0]>=-1e-10*max(g.diagonal().max(),1e-10)
                dense_count+=1
            d={name:np.array(g).diagonal() for name,g in grams.items()}
            details.append({'source':sid,'site':site,
                 'energy_ratio':{k:float(v[1]/v[0]) for k,v in d.items()},
                 'self_residual_minus_gate_gram':(np.array(grams['self_residual'])-np.array(grams['attention_gate_response'])).tolist(),
                 'cross_response_to_self_residual_energy':(d['cross_response']/d['self_residual']).tolist(),
                 'live_relative_l2':tail['live_relative_l2']})
    text=(source/'attention_joint_action.py').read_text()
    assert 'native_ffn(held(native["ffn_input"]), ffn)' in text
    assert 'state = ffn_fit(ffn_operands(actual["input"], cache_probe, weights), actual["label"], weights, inverse)' in text
    assert 'return (rows @ rows.T).cpu().tolist()' in text
    result={'status':'ARCHIVED_RECEIPT_AND_SOURCE_SEMANTICS_VERIFIED',
        'evidence_commit':'6baa2a44ffaac5cd90d9a4f4df81809a651a784d',
        'job':record['job'],'receipt_sha256':hashlib.sha256(raw).hexdigest(),
        'new_model_runs':0,'new_video_decodes':0,
        'independent_sources':2,'nested_sites':16,'stage_grams_2x2':stage_count,'dense_grams_3x3':dense_count,
        'actual_extra_dense_ffn_query_rows':16*448,
        'max_shared_probe_vs_live_relative_l2':max_probe,'max_native_tail_vs_live_relative_l2':max_live,
        'stage_matrix_minimum_eigenvalue':min(eigenmins),
        'files':{p.name:sha(p) for p in source.glob('*.py')},'sites':details,
        'semantics':{'reference':'same current block input; native attention plus fixed wide8192 readout',
          'dense_background':'native attention at that same block input; native FFN evaluated only on H448',
          'ffn_teacher':'captured live native P64 response, fitted once and shared by all replay branches',
          'attention_teacher':'full-query native P64 labels; short-query result is an audited numerical witness',
          'gram_inner_products':'FP64 accumulation of differences formed after FP32 promotion of archived path outputs',
          'stage_composition':'same-stage Grams do not contain cross-stage residual/FFN inner products; BF16 residual additions prevent inferring them by naive subtraction'}}
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in {'sites','files'}},indent=2))

if __name__=='__main__':main()
