"""Static per-position packed HiF4 reference with calibration-only gain/bias."""
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from ar2_checkpoint import CheckpointStore, digest
from run_ar2_restart import tensor_sha


def create_runtime(cache_path, policy_path):
    from HiFloat4.ptq.adapters.lightx2v_wan_distill import HIF4CacheRuntime
    from HiFloat4.ptq.adapters.wan_i2v import native_hif4_activation_qdq as qdq

    class TrainedPackedRuntime(HIF4CacheRuntime):
        def __init__(self):
            super().__init__(cache_path)
            self.policy = json.loads(Path(policy_path).read_text())
            if self.policy['kind'] != 'CALIBRATION_STATIC_PACKED_REFERENCE':
                raise RuntimeError('Unsupported trained packed policy')
            self.policy_sha = digest(policy_path)
            cache_binding = self.policy['cache_manifest']
            if digest(cache_binding['path']) != cache_binding['sha256']:
                raise RuntimeError('Policy cache manifest changed')
            cache_manifest = json.loads(Path(cache_binding['path']).read_text())
            if not Path(cache_path).samefile(cache_manifest['path']):
                raise RuntimeError('Runtime uses a different quantized cache')
            self.coverage, self.positions, self.channels_cuda = {}, {}, {}
            self.residual_parameters, self.pending_residual, self.residual_coverage = {}, {}, {}
            self.fit_identities = []
            for expert in ['high_noise_model','low_noise_model']:
                binding = self.policy['fit_reports'][expert]
                if digest(binding['path']) != binding['sha256']:
                    raise RuntimeError('Training fit report changed')
                report = json.loads(Path(binding['path']).read_text())
                root = Path(report['checkpoint_root'])
                identity = json.loads((root/'identity.json').read_text())
                store = CheckpointStore(root,identity['identity'])
                if identity['identity']['cache_manifest_sha256'] != cache_binding['sha256']:
                    raise RuntimeError('Training and deployment quantized cache identities differ')
                if report['status'] != 'COMPLETED_CALIBRATION_PARAMETER_FIT' or report['expert'] != expert or store.fingerprint != report['fingerprint']:
                    raise RuntimeError('Incomplete or inconsistent learned parameters')
                self.fit_identities.append({'expert':expert,'fingerprint':store.fingerprint})
                steps = [0,1] if expert == 'high_noise_model' else [2,3]
                for step in steps:
                    for block in range(40):
                        part = store.load('positions',f'step{step}_block{block:02d}')
                        if part is None:
                            raise RuntimeError('Missing trained position')
                        arrays, info = part['arrays'],part['metadata']
                        family,mode = self.policy['support'],self.policy['mode']
                        groups = torch.from_numpy(arrays[f'{family}_groups']).long().sort().values
                        if groups.shape != (108,) or len(groups.unique()) != 108 or bool((groups<0).any()) or bool((groups>=216).any()):
                            raise RuntimeError('Invalid learned whole-group support')
                        channels = (groups[:,None]*64+torch.arange(64)).reshape(-1)
                        parameter = info['policies'][family]['modes'][mode]
                        b2 = torch.from_numpy(arrays[f'{family}_{mode}_b2']).to(torch.bfloat16)
                        if tensor_sha(b2) != parameter['b2_sha256']:
                            raise RuntimeError('Learned folded bias changed')
                        record0 = dict(self.records[expert][f'blocks.{block}.ffn.0'])
                        record2 = dict(self.records[expert][f'blocks.{block}.ffn.2'])
                        if tensor_sha(record2['weight_qdq']) != info['weight_sha256']:
                            raise RuntimeError('Learned policy effective W2 differs from runtime cache')
                        record0['weight_qdq'] = record0['weight_qdq'].index_select(0,channels).contiguous()
                        record2['weight_qdq'] = record2['weight_qdq'].index_select(1,channels).contiguous()
                        record2['smooth_scale'] = record2['smooth_scale'].index_select(0,channels).contiguous()
                        self.positions[step,block] = {'records':{0:record0,2:record2},'channels':channels,
                            'alpha':parameter['alpha'],'b2':b2,'original_b2_sha256':info['original_b2_sha256']}
                # Both half-width step variants together occupy the original FFN
                # storage. Drop unused dense FFN tensors from the resident CPU map.
                for block in range(40):
                    for slot in [0,2]:
                        self.records[expert][f'blocks.{block}.ffn.{slot}'] = self.positions[steps[0],block]['records'][slot]
            if 'residual_fit' in self.policy:
                from reduced_rank_residual import predict_y
                self.predict_residual = predict_y
                binding = self.policy['residual_fit']
                if digest(binding['path']) != binding['sha256']:
                    raise RuntimeError('Residual fit report changed')
                report = json.loads(Path(binding['path']).read_text())
                root = Path(report['checkpoint_root'])
                identity = json.loads((root/'identity.json').read_text())
                store = CheckpointStore(root,identity['identity'])
                if (report['status'] != 'COMPLETED_RR1_FULL160_CALIBRATION_FIT'
                        or len(report['positions']) != 160 or report['fingerprint'] != store.fingerprint
                        or identity['identity']['cache_manifest_sha256'] != cache_binding['sha256']):
                    raise RuntimeError('Residual parameters incomplete or inconsistent')
                trained_base = {row['fingerprint'] for row in identity['identity']['base_fits']}
                if trained_base != {row['fingerprint'] for row in self.fit_identities}:
                    raise RuntimeError('Residual correction was trained for a different base')
                if self.policy['support'] != 'C' or self.policy['mode'] != 'centered_gain_bias':
                    raise RuntimeError('Residual requires its fixed C-centered base')
                for step in range(4):
                    for block in range(40):
                        part = store.load('positions',f'step{step}_block{block:02d}')
                        if part is None:
                            raise RuntimeError('Missing residual position')
                        parameters = {k:torch.from_numpy(v).to(torch.bfloat16) for k,v in part['arrays'].items()}
                        if {k:tensor_sha(v) for k,v in parameters.items()} != part['metadata']['parameter_sha256']:
                            raise RuntimeError('Native residual parameter encoding changed')
                        self.residual_parameters[step,block] = parameters
                self.residual_rank = report['selected']['rank']
                self.residual_fingerprint = store.fingerprint

        def begin_request(self):
            if self.pending_residual:
                raise RuntimeError('Previous request left unmatched residual predictions')
            self.coverage.clear()
            self.residual_coverage.clear()

        def apply(self,mm_weight,input_tensor,*,expert,step_index,block_index,module_name):
            if not module_name.endswith(('.ffn.0','.ffn.2')):
                return super().apply(mm_weight,input_tensor,expert=expert,step_index=step_index,block_index=block_index,module_name=module_name)
            if expert != ('high_noise_model' if step_index<2 else 'low_noise_model'):
                raise RuntimeError('Expert/step contract mismatch')
            position = self.positions[step_index,block_index]
            slot = 0 if module_name.endswith('.ffn.0') else 2
            record = position['records'][slot]
            expected_in,expected_out = (5120,6912) if slot==0 else (6912,5120)
            if input_tensor.shape[-1] != expected_in:
                raise RuntimeError('Packed FFN input width mismatch')
            x = (input_tensor.float()/record['smooth_scale'].to(input_tensor.device)).to(input_tensor.dtype)
            x = qdq(x,force_fp32=True)
            self.counters['activation_qdq_calls'] += 1
            if slot==0 and self.residual_parameters:
                pending_key = (step_index,block_index)
                if pending_key in self.pending_residual:
                    raise RuntimeError('Duplicate residual prediction')
                # The predictor sees actual post-Q1 V1, before the main FFN0.
                # Count both auxiliary GEMMs and their transfers independently.
                parameters = {k:v.to(x.device) for k,v in self.residual_parameters[pending_key].items()}
                self.pending_residual[pending_key] = self.predict_residual(x,parameters)
                self.residual_coverage[f'step{step_index}_block{block_index}'] = {
                    'rows':x.numel()//5120,'auxiliary_GEMM_shapes':[[self.residual_rank,5120],[5120,self.residual_rank]],
                    'auxiliary_linear_calls':2,'auxiliary_parameter_H2D_bytes':sum(v.numel()*v.element_size() for v in parameters.values()),
                    'auxiliary_MACs_over_original_dense_FFN':self.residual_rank/13824}
            if slot==2:
                x = (x.float()*position['alpha']).to(x.dtype)
                bias = position['b2'].to(x.device)
            else:
                original = mm_weight._get_actual_bias()
                key = (step_index,block_index,str(x.device))
                if key not in self.channels_cuda:
                    self.channels_cuda[key] = position['channels'].to(x.device)
                bias = None if original is None else original.to(x.device).index_select(0,self.channels_cuda[key])
            weight = self._weight(expert=expert,block_index=block_index,module_name=module_name,
                                  record=record,device=x.device,dtype=x.dtype)
            result = F.linear(x,weight,bias)
            if slot==2 and self.residual_parameters:
                pending_key = (step_index,block_index)
                if pending_key not in self.pending_residual:
                    raise RuntimeError('FFN2 lacks its matched post-Q1 residual prediction')
                correction = self.pending_residual.pop(pending_key)
                if correction.shape != result.shape or correction.dtype != result.dtype:
                    raise RuntimeError('Residual output shape/dtype differs from main FFN2')
                result = result + correction
                if step_index==3 and block_index==39 and (len(self.residual_coverage)!=160 or self.pending_residual):
                    raise RuntimeError('Final FFN lacks complete160 residual corrections')
            key=f'step{step_index}_block{block_index}_ffn{slot}'
            if result.shape[-1]!=expected_out or key in self.coverage:
                raise RuntimeError('Packed FFN duplicate/shape failure')
            self.coverage[key]={'rows':input_tensor.numel()//expected_in,'actual_weight_shape':list(weight.shape),'main_GEMM_density':.5}
            self.counters['linear_forward_calls']+=1
            return result

        def report(self):
            return {**super().report(),'packed_reference':{'policy':self.policy,'policy_sha256':self.policy_sha,
                'source_sha256':digest(Path(__file__)),'fit_identities':self.fit_identities,
                'actual_FFN_linear_calls':len(self.coverage),'expected_FFN_linear_calls':320,'coverage':self.coverage,
                'weight_packing':'CPU per-position before generation; transfers use original block cache',
                'residual_correction':{'active':bool(self.residual_parameters),
                    'fingerprint':getattr(self,'residual_fingerprint',None),
                    'actual_positions':len(self.residual_coverage),'expected_positions':160 if self.residual_parameters else 0,
                    'pending_predictions':len(self.pending_residual),'coverage':self.residual_coverage,
                    'precision':'two BF16 auxiliary linears, BF16 sum before original gate/writeback',
                    'timing_includes_auxiliary_cost':False},
                'full_FFN_speed_acceptance':False,'formal20_quality_acceptance':False}}

    return TrainedPackedRuntime()
