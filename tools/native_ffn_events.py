"""Native FFN events without per-block synchronization or CPU tensor reads."""
import torch


class NativeFFNEvents:
    def __init__(self):
        self.rows = []
        self.pending = {}
        self.originals = []

    def begin(self):
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    def finish(self, start, kind, infer):
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        self.rows.append((kind, int(infer.scheduler.step_index), int(infer.block_idx), start, end))

    def __enter__(self):
        from lightx2v.models.networks.wan.infer.transformer_infer import WanTransformerInfer
        owner = self
        for name in ('pre_process', 'infer_ffn', 'post_process'):
            self.originals.append((WanTransformerInfer, name, getattr(WanTransformerInfer, name)))
        old_pre, old_ffn, old_post = [row[2] for row in self.originals]

        def pre(infer, *args, **kwargs):
            start = owner.begin()
            result = old_pre(infer, *args, **kwargs)
            owner.finish(start, 'shared_modulation_preparation', infer)
            return result

        def ffn(infer, *args, **kwargs):
            if id(infer) in owner.pending:
                raise RuntimeError('Nested native FFN event boundary')
            owner.pending[id(infer)] = owner.begin()
            return old_ffn(infer, *args, **kwargs)

        def post(infer, *args, **kwargs):
            result = old_post(infer, *args, **kwargs)
            owner.finish(owner.pending.pop(id(infer)), 'FFN_norm_through_gate_residual', infer)
            return result

        WanTransformerInfer.pre_process = pre
        WanTransformerInfer.infer_ffn = ffn
        WanTransformerInfer.post_process = post
        return self

    def __exit__(self, *error):
        for cls, name, function in reversed(self.originals):
            setattr(cls, name, function)
        if not any(error) and self.pending:
            raise RuntimeError('Unmatched full FFN event boundary')

    def report_after_request_synchronized(self):
        result = []
        for kind, step, block, start, end in self.rows:
            if not end.query():
                raise RuntimeError('Read timing only after the complete native request is synchronized')
            result.append({'kind': kind, 'step': step, 'block': block, 'cuda_ms': start.elapsed_time(end)})
        expected = {(s, b) for s in range(4) for b in range(40)}
        for kind in ('shared_modulation_preparation', 'FFN_norm_through_gate_residual'):
            rows = [r for r in result if r['kind'] == kind]
            if len(rows) != 160 or {(r['step'], r['block']) for r in rows} != expected:
                raise RuntimeError('Missing actual full160 native timing regions')
        return {'rows': result, 'instrumentation_per_block_device_synchronization': False,
                'instrumentation_CPU_tensor_reads_during_request': False,
                'operator_internal_synchronization_excluded_from_cost': False,
                'native_core_cuda_ms': sum(r['cuda_ms'] for r in result if r['kind'] == 'FFN_norm_through_gate_residual'),
                'shared_modulation_cuda_ms': sum(r['cuda_ms'] for r in result if r['kind'] == 'shared_modulation_preparation')}
