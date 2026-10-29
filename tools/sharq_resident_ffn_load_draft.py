"""Unregistered engineering draft: selectively omit BF16 FFN W copies.

No model imports or execution authorization. Caller must bind cache, target MM
objects, actual source resolver, stream discipline, and operation ledger.
"""
import types


def need(value, message):
    if not value:
        raise ValueError(message)


class SelectiveFFNLoad:
    def __init__(self, entries, cache, resolve, charge, observe, check_context, bytes_of, validate_source):
        self.entries, self.cache, self.resolve = entries, cache, resolve
        self.charge, self.observe, self.check_context, self.bytes_of, self.validate_source = charge, observe, check_context, bytes_of, validate_source
        need(callable(check_context) and callable(bytes_of) and callable(validate_source), 'context, source and size gates required')
        self.saved = []
        self.enabled = False

    def __enter__(self):
        try:
            for mm, expert, suffix in self.entries:
                need(suffix in ('ffn.0', 'ffn.2') and expert in (0, 1), 'FFN identity')
                need(not mm.has_lora_branch and not mm.has_diff and not mm.is_post_adapter, 'unsupported adapter')
                need(len(mm.base_attrs) == 2 and {x[1] for x in mm.base_attrs} == {'weight', 'bias'}, 'unsupported base attributes')
                need(not any(hasattr(mm, a) for a in ('lora_alpha','lora_down','lora_up','weight_diff','bias_diff')), 'unsupported live adapter')
                old = mm.load_state_dict
                present = 'load_state_dict' in mm.__dict__
                previous = mm.__dict__.get('load_state_dict')
                self.saved.append((mm, present, previous))
                owner = self
                def load(obj, destination, block, adapter_block_index=None, *, _old=old, _e=expert, _s=suffix):
                    need(adapter_block_index is None and type(block) is int and 0 <= block < 40, 'invalid block')
                    key = (_e, block, _s)
                    names = {attr: owner.resolve(name, block, None, obj.is_post_adapter)
                             for name, attr, _ in obj.base_attrs}
                    need(all(n in destination for n in names.values()), 'missing source weight or bias')
                    need(not any(owner.resolve(getattr(obj, attr_name),block) in destination for attr_name in obj.lora_attrs.values()), 'unexpected adapter source')
                    need(obj.weight_cuda_buffer is not None and obj.bias_cuda_buffer is not None, 'missing original buffers')
                    owner.validate_source(obj,key,destination[names['weight']],destination[names['bias']])
                    weight_bytes=owner.bytes_of(destination[names['weight']]);bias_bytes=owner.bytes_of(destination[names['bias']])
                    need(type(weight_bytes) is int and weight_bytes>0 and type(bias_bytes) is int and bias_bytes>0, 'logical source bytes')
                    if not owner.enabled:
                        value=owner.charge(key,weight_bytes,bias_bytes,False,lambda:_old(destination,block,adapter_block_index))
                        owner.observe({'key':key,'weight_copy':'ISSUED','bias_copy':'ISSUED','requested_weight_logical_bytes':weight_bytes,'requested_bias_logical_bytes':bias_bytes,'physical_bus_bytes':'UNKNOWN'})
                        return value
                    owner.check_context(obj, key)
                    need(key in owner.cache and len(owner.cache[key]) == 4, 'missing immutable packed cache')
                    # This is metadata only: actual BF16 W contents are stale.
                    obj.weight = obj.weight_cuda_buffer
                    def copy_bias():
                        obj.bias = obj.bias_cuda_buffer.copy_(destination[names['bias']], non_blocking=True)
                    owner.charge(key, weight_bytes, bias_bytes, True, copy_bias)
                    owner.observe({'key':key, 'weight_copy':'SKIPPED', 'bias_copy':'ISSUED',
                                   'skipped_weight_logical_bytes':weight_bytes, 'requested_bias_logical_bytes':bias_bytes, 'physical_bus_bytes':'UNKNOWN', 'weight_buffer_contents':'STALE_DO_NOT_CONSUME'})
                mm.load_state_dict = types.MethodType(load, mm)
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, typ, value, tb):
        for mm, present, previous in reversed(self.saved):
            if present:
                mm.__dict__['load_state_dict'] = previous
            else:
                mm.__dict__.pop('load_state_dict', None)
        self.saved.clear()
