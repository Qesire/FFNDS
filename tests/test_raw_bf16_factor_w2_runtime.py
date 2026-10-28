import unittest
from contextlib import contextmanager
from types import SimpleNamespace as NS
from tools.raw_bf16_factor_w2_runtime import FactorW2Runtime,ORIGINAL_W2_BYTES

class Tensor:
    def __init__(self,shape):self.shape=shape
    def element_size(self):return 2

class MMWeight:
    has_lora_branch=False;has_diff=False
    def __init__(self):self.loads=0;self.calls=0
    def apply(self,x):self.calls+=1;return 'dense'
    def load_state_dict(self,destination,block_index,adapter_block_index=None):self.loads+=1

class Active:
    block=(0,3)
    def step(self,expert):return 0
    def mm_call(self,mm,value,meta,original):self.meta=meta;return original(mm,value)

class RuntimeTest(unittest.TestCase):
    def test_dense_delegation_factor_skip_marker_and_restore(self):
        objects=[MMWeight() for _ in range(4)];models=[]
        for i in range(2):
            buffers=[NS(compute_phases=[None,None,NS(ffn_2=objects[2*i+j])]) for j in range(2)]
            models.append(NS(transformer_infer=NS(offload_manager=NS(cuda_buffers=buffers))))
        pipe=NS(runner=NS(model=NS(model=models)));guard=NS(active=Active());factors={'left.weight':Tensor((4,13824)),'right.weight':Tensor((5120,4)),'right.bias':Tensor((5120,))}; events=[];sync=[];record=[];gemms=[]
        @contextmanager
        def meter(name):
            gemms.append(name);yield
        def forward(value,bank,m):
            self.assertIs(m,meter)
            with m('BF16_factor_GEMM'):pass
            with m('BF16_factor_GEMM'):pass
            return 'factor'
        runtime=FactorW2Runtime(pipe,guard,{(0,0,3):factors},factor_forward=forward,meter=meter,synchronize=lambda:sync.append(1),record_stream=record.append,emit=events.append)
        original=objects[0].apply
        with runtime:
            self.assertEqual(objects[0].apply(Tensor((1,13824))),'dense')
            runtime.set_mode('factor');dest={'blocks.3.ffn.2.weight':Tensor((13824,5120)),'blocks.3.ffn.2.bias':Tensor((5120,))}
            objects[0].load_state_dict(dest,3);self.assertEqual(objects[0].loads,0)
            self.assertEqual(objects[0].apply(Tensor((1,13824))),'factor');self.assertEqual(len(record),3);self.assertEqual(events[-1]['bytes'],ORIGINAL_W2_BYTES);self.assertEqual(gemms,['BF16_factor_GEMM']*2)
            self.assertEqual(guard.active.meta['call_label'],'factor_W2_two_GEMMs')
            guard.active.block=(0,4)
            with self.assertRaises(ValueError):objects[0].apply(Tensor((1,13824)))
            guard.active.block=(0,3);runtime.set_mode('dense')
            with self.assertRaises(ValueError):objects[0].apply(Tensor((1,13824)))
            objects[0].load_state_dict(dest,3);self.assertEqual(objects[0].apply(Tensor((1,13824))),'dense')
        self.assertEqual(objects[0].apply.__func__,original.__func__);self.assertGreaterEqual(len(sync),4)

if __name__=='__main__':unittest.main()
