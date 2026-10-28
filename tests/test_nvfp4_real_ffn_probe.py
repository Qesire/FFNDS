"""Synthetic CPU-only controls for the packed NVFP4 execution contract."""
import importlib.util
from pathlib import Path
import unittest
import torch

spec=importlib.util.spec_from_file_location('probe',Path(__file__).resolve().parents[1]/'tools/nvfp4_real_ffn_probe.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
torch.set_num_threads(1)


def layer(inputs, outputs, alpha=0.25, global_scale=4.):
    return {'weight':torch.zeros((outputs,inputs//2),dtype=torch.uint8),
            'weight_scale':torch.ones((128,((inputs//16+3)//4)*4),dtype=torch.float8_e4m3fn),
            'alpha':torch.tensor(alpha), 'input_global_scale':torch.tensor(global_scale),
            'bias':torch.zeros(outputs,dtype=torch.bfloat16)}


class ProbeControls(unittest.TestCase):
    def test_two_tiles_and_row32_128_swizzle(self):
        rows,k=129,128
        flat=torch.zeros(256*8,dtype=torch.float8_e4m3fn)
        for row in range(rows):
            for group in range(k//16):
                # C++ physical tile decomposition, independently of reshape.
                offset=(row//128)*1024+(group//4)*512+(row%32)*16+((row%128)//32)*4+group%4
                self.assertEqual(m.scale_offset(row,group,k),offset)
                flat[offset]=float(1+(row+group)%7)
        actual=m.unswizzle_scales(flat.reshape(256,8),rows,k)
        for row in (0,31,32,63,64,95,96,127,128):
            for group in range(8):
                self.assertEqual(float(actual[row,group]),float(1+(row+group)%7))

    def test_nibbles_table_and_signed_zero(self):
        packed=torch.tensor([[0x10,0x32,0x54,0x76,0x98,0xba,0xdc,0xfe]],dtype=torch.uint8)
        scales=torch.ones((128,4),dtype=torch.float8_e4m3fn)
        decoded=m.decode_unscaled(packed,scales)[0]
        self.assertEqual(decoded.tolist(),list(m.E2M1)+[-v for v in m.E2M1])
        self.assertFalse(bool(torch.signbit(decoded[0])))
        self.assertTrue(bool(torch.signbit(decoded[8])))

    def test_global_alpha_and_q2_uses_new_hidden(self):
        calls=[]; x=torch.tensor([[1.]*16],dtype=torch.bfloat16)
        layers=[layer(16,16,0.25,4.),layer(16,16,0.5,2.)]
        class API:
            def quant(self,value,global_scale):
                calls.append(('quant',value.clone(),global_scale))
                return value,global_scale
            def gemm(self,a,w,sa,sw,alpha,bias):
                calls.append(('gemm',alpha,sa))
                return (a.float()*sa*alpha).to(torch.bfloat16)+bias
            def gelu(self,value):
                calls.append(('gelu',value.clone()))
                return (value+2).contiguous()
        result=m.packed_ffn(x,layers,API(),capture=True)
        self.assertEqual([c[0] for c in calls],['quant','gemm','gelu','quant','gemm'])
        self.assertTrue(torch.equal(calls[3][1],result['h']))
        self.assertTrue(torch.equal(result['h'],result['pre_GELU']+2))
        self.assertIs(calls[1][1],layers[0]['alpha']);self.assertIs(calls[3][2],layers[1]['input_global_scale'])
        self.assertTrue(torch.equal(result['Y'],x+2)) # G * alpha = 1 for each synthetic stage

    def test_shape_and_dtype_rejected(self):
        x=torch.ones((1,16),dtype=torch.bfloat16); good=layer(16,16)
        bad=dict(good,weight=good['weight'].float())
        with self.assertRaises(ValueError): m.check_linear(x,bad)
        with self.assertRaises(ValueError): m.check_linear(x.float(),good)
        with self.assertRaises(ValueError): m.unswizzle_scales(torch.ones((128,4)),1,16)
        with self.assertRaises(ValueError): m.scale_offset(0,1,16)


if __name__=='__main__': unittest.main()
