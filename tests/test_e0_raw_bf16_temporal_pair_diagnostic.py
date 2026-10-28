"""Tiny synthetic NumPy controls; no real saved operands or GPU."""
import hashlib
import json
import struct
import unittest
import numpy as np
from tools import e0_raw_bf16_temporal_pair_diagnostic as m


class TemporalControls(unittest.TestCase):
    def test_pairs_oriented_cyclic_and_missing(self):
        # 3 temporal slots, 2x2 spatial grid; rows intentionally unsorted.
        rows=[9,0,5,8,6,1]
        pairs=m.temporal_pairs(rows,(3,2,2))
        self.assertEqual([(rows[p['ia']],rows[p['ib']]) for p in pairs],[(0,8),(1,5),(1,9),(5,9)])
        self.assertTrue(all(p['tau_a']<p['tau_b'] for p in pairs))
        for p in pairs:
            self.assertEqual(rows[p['control_ib']]//4,p['tau_b'])
            self.assertNotEqual(rows[p['control_ib']]%4,rows[p['ib']]%4)
        self.assertIsNone(m.temporal_pairs([0,4],(2,2,2))[0]['control_ib'])
        with self.assertRaises(ValueError):m.temporal_pairs([0,0],(2,2,2))

    def test_bf16_decode_and_synthetic_packet_integrity(self):
        def bf(values):return (np.asarray(values,dtype=np.float32).view(np.uint32)>>16).astype('<u2').tobytes()
        self.assertEqual(m.decode_bf16(bf([1.,-2.]),[2],np).tolist(),[1.,-2.])
        with self.assertRaises(ValueError):m.decode_bf16(struct.pack('<H',0x7f80),[1],np)
        shapes={'V1':[2,2],'gate':[2],'rows':[2]};payloads={'V1':bf([1,2,3,4]),'gate':bf([1,2]),'rows':struct.pack('<2q',0,4)}
        header={};meta={};offset=0
        for name,shape in shapes.items():
            data=payloads[name];header[name]={'dtype':'I64' if name=='rows' else 'BF16','shape':shape,'data_offsets':[offset,offset+len(data)]}
            meta[name]={'shape':shape,'dtype':'int64' if name=='rows' else 'bfloat16','raw_tensor_sha256':hashlib.sha256(data).hexdigest()};offset+=len(data)
        h=json.dumps(header).encode();raw=struct.pack('<Q',len(h))+h+b''.join(payloads.values())
        entry={'payload':{'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()},'tensor_metadata':meta}
        packet=m.decode_packet(raw,entry,[0,4],np,shapes);self.assertEqual(packet['rows'],[0,4]);np.testing.assert_array_equal(packet['V1'],[[1,2],[3,4]])
        with self.assertRaises(ValueError):m.decode_packet(raw,entry,[0,5],np,shapes)
        with self.assertRaises(ValueError):m.decode_packet(raw[:-1]+b'\1',entry,[0,4],np,shapes,whole_SHA_already_stream_verified=True)

    def test_gate_ln_pair_metric_and_missing_cosine(self):
        p={'V1':np.array([[1.,0.],[0.,1.],[0.,0.]],np.float32),'h':np.ones((3,2),np.float32),
           'Y':np.array([[1.,2.],[2.,1.],[0.,0.]],np.float32),'gate':np.array([2.,-.5],np.float32),
           'r':np.ones((3,2),np.float32),'u':np.array([[1.,3.],[3.,1.],[2.,2.]],np.float32)}
        values=m.geometric_operands(p,np);np.testing.assert_array_equal(values['gY'],[[2,-1],[4,-.5],[0,0]])
        np.testing.assert_allclose(values['rowLN_u'][0],np.array([-1,1])/np.sqrt(1+1e-6))
        pairs=[{'ia':0,'ib':1,'control_ib':2},{'ia':0,'ib':2,'control_ib':None}]
        a=m.pair_geometry(values['X'],pairs,np)
        self.assertAlmostEqual(a[0,0],1.);self.assertAlmostEqual(a[0,1],0.)
        self.assertTrue(np.isnan(a[0,5]));self.assertTrue(np.isnan(a[1,1]));self.assertTrue(np.isnan(a[1,4:]).all())

    def test_position_then_source_equal_not_pair_weighted(self):
        summary=m.SourceEqualSummary(np,ordinals=(10,20))
        def add(source,block,count,value):
            pairs=[{'tau0_participates':True,'delta_tau':1} for _ in range(count)]
            summary.add(source,0,block,pairs,np.full((len(m.OPERANDS),count,len(m.METRICS)),value))
        add(10,0,1,0.);add(10,1,9,2.);add(20,0,3,4.)
        r=summary.result();self.assertAlmostEqual(r['source_means'][0][0][0][0][0],1.)
        self.assertAlmostEqual(r['source_equal_means_axis_without_source'][0][0][0][0],2.5)
        self.assertEqual(r['available_source_counts'][0][0][0][0],2)
        with self.assertRaises(ValueError):add(10,0,1,3.)


if __name__=='__main__':unittest.main()
