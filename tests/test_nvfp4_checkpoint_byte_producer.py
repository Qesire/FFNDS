import hashlib
import json
import os
import struct
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from tools.nvfp4_checkpoint_byte_producer import file_hash_header,check_cache,FILES

class HeaderTest(unittest.TestCase):
    def test_byte_identity_and_header(self):
        header=json.dumps({'blocks.0.ffn.0.weight':{'dtype':'U8','shape':[2],'data_offsets':[0,2]}}).encode();data=struct.pack('<Q',len(header))+header+b'\x01\x02';binding={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x';p.write_bytes(data);count=[];result=file_hash_header(p,binding,count.append)
            self.assertEqual(sum(count),len(data));self.assertEqual(result['ffn_tensor_count'],1)
            p.write_bytes(data[:-1]+b'\x03')
            with self.assertRaises(ValueError):file_hash_header(p,binding)
    def test_changed_stat_and_oversized_cache(self):
        header=json.dumps({'x':{'dtype':'U8','shape':[2],'data_offsets':[0,2]}}).encode();data=struct.pack('<Q',len(header))+header+b'ab'
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'x';p.write_bytes(data);binding={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}; changed=[]
            def alter(_):
                if not changed:
                    changed.append(True);os.utime(p,ns=(1,1))
            with self.assertRaises(ValueError):file_hash_header(p,binding,alter)
            blobs=root/'hub'/'models--lightx2v--LightWan2.2-A14B'/'blobs';blobs.mkdir(parents=True);(blobs/FILES[0]['sha256']).write_bytes(b'12345')
            with patch('tools.nvfp4_checkpoint_byte_producer.SIZE',4):
                with self.assertRaises(ValueError):check_cache(root)

    def test_bad_offsets(self):
        header=json.dumps({'blocks.0.ffn.2.weight':{'dtype':'U8','shape':[2],'data_offsets':[1,3]}}).encode();data=struct.pack('<Q',len(header))+header+b'abc'
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x';p.write_bytes(data)
            with self.assertRaises(ValueError):file_hash_header(p,{'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})

if __name__=='__main__':unittest.main()
