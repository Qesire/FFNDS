import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path
from tools.raw_bf16_full_depth_audit import audit_packet,draw,validate_finalize,check_generation,stream_video
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
from ffnds_experiment_plans import row_draw

class AuditTest(unittest.TestCase):
    def test_streaming_raw_and_corruption(self):
        shapes={'rows':[2],'gate':[2]}; payloads={'rows':struct.pack('<qq',1,7),'gate':b'\x80\x3f\x00\x40'}; offset=0;hdr={};meta={}
        for key in ('gate','rows'):
            raw=payloads[key];hdr[key]={'dtype':'I64' if key=='rows' else 'BF16','shape':shapes[key],'data_offsets':[offset,offset+len(raw)]};offset+=len(raw)
            meta[key]={'dtype':'int64' if key=='rows' else 'bfloat16','shape':shapes[key],'raw_tensor_sha256':hashlib.sha256(raw).hexdigest()}
        header=json.dumps(hdr,separators=(',',':')).encode(); data=struct.pack('<Q',len(header))+header+payloads['gate']+payloads['rows']
        binding={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'pack';p.write_bytes(data);audit_packet(p,binding,meta,[1,7],shapes)
            with self.assertRaises(ValueError):audit_packet(p,binding,meta,[7,1],shapes)
            wrong=dict(meta);wrong['gate']={**meta['gate'],'shape':[1]}
            with self.assertRaises(ValueError):audit_packet(p,binding,wrong,[1,7],shapes)
            p.write_bytes(data[:-1]);seen=[]
            with self.assertRaises(ValueError):audit_packet(p,binding,meta,[1,7],shapes,count=seen.append)
            self.assertEqual(sum(seen),0)
            p.write_bytes(data[:-1]+bytes([data[-1]^1]))
            with self.assertRaises(ValueError):audit_packet(p,binding,meta,[1,7],shapes)
    def test_finalize_rejects_wrong_binding(self):
        proof={'protocol':{'path':'wrong','bytes':2660,'sha256':'x'},'RUN':{'path':'x','bytes':1978,'sha256':'x'},'exit':{'path':'x','bytes':528,'sha256':'x'}}
        with self.assertRaises(ValueError): validate_finalize(proof,{}, {}, {}, {})

    def test_generation_source_and_core_mismatch_rejected(self):
        case={'source_id':'s','native_source_ordinal':897,'consumer20_index':3,'image_sha256':'img','prompt_sha256':'prompt'}
        video={'sha256':'video'};phase={'run_id':'raw_full_depth_ord897_seed42_dense','video':video,'generation_receipt':{'sha256':'g'}}
        gen={'run_id':phase['run_id'],'protocol_sha256':'p','source_id':'s','native_source_ordinal':897,'source_index':3,'status':'COMPLETE_RAW_BF16_TEACHER_STATE_FULL160','numeric_path':'BF16_ORIGINAL','video':video}
        core={'identity':{'run_sha256':'g','source_id':'img','input_prompt_digest':'prompt','protocol_sha256':'p','seed':42},'candidate':{'sha256':'video'},'metrics':{'artifact':{'verified_full_decode_frames':81}}}
        check_generation(gen,core,phase,case,'p')
        for field in ('source_id','native_source_ordinal','protocol_sha256'):
            bad=dict(gen);bad[field]='wrong'
            with self.assertRaises(ValueError):check_generation(bad,core,phase,case,'p')
        bad={**core,'identity':{**core['identity'],'input_prompt_digest':'wrong'}}
        with self.assertRaises(ValueError):check_generation(gen,bad,phase,case,'p')

    def test_oversize_video_rejected_before_read(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'video';p.write_bytes(b'12345');seen=[]
            with self.assertRaises(ValueError):stream_video(p,{'bytes':4,'sha256':'wrong'},seen.append)
            self.assertEqual(seen,[])

    def test_draw_identity(self):
        for ordinal in (894,895,911,927):
            for step in (0,2,3):
                for block in (0,17,39):
                    self.assertEqual(draw(ordinal,step,block),row_draw(31668,256,base_seed=42,source_index=ordinal,step=step,block=block,state_pair=True)[0])

if __name__=='__main__':unittest.main()
