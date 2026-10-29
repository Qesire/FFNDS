import base64,copy,hashlib,importlib.util,io,json,sys,tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('bytebridge',ROOT/'tools/e0_teal_llama3_tokenizer_byte_bridge.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def fixture():
    inv=m.byte_unicode_inverse();enc={b:c for c,b in inv.items()}
    vocab={enc[i]:255-i for i in range(256)}
    return {'model':{'type':'BPE','vocab':vocab},'added_tokens':[{'id':256,'content':'<special>','special':True}]}
def encode(x):return json.dumps(x,ensure_ascii=False).encode('utf-8')
class Tests(unittest.TestCase):
    def test_bytelevel_bijection_and_controls(self):
        inv=m.byte_unicode_inverse();self.assertEqual(len(inv),256);self.assertEqual(set(inv.values()),set(range(256)))
        chars={v:k for k,v in inv.items()};self.assertEqual(chars[33],'!');self.assertNotEqual(chars[0],chr(0));self.assertEqual(chars[255],'ÿ')
        rank,meta=m.parse_tokenizer(encode(fixture()));self.assertEqual(rank[255],b'\x00');self.assertEqual(rank[0],b'\xff');self.assertEqual(meta['byte_coverage'],256)
    def test_canonical_sort_roundtrip_and_exact_newlines(self):
        ranks,_=m.parse_tokenizer(encode(fixture()));out=io.BytesIO();count=m.write_canonical(ranks,out,100000)
        lines=out.getvalue().splitlines(keepends=True);self.assertEqual(lines[0],base64.b64encode(b'\xff')+b' 0\n');self.assertEqual(lines[-1],b'AA== 255\n');self.assertEqual(count,len(out.getvalue()))
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'dump';p.write_bytes(out.getvalue());self.assertEqual(m.verify_canonical(p,ranks),256)
            p.write_bytes(out.getvalue().replace(b' 0\n',b' 00\n',1))
            with self.assertRaises(ValueError):m.verify_canonical(p,ranks)
    def test_duplicate_json_and_invalid_ranks(self):
        with self.assertRaises(ValueError):m.parse_tokenizer(b'{"model":{},"model":{}}')
        for bad in (True,-1,1.5,256):
            x=fixture();key=next(iter(x['model']['vocab']));x['model']['vocab'][key]=bad
            with self.assertRaises(ValueError):m.parse_tokenizer(encode(x))
        x=fixture();keys=list(x['model']['vocab']);x['model']['vocab'][keys[0]]=x['model']['vocab'][keys[1]]
        with self.assertRaises(ValueError):m.parse_tokenizer(encode(x))
    def test_unknown_codepoints_empty_and_special_conflicts(self):
        x=fixture();x['model']['vocab']['🙂']=256
        with self.assertRaises(ValueError):m.parse_tokenizer(encode(x))
        x=fixture();x['model']['vocab']['']=256
        with self.assertRaises(ValueError):m.parse_tokenizer(encode(x))
        x=fixture();x['added_tokens'][0]['id']=0
        with self.assertRaises(ValueError):m.parse_tokenizer(encode(x))
        x=fixture();x['added_tokens'][0]['content']=next(iter(x['model']['vocab']))
        with self.assertRaises(ValueError):m.parse_tokenizer(encode(x))
        x=fixture();x['added_tokens'].append(copy.deepcopy(x['added_tokens'][0]))
        with self.assertRaises(ValueError):m.parse_tokenizer(encode(x))
        x=fixture();x['added_tokens'][0]['special']=False
        with self.assertRaises(ValueError):m.parse_tokenizer(encode(x))
        x=fixture();x['added_tokens'][0]['id']=1000000
        with self.assertRaises(ValueError):m.parse_tokenizer(encode(x))
    def test_output_limit_and_size_sha_mismatch(self):
        ranks,_=m.parse_tokenizer(encode(fixture()))
        with self.assertRaises(ValueError):m.write_canonical(ranks,io.BytesIO(),5)
        payload=b'small';self.assertFalse(len(payload)==m.TARGET_BYTES and hashlib.sha256(payload).hexdigest()==m.TARGET_SHA)
    def test_url_confinement(self):
        self.assertTrue(m.allowed_url(m.INPUT_URL))
        self.assertTrue(m.allowed_url('https://huggingface.co/api/resolve-cache/models/NousResearch/Meta-Llama-3-8B/'+m.REV+'/tokenizer.json?x=1'))
        for url in ('http://huggingface.co/x','https://evil.example/x','https://user@huggingface.co/x','https://huggingface.co/other','https://huggingface.co/NousResearch/Meta-Llama-3-8B/resolve/main/tokenizer.json'):
            self.assertFalse(m.allowed_url(url))
        self.assertFalse(m.allowed_url('https://huggingface.co:bad/x'))
    def test_bounded_anon_download_and_failed_length(self):
        data=b'abc';blob=hashlib.sha1(b'blob 3\0'+data).hexdigest()
        class Response:
            status=200
            def __init__(self):self.stream=io.BytesIO(data)
            def read(self,n):return self.stream.read(n)
            def getheader(self,k):return '3' if k=='Content-Length' else None
        class Conn:
            def __init__(self,*a,**kw):self.response=Response()
            def request(self,method,path,headers):self.headers=headers;self.method=method;self.path=path;self.assert_no_auth()
            def assert_no_auth(self):
                assert not any('authorization' in k.lower() or 'cookie' in k.lower() for k in self.headers)
            def getresponse(self):return self.response
            def close(self):pass
        with tempfile.TemporaryDirectory() as tmp,patch.object(m,'INPUT_BYTES',3),patch.object(m,'INPUT_BLOB',blob),patch.object(m.http.client,'HTTPSConnection',Conn):
            report={'predebited':dict.fromkeys(m.LIMITS,0),'completed':dict.fromkeys(m.LIMITS,0),'inflight':None};saved=[];p=Path(tmp)/'in'
            m.download(m.INPUT_URL,p,report,lambda:saved.append(copy.deepcopy(report)))
            self.assertEqual(p.read_bytes(),data);self.assertEqual(report['download']['git_blob_sha1'],blob)
            self.assertEqual(report['completed']['input_download'],1);self.assertEqual(report['download']['bytes_received'],3)
            with self.assertRaises(FileExistsError):m.download(m.INPUT_URL,p,{'predebited':dict.fromkeys(m.LIMITS,0),'completed':dict.fromkeys(m.LIMITS,0),'inflight':None},lambda:None)
    def test_http_failure_retains_predebit_without_output(self):
        class Response:
            status=403
            def getheader(self,key):return None
        class Conn:
            def __init__(self,*a,**kw):pass
            def request(self,*a,**kw):pass
            def getresponse(self):return Response()
            def close(self):pass
        with tempfile.TemporaryDirectory() as tmp,patch.object(m.http.client,'HTTPSConnection',Conn):
            report={'predebited':dict.fromkeys(m.LIMITS,0),'completed':dict.fromkeys(m.LIMITS,0),'inflight':None};saved=[];p=Path(tmp)/'input'
            with self.assertRaises(ValueError):m.download(m.INPUT_URL,p,report,lambda:saved.append(copy.deepcopy(report)))
            self.assertEqual(saved[-1]['predebited']['input_download'],1);self.assertEqual(saved[-1]['completed']['input_download'],0)
            self.assertEqual(saved[-1]['download']['requests'][0]['status'],403);self.assertFalse(p.exists())
    def test_download_interruption_keeps_partial_application_ledger(self):
        class Response:
            status=200
            def __init__(self):self.calls=0
            def getheader(self,key):return None
            def read(self,n):
                self.calls+=1
                if self.calls==1:return b'abc'
                raise OSError('synthetic interruption')
        class Conn:
            def __init__(self,*a,**kw):self.response=Response()
            def request(self,*a,**kw):pass
            def getresponse(self):return self.response
            def close(self):pass
        with tempfile.TemporaryDirectory() as tmp,patch.object(m.http.client,'HTTPSConnection',Conn):
            report={'predebited':dict.fromkeys(m.LIMITS,0),'completed':dict.fromkeys(m.LIMITS,0),'inflight':None};saved=[];p=Path(tmp)/'input'
            with self.assertRaises(OSError):m.download(m.INPUT_URL,p,report,lambda:saved.append(copy.deepcopy(report)))
            self.assertEqual(p.read_bytes(),b'abc');self.assertEqual(saved[-1]['download']['bytes_written'],3)
            self.assertEqual(saved[-1]['download']['unrecorded_application_bytes_max'],65536)
            self.assertIn('NOT_POWER_LOSS_DURABLE',saved[-1]['download']['counter_semantics'])
            self.assertEqual(saved[-1]['inflight'],'input_download')
    def test_serialization_interruption_leaves_prior_progress(self):
        ranks,_=m.parse_tokenizer(encode(fixture()));progress=[]
        class FailingWriter:
            def __init__(self):self.data=bytearray()
            def write(self,b):
                if len(self.data)>=50:raise OSError('synthetic interruption')
                self.data.extend(b);return len(b)
        sink=FailingWriter()
        with self.assertRaises(OSError):m.write_canonical(ranks,sink,100000,progress.append)
        self.assertTrue(progress);self.assertEqual(progress[-1],len(sink.data))
        self.assertEqual(m.DESIGN['chunk_bytes'],65536)
    def test_draft_inert_and_no_torch(self):
        p=m.draft();m.validate(p)
        with self.assertRaises(ValueError):m.authorize(p,'fake')
        self.assertNotIn('torch',sys.modules)
if __name__=='__main__':unittest.main()
