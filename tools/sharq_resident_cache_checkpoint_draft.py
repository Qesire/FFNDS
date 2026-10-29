"""Track in-process packed-cache object/storage identities, not tensor bytes."""
import hashlib,json

def cache_checkpoint(cache):
 expected={(e,b,s) for e in (0,1) for b in range(40) for s in ('ffn.0','ffn.2')}
 if set(cache)!=expected:raise ValueError('incomplete cache')
 rows=[];logical_bytes=0
 for key in sorted(cache):
  value=cache[key]
  if not isinstance(value,(tuple,list)) or len(value)!=4:raise ValueError('cache tuple')
  members=[]
  for t in value:
   num=t.numel();size=t.element_size();logical_bytes+=num*size
   members.append({'object_id':id(t),'data_ptr':t.data_ptr(),'shape':list(t.shape),'stride':list(t.stride()),'dtype':str(t.dtype),'device':str(t.device),'bytes':num*size})
  rows.append({'key':key,'members':members})
 # This digest is process-local metadata and explicitly does not certify data.
 digest=hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest()
 return {'entries':len(cache),'logical_bytes':logical_bytes,'process_metadata_sha256':digest,'tensor_content_identity':'NOT_PROVED_BY_METADATA'}
