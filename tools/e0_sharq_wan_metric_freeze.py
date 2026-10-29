"""Freeze a reviewed SharQ metric draft after its real parent EXIT0.

Metadata-only; byte preflight, independent review and ACTIVE remain mandatory.
"""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import e0_sharq_wan_metric_adapter as adapter

def freeze(kind,p,hf):
 engine=adapter.install(kind) if kind!='visual' else __import__('e0_sharq_wan_visual_cpu');engine.validate(p) if kind!='visual' else engine.validate(p,hf);engine.need(p['status']==('DRAFT' if kind=='visual' else 'DRAFT_NOT_AUTHORIZED') and p['assets_finalized'] is False and (kind=='visual' or p['binding_status']=='BOUND_DRAFT_NOT_REGISTERED'),'fresh draft required');engine.need(adapter.line.parent_data(hf,p['part'] if kind=='visual' else p['shard'])==(p['parent'] if kind=='visual' else p['parents'][0]) and adapter.line.closed_rows((p['parent'] if kind=='visual' else p['parents'][0]),hf,p['part'] if kind=='visual' else p['shard'])==p['videos'],'actual closed parent changed');q=dict(p,status='FROZEN_PENDING_REGISTRATION',assets_finalized=True);
 if kind=='visual':engine.validate(q,hf,True)
 else:q['binding_status']='BOUND_READY_FOR_REGISTRATION';engine.validate(q,True)
 return q

def main():
 a=argparse.ArgumentParser();a.add_argument('kind',choices=('quality','raft','visual'));a.add_argument('--protocol',type=Path,required=True);a.add_argument('--hf-root',type=Path,default=Path('/home/xiang/.cache/huggingface'));a.add_argument('--out',type=Path,required=True);x=a.parse_args();engine=adapter.install(x.kind) if x.kind!='visual' else __import__('e0_sharq_wan_visual_cpu');engine.need(not x.out.exists(),'fresh output required');q=freeze(x.kind,engine.base.read(x.protocol),x.hf_root);x.out.parent.mkdir(parents=True,exist_ok=True);engine.base.write(x.out,q);print(json.dumps({'id':q['id'],'status':q['status'],'requires':'independent review and ACTIVE'}))
if __name__=='__main__':main()
