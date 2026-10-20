#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path

def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument('--report', required=True)
    ap.add_argument('--expected', choices=['a4','a16'], required=True)
    a=ap.parse_args()
    obj=json.loads(Path(a.report).read_text(encoding='utf-8'))
    calls=int(obj.get('activation_qdq_calls',-1))
    forwards=int(obj.get('linear_forward_calls',-1))
    if forwards <= 0: raise RuntimeError(f'no prepared Linear forwards recorded: {forwards}')
    if a.expected=='a16' and calls != 0:
        raise RuntimeError(f'A16/BF16 runtime expected activation_qdq_calls=0, got {calls}')
    if a.expected=='a4' and calls <= 0:
        raise RuntimeError(f'A4 runtime expected activation_qdq_calls>0, got {calls}')
    print(json.dumps({'report':str(Path(a.report).resolve()),'expected':a.expected,'activation_qdq_calls':calls,'linear_forward_calls':forwards},indent=2))
    print('RUNTIME_ACTIVATION_DOMAIN_V48_OK')
if __name__=='__main__': main()
