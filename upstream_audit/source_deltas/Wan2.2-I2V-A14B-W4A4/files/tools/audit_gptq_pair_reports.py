#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, statistics
from pathlib import Path


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument('--input-root', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--min-fraction-lt1', type=float, default=0.70)
    ap.add_argument('--max-median-rho', type=float, default=0.95)
    ap.add_argument('--min-median-different-rate', type=float, default=1e-6)
    ap.add_argument('--no-fail', action='store_true')
    a=ap.parse_args()
    files=sorted(Path(a.input_root).rglob('*.pair.json'))
    if not files: raise FileNotFoundError(f'no *.pair.json under {a.input_root}')
    modules=[]
    domains=set()
    for f in files:
        obj=json.loads(f.read_text(encoding='utf-8'))
        domains.add(str(obj.get('curvature_domain','')))
        modules.extend(obj.get('modules',[]))
    if len(domains)!=1: raise RuntimeError(f'mixed curvature domains in audit: {sorted(domains)}')
    rhos=[float(x['rho']) for x in modules]
    diffs=[float(x['rtn_gptq_weight_different_rate']) for x in modules]
    report={
      'artifact':'gptq_pair_gate_v48','curvature_domain':next(iter(domains)),
      'report_files':len(files),'modules':len(modules),
      'rho_fraction_lt_1':sum(r<1 for r in rhos)/len(rhos),
      'median_rho':statistics.median(rhos),
      'median_weight_different_rate':statistics.median(diffs),
      'thresholds':{
        'min_fraction_lt1':a.min_fraction_lt1,'max_median_rho':a.max_median_rho,
        'min_median_different_rate':a.min_median_different_rate,
      },
    }
    report['passed']=(report['rho_fraction_lt_1']>=a.min_fraction_lt1 and report['median_rho']<=a.max_median_rho and report['median_weight_different_rate']>=a.min_median_different_rate)
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2,ensure_ascii=False))
    if not report['passed'] and not a.no_fail: raise SystemExit(3)
    print('GPTQ_PAIR_GATE_V48_OK')
if __name__=='__main__': main()
