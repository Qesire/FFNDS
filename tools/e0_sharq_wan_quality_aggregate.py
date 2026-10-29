"""Pure source-equal reduction of already independently closed metric records.

No I/O, metric computation or verification here. The caller must bind both
actual per-source RUN and verifier EXIT0 before using these reductions.
"""
import math
ORDINALS=(1040,979);MODES=('dense','dense32','sharq')
def need(x,msg):
 if not x:raise ValueError(msg)
def reduce_vectors(rows,dimensions):
 rows=list(rows);dimensions=tuple(dimensions);need(len(rows)==2 and {r['ordinal'] for r in rows}==set(ORDINALS) and len(set(dimensions))==len(dimensions)>0,'source/dimension coverage differs');by={r['ordinal']:r for r in rows}
 for r in rows:
  need(set(r['scores'])==set(MODES) and all(set(r['scores'][m])==set(dimensions) for m in MODES),'arm/axis coverage differs');need(all(math.isfinite(float(r['scores'][m][d])) for m in MODES for d in dimensions),'nonfinite metric')
 ordered=[by[o] for o in ORDINALS];means={m:{d:sum(float(r['scores'][m][d]) for r in ordered)/2 for d in dimensions} for m in MODES};deltas={m:{d:sum(float(r['scores'][m][d])-float(r['scores']['dense'][d]) for r in ordered)/2 for d in dimensions} for m in MODES[1:]}
 return {'independent_source_count':2,'source_role':'EXPOSED_DEVELOPMENT','seeds_per_source':1,'source_ordinals':list(ORDINALS),'source_equal_means':means,'source_equal_candidate_minus_dense':deltas,'per_source':ordered,'quality_accepted':False,'heldout':False,'human_review':False,'official_five_independent_samples':False}
