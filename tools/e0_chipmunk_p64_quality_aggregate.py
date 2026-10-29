"""Pure aggregation of two already independently verified source reports."""
import math,statistics
SOURCES=(898,96);MODES=('original_dense','global_p64');DIMENSIONS=('imaging_quality','aesthetic_quality','motion_smoothness','dynamic_degree','subject_consistency','background_consistency','i2v_subject','i2v_background')
def aggregate(rows):
 if len(rows)!=2 or {r.get('ordinal') for r in rows}!=set(SOURCES):raise ValueError('two unique source clusters required')
 for row in rows:
  if set(row['scores'])!=set(MODES):raise ValueError('two actual modes required')
  for score in row['scores'].values():
   if set(score)!=set(DIMENSIONS) or any(not isinstance(v,(int,float)) or isinstance(v,bool) or not math.isfinite(v) for v in score.values()):raise ValueError('eight finite axes required')
 axes={}
 for d in DIMENSIONS:
  values={m:statistics.fmean(r['scores'][m][d] for r in rows) for m in MODES}
  delta=statistics.fmean(r['scores']['global_p64'][d]-r['scores']['original_dense'][d] for r in rows)
  axes[d]={'source_equal_means':values,'source_equal_paired_difference':delta,'source_count':2}
 return {'per_source':rows,'per_axis':axes,'independent_image_sources':2,'samples_per_source_per_arm':1,'official_five_independent_samples':False,'quality_accepted':False}
