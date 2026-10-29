"""Pure four-arm, source-equal factorial contrasts; no inference or I/O."""
import math
import statistics
MODES=('none','high_only','low_only','both')
SOURCES=(898,96)
DIMENSIONS=('imaging_quality','aesthetic_quality','motion_smoothness','dynamic_degree','subject_consistency','background_consistency','i2v_subject','i2v_background')
def aggregate(rows, dimensions=DIMENSIONS):
 if len(rows)!=2 or {r.get('ordinal') for r in rows}!=set(SOURCES):raise ValueError('two unique exposed sources required')
 for r in rows:
  if set(r['scores'])!=set(MODES):raise ValueError('four actual arms required')
  for v in r['scores'].values():
   if set(v)!=set(dimensions) or any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for x in v.values()):raise ValueError('finite complete dimensions required')
 result={}
 for d in dimensions:
  per=[]
  for r in rows:
   s={m:r['scores'][m][d] for m in MODES}; n,h,l,b=(s[m] for m in MODES)
   per.append({'ordinal':r['ordinal'],'scores':s,'high_minus_none':h-n,'low_minus_none':l-n,'interaction':b-h-l+n})
  result[d]={'per_source':per,'source_equal_means':{m:statistics.fmean(p['scores'][m] for p in per) for m in MODES},'source_equal_high_minus_none':statistics.fmean(p['high_minus_none'] for p in per),'source_equal_low_minus_none':statistics.fmean(p['low_minus_none'] for p in per),'source_equal_interaction':statistics.fmean(p['interaction'] for p in per)}
 return {'per_source':rows,'dimensions':result,'independent_sources':2,'exposed_development_single_seed':True,'both_is_prior_video_reused':True,'none_is_byte_equal_prior_dense':True,'repeated_video_evaluation_is_not_new_sample':True,'official_five_independent_samples':False,'quality_accepted':False,'interaction_interpretation':'policy-level: same dynamic P64 policy, masks can differ under different student prefixes; not fixed-mask linear decomposition'}

def repeated_video_scores(new_scores, old_scores):
 """Numeric differences for repeated NONE/old-Dense and BOTH/old-P64 evaluations."""
 if set(new_scores)!=set(old_scores) or not new_scores:raise ValueError('matching metric dimensions required')
 result={}
 for key in new_scores:
  a,b=new_scores[key],old_scores[key]
  if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in (a,b)):raise ValueError('finite numeric scores required')
  result[key]={'new':a,'old':b,'difference':a-b,'exact_equal':a==b}
 return {'per_dimension':result,'all_exact_equal':all(v['exact_equal'] for v in result.values()),'interpretation':'repeat evaluation of same video bytes, not an independent sample'}

def compare_repeated_source_scores(new_rows, prior_rows):
 """Compare both reevaluated same-video arms to prior source-specific scores."""
 if {x['ordinal'] for x in new_rows}!={898,96} or {x['ordinal'] for x in prior_rows}!={898,96} or len(new_rows)!=2 or len(prior_rows)!=2:raise ValueError('exact two matching source identities required')
 old={r['ordinal']:r for r in prior_rows}
 return [{'ordinal':r['ordinal'],'none_vs_previous_dense':repeated_video_scores(r['scores']['none'],old[r['ordinal']]['scores']['original_dense']),'both_vs_previous_p64':repeated_video_scores(r['scores']['both'],old[r['ordinal']]['scores']['global_p64'])} for r in new_rows]
