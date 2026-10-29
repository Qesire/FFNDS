"""Deterministic safe adaptation of Chipmunk's first-1024 selector.

Author CURAND/thread/atomic order is not reproduced. Padding picks distinct
lowest invalid indices and updates them as real channels. No tensor imports.
"""
import math,random,hashlib,json
SAMPLE=1024;QUANTILE=0.7;RANDOM_RATE=0.05;ALIGN=256
PROXY_SEED=46829;RANDOM_SEEDS=(101,202,303)

def validate_indices(v,h):
    if not isinstance(v,list) or any(type(i) is not int or not 0<=i<h for i in v) or len(set(v))!=len(v):raise ValueError('Invalid or duplicate selection')
    return v

def selector(scores,seed=PROXY_SEED,sample=SAMPLE,quantile=QUANTILE,random_rate=RANDOM_RATE,align=ALIGN):
    h=len(scores)
    if not h or sample<=0 or h<sample or not 0<=quantile<=1 or not 0<=random_rate<=1 or align<=0:raise ValueError('Invalid selector inputs')
    if any(not math.isfinite(float(x)) for x in scores):raise ValueError('Nonfinite score')
    if quantile==0:return {'indices':list(range(h)),'K':h,'cutoff':None,'base_count':h,'padding_count':0,'edge':'AUTHOR_QUANTILE_ZERO_FULL'}
    if quantile==1:return {'indices':[],'K':0,'cutoff':None,'base_count':0,'padding_count':0,'edge':'AUTHOR_QUANTILE_ONE_EMPTY'}
    cutoff=sorted(float(x) for x in scores[:sample])[int(sample*quantile)]
    rng=random.Random(seed);chosen=[];invalid=[];threshold_count=0;random_added=0
    for i,score in enumerate(scores):
        if float(score)>=cutoff:chosen.append(i);threshold_count+=1
        elif rng.random()<random_rate:chosen.append(i);random_added+=1
        else:invalid.append(i)
    base=len(chosen);target=min(h,((base+align-1)//align)*align);padding=target-base
    chosen.extend(invalid[:padding]);chosen.sort();validate_indices(chosen,h)
    return {'indices':chosen,'K':target,'cutoff':cutoff,'base_count':base,'padding_count':padding,
            'threshold_count':threshold_count,'threshold_tie_count':sum(float(x)==cutoff for x in scores),
            'random_added':random_added,'edge':'NORMAL',
            'rng':'PYTHON_MT19937_FIXED_SEED_NOT_AUTHOR_CURAND','padding':'ASCENDING_INVALID_BOUNDED_NOT_AUTHOR_ATOMIC_ORDER'}

def masks(k,h):
    if not 0<=k<=h:raise ValueError('K outside range')
    return {**{'random_'+str(seed):sorted(random.Random(seed).sample(range(h),k)) for seed in RANDOM_SEEDS},
            'fixed':list(range(k)),'full':list(range(h)),'empty':[]}

def digest(indices):return hashlib.sha256(json.dumps(indices,separators=(',',':')).encode()).hexdigest()
