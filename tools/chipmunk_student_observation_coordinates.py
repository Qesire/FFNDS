"""Deterministic, disjoint selection/evaluation coordinates for frozen grid."""
import random

def coordinates(selector,*,seed=42,t=21,h=29,w=52,n=256):
 m=t*h*w; selected=set(selector)
 if len(selected)!=len(selector) or any(type(x) is not int or x<0 or x>=m for x in selected):raise ValueError('selector coordinates differ')
 cells=[]
 for tau in range(t):
  for yhalf in range(2):
   for xhalf in range(2):
    ids=[tau*h*w+y*w+x for y in range(h) for x in range(w) if (y>=h//2)==bool(yhalf) and (x>=w//2)==bool(xhalf) and tau*h*w+y*w+x not in selected]
    cells.append(ids)
 rng=random.Random(seed)
 for ids in cells:rng.shuffle(ids)
 counts=[0]*len(cells)
 for j in range(n):
  i=j%len(cells)
  if counts[i]>=len(cells[i]):raise ValueError('insufficient disjoint rows')
  counts[i]+=1
 result=[]
 for i,ids in enumerate(cells):
  for idx in ids[:counts[i]]:
   tau,rem=divmod(idx,h*w);y,x=divmod(rem,w)
   result.append({'row':idx,'tau':tau,'y':y,'x':x,'stratum':i,'eligible_population':len(ids),'sample_count':counts[i],'inclusion_probability':counts[i]/len(ids)})
 return result
