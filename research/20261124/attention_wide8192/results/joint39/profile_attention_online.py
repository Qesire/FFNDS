"""Measure attention components in the timing job's instrumented warm-ups."""
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from statistics import median
import sys
from types import SimpleNamespace

import torch

import attention_online_structured as online


original_bind = online.OnlineAttention.bind


@contextmanager
def instrumented_bind(self,arm,audit=None,profile=False,archive_routes=False):
    if not profile:
        with original_bind(self,arm,audit=audit,profile=False,archive_routes=archive_routes) as result:
            yield result
        return

    events=defaultdict(list)

    def measured(name,function):
        def timed(*args,**kwargs):
            start,stop=(torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            value=function(*args,**kwargs)
            stop.record()
            events[name].append((start,stop))
            return value
        return timed

    route,selected,approximate=self.route,self.selected,self.approximate
    flash,fit,original_functional=online.native_flash,online.schur_fit,online.F
    self.route=measured("route",route)
    self.selected=measured("selected_and_statistics",selected)
    online.native_flash=measured("low_native_anchor_O_LSE",flash)
    online.schur_fit=measured("probe_schur_fit",fit)
    teacher=measured("native_P64_full_KV",original_functional.scaled_dot_product_attention)
    total=measured("selected_probe_recovery_total",approximate)

    def candidate(*args, **kwargs):
        online.F=SimpleNamespace(pad=original_functional.pad,scaled_dot_product_attention=teacher)
        try:
            return total(*args,**kwargs)
        finally:
            online.F=original_functional

    self.approximate=candidate
    try:
        with original_bind(self,arm,audit=audit,profile=True,archive_routes=archive_routes) as result:
            yield result
    finally:
        self.route,self.selected,self.approximate=route,selected,approximate
        online.native_flash,online.schur_fit=flash,fit

    torch.cuda.synchronize()
    normal={name:[a.elapsed_time(b) for a,b in rows] for name,rows in events.items()}
    layout={"route":(0,2),"selected_and_statistics":(1,2,3),"low_native_anchor_O_LSE":(2,),
            "probe_schur_fit":(1,3),"native_P64_full_KV":(1,3),"selected_probe_recovery_total":(1,3)}
    assert set(normal)==set(layout)
    components={}
    for label,steps in layout.items():
        values=normal[label]
        assert len(values)==40*len(steps)
        components[label]={str(step):median(values[40*i+4:40*(i+1)]) for i,step in enumerate(steps)}
    dest=Path("output")/f"PROFILE_{arm}.json"
    online.save(dest,dict(arm=arm,metric="CUDA event median milliseconds per layer at layers 4-39 of instrumented warm-up",
                          overlapping_components=True,profile_not_used_for_formal_timing=True,components=components))

if __name__=="__main__":
    online.OnlineAttention.bind=instrumented_bind
    online.main(*(Path(p).resolve() for p in sys.argv[1:4]),sys.argv[4])
