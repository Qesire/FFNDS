from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import torch
import torch.nn as nn
from .nvfp4_rtn_backend import global_scale_from_amax
from .nvfp4_rtn_linear import iter_selected_linears

class NVFP4ActivationCalibrator:
    def __init__(self,*,include_regex=None,exclude_regex=None,margin=1.0):
        self.include_regex=include_regex; self.exclude_regex=exclude_regex; self.margin=float(margin)
        self.maxima={}; self.calls={}; self.selected=[]; self._handles=[]
    def attach(self,model:nn.Module,*,prefix:str):
        names=[]
        for _,full_name,mod in iter_selected_linears(model,prefix=prefix,include_regex=self.include_regex,exclude_regex=self.exclude_regex):
            names.append(full_name); self.selected.append(full_name)
            def hook(_module,inputs,name=full_name):
                if not inputs or not torch.is_tensor(inputs[0]): return
                cur=inputs[0].detach().abs().amax().to(torch.float32); prev=self.maxima.get(name)
                self.maxima[name]=cur if prev is None else torch.maximum(prev.to(cur.device),cur)
                self.calls[name]=self.calls.get(name,0)+1
            self._handles.append(mod.register_forward_pre_hook(hook))
        return {"prefix":prefix,"selected":len(names),"selected_names":names}
    def remove(self):
        for h in self._handles: h.remove()
        self._handles.clear()
    def finalize(self,*,scales_output,report_output,require_all_observed=True):
        self.remove(); selected=sorted(set(self.selected)); observed=sorted(self.maxima); missing=sorted(set(selected)-set(observed))
        scales={}; amax_tensors={}; rows=[]
        for name in selected:
            if name not in self.maxima: continue
            amax=float(self.maxima[name].detach().float().cpu().item()); g=global_scale_from_amax(amax,margin=self.margin)
            scales[f"{name}.input_global_scale"]=torch.tensor([g],dtype=torch.float32)
            amax_tensors[f"{name}.input_amax"]=torch.tensor([amax],dtype=torch.float32)
            rows.append({"name":name,"calls":int(self.calls.get(name,0)),"input_amax":amax,"input_global_scale":g})
        sp=Path(scales_output); rp=Path(report_output); sp.parent.mkdir(parents=True,exist_ok=True); rp.parent.mkdir(parents=True,exist_ok=True)
        from safetensors.torch import save_file
        save_file(scales,str(sp),metadata={"format":"nvfp4_activation_global_scales"})
        ap=sp.with_name('nvfp4_activation_amax.safetensors'); save_file(amax_tensors,str(ap),metadata={"format":"nvfp4_activation_amax"})
        report={"format":"NVFP4","definition":"per-Linear G=448*6/calibration_amax; runtime group16 FP8-E4M3FN local scale",
            "selection_policy":"exact_match_to_successful_HiF4_selector","margin":self.margin,
            "selected_count":len(selected),"observed_count":len(observed),"scale_count":len(scales),
            "missing_count":len(missing),"missing":missing,"scales_output":str(sp),"amax_output":str(ap),"layers":rows}
        rp.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
        if require_all_observed and missing: raise RuntimeError(f"Calibration missed {len(missing)} layers; first={missing[:10]}")
        return report
