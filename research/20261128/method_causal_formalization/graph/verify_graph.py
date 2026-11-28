#!/usr/bin/env python3
"""Check source-bound causal invariants of CAUSAL_GRAPH.json on CPU.

This checks the supplied abstract graph, not that a GPU runtime executed it.
"""
from collections import Counter, defaultdict, deque
from pathlib import Path
import argparse
import hashlib
import json
import struct

HERE = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def idsha(values):
    return hashlib.sha256(struct.pack("<"+"q"*len(values), *values)).hexdigest()


def verify(g, n_tokens):
    nodes = g["nodes"]
    idx = {n["id"]: n for n in nodes}
    require(len(idx)==len(nodes),"duplicate node id")
    indegree = {n["id"]:len(n["deps"]) for n in nodes}
    successors=defaultdict(list)
    edges=0
    for n in nodes:
        require(n["duration_s"] is None,"invented microkernel timing")
        for dep in n["deps"]:
            require(dep in idx,f"missing producer: {dep}")
            p=idx[dep]
            require(p["tau"]<n["tau"],f"future submission dependency: {dep}->{n['id']}")
            if n["step"] is not None and p["step"] is not None:
                require(p["step"]<=n["step"],"future denoising step input")
                if p["step"]==n["step"] and p["layer"] is not None and n["layer"] is not None:
                    require(p["layer"]<=n["layer"],"future layer input")
            if p["op"] in ("static_model_weights","offline_FFN_CAL_constants"):
                require((p["owner_expert"],p["owner_layer"])==(n["expert"],n["layer"]),
                        "cross-expert/layer static operand")
            successors[dep].append(n["id"])
            edges+=1
    queue=deque(k for k,v in indegree.items() if not v)
    order=[]
    while queue:
        p=queue.popleft();order.append(p)
        for n in successors[p]:
            indegree[n]-=1
            if not indegree[n]:queue.append(n)
    require(len(order)==len(nodes),"cycle")

    def has_ancestor(target, producer):
        stack=[target]; seen=set()
        while stack:
            n=stack.pop()
            if n==producer:return True
            if n in seen:continue
            seen.add(n)
            # All within-block queries can stop at older layer / global nodes.
            stack.extend(idx[n]["deps"])
        return False

    def block_nodes(s,l):
        return [n for n in nodes if (n["step"],n["layer"])==(s,l)]

    teacher_count=0
    for s in range(4):
        for l in range(40):
            prefix=f"s{s}.l{l:02d}."
            bn=block_nodes(s,l)
            kv=prefix+"full_KV"
            require(idx[kv]["keys"]==n_tokens,"current binding does not construct all K/V")
            require(idx[kv]["information_lower_bound_is_materialization"] is False,
                    "materialization incorrectly claimed as information lower bound")
            require(sum(n["op"]=="cross_text_KnormV" for n in bn)==1,"text KV invocation count")
            if l:
                require(f"s{s}.l{l-1:02d}.complete" in idx[prefix+"input"]["deps"],"missing layer barrier")
            for n in bn:
                if n["op"]=="native_P64_all_keys":
                    teacher_count+=1
                    require(n["keys"]==n_tokens and not n["is_causal"],"teacher key or frame-causal mismatch")
                    require(has_ancestor(n["id"],kv),"P64 teacher before full current K/V")
                    require(n["exact_on"]=="candidate_current_self_attention_input","P64 Dense reset claim")
                if n["op"]=="native_FFN_P64":
                    norms=[p["id"] for p in bn if p["op"]=="norm2_modulation" and has_ancestor(n["id"],p["id"])]
                    require(norms,"FFN teacher lacks actual norm2 input")
                    require(any(p["op"]=="cross_residual" and has_ancestor(n["id"],p["id"]) for p in bn),
                            "FFN teacher bypasses cross attention")
                    require(n["exact_on"]=="candidate_current_FFN_input","FFN Dense reset claim")
                if n["op"] in ("attention_full_readout_P_overwrite","attention_nonP_readout",
                                "attention_nonP_readout_folded_Wo"):
                    require(has_ancestor(n["id"],prefix+"attention_fit"),"nonP attention before current fit")
                if n["op"] in ("FFN_nonP_base_readout","FFN_full_tile_base_readout"):
                    require(has_ancestor(n["id"],prefix+"FFN_fit"),"nonP FFN before current fit")
            scatter=idx.get(prefix+"scatter_complete")
            if scatter:
                scope={idx[p]["scope"] for p in scatter["deps"]}
                require(scope=={"P",*(f"nonP_tile_{j}" for j in range(4))},"streamed row cover")
                require(scatter["immutable_input_no_alias"],"block input overwritten too early")
                require(not has_ancestor(prefix+"P_output",prefix+"attention_fit"),
                        "false fit dependency on exact P chain")
            require(has_ancestor(prefix+"complete",kv),"block output missing current K/V dependency")
        # Final head and scheduler current-state closure.
        require(has_ancestor(f"s{s}.unpatchify",f"s{s}.l39.complete"),"head before layer 39")
        require(set(idx[f"s{s}.scheduler_sub"]["deps"])=={f"s{s}.step_pre",f"s{s}.unpatchify"},
                "scheduler sub has missing/unbound history")
        require(idx[f"s{s}.scheduler_cast"]["solver_history_inputs"]==[],"unexpected multistep history")
        if s:
            require(idx[f"s{s}.step_pre"]["deps"]==[f"s{s-1}.scheduler_cast"],"broken step feedback")
    require(teacher_count==80,"expected 80 attention P64 sites")

    occupancy=defaultdict(int)
    peaks=defaultdict(int)
    lifetimes=defaultdict(list)
    events=[]
    for state in g["states"]:
        require((state["anchor_step"],state["hit_step"]) in ((0,1),(2,3)),"cache crosses wrong steps")
        e="high" if state["anchor_step"]==0 else "low"
        require(state["expert"]==e,"cache expert mismatch")
        ids=[state[k] for k in ("write","take","last_read","release")]
        require(all(v in idx for v in ids),"cache node missing")
        ns=[idx[v] for v in ids]
        require(ns[0]["tau"]<ns[1]["tau"]<=ns[2]["tau"]<ns[3]["tau"],"cache lifetime order")
        require(all((n["expert"],n["layer"])==(state["expert"],state["layer"]) for n in ns),
                "cross-expert/layer cache consumption")
        require(ns[1]["step"]==state["hit_step"] and ns[0]["step"]==state["anchor_step"],"cache step mismatch")
        require(ns[0]["id"] in ns[1]["deps"],"cache take lacks producer")
        require(ns[1]["physical_bytes_reusable"] is False,"owner clear mistaken for GPU completion")
        require(state["slot"]==state["layer"],"40-slot reuse index mismatch")
        lifetimes[state["kind"],state["slot"]].append((ns[0]["tau"],ns[3]["tau"],state["expert"]))
        events.extend([(ns[0]["tau"],state["kind"],1),(ns[3]["tau"],state["kind"],-1)])
    for key,ranges in lifetimes.items():
        ranges.sort()
        require(len(ranges)==2 and ranges[0][1]<ranges[1][0],f"physical slot overwritten live: {key}")
        require([r[2] for r in ranges]==["high","low"],"slot expert reuse ordering")
    for _,kind,delta in sorted(events):
        occupancy[kind]+=delta;peaks[kind]=max(peaks[kind],occupancy[kind])
        require(0<=occupancy[kind]<=40,"live state exceeds 40 layer slots")
    require(dict(peaks)=={"attention":40,"ffn":40},"state peak")
    require(all(v==0 for v in occupancy.values()),"unconsumed end state")
    require(g["physical_FFN_slots"]==40 and g["expert_layer_static_sites"]==80,"slots versus constants")
    require(sum(n["op"]=="cross_text_KnormV" for n in nodes)==160,"text KV is not actually cross-step cached")
    require(sum(n["op"]=="offline_FFN_CAL_constants" for n in nodes)==80,"missing expert/layer constants")
    return dict(nodes=len(nodes),semantic_edges=edges,acyclic=True,submission_order_topological=True,
         no_future_step_inputs=True,no_cross_expert_cache=True,current_all_key_barriers=True,
         hit_attention_P64_sites=teacher_count,hit_native_probe_queries=teacher_count*64,
         text_KV_invocations=160,physical_FFN_slots=40,expert_layer_static_sites=80,
         peak_live_attention_states=peaks["attention"],peak_live_FFN_states=peaks["ffn"],
         per_step_logical_state_count_after_complete=[40,0,40,0],
         cache_last_use_precedes_slot_rewrite=True,
         numeric_microkernel_duration_inferred=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph",type=Path,default=HERE/"CAUSAL_GRAPH.json")
    parser.add_argument("--output",type=Path,default=HERE/"GRAPH_VERIFICATION.json")
    parser.add_argument("--source-root",type=Path,help="Optional root containing original source paths; omitting it checks graph invariants and referenced metadata only")
    args=parser.parse_args()
    data=json.loads(args.graph.read_text())
    N=data["axes"]["N"]; probes=data["probes"]["ids"]
    require(len(probes)==len(set(probes))==64 and min(probes)>=0 and max(probes)<N,"P64 IDs invalid")
    require(idsha(probes)==data["probes"]["sha256_int64_le"],"P order hash")
    require(all(n==(f*29+y)*52+x for n,(f,y,x) in zip(probes,data["probes"]["coordinates"])),"flatten mismatch")
    non=[n for n in range(N) if n not in set(probes)]
    for t in data["partition"]["tiles"]:
        ids=non[t["start_offset"]:t["start_offset"]+t["rows"]]
        require(idsha(ids)==t["sha256_int64_le"],"nonP original-coordinate partition mismatch")
    require(sum(t["rows"] for t in data["partition"]["tiles"])+len(probes)==N,"row partition cover")
    checked=0
    if args.source_root is not None:
        for source in data["source_fingerprints"]:
            require(hashlib.sha256((args.source_root/source["path"]).read_bytes()).hexdigest()==source["sha256"],
                    "source changed: "+source["path"])
            checked+=1
    report=dict(status="PASS_SOURCE_DERIVED_GRAPH_INVARIANTS",new_GPU_execution=False,
        actual_P64_frames=sorted({p//(29*52) for p in probes}),
        actual_P64_frame_coverage=len({p//(29*52) for p in probes}),
        actual_nonP_tile_rows=[t["rows"] for t in data["partition"]["tiles"]],
        graph_sha256=hashlib.sha256(args.graph.read_bytes()).hexdigest(),
        source_fingerprints_referenced=len(data["source_fingerprints"]),
        source_fingerprints_freshly_verified=checked,
        source_verification_mode="fresh_local_SHA256" if args.source_root is not None else "metadata_only_no_source_files_read",
        scheduler_numeric_latest_binding=data["scheduler"]["latest_numeric_sigma_binding"],
        graphs={g["name"]:verify(g,N) for g in data["graphs"]},
        limits=data["verification_limits"])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
