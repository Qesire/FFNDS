#!/usr/bin/env python3
"""Build source-derived HHLL execution DAGs; no torch/model/GPU execution.

The nodes are operator boundaries, not profiler events. `tau` is the observed
Python submission order of those boundaries. `deps` are information/lifetime
dependencies. Measured seconds are attached only to whole-step macro records.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import struct

OUT = Path(__file__).resolve().parent
N, D, H, DH, L, P, TILE = 31668, 5120, 40, 128, 40, 64, 8192
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ids_sha(ids):
    return hashlib.sha256(struct.pack("<" + "q" * len(ids), *ids)).hexdigest()


class Graph:
    def __init__(self, name, streamed):
        self.name, self.streamed = name, streamed
        self.nodes, self.states, self.macros = [], [], []
        self.index, self.cache = {}, {}
        self.step = self.layer = self.expert = None

    def add(self, suffix, op, deps=(), scope="all", **attrs):
        prefix = "global" if self.step is None else f"s{self.step}"
        if self.layer is not None:
            prefix += f".l{self.layer:02d}"
        name = prefix + "." + suffix
        assert name not in self.index, name
        node = dict(id=name, tau=len(self.nodes), step=self.step, layer=self.layer,
                    expert=self.expert, op=op, scope=scope, deps=list(dict.fromkeys(deps)),
                    duration_s=None, **attrs)
        self.nodes.append(node)
        self.index[name] = node
        return name

    def write_state(self, kind, deps, fields):
        node = self.add(kind + "_anchor_write", "anchor_write", deps,
                        state_kind=kind, fields=fields, physical_slot=self.layer)
        rec = dict(id=f"{kind}.{self.expert}.l{self.layer:02d}", kind=kind,
                   expert=self.expert, layer=self.layer, anchor_step=self.step,
                   hit_step=self.step + 1, slot=self.layer, write=node,
                   take=None, last_read=None, release=None, fields=fields)
        self.states.append(rec)
        self.cache[kind, self.expert, self.layer] = rec
        return node

    def take_state(self, kind):
        rec = self.cache[kind, self.expert, self.layer]
        rec["take"] = self.add(kind + "_anchor_take", "anchor_take", [rec["write"]],
                               state_kind=kind, state_id=rec["id"],
                               physical_slot=self.layer,
                               logical_owner_released=True,
                               physical_bytes_reusable=False)
        return rec["take"]

    def release_state(self, kind, last_read):
        rec = self.cache[kind, self.expert, self.layer]
        rec["last_read"] = last_read
        rec["release"] = self.add(kind + "_borrow_end", "borrow_end", [last_read, rec["take"]],
                                  state_id=rec["id"], state_kind=kind,
                                  physical_slot=self.layer, physical_bytes_reusable=True)
        return rec["release"]

    def text_kv(self, context, weights, suffix="text_kv"):
        return self.add(suffix, "cross_text_KnormV", [context, weights],
                        scope="text", axes="[text_tokens,H,Dh]",
                        recomputed_this_invocation=True,
                        cross_step_cache_implemented=False)

    def tail(self, suffix, x, attention, text_kv, modulation, weights, scope,
             already_projected=False, context=None):
        y = attention if already_projected else self.add(suffix + "_Wo", "self_Wo",
                [attention, weights], scope, axes="[rows,d]")
        r = self.add(suffix + "_self_gate", "self_gate_residual", [x, y, modulation], scope)
        n = self.add(suffix + "_norm3", "norm3", [r, weights], scope)
        q = self.add(suffix + "_cross_Qnorm", "cross_Q_projection_norm", [n, weights], scope)
        if text_kv is None:
            text_kv = self.text_kv(context, weights)
        a = self.add(suffix + "_cross_attention", "cross_attention", [q, text_kv], scope)
        a = self.add(suffix + "_cross_Wo", "cross_Wo", [a, weights], scope)
        r = self.add(suffix + "_cross_residual", "cross_residual", [r, a], scope)
        u = self.add(suffix + "_norm2_mod", "norm2_modulation", [r, weights, modulation], scope,
                     axes="[rows,d]", exact_input_boundary="candidate_current_FFN_input")
        return r, u

    def finish(self, suffix, r, y, mod, scope, native=False):
        return self.add(suffix + "_output", "FFN_gate_residual", [r, y, mod], scope,
                        implementation="native_rows" if native else "fused_full_field")

    def baseline_block(self, x, context, embed, weights, static, rope):
        mod = self.add("modulation", "six_shift_scale_gate_fields", [embed, weights])
        u = self.add("norm1_mod", "norm1_modulation", [x, mod, weights], axes="[N,d]")
        q = self.add("Q", "self_Q_projection_norm", [u, weights], axes="[N,H,Dh]")
        k = self.add("K", "self_K_projection_norm", [u, weights], axes="[N,H,Dh]")
        v = self.add("V", "self_V_projection", [u, weights], axes="[N,H,Dh]")
        qr = self.add("QK_RoPE", "paired_QK_RoPE", [q, k, rope], original_token_ids=True)
        kv = self.add("full_KV", "full_KV_materialized", [qr, v],
                      keys=N, all_key_contributions_available=True,
                      implementation_requirement=True,
                      information_lower_bound_is_materialization=False)
        if self.step in (0, 2):
            route = self.add("route", "refresh_route", [qr, static], query_block=128, top_blocks=31)
            if self.step == 0 and not self.streamed:
                ca = self.write_state("attention", [route], ["route"])
            if self.step == 2 and self.streamed:
                moments = self.add("anchor_value_moments", "value_global_mean_scale", [v])
                att = self.add("native_attention", "partition_selected_omitted_full_recompose",
                               [qr,kv,route,moments], exact_on="candidate_current_input_in_real_arithmetic",
                               source_boundary="partition_attention",
                               finite_precision_equivalence="real_equivalent_only")
            else:
                att = self.add("native_attention", "native_full_attention", [qr, kv],
                               exact_on="candidate_current_input")
            if self.step == 2:
                if self.streamed:
                    selected = self.add("anchor_selected", "partition_selected_view", [att])
                else:
                    selected = self.add("anchor_selected", "selected_full_field", [qr, kv, route])
                    moments = self.add("anchor_value_moments", "value_global_mean_scale", [v])
                hist = self.add("low_history", "partition_rho_centered_C_projected_A" if self.streamed else "rho_centered_C_projected_A", [att, selected, moments, static],
                                precision="rho_FP32_C_A_BF16")
                ca = self.write_state("attention", [route, hist], ["route", "rho", "C", "A"])
            elif self.streamed:
                ca = self.write_state("attention", [route, att], ["route"])
            self.index[att]["cache_completion"] = ca
        else:
            state = self.take_state("attention")
            selected = self.add("selected", "selected_full_field", [qr, kv, state])
            design = self.add("attention_design", "attention_design_32", [selected, state, static])
            truth = self.add("attention_P64", "native_P64_all_keys", [qr, kv], "P",
                             rows=P, keys=N, is_causal=False,
                             exact_on="candidate_current_self_attention_input")
            bd = [selected, state]
            if self.step == 3:
                moments = self.add("value_moments", "value_global_mean_scale", [v])
                bd.append(moments)
            base = self.add("attention_base", "attention_base", bd)
            fit = self.add("attention_fit", "attention_P64_global_fit", [truth, design, base, static], "P",
                           current_step_only=True, readout_shape="[H,32,Dh]")
            att = self.add("attention_readout", "attention_full_readout_P_overwrite", [base, design, fit, truth],
                           axes="[N,H,Dh]", native_probe_overwrite=True)
            self.release_state("attention", att)
        # Native code computes context K/V inside infer_cross_attn for every invocation.
        r, u2 = self.tail("all", x, att, None, mod, weights, "all", context=context)
        if self.step in (0, 2):
            z = self.add("FFN_W1", "FFN_W1_bias", [u2, weights])
            zsel = self.add("FFN_selected_preact", "selected_2560_preactivation", [z, static])
            act = self.add("FFN_gelu", "GELU_tanh", [z])
            y = self.add("FFN_W2", "FFN_W2_bias_direct_anchor", [act, weights],
                         direct_anchor_already_in_baseline=True)
            cf = self.write_state("ffn", [u2, zsel, y], ["u0", "z0k", "y0"])
            ydep = self.add("FFN_anchor_ready", "anchor_output_alias", [y, cf])
        else:
            state = self.take_state("ffn")
            operands = self.add("FFN_P_operands", "FFN_base_modes_kernel_features", [u2, state, static], "P", rows=P)
            truth = self.add("FFN_P64", "native_FFN_P64", [u2, weights], "P", rows=P,
                             exact_on="candidate_current_FFN_input")
            fit = self.add("FFN_fit", "FFN_P64_global_fit", [truth, operands, static], "P",
                           native_observations=64, shared_modes=16, joint_design_columns=80,
                           point_edge_metric_rows=106, new_current_native_observation_rank=64)
            parts = []
            for j in range(math.ceil(N / TILE)):
                parts.append(self.add(f"FFN_tile{j}", "FFN_full_tile_base_readout",
                    [u2, state, static, fit], f"full_tile_{j}",
                    tile_start=j*TILE, rows=min(TILE, N-j*TILE), includes_P_then_overwritten=True))
            ydep = self.add("FFN_gather_P_overwrite", "FFN_full_output_P_overwrite", [*parts, truth],
                            native_probe_overwrite=True)
            self.release_state("ffn", ydep)
        return self.finish("all", r, ydep, mod, "all")

    def streamed_block(self, x, context, embed, weights, static, rope):
        # Anchors still use full-field path, with 09's low partition alternative.
        if self.step in (0, 2):
            return self.baseline_block(x, context, embed, weights, static, rope)
        attention_state, ffn_state = self.take_state("attention"), self.take_state("ffn")
        mod = self.add("modulation", "six_shift_scale_gate_fields", [embed, weights])
        pieces = []
        for j in range(math.ceil(N/TILE)):
            u = self.add(f"KV{j}_norm1", "norm1_modulation", [x, mod, weights], f"full_tile_{j}",
                         rows=min(TILE,N-j*TILE))
            k = self.add(f"KV{j}_Knorm", "self_K_projection_norm", [u, weights], f"full_tile_{j}")
            v = self.add(f"KV{j}_V", "self_V_projection", [u, weights], f"full_tile_{j}")
            pieces.append(self.add(f"KV{j}_RoPE", "K_RoPE_private_paired_fallback", [k,v,rope],
                                   f"full_tile_{j}", original_token_ids=True,
                                   unused_paired_RoPE_output=True))
        kv = self.add("full_KV", "full_KV_materialized", pieces, keys=N,
                      all_key_contributions_available=True, implementation_requirement=True,
                      information_lower_bound_is_materialization=False)
        text_kv = self.text_kv(context, weights)

        def query(label, scope):
            u = self.add(label+"_norm1", "norm1_modulation_recomputed", [x,mod,weights], scope)
            q = self.add(label+"_Qnorm", "self_Q_projection_norm", [u,weights], scope)
            return self.add(label+"_Q_RoPE", "Q_RoPE_private_paired_fallback", [q,rope], scope,
                            original_token_ids=True, unused_paired_RoPE_output=True)

        qp = query("P", "P")
        extra = []
        if self.step == 3:
            extra.append(self.add("value_moments", "value_global_mean_scale", [kv]))
        selected = self.add("selected_P", "selected_P64", [qp,kv,attention_state], "P", rows=P)
        truth = self.add("attention_P64", "native_P64_all_keys", [qp,kv], "P", rows=P,
                         keys=N, is_causal=False, exact_on="candidate_current_self_attention_input")
        afit = self.add("attention_fit", "attention_P64_global_fit",
                        [truth,selected,attention_state,static,*extra], "P",
                        current_step_only=True, readout_shape="[H,32,Dh]")
        fold = None
        if self.step == 1:
            fold = self.add("high_fold", "high_readout_Wo_fold_setup", [afit,weights,static],
                            scope="weights", effective_projection_shape=[D,D],
                            finite_precision_equivalence="real_equivalent_only")
        # P tail has no semantic dependence on afit/fold. Python prepares fit
        # first; this order is represented by tau, not a false information edge.
        rp, up = self.tail("P", x, truth, text_kv, mod, weights, "P")
        fop = self.add("FFN_P_operands", "FFN_base_modes_kernel_features", [up,ffn_state,static], "P")
        ftrue = self.add("FFN_P64", "native_FFN_P64", [up,weights], "P", rows=P,
                         exact_on="candidate_current_FFN_input")
        ffit = self.add("FFN_fit", "FFN_P64_global_fit", [ftrue,fop,static], "P",
                        native_observations=64, shared_modes=16, joint_design_columns=80,
                        point_edge_metric_rows=106, new_current_native_observation_rank=64)
        outputs = [self.finish("P",rp,ftrue,mod,"P",native=True)]
        last_attention = None
        for j in range(math.ceil((N-P)/TILE)):
            scope = f"nonP_tile_{j}"
            q = query(f"tile{j}",scope)
            sel = self.add(f"tile{j}_selected", "selected_nonP_tile", [q,kv,attention_state],scope,
                           original_token_ids=True, rows=min(TILE,N-P-j*TILE))
            att = self.add(f"tile{j}_attention_readout",
                "attention_nonP_readout_folded_Wo" if fold else "attention_nonP_readout",
                [sel,attention_state,static,afit,*extra]+([fold] if fold else []),scope)
            last_attention = att
            r,u = self.tail(f"tile{j}",x,att,text_kv,mod,weights,scope,already_projected=bool(fold))
            y = self.add(f"tile{j}_FFN", "FFN_nonP_base_readout", [u,ffn_state,static,ffit],scope)
            outputs.append(self.finish(f"tile{j}",r,y,mod,scope,native=True))
        self.release_state("attention",last_attention)
        self.release_state("ffn",outputs[-1])
        return self.add("scatter_complete", "all_original_rows_scattered", outputs,
                        rows=N, immutable_input_no_alias=True)

    def build(self, observed):
        cond = self.add("condition", "request_image_text_encoding", scope="request", opaque=True)
        z = self.add("initial_latent", "seed42_initial_latent", scope="latent", axes="[16,21,58,104]")
        rope = self.add("RoPE_grid", "position_cache_shape_only", scope="positions",
                        grid=[21,29,52], flatten="n=(f*29+y)*52+x")
        weights, static, nonblock_weights = {}, {}, {}
        for expert in ("high","low"):
            nonblock_weights[expert] = self.add(f"nonblock_weights_{expert}", "expert_prepost_weights",
                                                scope="weights", owner_expert=expert)
            for layer in range(L):
                weights[expert,layer] = self.add(f"weights_{expert}_{layer}", "static_model_weights",
                                                scope="weights", owner_expert=expert,owner_layer=layer)
                static[expert,layer] = self.add(f"fit_{expert}_{layer}", "offline_FFN_CAL_constants",
                                               scope="constants",owner_expert=expert,owner_layer=layer,
                                               depends_on_future_online_readout=False)
        for step in range(4):
            self.step,self.expert,self.layer = step,"high" if step<2 else "low",None
            start = self.add("step_pre", "native_latent_cast_and_timestep", [z], scope="latent",
                             native_latent_dtype="BF16", schedule_index=step)
            ew = nonblock_weights[self.expert]
            emb = self.add("time_embedding", "timestep_embedding_projection", [start,ew],scope="condition")
            context = self.add("text_embedding", "expert_text_embedding", [cond,ew],scope="text",
                               recomputed_this_step=True)
            x = self.add("patch_embedding", "concat_condition_patch_embedding", [start,cond,ew],
                         axes="[N,d]", patch_size=[1,2,2], scheduler_latent_overwrite=False)
            blocks = []
            for layer in range(L):
                self.layer = layer
                blockin = self.add("input", "complete_layer_input", [x], axes="[N,d]",
                                   all_rows_required_by_current_binding=True)
                fn = self.streamed_block if self.streamed else self.baseline_block
                end = fn(blockin,context,emb,weights[self.expert,layer],static[self.expert,layer],rope)
                x = self.add("complete", "layer_barrier", [end], axes="[N,d]", rows=N,
                             required_before_next_layer_current_KV=True)
                blocks.append(x)
            self.layer = None
            head = self.add("head", "final_norm_time_modulation_linear_head", [x,emb,ew],axes="[N,64]")
            flow = self.add("unpatchify", "unpatchify_FP32_noise_pred", [head],scope="latent",axes="[16,21,58,104]")
            sub = self.add("scheduler_sub", "FP32_z_minus_sigma_v", [start,flow],scope="latent")
            if step < 3:
                upd = self.add("scheduler_add", "FP32_add_next_sigma_v", [sub,flow],scope="latent")
            else:
                upd = sub
            z = self.add("scheduler_cast", "cast_update_to_native_latent_dtype", [upd],scope="latent",
                         native_latent_dtype="BF16", solver_history_inputs=[],
                         scheduler="Wan22StepDistillScheduler.step_post")
            self.macros.append(dict(step=step,expert=self.expert,mode="anchor" if step%2==0 else "hit",
                start=start,end=z,layer_barriers=blocks,
                observed_mean_step_event_s=observed[step],
                timing_scope="dispatch event through step_post and symmetric flow/latent witness clones",
                per_kernel_durations_known=False))
        self.step=self.layer=self.expert=None
        self.add("video", "VAE_decode_and_video_encode", [z,cond],scope="request",opaque=True)
        return dict(name=self.name,description="source-derived boundaries, not a captured kernel trace",
                    status="measured_implemented_schedule",nodes=self.nodes,states=self.states,macros=self.macros,
                    physical_FFN_slots=40,expert_layer_static_sites=80,
                    text_KV_cross_step_cache_implemented=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs",type=Path,default=OUT/"GRAPH_INPUTS.json")
    parser.add_argument("--output",type=Path,default=OUT/"CAUSAL_GRAPH.json")
    args=parser.parse_args()
    inputs=json.loads(args.inputs.read_text())
    expected=dict(N=N,d=D,H=H,Dh=DH,layers=L,steps=4,grid=[21,29,52],tile_rows=TILE)
    if inputs["geometry"]!=expected:
        raise ValueError("This source-derived builder requires the frozen wide8192 geometry")
    probes=inputs["observer_ids"]
    if ids_sha(probes)!=inputs["observer_ids_sha256_int64_le"]:
        raise ValueError("P64 original-order hash differs from input binding")
    pset=set(probes)
    non=[n for n in range(N) if n not in pset]
    paired=inputs["paired_macro_timing"]["records"]
    graphs=[Graph("frozen_centered_wide8192",False).build(paired["reference_step_mean_s"]),
            Graph("combined_graph_09_probe_first",True).build(paired["candidate_step_mean_s"])]
    data=dict(schema_version=1,evidence_HEAD=inputs["evidence_HEAD"],
        measurements_commit=inputs["measurements_commit"],new_GPU_execution=False,
        portable_inputs_sha256=sha(args.inputs),
        production_modified=False,axes=dict(denoise_step="s=0,1,2,3",latent_frame="f=0,...,20",
        logical_execution_order="tau=node submission index; not elapsed time",grid=[21,29,52],
        token_index="n=(f*29+y)*52+x",N=N,d=D,H=H,Dh=DH,rows_per_tile=TILE),
        probes=dict(ids=probes,sha256_int64_le=ids_sha(probes),original_order_preserved=True,
                    coordinates=[[n//(29*52),(n//52)%29,n%52] for n in probes]),
        partition=dict(nonP_order="ascending original global token id",tiles=[dict(
            index=j,start_offset=j*TILE,rows=len(non[j*TILE:(j+1)*TILE]),
            sha256_int64_le=ids_sha(non[j*TILE:(j+1)*TILE])) for j in range(math.ceil(len(non)/TILE))]),
        source_fingerprints=inputs["source_fingerprints"],
        source_fingerprints_rechecked_during_build=False,
        timing_input_binding=inputs["paired_macro_timing"],
        scheduler=inputs["scheduler_binding"],
        graphs=graphs,
        unimplemented_mathematical_rewrites=[
          dict(name="next_layer_KV_producer_pipeline",implemented=False,
               allowable="produce rowwise norm/modulation/Knorm/RoPE/V of layer l+1 from each completed layer-l tile",
               remaining_barrier="native P64 attention output requires contributions from all layer-(l+1) keys; current API additionally needs full contiguous K/V",
               caveats=["global token coordinates fixed","new buffer/event ownership required","online softmax can accumulate key tiles but final normalization waits all contributions","no measured speed claim"]),
          dict(name="probe_tail_fit_overlap",implemented=False,
               allowable="P tail depends on native P64 values, not nonP readout fit; it may be scheduled while readout fitting completes",
               remaining_barrier="nonP reconstruction requires fit; resource overlap is hardware dependent"),
          dict(name="text_KV_same_expert_cache",implemented=False,
               allowable="reuse only same source/condition, expert, layer, weight version, dtype and context row order",
               remaining_barrier="high and low experts have distinct embeddings and projection weights; existing code recomputes every invocation")],
        verification_limits=["No GPU kernels, asynchronous completion, or profiler trace verified here.",
            "tau is source-order abstraction; scalar nodes may represent several launches.",
            "No microkernel duration or optimal wall-clock inferred from step events.",
            "Real arithmetic row scheduling equivalence does not imply BF16 bitwise equivalence.",
            "Candidate native anchors/probes are exact on their own current inputs, not resets to Dense trajectory."])
    dest=args.output
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"path":str(dest),"nodes":{g["name"]:len(g["nodes"]) for g in graphs},
                      "states":{g["name"]:len(g["states"]) for g in graphs}},ensure_ascii=False))


if __name__=="__main__":
    main()
