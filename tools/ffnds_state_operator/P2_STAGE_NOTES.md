# P2 production scope preparation

Run the generator on the school filesystem once P1 has an actual successful Slurm EXIT and BF16 acceptance result. Put `prepare_p2.py`, `stage_renderer.py`, and `wan_state_operator_job.sh` in the same tooling directory. Preserve their byte hashes when transferring.

```bash
work="$HOME/projects/wan-quant/work"
experiment="$work/ffnds_state_operator_20261118_3011ca9"
strict="$work/ffnds_strict_reproduction_20261116_336b7e1"
new_stage_root="$experiment/p2_new"
python3.11 -B prepare_p2.py --p1 "$experiment/p1_v2" --strict "$strict" --out "$new_stage_root"
```

The generator writes four immutable scope candidates under `p2/l0`, `p2/lplus`, `p2/q32`, and `p2/n32`, with an inventory in `p2/PREPARED.json`. Each source is seed 42 on the archived HHLL scheduler, with 81 frames at 832 × 464 and 16 fps. The condition map ties image bytes, prompt, cluster, Dense noise, and the Dense/Q1/Global-P64 media and final latent to the existing receipts. The candidate scope requires the reviewed P0 six-site BF16 Q1 identity and the completed P1 fit, replay, and allocation receipts.

Each arm requests one 80 GB A100, 4 CPUs, and 160 GiB host memory for at most 90 minutes. The four arms together have a 12 D4 and 48 NFE ceiling. Dispatch must keep at most two P2 GPU jobs active and use zero retries. Check shared disk space before concurrent launch. A single job may create roughly 13 GB in live Q1/candidate residuals, and each checks a 12 GiB in-flight floor. The scope freezes those limits but the dispatcher is responsible for the two-job ceiling.

Each video should yield 40 FFN calls per step, comprising 12 candidate calls and 148 calls through the exact archived Q1 action. Review the dense anchor return, same-expert hit consumption, and actual gate write evidence in production. P1 checks archived real pre-gate tensors and gate with a controlled residual addend. The live P2 hook compares the actual residual write against one multiplication by the current gate. The renderer adds flow/latent hashes and rejects missing traces or a video geometry mismatch.

Independently review the scope files and present a real matching review and ACTIVE record before dispatching each arm. The generated inventory records the pending review state. An arm that fails admission or execution consumes its frozen attempt under the plan.

The stage generator extends the exact P1 companion source `afa7b57ac5ddb3d16d89ecc71790a0e9084b7bdda6479c618bdf41841ae674dc` only in the generated P2 staging bundle. The resulting `wan_state_operator_job.py` has SHA256 `7156a6b686b83c58a26357a7aaed3211a7f2a5286e8f76a2cfebc35118ef8e30`. After constructing the pipeline and before dispatch, it requires the live `EFFECTIVE_CONFIG.json` to match the archived strict Q1 configuration byte for byte. It records both file hashes and the outcome in `output/RECIPE_PARITY.json`. The arm's `output/BINDINGS.json` binds this receipt on success. A discrepancy stops execution before any NFE.
