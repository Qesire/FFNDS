#!/bin/bash
set -euo pipefail
digest=$1
cd /public/home/scc/pb23061276/projects/wan-quant/work/ffnds_strict_reproduction_20261116_336b7e1/icc_cal_finalize_v3
sha256sum -c SHA256SUMS
printf '%s  scope.json\n' "$digest" | sha256sum -c -
test "$(hostname)" = tradmin-02
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1
printf '%s  %s\n' e51ea937396ddee310ad36ade10106fcc941efbf35b462666e59d772f4e294d0 /public/home/scc/pb23061276/projects/wan-quant/work/ffnds_strict_reproduction_20261116_336b7e1/icc_cal11_first_v1/scope.json | sha256sum -c -
grep -q '"exit_code":0,"slurm_job_id":"168767"' /public/home/scc/pb23061276/projects/wan-quant/work/ffnds_strict_reproduction_20261116_336b7e1/icc_cal11_first_v1/receipt/EXIT.json
test -s /public/home/scc/pb23061276/projects/wan-quant/work/ffnds_strict_reproduction_20261116_336b7e1/icc_cal11_first_v1/output/SUMMARY.json
printf '%s  %s\n' ed9e1436cb5174902c7f9dc3b4e6302e4baea2ba5b516f8aeec3f690f1c47c4c /public/home/scc/pb23061276/projects/wan-quant/work/ffnds_strict_reproduction_20261116_336b7e1/icc_cal11_cal3a_v1/scope.json | sha256sum -c -
grep -q '"exit_code":0,"slurm_job_id":"168855"' /public/home/scc/pb23061276/projects/wan-quant/work/ffnds_strict_reproduction_20261116_336b7e1/icc_cal11_cal3a_v1/receipt/EXIT.json
test -s /public/home/scc/pb23061276/projects/wan-quant/work/ffnds_strict_reproduction_20261116_336b7e1/icc_cal11_cal3a_v1/output/SUMMARY.json
printf '%s  %s\n' 13671cbb7ca208748ad93815c462cde9fc169082624818a672e1c03da2b6c93c /public/home/scc/pb23061276/projects/wan-quant/work/ffnds_strict_reproduction_20261116_336b7e1/icc_cal11_mains_a_v2/scope.json | sha256sum -c -
grep -q '"exit_code":0,"slurm_job_id":"168941"' /public/home/scc/pb23061276/projects/wan-quant/work/ffnds_strict_reproduction_20261116_336b7e1/icc_cal11_mains_a_v2/receipt/EXIT.json
test -s /public/home/scc/pb23061276/projects/wan-quant/work/ffnds_strict_reproduction_20261116_336b7e1/icc_cal11_mains_a_v2/output/SUMMARY.json
printf '%s  %s\n' 28730c9e7e91ac4b5a9ff7ed9890c3c556ca40ea2311c697ef31a4b87a43717a /public/home/scc/pb23061276/projects/wan-quant/work/ffnds_strict_reproduction_20261116_336b7e1/icc_cal11_cal3b_v2/scope.json | sha256sum -c -
grep -q '"exit_code":0,"slurm_job_id":"168942"' /public/home/scc/pb23061276/projects/wan-quant/work/ffnds_strict_reproduction_20261116_336b7e1/icc_cal11_cal3b_v2/receipt/EXIT.json
test -s /public/home/scc/pb23061276/projects/wan-quant/work/ffnds_strict_reproduction_20261116_336b7e1/icc_cal11_cal3b_v2/output/SUMMARY.json
printf '%s  %s\n' ea40e4342bca631421ed992b3b45293b8ea8c870850511bc6c921760871077c6 /public/home/scc/pb23061276/projects/wan-quant/work/ffnds_strict_reproduction_20261116_336b7e1/icc_cal11_mains_d_v2/scope.json | sha256sum -c -
grep -q '"exit_code":0,"slurm_job_id":"168943"' /public/home/scc/pb23061276/projects/wan-quant/work/ffnds_strict_reproduction_20261116_336b7e1/icc_cal11_mains_d_v2/receipt/EXIT.json
test -s /public/home/scc/pb23061276/projects/wan-quant/work/ffnds_strict_reproduction_20261116_336b7e1/icc_cal11_mains_d_v2/output/SUMMARY.json
printf '%s  %s\n' ea46a0cf32ea57c995c92c4da1bdad31b9da32aaea16247ddf2007fa68b57868 /public/home/scc/pb23061276/projects/wan-quant/work/ffnds_strict_reproduction_20261116_336b7e1/icc_cal11_mains_b_v2/scope.json | sha256sum -c -
grep -q '"exit_code":0,"slurm_job_id":"168944"' /public/home/scc/pb23061276/projects/wan-quant/work/ffnds_strict_reproduction_20261116_336b7e1/icc_cal11_mains_b_v2/receipt/EXIT.json
test -s /public/home/scc/pb23061276/projects/wan-quant/work/ffnds_strict_reproduction_20261116_336b7e1/icc_cal11_mains_b_v2/output/SUMMARY.json
printf '%s  %s\n' ce38415454ffa90db607b8b403e5c6bc6d0017c429bd54d910d29e1d340f0fcc /public/home/scc/pb23061276/projects/wan-quant/work/ffnds_strict_reproduction_20261116_336b7e1/icc_cal11_cal3c_v2/scope.json | sha256sum -c -
grep -q '"exit_code":0,"slurm_job_id":"168945"' /public/home/scc/pb23061276/projects/wan-quant/work/ffnds_strict_reproduction_20261116_336b7e1/icc_cal11_cal3c_v2/receipt/EXIT.json
test -s /public/home/scc/pb23061276/projects/wan-quant/work/ffnds_strict_reproduction_20261116_336b7e1/icc_cal11_cal3c_v2/output/SUMMARY.json
printf '%s  %s\n' 2b1671f060c5261421e0ea8de6ba7de8b13c03d63e48fbd0417f3d84eab6277b /public/home/scc/pb23061276/projects/wan-quant/work/ffnds_strict_reproduction_20261116_336b7e1/icc_cal11_cd_retry_v1/scope.json | sha256sum -c -
grep -q '"exit_code":0,"slurm_job_id":"171133"' /public/home/scc/pb23061276/projects/wan-quant/work/ffnds_strict_reproduction_20261116_336b7e1/icc_cal11_cd_retry_v1/receipt/EXIT.json
test -s /public/home/scc/pb23061276/projects/wan-quant/work/ffnds_strict_reproduction_20261116_336b7e1/icc_cal11_cd_retry_v1/output/SUMMARY.json
active_jobs=$(squeue -u "$USER" -h -o '%i')
if grep -qx 171133 <<< "$active_jobs"; then
    exit 1
fi
test -f /public/home/scc/pb23061276/projects/wan-quant/work/ffnds_strict_reproduction_20261116_336b7e1/icc_cal11_cd_retry_v1/slurm-171133.err
if grep -qiE 'CANCELLED|DUE TO SIGNAL|TIME LIMIT|error:.*oom' /public/home/scc/pb23061276/projects/wan-quant/work/ffnds_strict_reproduction_20261116_336b7e1/icc_cal11_cd_retry_v1/slurm-171133.err; then
    exit 1
fi
(cd /public/home/scc/pb23061276/projects/wan-quant/work/ffnds_strict_reproduction_20261116_336b7e1/icc_cal11_cd_retry_v1 && sha256sum -c receipt/SHA256SUMS)
mkdir launch
guardian=
trap 'rc=$?; printf "{\"exit_code\":%s,\"end_utc\":\"%s\"}\n" "$rc" "$(date -u +%FT%TZ)" > launch/EXIT.json' EXIT
terminate() {
    local code=$1
    if [[ -n "$guardian" ]]; then
        kill -TERM "$guardian" 2>/dev/null || true
        wait "$guardian" || true
    fi
    exit "$code"
}
trap 'terminate 143' TERM
trap 'terminate 130' INT
printf '{"scope_sha256":"%s","host":"%s","start_utc":"%s"}\n' "$digest" "$(hostname)" "$(date -u +%FT%TZ)" > launch/RUN.json
timeout --kill-after=5s 595s nice -n 10 /public/home/scc/pb23061276/miniconda3/envs/hif4/bin/python3.11 -B icc_calibration_finalize_job.py scope.json "$digest" > launch/runner.log 2>&1 &
guardian=$!
while kill -0 "$guardian" 2>/dev/null; do
    rss=$(ps --ppid "$guardian" -o rss= | awk '{total+=$1} END {print total+0}')
    if (( rss > 4194304 )); then
        printf 'CPU aggregate exceeded 4GiB RSS\n' >&2
        terminate 137
    fi
    sleep 1
done
wait "$guardian"
test "$(du -sb output | cut -f1)" -le 1073741824
sha256sum output/RUN.json output/EXIT.json output/SUMMARY.json output/SCALES.json output/CAL3_PAIRED.json output/VERIFIED_RECOVERY.json > launch/OUTPUT_SHA256SUMS
