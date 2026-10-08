#!/usr/bin/env bash
# Complete the missing paired 200-epoch P2+ADown seeds and benchmark frozen models.
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_dir"
python_bin=/home/b520/anaconda3/envs/YHP/bin/python
output_dir="$project_dir/runs/ablation_200_v1/p2_adown"

wait_for_idle_gpu() {
    local active_pids
    while true; do
        active_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
        if [[ -z "$active_pids" ]]; then
            return
        fi
        sleep 60
    done
}

printf '%s Waiting for an idle GPU before P2+ADown seeds 0 and 2.\n' "$(date --iso-8601=seconds)"
wait_for_idle_gpu
printf '%s Starting the frozen 200-epoch protocol.\n' "$(date --iso-8601=seconds)"
"$python_bin" -u reproduction/ablation/run_final_candidate.py --models p2_adown --seeds 0 2

printf '%s Waiting for an idle GPU before the same-device speed benchmark.\n' "$(date --iso-8601=seconds)"
wait_for_idle_gpu
"$python_bin" -u reproduction/ablation/benchmark_frozen.py \
    --weights "YOLO11n=runs/final_baseline_yolo11n_200/train_seed1/weights/best.pt" \
    --weights "P2=runs/ablation_200_v1/p2/train_seed1/weights/best.pt" \
    --weights "P2_MSEF=runs/ablation_200_v1/p2_msef_paper/train_seed1/weights/best.pt" \
    --weights "P2_ADown=runs/ablation_200_v1/p2_adown/train_seed1/weights/best.pt" \
    --weights "P2_MSEF_ADown=runs/final_candidate_p2_msef_adown_s0_200/train_seed1/weights/best.pt" \
    --output "$output_dir/efficiency_200_seed1.csv"
printf '%s Training, frozen evaluation, and speed benchmark completed.\n' "$(date --iso-8601=seconds)"
