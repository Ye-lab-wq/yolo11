#!/usr/bin/env bash
# Run from any directory. The model list and order were frozen in PROTOCOL.md.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

"/home/b520/anaconda3/envs/YHP/bin/python" \
  external_benchmarks/uav_eagle/audit.py

"/home/b520/anaconda3/envs/YHP/bin/python" \
  reproduction/ablation/evaluate_frozen.py \
  --split external_test --device 0 --batch 16 \
  --output reproduction/results/uav_eagle_external_200_3seed/results.json \
  --weights yolo11n_s0=runs/clean_v2_scratch_pair/yolo11n_seed0/weights/best.pt \
  --weights yolo11n_s1=runs/final_baseline_yolo11n_200/train_seed1/weights/best.pt \
  --weights yolo11n_s2=runs/final_baseline_yolo11n_200/train_seed2/weights/best.pt \
  --weights p2_s0=runs/ablation_200_v1/p2/train_seed0/weights/best.pt \
  --weights p2_s1=runs/ablation_200_v1/p2/train_seed1/weights/best.pt \
  --weights p2_s2=runs/ablation_200_v1/p2/train_seed2/weights/best.pt \
  --weights p2_msef_s0=runs/ablation_200_v1/p2_msef_paper/train_seed0/weights/best.pt \
  --weights p2_msef_s1=runs/ablation_200_v1/p2_msef_paper/train_seed1/weights/best.pt \
  --weights p2_msef_s2=runs/ablation_200_v1/p2_msef_paper/train_seed2/weights/best.pt \
  --weights p2_adown_s0=runs/ablation_200_v1/p2_adown/train_seed0/weights/best.pt \
  --weights p2_adown_s1=runs/ablation_200_v1/p2_adown/train_seed1/weights/best.pt \
  --weights p2_adown_s2=runs/ablation_200_v1/p2_adown/train_seed2/weights/best.pt \
  --weights full_s0=runs/final_candidate_p2_msef_adown_s0_200/train_seed0/weights/best.pt \
  --weights full_s1=runs/final_candidate_p2_msef_adown_s0_200/train_seed1/weights/best.pt \
  --weights full_s2=runs/final_candidate_p2_msef_adown_s0_200/train_seed2/weights/best.pt
