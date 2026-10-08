# Controlled MEDA experiments

This directory contains only the new, auditable experiment layer. The author's uploaded model files and historical
checkpoints remain in their original project locations.

- `IMPLEMENTATION_AUDIT.md`: claimed mechanism versus uploaded executable code versus corrected/reference code.
- `PROTOCOL.md`: protocol frozen before the new screen, including advancement thresholds and test-set embargo.
- `generate_models.py`, `models/`: one graph builder and generated one-factor model YAML files.
- `run_screen.py`: resumable seed-0/60-epoch validation-only component screen.
- `analyze_screen.py`: direct-parent deltas and predeclared advancement decisions.
- `run_confirmatory.py`: resumable 200-epoch multi-seed confirmation after an explicit selection JSON is frozen.
- `evaluate_frozen.py`: final Ultralytics plus COCO-style AP/AP75/size-stratified evaluation; not used for screening.
- `benchmark_frozen.py`: same-GPU fused FP32 parameters/GFLOPs/latency/FPS/VRAM benchmark.
- `build_p2_msef_adown_visual_evidence.py`: reproducible seed-1 P2/MSEF/ADown qualitative and quantitative figure builder. Its figures, source table, raw predictions, checkpoint hashes, and selection manifest are under `reproduction/results/p2_msef_adown_visual_evidence_seed1/`.
- `audit_blur_msef.py`: exploratory, rank-based sharpness-proxy audit that reuses the frozen seed-1 predictions instead of rerunning inference. It exports per-image proxy scores, deterministic review images, and P2 versus P2+MSEFPaper COCO metrics by proxy third under `reproduction/results/blur_msef_audit_seed1/`. Low proxy scores are not treated as verified motion-blur labels.
- `run_adown_residual.py`: resumable validation-only 60-epoch screen of the predeclared deep residual ADown candidate. The unchanged seed-0 P2+ADown screen is its parent; no test split is accessed. See `ADOWN_RESIDUAL_PROTOCOL.md`.
- `build_motion_blur_benchmark.py` and `evaluate_motion_blur_benchmark.py`: build a deterministic three-severity motion-blur derivative of the 625-image public test split, then evaluate the frozen seed-1 P2 and P2+MSEFPaper checkpoints without retraining. This is a secondary robustness benchmark, not an independent dataset.
- `run_motion_blur_training.py`: resumable two-cell seed-0/60-epoch screen that applies identical online directional motion blur to P2 and P2+MSEFPaper. See `MOTION_BLUR_TRAINING_PROTOCOL.md`; test access is forbidden during training/model selection.
- `evaluate_motion_blur_training_multiseed.py`: resumable frozen evaluation and seed-wise/mean±SD aggregation for the three-seed P2/MSEF × standard/motion-blur-training factorial.
- `build_motion_blur_visual_analysis.py`: deterministic three-seed qualitative grids for blur gains and clean-image regressions, plus a publication-style quantitative robustness figure and complete per-image source data.

The only supported environment is `/home/b520/anaconda3/envs/YHP`, importing this repository's local
`ultralytics/`. New run artifacts follow the Ultralytics layout under `runs/ablation_screen_v1/` and
`runs/ablation_confirmatory_v1/`.

`INTERACTION_CONFIRMATION_PROTOCOL.md` and `run_interaction_confirmation.py`
define the held-out-seed 2x2 replication of the discovered MSEFPaper x
full-path ADown interaction. Seed 0 is reused as discovery evidence; seeds 1
and 2 are trained from scratch without test-set access.

The final public test and the 16-frame self-video test must remain separate. Validation chooses `best.pt`; test is
accessed only after the final method and analysis rules have been frozen.
