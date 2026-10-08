# Clean-v2 200-epoch three-seed result index

All five models were trained from scratch for 200 epochs at 640 pixels with the frozen AdamW protocol. Each `best.pt` was selected by validation mAP50–95. This index uses the same COCO-style evaluator on the 625-image / 685-instance public test of `DetectDataset_clean_v2`; the 16 self-video frames are reported separately in each run's `test_results/self_test.json` or `seed*/self_test.json`.

The public test has already been inspected repeatedly during method development. These results describe the existing benchmark; they are not a new untouched final test. An external zero-shot test is specified in `external_benchmarks/uav_eagle/PROTOCOL.md` and is pending evaluation.

| Model | AP50–95/% | AP50/% | AP75/% | AP 16–32 px/% |
|---|---:|---:|---:|---:|
| YOLO11n | 55.77 ± 0.26 | 94.41 ± 1.03 | 57.75 ± 1.90 | 38.01 ± 1.49 |
| +P2 | 56.87 ± 1.09 | 95.16 ± 0.22 | 60.97 ± 2.38 | 38.95 ± 0.96 |
| +P2+MSEFPaper | 56.77 ± 0.48 | 94.79 ± 0.29 | 60.90 ± 1.57 | 39.79 ± 1.42 |
| +P2+ADown | **57.20 ± 0.19** | 94.95 ± 0.18 | **62.84 ± 0.77** | **39.84 ± 1.08** |
| +P2+MSEFPaper+ADown | 56.86 ± 0.29 | 94.96 ± 0.70 | 61.44 ± 0.87 | 39.38 ± 1.78 |

The dispersions are sample SD across seeds 0, 1, 2, not confidence intervals. Direct paired AP50–95 differences, in seed order, are:

- P2+ADown minus YOLO11n: `+1.37, +1.42, +1.53` points (mean `+1.44 ± 0.08`).
- P2+ADown minus P2: `+1.10, +1.02, −1.13` points (mean `+0.33 ± 1.26`). The ADown effect over the P2 parent is not stable across these seeds.
- Full P2+MSEF+ADown minus P2+ADown: `−0.60, −0.18, −0.25` points (mean `−0.34 ± 0.22`). Adding MSEF to this parent does not improve AP50–95 in any of the three seeds.

The selected seed-1 checkpoints were measured on the same RTX 3090 using fused PyTorch FP32 forward passes at 640×640. File reading, preprocessing and NMS are excluded. Full raw measurements are in `runs/ablation_200_v1/p2_adown/efficiency_200_seed1.csv` and its adjacent JSON.

| Model | Params/M | GFLOPs | Batch-1 latency/ms | Batch-1 FPS | Batch-16 throughput/FPS |
|---|---:|---:|---:|---:|---:|
| YOLO11n | 2.590 | 6.44 | 5.76 | 173.5 | 1130.3 |
| P2 | 2.904 | 10.83 | 6.84 | 146.3 | 654.2 |
| P2+MSEF | 2.920 | 11.69 | 6.95 | 143.9 | 532.5 |
| P2+ADown | 2.278 | 9.47 | 7.37 | 135.7 | 608.5 |
| P2+MSEF+ADown | 2.295 | 10.33 | 7.56 | 132.2 | 501.6 |

ADown lowers parameters and GFLOPs relative to P2 but does **not** improve measured latency or throughput on this GPU. Relative to YOLO11n, P2+ADown has fewer parameters but more operations and lower FPS. No edge-device efficiency claim follows from these measurements.

## Where the source files are

- `runs/final_baseline_yolo11n_200/`: baseline seeds 1 and 2; seed 0 training is `runs/clean_v2_scratch_pair/yolo11n_seed0/`. All three baseline public-test JSON files are under `runs/final_baseline_yolo11n_200/test_results/`.
- `runs/ablation_200_v1/p2/`, `p2_msef_paper/`, `p2_adown/`: three-seed ablation runs, frozen checkpoint metadata, public/combined/self test JSON, args and epoch CSV. P2+ADown seeds 0 and 2 completed on 2026-10-07.
- `runs/final_candidate_p2_msef_adown_s0_200/`: full-combination three-seed run.
- `reproduction/results/p2_msef_adown_visual_evidence_seed1/`: deterministic qualitative comparisons and source data for the seed-1 structure experiment.
- `reproduction/results/motion_blur_visual_analysis_3seed/`: synthetic-blur qualitative and quantitative figures; this is a separate 60-epoch experiment and its numbers are not mixed with the 200-epoch table.

The `.pt` weights, source images, generated training batches and caches remain local and are excluded from Git. To rerun inference elsewhere, the identical validation-selected `best.pt` files must be transferred separately.
