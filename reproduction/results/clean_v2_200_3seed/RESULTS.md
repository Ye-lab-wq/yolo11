# Clean-v2 200-epoch three-seed result index

All five models were trained from scratch for 200 epochs at 640 pixels with the frozen AdamW protocol. Each `best.pt` was selected by validation mAP50–95. This index uses the same COCO-style evaluator on the 625-image / 685-instance original-source test portion of the **one** rebuilt `DetectDataset_clean_v2` dataset (code name `public_test`). The 16 added self-video frames are reported separately in each run's `test_results/self_test.json` or `seed*/self_test.json`; both portions together are the 641-image `combined_test`, not a second dataset.

The public test has already been inspected repeatedly during method development. These results describe the existing benchmark; they are not a new untouched final test. The predeclared external zero-shot test has since been completed separately: [`UAV-Eagle results`](../uav_eagle_external_200_3seed/RESULTS.md).

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

The following **separate Ultralytics validator** outputs provide the requested
point Precision and Recall on the same 625 images. They must not be combined
numerically with the COCO-style AP table above: the evaluators implement
different aggregation. Entries are three-seed mean ± sample SD in percent;
each seed's raw values are in the listed `runs/.../test_results/public_test.json`
files below.

| Model | Precision/% | Recall/% | Ultralytics mAP50/% | Ultralytics mAP50–95/% |
|---|---:|---:|---:|---:|
| YOLO11n | 94.22 ± 1.39 | 93.55 ± 0.41 | 96.05 ± 0.87 | 57.77 ± 0.33 |
| P2 | 94.27 ± 0.14 | 93.66 ± 0.36 | 96.28 ± 0.24 | 58.94 ± 0.96 |
| P2+MSEFPaper | 94.65 ± 0.80 | 93.80 ± 0.49 | 96.26 ± 0.49 | 58.49 ± 0.38 |
| P2+ADown | 94.40 ± 0.50 | 93.56 ± 0.29 | 96.13 ± 0.36 | 59.18 ± 0.27 |
| P2+MSEFPaper+ADown | 94.34 ± 0.73 | 93.92 ± 1.26 | 96.27 ± 0.74 | 58.98 ± 0.40 |

P2+ADown's point Precision/Recall and mAP50 are close to baseline; the
repeated directional gain is in stricter IoU outcomes, not a general
Precision/Recall improvement.

## Added self-video frames: report separately

The 16 `self_test` images contain 16 instances from the **same collected video**
as the 36 added training frames. They are unseen frames, not unseen-video or
independent-source generalization. The COCO-style AP50 is 100% for every
model/seed, so this subset is too small and saturated for model selection.
AP50–95 below is given only for transparent accounting (percent):

| Model | Seed 0 | Seed 1 | Seed 2 |
|---|---:|---:|---:|
| YOLO11n | 74.60 | 73.03 | 71.32 |
| P2 | 74.28 | 74.71 | 73.57 |
| P2+MSEFPaper | 72.46 | 74.28 | 70.70 |
| P2+ADown | 69.89 | 80.36 | 72.97 |
| P2+MSEFPaper+ADown | 74.13 | 72.34 | 72.06 |

These results are stored in each run's `test_results/self_test.json` or
`test_results/seed*/self_test.json`. They are not pooled into the 625-image
primary table or used to choose a checkpoint.

The selected seed-1 checkpoints were measured on the same RTX 3090 using fused PyTorch FP32 forward passes at 640×640. File reading, preprocessing and NMS are excluded. Full raw measurements are in `runs/ablation_200_v1/p2_adown/efficiency_200_seed1.csv` and its adjacent JSON.

| Model | Params/M | GFLOPs | Checkpoint/MiB | Batch-1 latency/ms | Batch-1 FPS | Batch-16 throughput/FPS | Batch-16 peak allocated/MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| YOLO11n | 2.590 | 6.44 | 5.24 | 5.76 | 173.5 | 1130.3 | 377.0 |
| P2 | 2.904 | 10.83 | 6.03 | 6.84 | 146.3 | 654.2 | 911.8 |
| P2+MSEF | 2.920 | 11.69 | 6.07 | 6.95 | 143.9 | 532.5 | 911.8 |
| P2+ADown | 2.278 | 9.47 | 4.85 | 7.37 | 135.7 | 608.5 | 912.8 |
| P2+MSEF+ADown | 2.295 | 10.33 | 4.89 | 7.56 | 132.2 | 501.6 | 912.3 |

ADown lowers parameters and GFLOPs relative to P2 but does **not** improve measured latency or throughput on this GPU. Relative to YOLO11n, P2+ADown has fewer parameters but more operations and lower FPS. No edge-device efficiency claim follows from these measurements.
Checkpoint size is the on-disk seed-1 `best.pt`; peak allocated GPU memory is
PyTorch's batch-16 forward-pass allocation, not total device occupancy or a
deployed application's peak VRAM.

## Where the source files are

- `runs/final_baseline_yolo11n_200/`: baseline seeds 1 and 2; seed 0 training is `runs/clean_v2_scratch_pair/yolo11n_seed0/`. All three baseline public-test JSON files are under `runs/final_baseline_yolo11n_200/test_results/`.
- `runs/ablation_200_v1/p2/`, `p2_msef_paper/`, `p2_adown/`: three-seed ablation runs, frozen checkpoint metadata, public/combined/self test JSON, args and epoch CSV. P2+ADown seeds 0 and 2 completed on 2026-10-07.
- `runs/final_candidate_p2_msef_adown_s0_200/`: full-combination three-seed run.
- `reproduction/results/p2_msef_adown_visual_evidence_seed1/`: deterministic qualitative comparisons and source data for the seed-1 structure experiment.
- `reproduction/results/motion_blur_visual_analysis_3seed/`: synthetic-blur qualitative and quantitative figures; this is a separate 60-epoch experiment and its numbers are not mixed with the 200-epoch table.

The `.pt` weights, source images, generated training batches and caches remain local and are excluded from Git. To rerun inference elsewhere, the identical validation-selected `best.pt` files must be transferred separately.

Audit note: all 15 training curves contain 200 epochs and all 45 internal
public/self/combined test JSONs are present. In 14 runs with explicit
`frozen_model*.json` records, each stored best validation mAP50–95 matches
the maximum of its `results.csv`; the remaining baseline seed 0 has the
standard Ultralytics `best.pt` and a complete curve but predates explicit
freeze metadata. The historical metadata field `best_epoch_zero_based` is
misnamed: its value equals the **one-based** `epoch` column in `results.csv`.
This is an indexing-label error, not an alternate model-selection rule; the
frozen records were not rewritten.
