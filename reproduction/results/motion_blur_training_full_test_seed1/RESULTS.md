# Seed-1 P2 blur-training audit on the full original test

The two frozen 60-epoch, from-scratch seed-1 P2 checkpoints were evaluated on `DetectDataset_clean_v2/data.yaml`'s unmodified `test` split: 641 images and 701 annotated drone instances (625 original-source test images plus 16 self-video test frames). No synthetic motion blur was applied at test time. Both checkpoints use the same `p2_nearest.yaml` architecture, input size 640, confidence floor 0.001, NMS IoU 0.70, and maximum 300 detections. The only intended training difference was online motion blur (`p=0.30`, kernel 3/5/7 px at 640) for `P2_blur_train`; model selection used clean validation. Evaluation ran on CPU with the retained YHP environment and batch 8.

Source data: [`results.json`](results.json). Evaluation command used `reproduction/ablation/evaluate_frozen.py --split combined_test` with these weights:

- `P2_standard`: `runs/ablation_interaction_confirm_v1/cell_00_s1/weights/best.pt`
- `P2_blur_train`: `runs/ablation_motion_blur_replicates_v1/seed1/p2_motion_blur_s1/weights/best.pt`

## Ultralytics detector metrics (full 641-image test)

| Metric / % | Standard P2 | Blur-trained P2 | Blur minus standard / pp |
|---|---:|---:|---:|
| Precision | 92.11 | 92.94 | +0.83 |
| Recall | 91.58 | 92.00 | +0.42 |
| mAP50 | 95.19 | 94.48 | -0.72 |
| mAP75 | 61.02 | 56.69 | -4.33 |
| mAP50:95 | 56.94 | 54.85 | -2.09 |

## Separate COCO-style evaluator (same full 641-image test)

This is the same evaluator used for the earlier 625-image motion-blur comparison. Its absolute AP values differ from Ultralytics' validator; compare within a block, never mix values across the two evaluators.

| Metric / % | Standard P2 | Blur-trained P2 | Blur minus standard / pp |
|---|---:|---:|---:|
| AP50 | 94.03 | 93.55 | -0.48 |
| AP75 | 59.07 | 56.75 | -2.31 |
| AP50:95 | 55.53 | 53.63 | -1.90 |
| AP tiny, <16 px | 17.45 | 17.98 | +0.53 |
| AP small, 16–32 px | 37.70 | 39.37 | +1.67 |
| AP medium, 32–96 px | 55.02 | 52.50 | -2.52 |
| AP large, ≥96 px | 61.03 | 59.33 | -1.70 |

The full-test result agrees in direction with the 625-image clean subset: same seed-1 COCO-style AP50:95 was 55.21 vs 53.31 there. The extra 16 self-video frames were included in the present full-test score but are not separately evaluated here. Point Precision/Recall increasing does not imply AP increasing: AP summarizes detection quality across confidence thresholds (and AP50:95 also across IoU thresholds). These are single-seed descriptive results, not a claim of statistical significance or real-motion-blur generalization.

## Relationship to the four-condition corruption audit

The four-condition audit used the **same 625 original-source test images four times**: once unchanged (`clean`), then as deterministic synthetic light, moderate and strong motion-blur copies (3/7/11 px line kernels at 640, angles assigned by source filename). All four conditions use the same unchanged labels and both frozen models. They are paired input conditions, not four disjoint image samples or four separately collected datasets. The 16 self-video test frames were not used in those synthetic-blur conditions. The present 641-image full-test evaluation used only the unmodified images.
