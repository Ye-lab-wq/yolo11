# Motion-blur training 2x2 — seed 0

Four frozen 60-epoch seed-0 checkpoints, selected only by clean-validation mAP50:95, were evaluated once on the 625-image/685-instance public test split and deterministic light/moderate/strong motion-blur derivatives. No test result was used to select a checkpoint.

## Primary COCO AP50:95

| Model | Clean/% | Light/% | Moderate/% | Strong/% | Blur mean/% |
|---|---:|---:|---:|---:|---:|
| P2_standard | 54.45 | 53.42 | 48.66 | 42.52 | 48.20 |
| P2_blur_train | 53.59 | 53.35 | 51.53 | 48.10 | 50.99 |
| P2_MSEF_standard | 55.49 | 54.84 | 50.21 | 42.73 | 49.26 |
| P2_MSEF_blur_train | 54.38 | 54.58 | 53.26 | 49.67 | 52.50 |

## Factorial AP50:95 effects

Values are percentage-point differences. Interaction is the MSEF effect after blur training minus the MSEF effect under standard training.

| Condition | Blur-training effect on P2 | Blur-training effect on P2+MSEF | MSEF effect, standard | MSEF effect, blur-trained | Interaction |
|---|---:|---:|---:|---:|---:|
| clean | -0.86 | -1.11 | +1.03 | +0.79 | -0.25 |
| light | -0.07 | -0.26 | +1.42 | +1.23 | -0.19 |
| moderate | +2.88 | +3.05 | +1.55 | +1.73 | +0.17 |
| strong | +5.58 | +6.93 | +0.22 | +1.57 | +1.35 |

## Clean-relative degradation

A less-negative change means better corruption robustness relative to the same model's clean result.

| Model | Light/pp | Moderate/pp | Strong/pp |
|---|---:|---:|---:|
| P2_standard | -1.03 | -5.80 | -11.94 |
| P2_blur_train | -0.24 | -2.06 | -5.49 |
| P2_MSEF_standard | -0.65 | -5.28 | -12.75 |
| P2_MSEF_blur_train | +0.20 | -1.12 | -4.71 |

## AP50 and AP75 endpoints

| Model | Condition | AP50/% | AP75/% | Tiny AP/% | Small AP/% |
|---|---|---:|---:|---:|---:|
| P2_standard | clean | 94.02 | 57.95 | 18.88 | 38.43 |
| P2_standard | light | 93.58 | 56.17 | 17.02 | 37.80 |
| P2_standard | moderate | 88.92 | 48.11 | 15.42 | 29.45 |
| P2_standard | strong | 82.17 | 38.19 | 12.71 | 21.99 |
| P2_blur_train | clean | 93.82 | 54.30 | 17.88 | 39.60 |
| P2_blur_train | light | 93.22 | 53.72 | 13.63 | 39.30 |
| P2_blur_train | moderate | 92.15 | 52.01 | 17.04 | 36.94 |
| P2_blur_train | strong | 89.75 | 47.55 | 15.00 | 32.90 |
| P2_MSEF_standard | clean | 94.20 | 59.86 | 21.20 | 38.13 |
| P2_MSEF_standard | light | 93.41 | 58.96 | 20.26 | 36.02 |
| P2_MSEF_standard | moderate | 89.31 | 52.98 | 16.83 | 30.23 |
| P2_MSEF_standard | strong | 80.98 | 40.05 | 19.83 | 21.15 |
| P2_MSEF_blur_train | clean | 94.02 | 59.32 | 15.32 | 39.84 |
| P2_MSEF_blur_train | light | 93.98 | 57.66 | 16.57 | 38.98 |
| P2_MSEF_blur_train | moderate | 93.02 | 55.16 | 30.29 | 36.10 |
| P2_MSEF_blur_train | strong | 90.96 | 49.28 | 22.48 | 30.25 |

## Interpretation

- Dynamic-blur training is the dominant robustness intervention: mean blurred AP50:95 rises from 48.20% to 50.99% for P2 (+2.80 pp), and from 49.26% to 52.50% for P2+MSEF (+3.24 pp).
- The cost is small on clean test images: -0.86 pp for P2 and -1.11 pp for P2+MSEF.
- Under identical blur training, MSEF adds +1.23, +1.73, and +1.57 pp at light/moderate/strong blur.
- The architecture-by-training interaction is small at light blur and becomes positive at moderate/strong blur; the strong-blur interaction is +1.35 pp. This is suggestive of complementarity, not definitive proof from one seed.
- P2+MSEF with blur training nearly preserves clean AP versus standard P2 (54.38% versus 54.45%, -0.08 pp) while improving strong-blur AP by +7.15 pp.

## Scope

This is a synthetic corruption benchmark derived from the public test images, not an independent real-blur dataset. Because the public test has now been inspected, further method or hyperparameter choices should be made on a separately generated validation-blur benchmark; the public test should not be reused for iterative tuning.
