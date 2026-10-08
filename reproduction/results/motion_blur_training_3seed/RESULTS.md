# Motion-blur training 2x2 — three-seed public-test audit

All checkpoints are 60-epoch from-scratch runs selected only by clean-validation mAP50:95. Seed 0 results are reused from the frozen identical-protocol evaluation; seeds 1 and 2 were evaluated on the same 625-image/685-instance public test and deterministic derivatives.

## AP50:95 by seed

| Seed | Model | Clean/% | Light/% | Moderate/% | Strong/% | Blur mean/% |
|---:|---|---:|---:|---:|---:|---:|
| 0 | P2_standard | 54.45 | 53.42 | 48.66 | 42.52 | 48.20 |
| 0 | P2_blur_train | 53.59 | 53.35 | 51.53 | 48.10 | 50.99 |
| 0 | P2_MSEF_standard | 55.49 | 54.84 | 50.21 | 42.73 | 49.26 |
| 0 | P2_MSEF_blur_train | 54.38 | 54.58 | 53.26 | 49.67 | 52.50 |
| 1 | P2_standard | 55.21 | 54.68 | 50.74 | 44.95 | 50.12 |
| 1 | P2_blur_train | 53.31 | 53.41 | 52.51 | 49.56 | 51.83 |
| 1 | P2_MSEF_standard | 54.87 | 53.72 | 49.69 | 42.82 | 48.74 |
| 1 | P2_MSEF_blur_train | 53.61 | 54.50 | 53.46 | 49.72 | 52.56 |
| 2 | P2_standard | 54.93 | 53.89 | 49.75 | 43.66 | 49.10 |
| 2 | P2_blur_train | 54.54 | 54.26 | 52.28 | 48.30 | 51.61 |
| 2 | P2_MSEF_standard | 54.60 | 53.87 | 49.12 | 41.85 | 48.28 |
| 2 | P2_MSEF_blur_train | 53.87 | 54.03 | 52.52 | 49.08 | 51.88 |

## AP50:95 mean ± sample SD

| Model | Clean/% | Light/% | Moderate/% | Strong/% |
|---|---:|---:|---:|---:|
| P2_standard | 54.87 ± 0.38 | 54.00 ± 0.64 | 49.72 ± 1.04 | 43.71 ± 1.22 |
| P2_blur_train | 53.81 ± 0.64 | 53.68 ± 0.51 | 52.11 ± 0.51 | 48.65 ± 0.79 |
| P2_MSEF_standard | 54.99 ± 0.45 | 54.15 ± 0.61 | 49.67 ± 0.54 | 42.47 ± 0.54 |
| P2_MSEF_blur_train | 53.95 ± 0.39 | 54.37 ± 0.30 | 53.08 ± 0.49 | 49.49 ± 0.35 |

## AP50:95 factorial effects by seed

Interaction = MSEF effect after blur training minus MSEF effect under standard training. Values are percentage points.

| Seed | Condition | Blur effect on P2 | Blur effect on P2+MSEF | MSEF effect, standard | MSEF effect, blur-trained | Interaction |
|---:|---|---:|---:|---:|---:|---:|
| 0 | clean | -0.86 | -1.11 | +1.03 | +0.79 | -0.25 |
| 0 | light | -0.07 | -0.26 | +1.42 | +1.23 | -0.19 |
| 0 | moderate | +2.88 | +3.05 | +1.55 | +1.73 | +0.17 |
| 0 | strong | +5.58 | +6.93 | +0.22 | +1.57 | +1.35 |
| 1 | clean | -1.91 | -1.26 | -0.34 | +0.31 | +0.65 |
| 1 | light | -1.27 | +0.78 | -0.96 | +1.09 | +2.06 |
| 1 | moderate | +1.77 | +3.77 | -1.05 | +0.95 | +2.00 |
| 1 | strong | +4.62 | +6.90 | -2.12 | +0.16 | +2.28 |
| 2 | clean | -0.40 | -0.74 | -0.33 | -0.67 | -0.34 |
| 2 | light | +0.37 | +0.16 | -0.02 | -0.23 | -0.21 |
| 2 | moderate | +2.53 | +3.40 | -0.63 | +0.24 | +0.87 |
| 2 | strong | +4.64 | +7.23 | -1.81 | +0.78 | +2.60 |

## Mean AP50:95 effects

| Condition | Blur effect on P2 | Blur effect on P2+MSEF | MSEF effect, standard | MSEF effect, blur-trained | Interaction |
|---|---:|---:|---:|---:|---:|
| clean | -1.06 ± 0.77 | -1.03 ± 0.27 | +0.12 ± 0.79 | +0.14 ± 0.74 | +0.02 ± 0.55 |
| light | -0.32 ± 0.85 | +0.23 ± 0.52 | +0.15 ± 1.20 | +0.70 ± 0.80 | +0.55 ± 1.30 |
| moderate | +2.39 ± 0.57 | +3.40 ± 0.36 | -0.04 ± 1.40 | +0.97 ± 0.74 | +1.01 ± 0.92 |
| strong | +4.94 ± 0.55 | +7.02 ± 0.19 | -1.24 ± 1.27 | +0.84 ± 0.71 | +2.08 ± 0.65 |

No p-value is reported because n=3 seeds is too small for reliable distributional inference. Report seed-wise effects and dispersion. This synthetic public-test corruption audit does not replace a separately collected real-motion-blur test set, and it must not be used for further model selection.
