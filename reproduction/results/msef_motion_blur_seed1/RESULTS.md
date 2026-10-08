# MSEFPaper motion-blur robustness audit

Frozen seed-1 P2 and P2+MSEFPaper checkpoints were evaluated on the original public test split and its deterministic motion-blur derivatives. No retraining was performed.

| Condition | P2 AP50:95/% | P2+MSEF AP50:95/% | MSEF-P2/pp | P2 AP75/% | P2+MSEF AP75/% | MSEF-P2/pp |
|---|---:|---:|---:|---:|---:|---:|
| clean | 56.35 | 56.54 | +0.19 | 58.44 | 61.38 | +2.94 |
| light | 55.43 | 54.97 | -0.46 | 58.64 | 58.79 | +0.16 |
| moderate | 50.32 | 49.68 | -0.64 | 51.03 | 51.41 | +0.38 |
| strong | 41.98 | 42.11 | +0.13 | 38.71 | 41.30 | +2.59 |

## Clean-relative degradation

Positive robustness advantage means MSEF loses fewer points than P2 relative to each model's own clean result.

| Condition | P2 AP change/pp | MSEF AP change/pp | MSEF robustness advantage/pp | P2 AP75 change/pp | MSEF AP75 change/pp | MSEF AP75 robustness advantage/pp |
|---|---:|---:|---:|---:|---:|---:|
| light | -0.91 | -1.56 | -0.65 | +0.20 | -2.59 | -2.78 |
| moderate | -6.03 | -6.86 | -0.83 | -7.41 | -9.97 | -2.56 |
| strong | -14.36 | -14.43 | -0.06 | -19.73 | -20.08 | -0.35 |

Interpretation: architecture-specific blur robustness requires a growing or consistently positive MSEF-minus-P2 advantage as blur severity increases. Absolute degradation alone only establishes that blur is difficult. This synthetic benchmark does not replace real blur labels.
