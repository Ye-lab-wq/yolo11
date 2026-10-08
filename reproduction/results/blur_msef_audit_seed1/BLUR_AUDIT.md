# Blur-proxy and MSEFPaper audit

This is an exploratory audit, not a blur-label ground truth. Variance of the Laplacian and Tenengrad are content-dependent: low texture, darkness, defocus, compression, and motion blur can overlap. Images are resized to 640x640 before scoring to reduce source-resolution confounding.

## Dataset proxy summary

| Split | Images | LapVar P10 | Median | P90 |
|---|---:|---:|---:|---:|
| train | 2911 | 611.4 | 2590.4 | 9902.0 |
| val | 625 | 956.8 | 2992.3 | 11872.7 |
| public_test | 625 | 706.4 | 3001.7 | 11613.6 |
| self_test | 16 | 8954.3 | 10394.9 | 13310.8 |

## MSEFPaper by rank-based proxy third

| Proxy third | Images / instances | Model | AP50:95/% | AP50/% | AP75/% | AP 16-32/% |
|---|---:|---|---:|---:|---:|---:|
| lower-sharpness third | 214 / 255 | +P2 | 50.69 | 92.88 | 48.84 | 40.49 |
| lower-sharpness third | 214 / 255 | +P2+MSEF | 52.14 | 93.24 | 51.16 | 41.88 |
| middle third | 214 / 226 | +P2 | 58.25 | 95.75 | 63.78 | 40.61 |
| middle third | 214 / 226 | +P2+MSEF | 57.72 | 94.03 | 67.09 | 38.40 |
| higher-sharpness third | 213 / 220 | +P2 | 62.02 | 98.05 | 67.03 | 42.26 |
| higher-sharpness third | 213 / 220 | +P2+MSEF | 61.20 | 97.24 | 69.19 | 31.42 |

Interpretation rule: a blur-targeting mechanism is supported only if its direct-parent gain is stronger and consistent in an independently defined blurred subset. A favorable lower-proxy result here is hypothesis-generating because the bins are proxy-defined and composition (including target size) may differ.

The audit uses frozen seed-1 predictions generated at confidence floor 0.001 and the same COCO-style evaluator as the five-model evidence package.
