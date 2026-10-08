# Figure notes

## Same-protocol results

Single-run seed-1 COCO-style results on 641 images / 701 instances:

| Model | AP50:95/% | AP50/% | AP75/% | AP 16-32 px/% | AP >=96 px/% | Params/M | GFLOPs |
|---|---:|---:|---:|---:|---:|---:|---:|
| YOLO11n | 56.39 | 93.61 | 60.53 | 36.61 | 64.00 | 2.590 | 6.4 |
| +P2 | 56.82 | 95.70 | 59.56 | 39.02 | 62.38 | 2.904 | 10.8 |
| +P2+MSEF | 56.95 | 94.79 | 62.37 | 38.75 | 63.57 | 2.920 | 11.7 |
| +P2+ADown | 57.86 | 95.07 | 64.17 | 40.96 | 64.58 | 2.278 | 9.5 |
| +P2+MSEF+ADown | 57.52 | 94.38 | 62.69 | 38.13 | 64.33 | 2.295 | 10.3 |

## Mechanism assessment

- **P2:** AP50 changes by +2.10 points and 16-32 px AP by +2.41 points, while AP75 changes by -0.97. This supports improved small-target/lenient-IoU detection, not uniformly better localization.
- **MSEF:** relative to P2, AP75 changes by +2.81 points, but AP50 by -0.91 and 16-32 px AP by -0.28. Its clearest observed effect is stricter box localization, not a broad small-target gain.
- **ADown without MSEF:** relative to P2, parameters change by -21.5% and GFLOPs by -12.0%, while AP50:95 changes by +1.04, AP75 by +4.61, and 16-32 px AP by +1.94.
- **ADown with MSEF:** relative to P2+MSEF, AP50:95 changes by +0.57, AP75 by +0.33, and 16-32 px AP by -0.62. The difference between the two ADown effects is the MSEF x ADown interaction; non-additivity means the modules must not be credited independently from only the full model.
- **Combination check:** adding MSEF to P2+ADown changes AP50:95 by -0.34, AP75 by -1.48, and 16-32 px AP by -2.84, while increasing GFLOPs by +8.4%. Under this protocol P2+ADown is the stronger compact candidate; the full combination is not justified as an additive improvement.

## Large-target threshold audit

The user-flagged `pic_1032` target is 457 x 600.5 px, occupies 67.0% of the image area, and lies at target-size percentile 98.43. P2+ADown produces a geometrically correct candidate at IoU=0.727, confidence=0.058; the full model produces IoU=0.728, confidence=0.013. Both survive low-floor NMS but fall below the 0.25 decision threshold. The separate diagnostic figure shows these candidates as dotted boxes.

This is not evidence of a systematic large-target collapse: among 120 targets with equivalent side >=300 px, the median final-minus-parent confidence change is +0.034; 5 parent detections cross below 0.25 and 4 cross above it. Large-target AP also changes by +0.76 points. The correct interpretation is an isolated score-calibration failure on an extreme-scale, unusual grayscale sample, not proof that ADown generally cannot detect large objects.

## Figure construction

The paper-style qualitative grid uses deterministic geometry-based selection rather than manual image picking: one P2 gain, one MSEF gain, one ADown gain, one combined-model gain, and one ADown regression. Ground truth is shown only in its own column with a green solid box; each model column uses red solid prediction boxes at confidence >=0.25. Low-score candidates are excluded from the main qualitative grid and shown only in the separately labelled diagnostic figure. Each row uses an identical native-pixel crop across models, with no brightness, contrast, denoising, or selective adjustment.

The quantitative figure reports AP-versus-IoU, size-stratified AP, direct-parent changes, and the parameter/AP75 trade-off. The <16 px subset is exploratory because it contains only nine instances. No uncertainty interval is shown because these figures use the requested single selected seed. All plotted values use the COCO-style evaluator consistently.

Alt text: Four-panel quantitative comparison and five-row qualitative crop grid compare seed-1 YOLO11n, YOLO11n+P2, YOLO11n+P2+MSEF, YOLO11n+P2+ADown, and the full combination. The main grid uses conventional separate solid ground-truth and prediction boxes. The figures expose both direct module effects and the non-additive MSEF-ADown interaction, retain an ADown regression, and separately diagnose the low-score but well-localized candidate for the user-flagged very large target.
