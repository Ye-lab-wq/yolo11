# Figure notes

## Files

- `qualitative_p2_adown_seed1.{png,pdf}`: identical-crop visual comparison with two P2 gains, two ADown-over-P2 gains, and one disclosed regression.
- `quantitative_p2_adown_seed1.{png,pdf}`: AP-versus-IoU, target-size AP, and localization/complexity trade-off.
- `figure_source_data.csv`: plotted values.
- `manifest.json`: dataset size, inference settings, checkpoint paths/SHA256, objective example-selection rules, selected boxes, and full metrics.
- `raw_predictions.json`: raw frozen-model predictions used by the evaluator.

## Same-protocol results

All values below are single-run seed-1 COCO-style results on the same 641-image/701-instance combined test set.

| Model | AP50:95/% | AP50/% | AP75/% | AP 16-32 px/% | Parameters/M | GFLOPs |
|---|---:|---:|---:|---:|---:|---:|
| YOLO11n | 56.39 | 93.61 | 60.53 | 36.61 | 2.590 | 6.3 |
| +P2 | 56.82 | 95.70 | 59.56 | 39.02 | 2.904 | 10.8 |
| +P2+ADown | 57.86 | 95.07 | 64.17 | 40.96 | 2.278 | 9.5 |

## Interpretation against the intended mechanisms

- **P2 mostly achieves its small-target purpose.** Relative to YOLO11n, P2 raises AP50 by 2.10 points and 16-32 px AP by 2.41 points. The first two qualitative rows show a missed target becoming a correct box. It does not improve every localization regime: AP75 changes by -0.97 points, parameters rise by 12.1%, and GFLOPs by 71.4%.
- **ADown achieves the intended P2-stage efficiency/localization trade-off, but not a universal gain.** Relative to P2, the combined model reduces parameters by 21.5% and GFLOPs by 12.0%, while AP75 rises by 4.61 points and AP50:95 by 1.04 points. The ADown rows show recovered/tighter boxes, while the retained counterexample shows that it can also suppress a valid detection.
- **The combined model is parameter-light, not compute-light, relative to baseline.** It has 12.0% fewer parameters than YOLO11n and gains 1.47 AP50:95 points and 3.64 AP75 points, but still requires 50.8% more GFLOPs.
- The <16 px subset contains only nine instances. Its lower combined-model AP is reported, but this subset is too small to support a stable claim about truly tiny targets.

These values use the COCO-style evaluator consistently. They should not be numerically mixed with Ultralytics summary metrics without explicitly naming the evaluator.

The qualitative grid uses deterministic, geometry-based example selection rather than manual image picking. Yellow dashed boxes are ground truth; solid colored boxes are predictions at confidence >=0.25. Each row uses an identical native-pixel crop across models and is magnified only for display. The counterexample is retained to avoid presenting improvements as universal.

The quantitative figure reports single-run seed-1 COCO-style metrics on 641 images/701 instances. Panel (a) shows AP as the IoU matching threshold becomes stricter; panel (b) reports size-stratified AP50:95, with the tiny subset explicitly marked exploratory because it contains only nine instances; panel (c) shows the parameter/AP75 trade-off, with marker area encoding GFLOPs. No uncertainty interval is shown because the requested figure uses one selected seed.

Alt text: Three-panel quantitative comparison and five-row qualitative crop grid compare YOLO11n, YOLO11n+P2, and YOLO11n+P2+ADown. P2 improves small-target AP, while ADown reduces parameter count relative to P2 and raises strict-IoU localization. Most selected crops show tighter or newly correct boxes, alongside one disclosed regression.
