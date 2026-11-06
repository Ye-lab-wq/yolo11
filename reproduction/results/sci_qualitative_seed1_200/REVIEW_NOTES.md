# 200-epoch seed-1 qualitative figure and reviewer audit

## Deliverables and scope

- `seed1_200_qualitative.pdf`, `.svg`, `.png`, `.tiff` are the same figure at manuscript/vector and raster resolutions.
- `manifest.json` records original image IDs and SHA256, GT, identical per-row crops, confidence and IoU for each model, checkpoint records, and screening rules.
- `build_sci_qualitative_seed1_200.py` in `reproduction/ablation/` rebuilds the figure from the previously saved predictions without new inference.
- All three image examples come from the original-source part of the **same** `DetectDataset_clean_v2` full test. The main aggregate result below uses all 641 test images and 701 GT instances, including the 16 self-video frames. No special metric or test-set exclusion is inferred from figure selection.

This is an **exploratory, post-hoc image plate**. It is not a random or independent sample of the test set. The 200-epoch seed-1 models were trained from scratch; each `best.pt` was chosen on validation. Display uses confidence ≥0.25 after NMS at IoU 0.7. Ground-truth and prediction boxes are drawn on identical original-pixel crops within each row, with no brightness, contrast, sharpening, blur removal, or replacement of image content. The crop differs across rows only to keep different-sized targets legible.

## Figure-level claim and evidence mapping

**Bounded claim:** compared with YOLO11n and P2 at the same seed and protocol, P2+ADown can produce tighter boxes for some 16–32-pixel-equivalent-side drones, but can also suppress a valid detection. The photographs illustrate, rather than establish, the aggregate AP results.

| Panel | Source target | YOLO11n IoU | P2 IoU | P2+ADown IoU | What it supports | What it does not support |
|---|---:|---:|---:|---:|---|---|
| a | 24.1 px | 0.624 | 0.696 | 0.785 | One small-target box becomes accurate enough to cross IoU 0.75. | No recall gain: all three models already detect the target at IoU 0.50. |
| b | 29.2 px | 0.610 | 0.612 | 0.777 | ADown-on-P2 is associated with a more accurate box in a different video/source. | Not proof of a general causal mechanism or a seed-stable gain. |
| c | 62.1 px | 0.784 | 0.777 | No detection at conf ≥0.25 | Important score-threshold failure boundary. | Not proof that ADown cannot geometrically localize the drone: its low-floor candidate has IoU 0.832 at confidence 0.013. |

The qualifying screen over the 625 original-source test images/685 targets found **six** 16–32 px targets with baseline IoU in `[0.50, 0.75)`, P2+ADown IoU ≥0.75 and P2+ADown confidence ≥0.40; panels a and b were selected from different source videos without a visible stock watermark. Under the recorded comparable-size regression screen, panel c was the one qualifying target. These counts describe *screening filters*, not the gain/loss prevalence in the whole test. Selection was made after test inspection; it must be disclosed as illustrative.

## Link to the aggregate experiment

The **same 641-image full-test, COCO-style evaluator** reports the following seed-1 outcomes. These are not per-image averages or values assembled from different checkpoints.

| Metric (%) | YOLO11n | P2 | P2+ADown | P2+ADown − YOLO11n | P2+ADown − P2 |
|---|---:|---:|---:|---:|---:|
| AP50:95 | 56.39 | 56.82 | 57.86 | +1.47 | +1.04 |
| AP75 | 60.53 | 59.56 | 64.17 | +3.64 | +4.61 |
| AP, 16–32 px | 36.61 | 39.02 | 40.96 | +4.35 | +1.94 |
| AP, <16 px | 18.12 | 19.42 | 14.52 | −3.60 | −4.90 |

Source: `reproduction/results/p2_msef_adown_visual_evidence_seed1/figure_source_data.csv` and its `manifest.json`. Values are rounded to two decimals; differences use the unrounded source. The <16 px category has only **nine targets**, so neither its decline nor any proposed gain supports a stable tiny-target conclusion. AP75 is a dataset-level precision–recall area at IoU 0.75, not the proportion of these three targets that pass IoU 0.75.

The image plate is consistent with the stronger seed-1 AP75 and 16–32 px AP, but does not explain them causally. The 200-epoch three-seed original-source tests showed that P2+ADown versus **YOLO11n** improved AP50:95 in all three seeds, whereas P2+ADown versus **P2** was not directionally stable across seeds. Therefore a paper must not use this seed-1 image plate to claim that ADown's incremental gain is robust across random seeds.

## Proposed English manuscript wording

**Results text:** “On the full rebuilt test set, the validation-selected 200-epoch seed-1 P2+ADown model achieved higher COCO-style AP75 than YOLO11n (64.17% versus 60.53%) and P2 (59.56%). Two illustrative 16–32-pixel targets show tighter boxes, crossing the IoU 0.75 threshold, whereas another image shows a valid target falling below the fixed display threshold. These examples illustrate localization and score-calibration differences; they do not establish universal small-target detection gains.”

**Figure legend:** “**Fig. X | Qualitative localization and failure cases for matched seed-1 models.** a,b Two 16–32-pixel-equivalent-side targets from different original-source videos are detected by all three validation-selected 200-epoch models, but the P2+ADown boxes align more closely with ground truth. c Counterexample where P2+ADown produces no displayed detection at confidence ≥0.25, although a correctly localized candidate remains at confidence 0.013. Left, full original image and crop boundary; right, identical native-pixel crops across models. Teal boxes show ground truth and red boxes show predictions. Images are enlarged only for display, without intensity or content modification. Examples were selected post hoc from the rebuilt dataset's 641-image test and are illustrative rather than representative; aggregate values are reported separately. Source coordinates, checkpoint identifiers, and selection rules are supplied with the figure.”

## Targeted SCI reviewer assessment

**Suitable after caption disclosure, not standalone proof.** The old plate mixed visually strong and weak examples, stock watermarks and a threshold-edge baseline “miss.” The new plate avoids that threshold-edge success claim and includes a failure case. Its two gains are genuine localization changes for already-detected targets; the aggregate AP75 gain points in the same direction. The P2+ADown failure is a **confidence-collapse** case, not an inability to predict a geometrically correct box. This distinction must stay in the Results and Discussion.

Reviewer objections that still matter:

1. The examples are post-hoc and the test set has been reused in method development. Treat the plate as explanatory illustration, never independent confirmation or proof of generalization.
2. The two successful examples are from original videos and the source frame licensing has not been verified. Before journal submission, verify permission or replace them with rights-cleared cases while preserving the same model and selection protocol. No stock watermark was erased.
3. The figure is single-seed and does not establish a stable incremental ADown benefit over P2. The manuscript should present the full quantitative ablation and state the seed/protocol limits.
4. The plate says nothing about MSEF. Do not use it to argue for that module, blur robustness, improved recall, faster inference, or performance on unseen datasets.

## Figure QA

- Figure contract: `image plate + quantitative cross-reference`; output width 182.9 mm, height 135.9 mm; a/b are positive localization examples and c is the conclusion-changing boundary case.
- Matplotlib (Python) is the sole rendering backend. PDF, editable-text SVG, 300-dpi PNG, and 600-dpi TIFF were exported.
- Source preflight: 21 pass, 0 warn, 0 fail. PDF glyph floor: 6.3 pt ≥5 pt. Panel alignment: PASS with 10 comparisons, no exemptions. Final PDF collision audit: PASS with 0 fail and 0 warn. The generated alignment overlay and collision overlay are QA artifacts, not paper figures.
- Visual inspection at the final 182.9-mm width found all GT/prediction boxes visible, row-wise crops identical, status text outside the image region, and no title/label overlap. The warning that image content or source rights may require replacement is substantive, not a rendering defect.
