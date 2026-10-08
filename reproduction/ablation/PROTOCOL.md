# MEDA implementation audit and controlled ablation protocol

Frozen before new ablation training on 2026-08-13. The cleaned, format-aware deduplicated
`DetectDataset_clean_v2` split is fixed. Test images are not used for screening or checkpoint selection.

## Claims under audit

| Factor | Baseline | Uploaded MEDA implementation | Audit issue |
|---|---|---|---|
| Detection scale | P3/P4/P5 | P2/P3/P4/P5 | Adds 160x160 head and 4.05x prediction sites |
| Downsampling | stride-2 Conv | ADown | Extra pooling/split/concat; latency must be measured |
| Upsampling | nearest | class named `DySample` | Uploaded class is static bilinear interpolation, not DySample |
| Neck fusion | C3k2 | custom `RepNCSPELAN4` | Uploaded class has no executable re-parameterization |
| Edge fusion | none | uploaded MSEF at P2/P5 | Uploaded equation differs from Chapter 3 |
| Attention | none | EMA at P2 | 48 parameters but high-resolution latency |
| Deep attention | C2PSA | removed | Removal is a separate factor, not automatically an improvement |
| Box loss | CIoU | uploaded focal-weighted IoU | Uploaded formula is not the paper/official Focaler-IoU |

`DySampleOfficial` follows the public ICCV 2023 LP implementation. `MSEFPaper` follows Chapter 3
Equations 3-11 to 3-13 using 3x3/5x5 smoothing and multiplicative edge gating. `focaler_ciou`
uses the official interval mapping with predeclared `d=0.00, u=0.95` and retains CIoU geometry.
These names are intentionally distinct from the uploaded implementations.

## Stage 1: exploratory screening

- From scratch, 60 epochs, seed 0, 640x640, batch 16, AdamW, no AMP.
- All augmentation and optimization settings are identical to the completed paired experiment.
- Primary screen outcome: best validation mAP50:95 within 60 epochs.
- Secondary outcomes: validation P/R/mAP50, parameters, GFLOPs, batch-1 and batch-16 latency.
- YOLO11n and full uploaded MEDA receive new, true 60-epoch runs with the identical scheduler and epoch-50 Mosaic
  closure. Earlier first-60 slices from 200-epoch runs are retained only as non-comparable audit records because their
  learning-rate horizon and Mosaic closure differ.
- Screening checkpoint selection uses only best validation mAP50:95 within the common 60-epoch window. Epoch 60 is
  a fixed compute budget, not an early-stopping result; all candidates therefore receive the same opportunity.
- No test evaluation is permitted in this stage.
- A factor advances if it improves its direct parent by at least 0.30 percentage points mAP50:95,
  or remains within 0.10 points while reducing latency by at least 10%. Borderline candidates may
  advance only when their mechanism is central to the stated tiny-target hypothesis.

The controlled P2 parent changes only the detection topology. Each child changes exactly one named
factor: uploaded bilinear upsampling, official DySample, ADown, uploaded/official RepNCSPELAN4, uploaded P2-MSEF,
equation-consistent P2-MSEF, P2-MSEF+EMA, P5-MSEF, C2PSA removal, uploaded focal loss, or official
Focaler-CIoU.

## Stage 2: confirmatory experiments

- At most one principal structural factor and one auxiliary strategy are selected.
- Selected ablation chain is trained for 200 epochs under the frozen paired protocol.
- YOLO11n and the final selected method receive seeds 0, 1, and 2; report mean, sample SD, and all runs.
- The already completed YOLO11n seed-0 200-epoch run is reused because its data, optimizer, augmentation and seed are
  identical to this frozen protocol; seed 1/2 are newly trained. No run is discarded or silently repeated for metrics.
- `best.pt` is selected solely by validation mAP50:95. No metric is assembled across epochs/checkpoints.
- After the final method is frozen, evaluate public test once. Self-video test is reported separately and
  is not pooled into the primary conclusion.
- Final reporting: P, R, mAP50, mAP50:95, AP75, APtiny/APsmall where sample counts permit, parameters,
  GFLOPs, checkpoint size, FP32 batch-1 latency/FPS, batch-16 throughput, and peak VRAM on RTX 3090.

Negative and failed results remain in the table. Any deviation from this protocol is timestamped and labeled
exploratory rather than silently incorporated into the confirmatory analysis.

## Recorded protocol correction

The initial screen launch incorrectly treated the first 60 rows of completed 200-epoch baseline/MEDA curves as
same-budget references. This was detected before confirmatory selection. Those rows use a 200-epoch learning-rate
schedule and do not close Mosaic within the first 60 epochs, whereas a true 60-epoch run closes Mosaic at epoch 50.
They were therefore removed from direct comparisons, retained in `noncomparable_first60_references.csv`, and replaced
with fresh 60-epoch baseline/MEDA runs. All one-factor candidate runs already used the same true 60-epoch protocol and
remain internally comparable.
