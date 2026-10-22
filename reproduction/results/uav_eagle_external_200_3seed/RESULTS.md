# UAV-Eagle frozen external zero-shot test (2026-10-20)

This evaluates the **predeclared** five architectures and seeds 0/1/2 from
[`external_benchmarks/uav_eagle/PROTOCOL.md`](../../../external_benchmarks/uav_eagle/PROTOCOL.md).
All 15 `best.pt` checkpoints were selected by internal validation before this
external inference. The frozen protocol records no UAV-Eagle fine-tuning or
checkpoint selection.
The YHP environment and the existing COCO-style evaluator were used at 640 px,
confidence floor 0.001, NMS IoU 0.70, max 300 detections, batch 16. The run's
full metrics and checkpoint SHA-256 hashes are in [`results.json`](results.json).

## Dataset audit

The supplied ZIP passed CRC testing. It contains 510 image/label pairs, 617
class-0 boxes and **no target-negative image**. All labels are valid normalized
YOLO boxes. Decoded RGB hashes found no overlap with any of the 4,213 internal
clean-v2 images. The external source has only 504 unique decoded images: six
identical-image pairs; four of those pairs have differing supplied box
coordinates. The labels and all 510 images were preserved as specified before
inference. See [`audit.json`](../../../external_benchmarks/uav_eagle/audit.json).

In native-image COCO area bins, this benchmark has zero `<32 px` instances,
282 medium and 335 large instances; tiny/small AP is therefore **undefined**,
not zero. As a separate input-scale diagnostic, resizing/letterboxing to 640
would place 166/617 boxes in the 16–32 px area bin, but the table below uses
the predeclared native-coordinate COCO metrics. This dataset is one external
source and cannot establish cross-aircraft or real-motion-blur generalization.

## Frozen results

All values are percentages; `AP` means COCO-style AP50–95. Mean and sample SD
describe the three seeds, **not** confidence intervals. No p-values are
reported with only three seeds.

| Model | AP50 | AP75 | AP50–95 | AP medium (native) | AP large (native) |
|---|---:|---:|---:|---:|---:|
| YOLO11n | 77.89 ± 0.61 | 17.85 ± 1.32 | 32.76 ± 0.79 | 26.21 ± 1.25 | 38.19 ± 0.51 |
| +P2 | 78.21 ± 1.17 | 22.85 ± 1.01 | 35.00 ± 0.71 | 27.76 ± 1.19 | 41.40 ± 0.18 |
| +P2+MSEFPaper | 77.74 ± 1.02 | 21.11 ± 0.44 | 34.47 ± 1.12 | 27.38 ± 1.03 | 40.42 ± 1.27 |
| +P2+ADown | **79.74 ± 0.41** | **24.58 ± 1.71** | **35.45 ± 0.75** | **28.03 ± 1.13** | 41.34 ± 0.39 |
| +P2+MSEFPaper+ADown | 78.03 ± 0.07 | 24.21 ± 3.51 | 34.87 ± 0.86 | 26.86 ± 0.80 | **41.75 ± 0.52** |

### Per-seed primary and localization outcomes

| Model | Seed 0 AP / AP75 | Seed 1 AP / AP75 | Seed 2 AP / AP75 |
|---|---:|---:|---:|
| YOLO11n | 33.02 / 18.37 | 31.86 / 16.36 | 33.39 / 18.83 |
| +P2 | 35.60 / 24.00 | 35.17 / 22.47 | 34.22 / 22.09 |
| +P2+MSEFPaper | 35.34 / 21.15 | 34.86 / 21.53 | 33.20 / 20.66 |
| +P2+ADown | 36.30 / 26.35 | 35.14 / 24.43 | 34.89 / 22.94 |
| +P2+MSEFPaper+ADown | 34.48 / 20.59 | 35.86 / 27.60 | 34.28 / 24.43 |

Paired AP50–95 differences in seed order, percentage points:

- P2 minus YOLO11n: `+2.58, +3.31, +0.83`; mean `+2.24 ± 1.27`.
- P2+ADown minus YOLO11n: `+3.29, +3.27, +1.51`; mean `+2.69 ± 1.02`.
- P2+ADown minus P2: `+0.70, −0.04, +0.68`; mean `+0.45 ± 0.42`.
- P2+MSEFPaper minus P2: `−0.26, −0.31, −1.02`; mean `−0.53 ± 0.42`.
- Full combination minus P2+ADown: `−1.82, +0.72, −0.62`; mean `−0.57 ± 1.27`.

P2+ADown exceeds the baseline on external AP50–95 and AP75 in every seed.
Its incremental AP over P2 is small and not positive in all three seeds, so
this comparison supports the **combined architecture** more strongly than an
independent ADown gain. MSEFPaper does not support a positive generalization
claim here. AP50–95 is substantially lower than on the internal 625-image
test for every model, consistent with an external-domain/annotation shift;
this experiment alone cannot identify the cause. Because all 510 frames contain
an annotated target, it cannot measure false alarms on target-free scenes.

## Reproduction

Run [`run_external.sh`](../../../external_benchmarks/uav_eagle/run_external.sh)
from any directory after placing and extracting the source ZIP and putting the
15 frozen checkpoints at the recorded paths. It re-audits the archive and data overlap before
running exactly the comparison above. Large source images and `.pt` files are
local only and excluded from Git.
