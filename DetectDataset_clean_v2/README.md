# DroneDetection clean frame v2

This dataset is a frame-level, format-aware rebuild of `/home/b520/Downloads/yelin/yolo11/DetectDataset`. Distinct
video frames may cross train/valid/test. High-confidence identical image content
cannot cross splits, including byte-identical files, identical decoded RGB pixels,
conservatively verified re-encodings, and variants sharing a normalized original
source ID. Different numbered video frames are not merged by perceptual similarity.

Public split sizes: train 2911, valid 625,
test 625. See `audit.json` and `split_manifest.csv` for the
complete protocol and provenance.

The self-collected video contributes 36 unique frames to `self_train` and 16
other unique frames to `self_test`. `data.yaml` combines these with the public
train/test splits. Use `data_public_test.yaml` and `data_self_test.yaml` to report
the two test domains separately. The self-video result is within-video unseen-frame
evaluation, not unseen-video generalization.

There is **one** rebuilt dataset, not a separate "combined dataset" and "public
dataset". Earlier reports use the code names below for evaluation scopes within
this same dataset:

| Scope | Images | Instances | Meaning |
|---|---:|---:|---|
| `public_test` | 625 | 685 | Test images inherited from the original dataset source; "public" is a local code name, not an external benchmark. |
| `self_test` | 16 | 16 | Unseen frames from the added self-collected video. |
| `combined_test` / `data.yaml` test | 641 | 701 | Both scopes together, not another dataset. |

Training combines 2,911 original-source images and 36 self-video frames (2,947
total); validation contains 625 original-source images. The deterministic
motion-blur benchmark is a **derived copy of the 625 `public_test` images**, not
additional training data or an independently collected test set.
