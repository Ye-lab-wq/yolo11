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
