# DroneDetection clean v1

This is a newly materialized dataset derived from `/home/b520/Downloads/yelin/yolo11/DetectDataset`. The source
directory was not modified.

Protocol: global exact-pixel deduplication, source/capture-disjoint grouping,
deterministic 70/15/15 allocation (seed 20260812), and exclusion of exact images
whose labels conflict. The legacy train/valid/test membership is not reused.

- `data.yaml`: main Ultralytics configuration; public clean train plus 36
  self-video frames for training, and public test plus 16 self-video frames for test
- `split_manifest.csv`: complete old-to-new provenance
- `audit.json`: protocol, exclusions, counts, sizes, and capture groups
- `train|valid|test/images|labels`: new YOLO dataset
- `self_train/images|labels`: 36 unique sequential frames from one self-collected
  1280x720 indoor-drone video, included in training
- `self_test/images|labels`: 16 other unique timestamp-sampled frames from the
  same video, included in final testing
- `data_self_test.yaml`: Ultralytics configuration for evaluating `self_test`
- `data_public_test.yaml`: Ultralytics configuration for evaluating only the
  public clean test while retaining the same combined training definition
- `self_video_manifest.csv`: recovered-file-to-clean-frame provenance
- `self_video_audit.json`: source-video hash, deduplication, and overlap audit

Important: old checkpoints trained on the legacy split are not a fair comparison
on this dataset. Baseline and MEDA must both be retrained from scratch with the
same optimization settings, seed policy, epoch budget, and model-selection rule.
The 16 self-test frames must not be used for early stopping, threshold tuning, or
model selection. Because self-train and self-test are different frames from the
same video, this is within-video unseen-frame evaluation, not unseen-video or
source-disjoint evaluation. Report public clean test and self-video test metrics
separately in addition to any combined metric.
