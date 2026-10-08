# UAV-Eagle external zero-shot test protocol

Frozen before opening the images or running inference on 2026-10-07.

## Source and scope

- Source: the official [UAV-Eagle repository](https://github.com/larics/UAV-Eagle), which links the downloadable image archive.
- The authors describe 510 annotated real images of one custom Eagle quadcopter. The labels are already in YOLO box format.
- This is a new external *source* for the current project. It is a zero-shot detection test: no images or labels from this source are used to train, select a checkpoint, tune confidence, or modify the model.
- Because the images are from one dataset and one aircraft type, this test does not establish general performance across drone types or independent videos. Motion effects are not a verified blur label.

## Fixed comparison

Evaluate the validation-selected 200-epoch `best.pt` checkpoints for seeds 0, 1 and 2 of:

1. YOLO11n;
2. YOLO11n + P2;
3. YOLO11n + P2 + MSEFPaper;
4. YOLO11n + P2 + full-path ADown;
5. YOLO11n + P2 + MSEFPaper + full-path ADown.

The P2+ADown 200-epoch seeds 0 and 2 finished on 2026-10-07; all five comparisons now have three validation-selected checkpoints. All models were trained from scratch on `DetectDataset_clean_v2` with the same 200-epoch schedule and image size 640. No UAV-Eagle-specific fine-tuning is allowed. External inference has not yet been run.

Use the existing `evaluate_frozen.py` COCO-style evaluator with inference size 640, confidence floor 0.001, NMS IoU 0.70 and max detections 300. Report AP50, AP75, AP50–95 and target-size AP for each seed, then descriptive mean and sample SD across seeds. The same-image paired contrasts are reported against YOLO11n and each module's direct parent. Keep per-seed values visible; do not choose a seed using this test.

Before inference, check archive integrity, image-label pairing and valid normalized coordinates, and compare decoded-image hashes against the internal clean dataset. Preserve the supplied labels. Any class-ID remapping must be recorded in the audit; no relabeling based on model predictions.

If there are no verified target-negative frames, precision and AP describe performance on target-positive frames only and cannot characterize background false alarms. Report that limitation explicitly.
