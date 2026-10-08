# Dataset audit examples

Red boxes are stored YOLO GT. Dashed yellow boxes are manual audit candidates and were not added to the dataset.

## Annotation examples

- `annotation_01_loose_box_video16_673.png` — Annotation audit 1: stored box appears substantially loose. Red is the dataset GT. The visual drone occupies only part of the stored box.
- `annotation_02_partial_box_video14_204.png` — Annotation audit 2: stored box covers only part of a large drone. The GT boundary is inconsistent with the full visible object extent.
- `annotation_05_ambiguous_region_pic1026.png` — Annotation audit 5: one very large region contains multiple visible structures. The single stored GT is not a tight one-object box, making localization metrics ambiguous.
- `annotation_06_loose_box_video17_1060.png` — Annotation audit 6: another loose-box example. This kind of GT tightness variation can depress AP75 even when detection is visually plausible.
- `annotation_03_missing_label_video18_1617.png` — a visible second drone has no stored GT; blue is an unmatched baseline prediction.
- `annotation_04_missing_label_video18_553.png` — one of several visible drones has no stored GT; blue is an unmatched baseline prediction.
- `annotation_07_same_image_conflicting_train_test_gt.png` — byte-identical image with different train/test GT boxes.

## Leakage examples

See `leakage_examples_overview.png` and `leakage_example_manifest.csv` for exact paths and SHA-256 hashes.
