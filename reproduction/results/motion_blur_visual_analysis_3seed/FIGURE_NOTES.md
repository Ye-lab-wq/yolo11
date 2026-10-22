# Motion-blur visual analysis

The figures use deterministic geometry-based selection, not manual image picking. A consensus example requires the paired direction to occur in at least two of three seeds. All model panels use the same native-pixel crop, confidence threshold 0.25, and NMS IoU 0.70. Thin solid green ground-truth boxes are overlaid in every model panel; predictions are thin solid vermillion boxes. `motion_blur_input_closeups` shows an identical native-pixel crop before and after the deterministic test corruption, magnified for display only; the largest annotated target in each selected image sets the crop center. No brightness, contrast, sharpening, denoising, or selective image adjustment was applied.

## Population counts

| Condition | Images | Consensus gain | Consensus loss | Any-seed gain | Any-seed loss |
|---|---:|---:|---:|---:|---:|
| clean | 625 | 35 | 34 | 138 | 148 |
| moderate | 625 | 75 | 24 | 213 | 122 |
| strong | 625 | 147 | 20 | 286 | 120 |


The broad gain/loss columns above are not mutually exclusive for multi-object images: one target can improve while another degrades. The threshold-specific consensus counts below are easier to interpret.

| Condition | TP50 gain/loss | TP75 gain/loss | mean-best-IoU gain/loss |
|---|---:|---:|---:|
| clean | 4/3 | 29/29 | 279/331 |
| moderate | 36/3 | 43/19 | 408/201 |
| strong | 93/5 | 84/15 | 434/163 |


The `blur_training_only_seed1` figure isolates online blur augmentation: same P2 architecture, seed 1, test input, confidence and NMS settings. Four rows show one clean and one light-blur regression, plus one moderate and one strong synthetic-blur gain. Cases are selected programmatically from single-target images with GT equivalent side 16–150 native pixels, then ranked by TP50/TP75/IoU difference; the clean and light cases are nearby frames from the same source video, not independent sources. The light-blur per-image predictions were newly computed on CPU with the same 640-pixel input, 0.25 confidence and 0.70 NMS IoU settings, using existing seed-1 checkpoints; no training or aggregate test metric was changed. The companion `p2_blur_training_test_seed1` figure plots all four conditions from `reproduction/results/motion_blur_training_3seed/seed_results.csv`, including the unfavorable light-blur result. Differences use unrounded CSV values; no uncertainty is shown because this is a single-seed plot. The older three-seed qualitative grids compare standard P2 with blur-trained P2+MSEF, so they do not isolate blur training. Per-box confidence labels and in-image statistics have been moved off the image to avoid obscuring boxes. These selected examples demonstrate mechanisms and counterexamples, not prevalence; prevalence is represented by the complete per-image CSV and the aggregate AP figure. Because the corruption is synthetic and derived from the original-source test images, these figures cannot establish real-world motion-blur generalization and must not be used for additional model selection.

Alt text: The quantitative two-panel figure shows three-seed AP50–95 curves across clean, light, moderate, and strong blur and paired effects for blur training, MSEF after blur training, and their interaction. The four-row case figure shows standard-trained and blur-trained predictions from the same P2 architecture and seed on clean, light, moderate and strong test inputs; the first two rows lose a true positive and the last two regain one. The older qualitative blur grid shows clean and blurred ground truth beside standard-P2 and blur-trained P2+MSEF predictions for all three seeds. The clear-regression grid shows counterexamples where standard P2 more consistently detects or localizes a clean-image target than the blur-trained P2+MSEF model.
