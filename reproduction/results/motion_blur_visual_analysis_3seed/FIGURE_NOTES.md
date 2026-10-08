# Motion-blur visual analysis

The figures use deterministic geometry-based selection, not manual image picking. A consensus example requires the paired direction to occur in at least two of three seeds. All model panels use the same native-pixel crop, confidence threshold 0.25, and NMS IoU 0.70. Ground truth is a solid green box in dedicated columns; predictions are solid vermillion boxes. No brightness, contrast, sharpening, denoising, or selective image adjustment was applied.

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


The qualitative examples demonstrate mechanisms and counterexamples, not prevalence; prevalence is represented by the complete per-image CSV and the aggregate AP figure. Because the corruption is synthetic and derived from the public test images, these figures cannot establish real-world motion-blur generalization and must not be used for additional model selection.

Alt text: The quantitative two-panel figure shows three-seed AP50–95 curves across clean, light, moderate, and strong blur and paired effects for blur training, MSEF after blur training, and their interaction. The qualitative blur grid shows clean and blurred ground truth beside standard-P2 and blur-trained P2+MSEF predictions for all three seeds. The clear-regression grid shows counterexamples where standard P2 more consistently detects or localizes a clean-image target than the blur-trained P2+MSEF model.
