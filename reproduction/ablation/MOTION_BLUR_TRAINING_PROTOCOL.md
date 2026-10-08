# Motion-blur training screen

## Question

Does identical online motion-blur exposure improve P2 and P2+MSEFPaper
robustness, and does MSEFPaper provide an effect beyond the augmentation itself?

## Frozen 2x2 design

| Architecture | Standard training | Motion-blur training |
|---|---|---|
| P2 | reuse completed `p2_nearest_ciou_s0` | train `p2_motion_blur_s0` |
| P2+MSEFPaper | reuse completed `p2_msef_paper_ciou_s0` | train `p2_msef_motion_blur_s0` |

The two new cells use the same seed-0, 60-epoch schedule and all other settings
as their completed parents. Online motion blur is applied after geometric/Mosaic
composition with probability 0.30, a uniformly sampled odd line-kernel length
from 3/5/7 pixels at the 640 input resolution, and a continuous random direction
from 0 to 180 degrees. Bounding boxes are unchanged.

Clean validation mAP50:95 selects `best.pt`. The public test and its synthetic
blur derivatives are forbidden during model selection. After both cells are
frozen, the primary robustness endpoint is mean AP50:95 over fixed light,
moderate, and strong derivatives of the public test split; clean public-test AP
is reported jointly. The architecture-specific MSEF effect is the difference
in differences between the two architecture rows, not the MSEF blurred score
in isolation.

This is an exploratory screen. It does not establish real-world motion-blur
generalization without a separately labelled real-blur subset.
