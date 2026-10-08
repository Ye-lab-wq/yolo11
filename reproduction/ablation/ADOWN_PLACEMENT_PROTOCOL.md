# ADown placement 2x2 screen

Frozen before placement training on 2026-08-14. This experiment decomposes the
six simultaneous ADown replacements in the selected P2+MSEFPaper graph into two
predeclared regions:

- **Backbone ADown:** the three P2->P3, P3->P4 and P4->P5 downsampling layers in
  the feature extractor;
- **Neck ADown:** the three P2->P3, P3->P4 and P4->P5 bottom-up downsampling
  layers after the P2 prediction feature.

The full 2x2 consists of two reused endpoints and two new runs:

| Backbone | Neck | Source |
|---|---|---|
| Conv | Conv | reuse completed `p2_msef_paper_ciou_s0` |
| ADown | Conv | new `adown_backbone_only_s0` |
| Conv | ADown | new `adown_neck_only_s0` |
| ADown | ADown | reuse completed `msefbase_adown_ciou_s0` |

Both new runs are from scratch, seed 0, 60 epochs, 640x640, batch 16, AdamW,
AMP off, and inherit every optimizer, augmentation, scheduler and validation
setting from the frozen secondary screen. Best validation mAP50:95 is the
primary exploratory outcome. P, R, mAP50 and parameter count are reported from
the same selected epoch/checkpoint. Test access is forbidden.

The interaction in percentage points is predeclared as:

`joint - backbone_only - neck_only + neither`.

No six-position search is permitted unless this grouped experiment produces a
materially contradictory result that cannot support a placement decision.
