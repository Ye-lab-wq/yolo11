# P2+MSEFPaper secondary one-factor screen

Frozen before secondary training on 2026-08-13. The direct parent is the completed
`p2_msef_paper_ciou_s0` run from `ablation_screen_v1` (seed 0, true 60-epoch
scheduler). The dataset and every optimization/augmentation setting remain fixed.

## Purpose

Determine whether any already-audited MEDA factor adds value after the strongest
screened structure, P2+MSEFPaper. This is an exploratory validation-only screen;
the public test split remains inaccessible for selection.

## One-factor children

1. replace all stride-2 Conv downsamplers with ADown;
2. replace nearest upsampling with the uploaded static bilinear implementation;
3. replace nearest upsampling with reference DySampleOfficial;
4. replace neck C3k2 fusion with the uploaded ordinary-Conv ELAN aggregation;
5. replace neck C3k2 fusion with RepNCSPELAN4Official;
6. append EMA only after the P2 MSEFPaper output;
7. replace only the P5 output block with MSEFPaper;
8. remove only the deep C2PSA block;
9. replace CIoU with the uploaded power-IoU loss;
10. replace CIoU with reference Focaler-CIoU (`d=0.00`, `u=0.95`).

The uploaded P2-MSEF is not repeated because it is an alternative implementation
of the parent factor and was already directly compared with MSEFPaper in stage 1.

## Frozen training and selection rule

- from scratch; 60 epochs; seed 0; image size 640; batch 16; AdamW; AMP off;
- same learning-rate horizon, Mosaic closure, augmentations and validation split;
- primary outcome: best validation mAP50:95 inside the common 60-epoch budget;
- secondary outcomes: P, R, mAP50, parameters, GFLOPs and fixed-protocol latency;
- advance for accuracy only at `>= +0.30` percentage points versus the direct parent;
- an accuracy-neutral factor may advance for efficiency only if it remains within
  `0.10` points and reduces measured batch-1 latency by at least 10%;
- all negative runs are retained; no checkpoint/metric mixing is permitted.

After all children finish, efficiency is measured before choosing at most one
additional factor for 200-epoch, three-seed confirmation.
