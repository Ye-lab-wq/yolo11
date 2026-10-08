# Deep residual ADown validation screen

## Question

Does a zero-initialized, full-channel, low-frequency residual on the two deep
backbone downsampling sites improve the validation performance and confidence
retention of `P2+ADown`, without materially weakening its lightweight benefit?

## Controlled comparison

- Parent: completed `p2_adown_ciou_s0` 60-epoch, seed-0 screen.
- Child: identical `p2_adown.yaml` except backbone layers 5 and 7 use
  `ADownResidual`; shallow backbone layer 3 and all three neck ADown sites are
  unchanged.
- Residual: `ADown(x) + tanh(alpha) * Project(AvgPool2d(x, 2, 2))`, with
  `alpha=0` at initialization. `Project` is identity when channel counts match
  and a linear 1x1 convolution plus batch normalization otherwise.
- Dataset, seed, optimizer, schedule, augmentations, loss, image size, and all
  other training arguments are copied from the parent screen.
- Model selection and screening use validation only. Test access is forbidden
  until a model is frozen for a later confirmatory experiment.

## Predeclared screen

The child is retained for a 200-epoch seed-1 confirmation only if all are true:

1. best validation mAP50:95 improves by at least 0.30 percentage points over
   the completed seed-0 parent;
2. best validation mAP50 does not fall by more than 0.50 percentage points;
3. parameters grow by no more than 2.0% over the parent;
4. the run completes all 60 epochs without NaN or resume/protocol mismatch.

This is an exploratory architecture screen. Passing it does not establish a
large-target or confidence-calibration mechanism; those require frozen-model
size-stratified and score-transition evaluation after confirmation.
