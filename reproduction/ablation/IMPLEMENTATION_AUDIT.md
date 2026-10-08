# MEDA module and implementation audit

This audit distinguishes three things that must not be conflated in a paper: the chapter's claimed mechanism,
the uploaded executable implementation, and a corrected/reference implementation. Experimental conclusions will
use the exact names below.

## Audited factors

| Factor | Intended purpose | Uploaded executable behavior | Audit verdict | Controlled experiment |
|---|---|---|---|---|
| P2 detection head | Preserve stride-4 detail for tiny drones | Adds a fourth 160x160 prediction level | Implemented and testable, but raises prediction sites at 640 from 8,400 to 34,000 (4.05x) | `p2_nearest_ciou_s0` versus YOLO11n |
| ADown | Downsample while retaining complementary average/max-pooled cues | Average pool, channel split, stride-2 3x3 branch plus max-pool/1x1 branch, then concatenate | Executable; the paper equation omits the max-pool operation required by the second branch | `p2_adown_ciou_s0` versus P2 parent |
| Uploaded `DySample` | Claimed content-aware learned sampling | A parameter-free call to static bilinear `interpolate` | **Not DySample**; it has no offsets and no `grid_sample` | `p2_author_bilinear_ciou_s0` |
| Reference `DySampleOfficial` | Learn grouped sampling offsets for content-aware upsampling | Offset convolution, initialized sampling positions, pixel shuffle and grouped `grid_sample` | Faithful LP reference path added under a distinct name; old checkpoints remain unchanged | `p2_official_dysample_ciou_s0` |
| Uploaded `RepNCSPELAN4` | Claimed CSP/ELAN aggregation with inference-time re-parameterization | Split/serial ordinary `Conv` blocks/concatenation | Aggregation exists, but **no RepConv and no train-to-deploy fusion path**; the re-parameterization claim is unsupported | `p2_rep_ciou_s0` |
| Unshadowed official `RepNCSPELAN4` | CSP/ELAN aggregation with RepCSP/RepConv branches | The repository's original Ultralytics implementation, exposed under the distinct name `RepNCSPELAN4Official` | Contains the re-parameterizable branches that the uploaded replacement removed | `p2_rep_official_ciou_s0` |
| Uploaded MSEF | Extract multi-scale high-frequency edge cues, mainly for blurred/tiny targets | 3/5/7 pooling branches; residual enters an `EdgeEnhancer` that returns `residual + sigmoid(Conv(residual))` | Executable but does not implement Chapter 3 Equations 3-11--3-13 | `p2_msef_author_ciou_s0`, `p2_p5_msef_author_ciou_s0` |
| Equation-consistent MSEF | Chapter's 3/5 smoothing plus multiplicative edge gating | `local + local * sigmoid(Conv1x1(local - pooled))` for 3/5 branches | Added under `MSEFPaper`; this is the implementation corresponding to the written equations | `p2_msef_paper_ciou_s0` |
| EMA on P2 | Suppress high-frequency background noise after P2 edge enhancement | Grouped height/width pooling and spatial attention at 160x160 | Executable; very few parameters do not imply low latency at this resolution | `p2_msef_author_ema_ciou_s0` versus P2-MSEF |
| C2PSA removal | Reduce deep attention cost/overcoupling | Final uploaded MEDA removes the official YOLO11n C2PSA block | A topology change and potential confounder, not an independently established innovation | `p2_no_c2psa_ciou_s0` |
| Uploaded “Focaler” loss | Claimed interval remapping of IoU difficulty | `1-IoU + (1-IoU)^2.5`, using plain IoU and no `d,u` interval | **Not Focaler-IoU** and also discards CIoU geometry in this term | `p2_nearest_author_focal_s0` |
| Reference Focaler-CIoU | Remap IoU to a declared focus interval while retaining CIoU geometry | `(1-CIoU) + IoU - clamp((IoU-d)/(u-d),0,1)`, frozen `d=0,u=0.95` | Corresponds to the Focaler-IoU formulation; implemented under an explicit new mode | `p2_nearest_focaler_ciou_s0` |

## Direct code evidence

- Uploaded static `DySample`: `ultralytics/nn/modules/dysample.py`, class `DySample`.
- Reference learned sampler: same file, class `DySampleOfficial`.
- Uploaded ADown, MSEF, EMA and `RepNCSPELAN4`: `ultralytics/nn/modules/meda_modules.py`.
- Equation-consistent MSEF: same file, class `MSEFPaper`.
- Uploaded loss formula and the three explicit loss modes: `ultralytics/utils/loss.py`, `focaler_iou_loss` and
  `BboxLoss.forward`.
- Uploaded full topology: `reproduction/dut/models/meda_pro_w025.yaml`.
- Generated one-factor controlled topologies: `reproduction/ablation/models/`.

## Scientific interpretation rule

The uploaded full MEDA is a bundle of at least eight coupled topology/training changes. Its aggregate result cannot
establish which mechanism works. A module will only be called effective if its one-factor validation comparison
passes the predeclared screen and survives the confirmatory run. A module that is merely executable, or whose full
bundle improves, will not be described as an innovation with demonstrated benefit.

The old uploaded names are retained for checkpoint compatibility. Corrected/reference implementations use new
names, preventing an old weight from silently changing semantics.

The official reference graph contains eight multi-branch `RepConv` objects before deployment fusion. A direct
`model.fuse()` audit converted all eight from `conv1/conv2` training branches to a single `conv` branch. The uploaded
replacement contains zero `RepConv` objects, so it cannot support the chapter's train-to-deploy claim.

CUDA `grid_sample` backward emits PyTorch's nondeterministic-algorithm warning. Ultralytics enables deterministic
mode with `warn_only=True`, so the reference DySample screen is seed-controlled but not guaranteed bitwise identical
across reruns. It must receive multi-seed confirmation if selected; a single run cannot support a stability claim.

## Seed-1 five-model effect audit

The 200-epoch seed-1 checkpoints were re-evaluated with one COCO-style protocol on the 641-image/701-instance
combined test set. This five-model comparison separates the two single branches from their combination:

| Model | AP50:95/% | AP50/% | AP75/% | AP 16-32 px/% | Params/M | GFLOPs |
|---|---:|---:|---:|---:|---:|---:|
| YOLO11n | 56.39 | 93.61 | 60.53 | 36.61 | 2.590 | 6.4 |
| P2 | 56.82 | 95.70 | 59.56 | 39.02 | 2.904 | 10.8 |
| P2+MSEFPaper | 56.95 | 94.79 | 62.37 | 38.75 | 2.920 | 11.7 |
| P2+ADown | **57.86** | **95.07** | **64.17** | **40.96** | **2.278** | **9.5** |
| P2+MSEFPaper+ADown | 57.52 | 94.38 | 62.69 | 38.13 | 2.295 | 10.3 |

| Factor and controlled parent | Intended effect | Observed direct-parent change | Verdict |
|---|---|---|---|
| P2 versus YOLO11n | Preserve stride-4 small-target detail | AP50 +2.10 points; 16-32 px AP +2.41; AP75 -0.97 | Small/lenient-IoU effect supported; strict localization not supported |
| MSEFPaper versus P2 | Strengthen multi-scale edge cues for blurred/tiny targets | AP75 +2.81; AP50 -0.91; 16-32 px AP -0.28 | Strict localization supported; blur/tiny mechanism remains unverified |
| ADown versus P2 | Retain complementary cues with a lighter downsampler | AP50:95 +1.04; AP75 +4.61; 16-32 px AP +1.94; params -21.5%; GFLOPs -12.0% | Strongly supported in the no-MSEF branch |
| MSEFPaper added to P2+ADown | Combine edge enhancement with efficient downsampling | AP50:95 -0.34; AP75 -1.48; 16-32 px AP -2.84; GFLOPs +8.4% | Negative interaction; the full combination is not supported by this seed-1 test |

The chapter specifically claims that dynamic motion/camera-follow blur suppresses high-frequency rotor and airframe
edges. It places MSEF before both P2 (distant tiny-target edge compensation) and P5 (large close-target motion-smear
compensation). The controlled implementation retained only the equation-consistent P2 placement: adding the same
MSEFPaper block at P5 reduced best seed-0/60-epoch validation mAP50:95 from 56.642% to 54.971% (-1.671 points).
Therefore the current selected P2 MSEFPaper is motivated by the chapter's high-frequency residual hypothesis but is
not a faithful retention of every claimed placement, and its blur-specific mechanism remains unproven.

All controlled reproduction runs used the same detection augmentation arguments: Mosaic probability 1.0 (disabled
for the final 10 epochs), HSV gains 0.015/0.7/0.4, translation 0.1, scale 0.5, and horizontal flip probability 0.5.
Rotation, shear, perspective, vertical flip, MixUp, CutMix, and Copy-Paste were disabled. The `auto_augment` and
`erasing` fields recorded in `args.yaml` belong to the classification pipeline and were not used by YOLO detection.
No explicit motion-blur augmentation was configured. The retained YHP environment also lacks the optional
Albumentations package, so Ultralytics' optional 1% Blur/MedianBlur defaults are inactive in this environment.

The important result is the interaction, not merely the individual module scores. Under this protocol,
`P2+ADown` is better than the full combination in AP50:95, AP50, AP75, small AP, large AP, parameter count, and
GFLOPs; the full model is only 0.04 point higher on medium AP. Therefore
MSEFPaper and ADown cannot be presented as additive improvements, and the full combination should not be selected
solely because both modules were individually motivated. The MSEF blur/edge claim still requires a predeclared
blur-labelled or controlled-corruption evaluation; AP75 alone does not establish the proposed mechanism.

The user-flagged very-large `pic_1032` target is not removed by NMS. `P2+ADown` retains an IoU 0.727 candidate at
confidence 0.058, and the full model retains an IoU 0.728 candidate at confidence 0.013; both fall below the 0.25
decision threshold. This is a score-suppression failure. It occurs in several isolated targets, but aggregate large
AP improves, so current evidence does not support a general large-target-collapse claim.

### Exploratory blur-proxy audit

The combined 641-image test set was ranked by variance of the Laplacian after every image was resized to 640x640,
then split into equal thirds. This is a sharpness proxy, not a motion-blur label: deterministic review of its lower
tail shows that large uniform skies and other low-texture scenes also receive low scores.

| Proxy third | P2 AP50:95/% | P2+MSEFPaper AP50:95/% | Change | P2 AP75/% | P2+MSEFPaper AP75/% | Change |
|---|---:|---:|---:|---:|---:|---:|
| Lower sharpness | 50.69 | 52.14 | +1.45 | 48.84 | 51.16 | +2.32 |
| Middle | 58.25 | 57.72 | -0.53 | 63.78 | 67.09 | +3.31 |
| Higher sharpness | 62.02 | 61.20 | -0.82 | 67.03 | 69.19 | +2.16 |

This exploratory pattern does not support claiming that the current dataset contains no blur, nor does it validate
the chapter's blur-specific mechanism. MSEFPaper's AP75 gain occurs in every proxy third, whereas its overall AP gain
is confined to the lower-proxy third. A controlled synthetic-motion-blur robustness set or manually annotated real
blur subset is required to separate motion blur from low texture and scene composition. Full source scores and
figures are in `reproduction/results/blur_msef_audit_seed1/`.

A subsequent deterministic corruption audit evaluated the frozen seed-1 P2 and P2+MSEFPaper checkpoints on 625
public-test images with 3/7/11-pixel directional motion-blur kernels at 640 input scale. Relative to each model's own
clean AP50:95, P2 lost 0.91/6.03/14.36 points and MSEFPaper lost 1.56/6.86/14.43 points. The corresponding MSEF
robustness advantages were -0.65/-0.83/-0.06 points; AP75 robustness advantages were likewise negative at
-2.78/-2.56/-0.35. Thus MSEFPaper retained an absolute strict-localization advantage in some conditions but did not
degrade more slowly under blur. The current frozen model does not validate the claimed blur-specific robustness.
Full results are in `reproduction/results/msef_motion_blur_seed1/`.

### Exploratory residual ADown screen

ADown first average-pools, splits the channels, processes one half through stride-2 3x3 convolution and the other
through max-pooling plus 1x1 convolution, then concatenates the halves. Unlike a standard stride-2 convolution,
each branch sees only half of the input channels. A defensible problem-first modification is therefore a lightweight
low-frequency residual path that sees all input channels:

`y = ADown(x) + tanh(alpha) * Project(AvgPool2d(x, 2, 2))`,

with learnable `alpha` initialized to zero and applied only at the two deep backbone downsampling sites. A controlled
seed-0/60-epoch validation screen changed best mAP50:95 from 55.555% to 56.483% (+0.928 points), recall from 89.635%
to 90.655% (+1.020), precision from 90.594% to 90.657% (+0.063), and mAP50 from 94.099% to 93.881% (-0.218).
Parameters increased from 2.278148M to 2.311430M (+1.46%) and GFLOPs from 9.4676 to 9.4946 (+0.29%), so the
candidate passed the predeclared screen.

Mechanistic attribution remains weak: the learned `tanh(alpha)` values were only -0.00749 and -0.00111, both small
and opposite to a simple additive-restoration narrative. The extra projection also changes later random
initialization draws even under the same seed. Therefore the screen supports a 200-epoch confirmation, not a claim
that full-channel semantic restoration caused the gain. The confirmation must retain the unchanged `P2+ADown`
parent, report the learned residual scales, and evaluate AP75, size-stratified AP, confidence transitions, and
latency before considering the block an innovation.

Project decision: because the learned residual scales are near zero and the intended restoration mechanism is not
supported, the residual candidate is retained as an exploratory record but is not advanced to a 200-epoch run. The
next controlled work focuses on dynamic-blur training and MSEF interaction instead.

## Code present but not part of final MEDA

`ELSNHead` is defined in `meda_modules.py` but is not referenced by the final MEDA YAML. CAA belongs to the earlier
YOLO11n-CAA/CAA-Focaler branch and is likewise absent from `meda_pro_w025.yaml`. Neither will be attributed to the
final MEDA architecture or mixed into its component ablation.
