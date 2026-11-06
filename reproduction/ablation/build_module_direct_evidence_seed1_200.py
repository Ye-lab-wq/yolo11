#!/usr/bin/env python3
"""Direct-parent qualitative audit from frozen 200-epoch seed-1 predictions.

The examples were selected after test inspection and are illustrations, not
independent validation. The transition audit uses every GT in the 641-image
test, one-to-one greedy matching at a stated confidence and IoU threshold.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

from audit_panel_alignment import require_matplotlib_panel_alignment


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "reproduction/results/p2_msef_adown_visual_evidence_seed1"
OUT = ROOT / "reproduction/results/module_direct_evidence_seed1_200"
CONF = 0.25
GT_COLOR, PRED_COLOR, CROP_COLOR = "#008C83", "#D1495B", "#E5A100"
EXAMPLES = (
    # Role, filename, target index in the YOLO label file, crop side, parent, child.
    ("P2: small-target score rescue", "video18_1628__8c8cb8266b2195d7.jpg", 0, 128, "YOLO11n", "+P2"),
    ("ADown: tighter small-target box", "video17_146__e86d0aaa615e4daf.jpg", 0, 128, "+P2", "+P2+ADown"),
    ("ADown: confidence-collapse counterexample", "video16_453__c10eeab514799863.jpg", 0, 176, "+P2", "+P2+ADown"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_gt(path: Path, width: int, height: int) -> np.ndarray:
    boxes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        cls, x, y, bw, bh = map(float, line.split())
        if cls != 0:
            raise ValueError(f"Unexpected class {cls}: {path}")
        boxes.append([(x - bw / 2) * width, (y - bh / 2) * height,
                      (x + bw / 2) * width, (y + bh / 2) * height])
    return np.asarray(boxes, dtype=float).reshape(-1, 4)


def ious(boxes: np.ndarray, gt: np.ndarray) -> np.ndarray:
    if len(boxes) == 0:
        return np.empty(0, dtype=float)
    intersection = np.maximum(np.minimum(boxes[:, 2:4], gt[2:]) -
                              np.maximum(boxes[:, :2], gt[:2]), 0).prod(axis=1)
    area = np.maximum(boxes[:, 2:4] - boxes[:, :2], 0).prod(axis=1)
    gt_area = np.maximum(gt[2:] - gt[:2], 0).prod()
    return intersection / np.maximum(area + gt_area - intersection, 1e-12)


def greedy_matches(pred: np.ndarray, gt: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Per-image, score-ordered, one-to-one GT matches; not COCO AP."""
    hit = np.zeros(len(gt), dtype=bool)
    for row in pred[np.argsort(-pred[:, 4])]:
        if row[4] < CONF:
            break
        overlaps = np.asarray([ious(row[None, :4], box)[0] for box in gt])
        overlaps[hit] = -1.0
        if len(overlaps) and overlaps.max() >= threshold:
            hit[int(np.argmax(overlaps))] = True
    return hit


def transition_audit(cache: dict) -> list[dict]:
    rows = []
    for rel in cache["YOLO11n"]:
        image_path = ROOT / rel
        with Image.open(image_path) as image:
            width, height = image.size
        gt = load_gt(image_path.parent.parent / "labels" / (image_path.stem + ".txt"), width, height)
        pred = {model: np.asarray(cache[model][rel], dtype=float).reshape(-1, 5)
                for model in ("YOLO11n", "+P2", "+P2+ADown")}
        hits50 = {model: greedy_matches(pred[model], gt, 0.5) for model in pred}
        hits75 = {model: greedy_matches(pred[model], gt, 0.75) for model in pred}
        for index, box in enumerate(gt):
            side = float(np.sqrt(np.prod(box[2:] - box[:2])))
            rows.append({"image": rel, "target_index": index, "side_px": side,
                         "YOLO11n_hit50": int(hits50["YOLO11n"][index]),
                         "P2_hit50": int(hits50["+P2"][index]),
                         "P2_ADown_hit50": int(hits50["+P2+ADown"][index]),
                         "YOLO11n_hit75": int(hits75["YOLO11n"][index]),
                         "P2_hit75": int(hits75["+P2"][index]),
                         "P2_ADown_hit75": int(hits75["+P2+ADown"][index])})
    return rows


def crop_for(gt: np.ndarray, side: int, width: int, height: int) -> tuple[int, int, int, int]:
    center = (gt[:2] + gt[2:]) / 2
    x0 = max(0, min(int(round(center[0] - side / 2)), width - side))
    y0 = max(0, min(int(round(center[1] - side / 2)), height - side))
    return x0, y0, x0 + side, y0 + side


def draw_box(ax, box: np.ndarray, offset: tuple[int, int], color: str, lw: float) -> None:
    ax.add_patch(Rectangle((box[0] - offset[0], box[1] - offset[1]),
                           box[2] - box[0], box[3] - box[1],
                           fill=False, edgecolor=color, linewidth=lw, joinstyle="miter"))


def record(cache: dict, role: str, filename: str, target_index: int, crop_side: int,
           parent: str, child: str) -> dict:
    path = ROOT / "DetectDataset_clean_v2/test/images" / filename
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"))
    height, width = rgb.shape[:2]
    gt = load_gt(path.parent.parent / "labels" / (path.stem + ".txt"), width, height)
    target = gt[target_index]
    crop = crop_for(target, crop_side, width, height)
    rel = str(path.relative_to(ROOT))
    model_data = {}
    for name in (parent, child):
        pred = np.asarray(cache[name][rel], dtype=float).reshape(-1, 5)
        kept = pred[pred[:, 4] >= CONF]
        overlaps = ious(kept[:, :4], target)
        index = int(np.argmax(overlaps)) if len(overlaps) else None
        raw_overlap = ious(pred[:, :4], target)
        raw_index = int(np.argmax(raw_overlap)) if len(raw_overlap) else None
        model_data[name] = {
            "pred": pred, "best_iou": float(overlaps[index]) if index is not None else 0.0,
            "best_conf": float(kept[index, 4]) if index is not None else 0.0,
            "raw_iou": float(raw_overlap[raw_index]) if raw_index is not None else 0.0,
            "raw_conf": float(pred[raw_index, 4]) if raw_index is not None else 0.0,
        }
    return {"role": role, "path": path, "rel": rel, "image": rgb, "gt": gt,
            "target_index": target_index, "target": target, "crop": crop,
            "side_px": float(np.sqrt(np.prod(target[2:] - target[:2]))),
            "parent": parent, "child": child, "models": model_data}


def render(rows: list[dict], prefix: Path) -> None:
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 7, "svg.fonttype": "none", "pdf.fonttype": 42,
                         "axes.linewidth": 0.6, "savefig.facecolor": "white"})
    fig = plt.figure(figsize=(7.2, 5.75), facecolor="white")
    gs = fig.add_gridspec(3, 3, left=0.06, right=0.985, bottom=0.12,
                          top=0.865, wspace=0.10, hspace=0.38)
    axes = [[fig.add_subplot(gs[i, j]) for j in range(3)] for i in range(3)]
    for i, row in enumerate(rows):
        image, target, crop = row["image"], row["target"], row["crop"]
        x0, y0, x1, y1 = crop
        context = axes[i][0]
        context.imshow(image, interpolation="nearest")
        context.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False,
                                    edgecolor=CROP_COLOR, linewidth=1.0))
        for box in row["gt"]:
            draw_box(context, box, (0, 0), GT_COLOR, 0.7)
        context.set_axis_off()
        context.set_title(f"{chr(97+i)}  {row['role']}  ({row['side_px']:.0f} px)",
                          loc="left", fontsize=7, fontweight="bold", pad=5)
        for j, model in enumerate((row["parent"], row["child"]), 1):
            ax = axes[i][j]
            ax.imshow(image[y0:y1, x0:x1], interpolation="nearest")
            for box in row["gt"]:
                if box[2] > x0 and box[0] < x1 and box[3] > y0 and box[1] < y1:
                    draw_box(ax, box, (x0, y0), GT_COLOR, 0.9)
            pred = row["models"][model]["pred"]
            centers = (pred[:, :2] + pred[:, 2:4]) / 2
            visible = pred[(pred[:, 4] >= CONF) & (centers[:, 0] >= x0)
                           & (centers[:, 0] <= x1) & (centers[:, 1] >= y0)
                           & (centers[:, 1] <= y1)]
            for box in visible:
                draw_box(ax, box[:4], (x0, y0), PRED_COLOR, 0.9)
            ax.set_xlim(0, x1-x0)
            ax.set_ylim(y1-y0, 0)
            ax.set_axis_off()
            metrics = row["models"][model]
            status = (f"IoU {metrics['best_iou']:.2f}  |  conf {metrics['best_conf']:.2f}"
                      if metrics["best_iou"] >= 0.5 else "No IoU≥0.50 detection")
            ax.text(0.5, -0.10, f"{model}: {status}", transform=ax.transAxes,
                    ha="center", va="top", fontsize=6.7)
    fig.text(0.06, 0.96, "Direct-parent module contrasts | matched 200-epoch seed-1 checkpoints",
             fontsize=8.2, fontweight="bold", va="top")
    fig.text(0.06, 0.93,
             "GT: teal solid box    Prediction: red solid box    Crop: amber    Display confidence ≥0.25",
             fontsize=6.7, va="top")
    fig.text(0.06, 0.053,
             "Same native-pixel crop within each row; no image enhancement. Post-hoc examples, not representative samples.",
             fontsize=6.2, va="bottom")
    require_matplotlib_panel_alignment(fig, json_out=str(prefix)+".alignment.json",
                                       overlay_svg=str(prefix)+".alignment-overlay.svg",
                                       tolerance_pt=1.5, gutter_tolerance_pt=1.5,
                                       require_panel_labels=False, strict=True)
    fig.savefig(str(prefix)+".pdf", dpi=300)
    fig.savefig(str(prefix)+".svg", dpi=300)
    fig.savefig(str(prefix)+".png", dpi=300)
    fig.savefig(str(prefix)+".tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cache_path = SOURCE / "raw_predictions.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    transitions = transition_audit(cache)
    if len(transitions) != 701:
        raise AssertionError(f"Expected 701 full-test targets, found {len(transitions)}")
    with (OUT / "target_transitions_conf025.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(transitions[0]))
        writer.writeheader()
        writer.writerows(transitions)
    rows = [record(cache, *example) for example in EXAMPLES]
    # Guard the intended contrast before publishing any selected plate.
    a, b, c = rows
    if not (16 <= a["side_px"] < 32 and a["models"]["YOLO11n"]["best_iou"] < 0.5
            and a["models"]["+P2"]["best_iou"] >= 0.5):
        raise AssertionError("P2 rescue no longer satisfies selection rule")
    if not (16 <= b["side_px"] < 32 and b["models"]["+P2"]["best_iou"] < 0.75
            and b["models"]["+P2+ADown"]["best_iou"] >= 0.75):
        raise AssertionError("ADown localization example no longer satisfies selection rule")
    if not (c["models"]["+P2"]["best_iou"] >= 0.75
            and c["models"]["+P2+ADown"]["best_iou"] < 0.5):
        raise AssertionError("ADown counterexample no longer satisfies selection rule")
    prefix = OUT / "direct_parent_examples"
    render(rows, prefix)
    manifest = {
        "source": "200-epoch, seed-1, validation-selected best.pt predictions",
        "full_test": "DetectDataset_clean_v2/test; 641 images, 701 targets",
        "display_confidence": CONF, "one_to_one_transition_ious": [0.5, 0.75],
        "cache_sha256": sha256(cache_path),
        "selection": "exploratory and post-hoc; one direct-parent rescue, one localization gain, one regression",
        "images": [{"role": row["role"], "path": row["rel"], "sha256": sha256(row["path"]),
                    "target_index": row["target_index"], "all_gt_xyxy": row["gt"].tolist(),
                    "crop_xyxy": row["crop"], "side_px": row["side_px"],
                    "parent": row["parent"], "child": row["child"],
                    "models": {name: {key: value for key, value in data.items() if key != "pred"}
                               for name, data in row["models"].items()}}
                   for row in rows],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    print(f"Saved {prefix}; audited {len(transitions)} targets")


if __name__ == "__main__":
    main()
