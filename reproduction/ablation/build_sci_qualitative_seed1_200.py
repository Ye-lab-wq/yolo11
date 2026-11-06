#!/usr/bin/env python3
"""Render a traceable, bounded qualitative comparison from frozen seed-1 outputs.

No model inference or image enhancement is performed here. The source is the
existing 200-epoch raw-prediction cache; the same crop is used for all three
models within each example. This figure illustrates localization differences,
not a population-level estimate or a proof of a module mechanism.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

from audit_panel_alignment import require_matplotlib_panel_alignment


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "reproduction/results/p2_msef_adown_visual_evidence_seed1"
OUTPUT = ROOT / "reproduction/results/sci_qualitative_seed1_200"
MODEL_KEYS = ("YOLO11n", "+P2", "+P2+ADown")
MODEL_TITLES = ("YOLO11n", "YOLO11n + P2", "YOLO11n + P2 + ADown")
CONFIDENCE = 0.25

# Exploratory examples, selected after viewing the test set. Their source IDs
# and selection limits are disclosed in the companion audit, not generalized.
EXAMPLES = (
    ("a", "Small target / tighter box", "video14_116__9262a227b8483f48.jpg", 128),
    ("b", "Small target / tighter box", "video17_146__e86d0aaa615e4daf.jpg", 128),
    ("c", "Boundary case / missed target", "video16_453__c10eeab514799863.jpg", 176),
)

GT_COLOR = "#008C83"
PRED_COLOR = "#D1495B"
CROP_COLOR = "#E5A100"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def label_box(path: Path, width: int, height: int) -> np.ndarray:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) != 1:
        raise ValueError(f"Expected exactly one target in {path}, found {len(lines)}")
    cls, xc, yc, bw, bh = map(float, lines[0].split())
    if cls != 0:
        raise ValueError(f"Unexpected class {cls} in {path}")
    return np.array(
        [(xc - bw / 2) * width, (yc - bh / 2) * height,
         (xc + bw / 2) * width, (yc + bh / 2) * height],
        dtype=float,
    )


def iou(boxes: np.ndarray, gt: np.ndarray) -> np.ndarray:
    if len(boxes) == 0:
        return np.empty(0, dtype=float)
    lt = np.maximum(boxes[:, :2], gt[:2])
    rb = np.minimum(boxes[:, 2:4], gt[2:])
    intersection = np.maximum(rb - lt, 0).prod(axis=1)
    area = np.maximum(boxes[:, 2:4] - boxes[:, :2], 0).prod(axis=1)
    gt_area = np.maximum(gt[2:] - gt[:2], 0).prod()
    return intersection / np.maximum(area + gt_area - intersection, 1e-12)


def crop_box(gt: np.ndarray, side: int, width: int, height: int) -> tuple[int, int, int, int]:
    center = (gt[:2] + gt[2:]) / 2
    x0 = int(round(center[0] - side / 2))
    y0 = int(round(center[1] - side / 2))
    x0 = max(0, min(x0, width - side))
    y0 = max(0, min(y0, height - side))
    return x0, y0, x0 + side, y0 + side


def draw_box(ax, box: np.ndarray, offset: tuple[float, float], color: str, lw: float) -> None:
    x0, y0 = offset
    ax.add_patch(Rectangle(
        (box[0] - x0, box[1] - y0), box[2] - box[0], box[3] - box[1],
        fill=False, edgecolor=color, linewidth=lw, joinstyle="miter",
    ))


def all_prediction_boxes_in_crop(pred: np.ndarray, crop: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = crop
    centers = (pred[:, :2] + pred[:, 2:4]) / 2
    return pred[(pred[:, 4] >= CONFIDENCE)
                & (centers[:, 0] >= x0) & (centers[:, 0] <= x1)
                & (centers[:, 1] >= y0) & (centers[:, 1] <= y1)]


def source_record(filename: str, crop_side: int, cache: dict) -> dict:
    path = ROOT / "DetectDataset_clean_v2/test/images" / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as source_image:
        image = np.asarray(source_image.convert("RGB"))
    height, width = image.shape[:2]
    gt = label_box(path.parent.parent / "labels" / f"{path.stem}.txt", width, height)
    crop = crop_box(gt, crop_side, width, height)
    rel = str(path.relative_to(ROOT))
    models = {}
    for name in MODEL_KEYS:
        pred = np.asarray(cache[name][rel], dtype=float).reshape(-1, 5)
        visible = pred[pred[:, 4] >= CONFIDENCE]
        overlaps = iou(visible, gt)
        best_index = int(np.argmax(overlaps)) if len(overlaps) else None
        models[name] = {
            "all": pred,
            "best_iou": float(overlaps[best_index]) if best_index is not None else 0.0,
            "best_confidence": float(visible[best_index, 4]) if best_index is not None else 0.0,
            "best_box": visible[best_index, :4].tolist() if best_index is not None else None,
        }
    return {
        "path": path, "relative_path": rel, "image": image, "gt": gt,
        "crop": crop, "side_px": float(np.sqrt((gt[2] - gt[0]) * (gt[3] - gt[1]))),
        "models": models,
    }


def validate_selected(rows: list[dict]) -> None:
    for row in rows[:2]:
        baseline = row["models"]["YOLO11n"]["best_iou"]
        improved = row["models"]["+P2+ADown"]
        if not (16 <= row["side_px"] < 32 and 0.5 <= baseline < 0.75
                and improved["best_iou"] >= 0.75 and improved["best_confidence"] >= 0.4):
            raise AssertionError(f"Positive example no longer satisfies audit rule: {row['path']}")
    negative = rows[2]
    if not (16 <= negative["side_px"] < 96
            and negative["models"]["YOLO11n"]["best_iou"] >= 0.75
            and negative["models"]["YOLO11n"]["best_confidence"] >= 0.4
            and negative["models"]["+P2+ADown"]["best_iou"] < 0.5):
        raise AssertionError("Boundary case no longer satisfies audit rule")


def screening_counts(cache: dict) -> dict[str, int]:
    """Count qualifying original-source targets before image-quality selection."""
    counts = {"source_images": 0, "targets": 0, "small_localization_gains": 0,
              "comparable_size_regressions": 0}
    for rel in cache["YOLO11n"]:
        if not rel.startswith("DetectDataset_clean_v2/test/images/"):
            continue
        counts["source_images"] += 1
        path = ROOT / rel
        with Image.open(path) as image:
            width, height = image.size
        lines = (path.parent.parent / "labels" / f"{path.stem}.txt").read_text(
            encoding="utf-8"
        ).splitlines()
        base = np.asarray(cache["YOLO11n"][rel], dtype=float).reshape(-1, 5)
        adown = np.asarray(cache["+P2+ADown"][rel], dtype=float).reshape(-1, 5)
        for line in lines:
            cls, xc, yc, bw, bh = map(float, line.split())
            if cls != 0:
                raise ValueError(f"Unexpected class in {path}")
            gt = np.array([(xc - bw / 2) * width, (yc - bh / 2) * height,
                           (xc + bw / 2) * width, (yc + bh / 2) * height])
            side = float(np.sqrt((gt[2] - gt[0]) * (gt[3] - gt[1])))
            base_kept = base[base[:, 4] >= CONFIDENCE]
            adown_kept = adown[adown[:, 4] >= CONFIDENCE]
            base_iou = float(max(iou(base_kept, gt), default=0.0))
            adown_ious = iou(adown_kept, gt)
            adown_iou = float(max(adown_ious, default=0.0))
            adown_conf = float(adown_kept[int(np.argmax(adown_ious)), 4]) if len(adown_ious) else 0.0
            base_ious = iou(base_kept, gt)
            base_conf = float(base_kept[int(np.argmax(base_ious)), 4]) if len(base_ious) else 0.0
            counts["targets"] += 1
            if (16 <= side < 32 and 0.5 <= base_iou < 0.75
                    and adown_iou >= 0.75 and adown_conf >= 0.4):
                counts["small_localization_gains"] += 1
            if (16 <= side < 96 and base_iou >= 0.75
                    and base_conf >= 0.4 and adown_iou < 0.5):
                counts["comparable_size_regressions"] += 1
    return counts


def build() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cache_path = SOURCE / "raw_predictions.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    counts = screening_counts(cache)
    rows = [source_record(filename, side, cache) for _, _, filename, side in EXAMPLES]
    validate_selected(rows)

    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 7, "svg.fonttype": "none", "pdf.fonttype": 42,
        "axes.linewidth": 0.6, "savefig.facecolor": "white",
    })
    fig = plt.figure(figsize=(7.2, 5.35), facecolor="white")
    gs = fig.add_gridspec(3, 4, left=0.055, right=0.985, bottom=0.092,
                          top=0.855, wspace=0.09, hspace=0.35)
    axes = [[fig.add_subplot(gs[r, c]) for c in range(4)] for r in range(3)]
    for row_index, ((letter, description, _, _), record) in enumerate(zip(EXAMPLES, rows)):
        image = record["image"]
        x0, y0, x1, y1 = record["crop"]
        context = axes[row_index][0]
        context.imshow(image, interpolation="nearest")
        context.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                    edgecolor=CROP_COLOR, linewidth=1.0))
        draw_box(context, record["gt"], (0, 0), GT_COLOR, 0.65)
        context.set_axis_off()
        context.set_title(f"{letter}  {description}  ({record['side_px']:.0f} px)",
                          loc="left", fontsize=7, fontweight="bold", pad=5)

        for model_index, name in enumerate(MODEL_KEYS, 1):
            ax = axes[row_index][model_index]
            ax.imshow(image[y0:y1, x0:x1], interpolation="nearest")
            draw_box(ax, record["gt"], (x0, y0), GT_COLOR, 0.9)
            for pred in all_prediction_boxes_in_crop(record["models"][name]["all"], record["crop"]):
                draw_box(ax, pred[:4], (x0, y0), PRED_COLOR, 0.9)
            ax.set_xlim(0, x1 - x0)
            ax.set_ylim(y1 - y0, 0)
            ax.set_axis_off()
            best = record["models"][name]
            summary = (f"IoU {best['best_iou']:.2f}  |  conf {best['best_confidence']:.2f}"
                       if best["best_iou"] >= 0.5 else "No IoU≥0.50 detection")
            ax.text(0.5, -0.095, summary, transform=ax.transAxes,
                    ha="center", va="top", fontsize=6.5, color="black")

    fig.canvas.draw()
    for column, header in enumerate(("Scene and crop", *MODEL_TITLES)):
        box = axes[0][column].get_position()
        fig.text((box.x0 + box.x1) / 2, 0.892, header,
                 ha="center", va="bottom", fontsize=7.3, fontweight="bold")
    fig.text(0.055, 0.958, "Matched 200-epoch seed-1 checkpoints | original-source test images",
             ha="left", va="top", fontsize=8, fontweight="bold")
    fig.text(0.055, 0.93,
             "GT: teal solid box    Prediction: red solid box    Crop boundary: amber    Display confidence ≥0.25",
             ha="left", va="top", fontsize=6.6)
    fig.text(0.055, 0.035,
             "Identical native-pixel crop across models within each row; no image enhancement. Examples are illustrative, not a population estimate.",
             ha="left", va="bottom", fontsize=6.3)

    prefix = OUTPUT / "seed1_200_qualitative"
    require_matplotlib_panel_alignment(
        fig, json_out=str(prefix) + ".alignment.json",
        overlay_svg=str(prefix) + ".alignment-overlay.svg",
        tolerance_pt=1.5, gutter_tolerance_pt=1.5,
        require_panel_labels=False, strict=True,
    )
    fig.savefig(str(prefix) + ".pdf", dpi=300)
    fig.savefig(str(prefix) + ".svg", dpi=300)
    fig.savefig(str(prefix) + ".png", dpi=300)
    fig.savefig(str(prefix) + ".tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)

    manifest = {
        "source": "200-epoch validation-selected seed-1 best.pt predictions",
        "split": "DetectDataset_clean_v2/test (original-source part of the 641-image full test)",
        "full_test_images": 641, "full_test_instances": 701,
        "prediction_cache": str(cache_path.relative_to(ROOT)),
        "prediction_cache_sha256": sha256(cache_path),
        "source_checkpoint_records": {
            name: json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8"))["weights"][name]
            for name in MODEL_KEYS
        },
        "display_confidence": CONFIDENCE,
        "screening_counts": counts,
        "selection": {
            "type": "exploratory, post-hoc qualitative illustration",
            "positive_rule": "16–32 px equivalent side; baseline IoU in [0.50,0.75); P2+ADown IoU>=0.75, confidence>=0.40",
            "negative_rule": "baseline IoU>=0.75; P2+ADown IoU<0.50; comparable target size",
            "visual_exclusions": "avoided visible stock watermarks; used different video/source IDs for the two gains",
            "not_claimed": "representative random sample, independent test evidence, module mechanism proof",
        },
        "images": [{
            "panel": letter, "role": description,
            "path": row["relative_path"], "sha256": sha256(row["path"]),
            "gt_xyxy": row["gt"].tolist(), "equivalent_side_px": row["side_px"],
            "crop_xyxy": list(row["crop"]),
            "models": {name: {
                "best_iou_at_conf025": row["models"][name]["best_iou"],
                "best_confidence": row["models"][name]["best_confidence"],
                "best_box_xyxy": row["models"][name]["best_box"],
            } for name in MODEL_KEYS},
        } for (letter, description, _, _), row in zip(EXAMPLES, rows)],
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {prefix}")


if __name__ == "__main__":
    build()
