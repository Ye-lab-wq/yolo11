#!/usr/bin/env python3
"""Audit blur proxies and MSEFPaper effects without rerunning inference.

The variance of the Laplacian is used only as an exploratory sharpness proxy.
It is content dependent: low-texture, dark, defocused, and motion-blurred images
can all receive low scores.  Therefore the script uses rank-based thirds and
exports review montages instead of declaring an arbitrary score a blur label.
"""

from __future__ import annotations

import contextlib
import csv
import io
import json
import sys
from collections import OrderedDict
from pathlib import Path

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reproduction.ablation.evaluate_frozen import coco_metrics, make_ground_truth


DATASET = ROOT / "DetectDataset_clean_v2"
RAW_PREDICTIONS = (
    ROOT
    / "reproduction/results/p2_msef_adown_visual_evidence_seed1/raw_predictions.json"
)
OUTPUT_DIR = ROOT / "reproduction/results/blur_msef_audit_seed1"
SPLITS = OrderedDict(
    [
        ("train", DATASET / "train/images"),
        ("val", DATASET / "valid/images"),
        ("public_test", DATASET / "test/images"),
        ("self_test", DATASET / "self_test/images"),
    ]
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MODELS = ("+P2", "+P2+MSEF")


def paths_in(directory: Path) -> list[Path]:
    return sorted(
        path.resolve()
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def label_summary(path: Path, width: int, height: int) -> tuple[int, float | None]:
    label_path = path.parent.parent / "labels" / f"{path.stem}.txt"
    sides = []
    if label_path.is_file():
        for line in label_path.read_text(encoding="utf-8").splitlines():
            values = line.split()
            if len(values) != 5:
                raise ValueError(f"Invalid label in {label_path}: {line!r}")
            _, _, _, box_width, box_height = map(float, values)
            sides.append(np.sqrt(box_width * width * box_height * height))
    return len(sides), float(np.median(sides)) if sides else None


def sharpness_record(path: Path, split: str) -> dict[str, object]:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read {path}")
    height, width = bgr.shape[:2]
    # Fixed dimensions make the proxy more comparable across source resolutions.
    gray = cv2.cvtColor(cv2.resize(bgr, (640, 640), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F, ksize=3).var())
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    tenengrad = float(np.mean(gx * gx + gy * gy))
    instances, median_target_side = label_summary(path, width, height)
    return {
        "split": split,
        "path": str(path.relative_to(ROOT)),
        "width": width,
        "height": height,
        "instances": instances,
        "median_target_side_px": median_target_side,
        "laplacian_variance_640": laplacian_variance,
        "tenengrad_640": tenengrad,
    }


def detection_records(
    model_predictions: dict[str, list[list[float]]], paths: list[Path], path_to_id: dict[str, int]
) -> list[dict[str, object]]:
    records = []
    for path in paths:
        relative = str(path.relative_to(ROOT))
        for prediction in model_predictions[relative]:
            x1, y1, x2, y2, score = map(float, prediction)
            records.append(
                {
                    "image_id": path_to_id[str(path)],
                    "category_id": 1,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": score,
                }
            )
    return records


def evaluate_bins(
    rows: list[dict[str, object]], raw: dict[str, dict[str, list[list[float]]]]
) -> list[dict[str, object]]:
    test_rows = [row for row in rows if row["split"] in {"public_test", "self_test"}]
    test_rows.sort(key=lambda row: (float(row["laplacian_variance_640"]), str(row["path"])))
    index_groups = np.array_split(np.arange(len(test_rows)), 3)
    names = ("lower-sharpness third", "middle third", "higher-sharpness third")
    results = []
    for name, indices in zip(names, index_groups):
        subset_rows = [test_rows[int(index)] for index in indices]
        subset_paths = [(ROOT / str(row["path"])).resolve() for row in subset_rows]
        with contextlib.redirect_stdout(io.StringIO()):
            ground_truth, path_to_id = make_ground_truth(subset_paths)
        sides = [
            float(row["median_target_side_px"])
            for row in subset_rows
            if row["median_target_side_px"] is not None
        ]
        for model in MODELS:
            records = detection_records(raw[model], subset_paths, path_to_id)
            with contextlib.redirect_stdout(io.StringIO()):
                metrics = coco_metrics(ground_truth, records)
            results.append(
                {
                    "sharpness_bin": name,
                    "model": model,
                    "images": len(subset_paths),
                    "instances": int(metrics["instances"]),
                    "laplacian_min": min(float(row["laplacian_variance_640"]) for row in subset_rows),
                    "laplacian_median": float(
                        np.median([float(row["laplacian_variance_640"]) for row in subset_rows])
                    ),
                    "laplacian_max": max(float(row["laplacian_variance_640"]) for row in subset_rows),
                    "median_target_side_px": float(np.median(sides)) if sides else None,
                    "AP50_95_percent": 100.0 * float(metrics["AP"]),
                    "AP50_percent": 100.0 * float(metrics["AP50"]),
                    "AP75_percent": 100.0 * float(metrics["AP75"]),
                    "AP_16_32_percent": 100.0 * float(metrics["AP_small_16_32"]),
                }
            )
    return results


def save_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def distribution_figure(rows: list[dict[str, object]], output_dir: Path) -> None:
    with mpl.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
        }
    ):
        fig, axes = plt.subplots(1, 2, figsize=(7.08, 2.7), layout="constrained")
        for split, color in zip(SPLITS, ("#0072B2", "#E69F00", "#009E73", "#CC79A7")):
            values = np.asarray(
                [float(row["laplacian_variance_640"]) for row in rows if row["split"] == split]
            )
            axes[0].plot(
                np.linspace(0, 100, len(values)),
                np.sort(values),
                label=f"{split} (n={len(values)})",
                color=color,
                linewidth=1.5,
            )
        axes[0].set(
            xlabel="Percentile (%)",
            ylabel="Laplacian variance at 640 px",
            title="a  Sharpness-proxy distributions",
            yscale="log",
        )
        axes[0].grid(axis="y", color="#DDDDDD", linewidth=0.6)
        axes[0].legend(frameon=False, fontsize=6.8)

        test_rows = [row for row in rows if row["split"] in {"public_test", "self_test"}]
        axes[1].scatter(
            [float(row["laplacian_variance_640"]) for row in test_rows],
            [float(row["tenengrad_640"]) for row in test_rows],
            s=7,
            alpha=0.45,
            color="#0072B2",
            edgecolors="none",
        )
        axes[1].set(
            xlabel="Laplacian variance at 640 px",
            ylabel="Tenengrad at 640 px",
            title="b  Two content-dependent proxies",
            xscale="log",
            yscale="log",
        )
        axes[1].grid(color="#DDDDDD", linewidth=0.6)
        fig.savefig(output_dir / "blur_proxy_distributions.png", dpi=400)
        fig.savefig(output_dir / "blur_proxy_distributions.pdf")
        plt.close(fig)


def review_figure(rows: list[dict[str, object]], output_dir: Path) -> None:
    test_rows = sorted(
        (row for row in rows if row["split"] in {"public_test", "self_test"}),
        key=lambda row: float(row["laplacian_variance_640"]),
    )
    n = len(test_rows)
    # Deterministic samples from the lower tail, around the median, and upper tail.
    indices = [0, max(1, n // 30), max(2, n // 10), n // 2 - 1, n // 2, n // 2 + 1, n - 3, n - 2, n - 1]
    labels = ["lower proxy"] * 3 + ["median proxy"] * 3 + ["higher proxy"] * 3
    with mpl.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
        }
    ):
        fig, axes = plt.subplots(3, 3, figsize=(7.08, 5.35), layout="constrained")
        for ax, index, label in zip(axes.flat, indices, labels):
            row = test_rows[index]
            with Image.open(ROOT / str(row["path"])) as image:
                ax.imshow(image.convert("RGB"))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(
                f"{label} | LapVar={float(row['laplacian_variance_640']):.1f}\n"
                f"{Path(str(row['path'])).name[:30]}",
                fontsize=6.4,
            )
        fig.suptitle(
            "Deterministic sharpness-proxy review (low score does not prove motion blur)",
            fontsize=8,
        )
        fig.savefig(output_dir / "blur_proxy_review.png", dpi=400)
        fig.savefig(output_dir / "blur_proxy_review.pdf")
        plt.close(fig)


def performance_figure(results: list[dict[str, object]], output_dir: Path) -> None:
    bins = list(dict.fromkeys(str(row["sharpness_bin"]) for row in results))
    metrics = ("AP50_95_percent", "AP50_percent", "AP75_percent")
    labels = ("AP50:95", "AP50", "AP75")
    colors = {"+P2": "#0072B2", "+P2+MSEF": "#CC79A7"}
    with mpl.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 8.5,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
        }
    ):
        fig, axes = plt.subplots(1, 3, figsize=(7.08, 2.45), layout="constrained")
        x = np.arange(len(bins))
        for ax, metric, label in zip(axes, metrics, labels):
            for offset, model in zip((-0.18, 0.18), MODELS):
                values = [
                    float(next(row[metric] for row in results if row["sharpness_bin"] == bin_name and row["model"] == model))
                    for bin_name in bins
                ]
                ax.bar(x + offset, values, 0.36, color=colors[model], label=model, edgecolor="black", linewidth=0.35)
            ax.set_xticks(x, ["lower", "middle", "higher"])
            ax.set(ylabel=f"{label} (%)", title=label)
            ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
        axes[0].legend(frameon=False, fontsize=7)
        fig.suptitle("Seed-1 MSEFPaper audit by image sharpness-proxy third", fontsize=8.5)
        fig.savefig(output_dir / "msef_by_blur_proxy.png", dpi=400)
        fig.savefig(output_dir / "msef_by_blur_proxy.pdf")
        plt.close(fig)


def write_report(rows: list[dict[str, object]], results: list[dict[str, object]], output_dir: Path) -> None:
    lines = [
        "# Blur-proxy and MSEFPaper audit",
        "",
        "This is an exploratory audit, not a blur-label ground truth. Variance of the Laplacian and Tenengrad are content-dependent: low texture, darkness, defocus, compression, and motion blur can overlap. Images are resized to 640x640 before scoring to reduce source-resolution confounding.",
        "",
        "## Dataset proxy summary",
        "",
        "| Split | Images | LapVar P10 | Median | P90 |",
        "|---|---:|---:|---:|---:|",
    ]
    for split in SPLITS:
        values = np.asarray(
            [float(row["laplacian_variance_640"]) for row in rows if row["split"] == split]
        )
        lines.append(
            f"| {split} | {len(values)} | {np.percentile(values, 10):.1f} | {np.median(values):.1f} | {np.percentile(values, 90):.1f} |"
        )
    lines.extend(
        [
            "",
            "## MSEFPaper by rank-based proxy third",
            "",
            "| Proxy third | Images / instances | Model | AP50:95/% | AP50/% | AP75/% | AP 16-32/% |",
            "|---|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in results:
        lines.append(
            f"| {row['sharpness_bin']} | {row['images']} / {row['instances']} | {row['model']} | "
            f"{row['AP50_95_percent']:.2f} | {row['AP50_percent']:.2f} | {row['AP75_percent']:.2f} | {row['AP_16_32_percent']:.2f} |"
        )
    lines.extend(
        [
            "",
            "Interpretation rule: a blur-targeting mechanism is supported only if its direct-parent gain is stronger and consistent in an independently defined blurred subset. A favorable lower-proxy result here is hypothesis-generating because the bins are proxy-defined and composition (including target size) may differ.",
            "",
            "The audit uses frozen seed-1 predictions generated at confidence floor 0.001 and the same COCO-style evaluator as the five-model evidence package.",
        ]
    )
    (output_dir / "BLUR_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [
        sharpness_record(path, split)
        for split, directory in SPLITS.items()
        for path in paths_in(directory)
    ]
    raw = json.loads(RAW_PREDICTIONS.read_text(encoding="utf-8"))
    results = evaluate_bins(rows, raw)
    save_csv(OUTPUT_DIR / "image_blur_proxies.csv", rows)
    save_csv(OUTPUT_DIR / "msef_blur_proxy_results.csv", results)
    distribution_figure(rows, OUTPUT_DIR)
    review_figure(rows, OUTPUT_DIR)
    performance_figure(results, OUTPUT_DIR)
    write_report(rows, results, OUTPUT_DIR)
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
