#!/usr/bin/env python3
"""Compact journal-style P2/ADown comparison using frozen seed-1 predictions.

Publisher figure layouts were consulted for *structure* only: matched scenes,
adjacent method columns, and repeated equal-size zoom crops. No published image
assets or figure styling files are copied. The examples remain post-hoc.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from audit_panel_alignment import require_matplotlib_panel_alignment
from build_module_direct_evidence_seed1_200 import ROOT, SOURCE, OUT, CONF, record


GT = "#00A99D"
PRED = "#E15157"
ROI = "#F5B227"
ROWS = (
    ("video18_1628__8c8cb8266b2195d7.jpg", 0, 128, "YOLO11n", "+P2"),
    ("video17_146__e86d0aaa615e4daf.jpg", 0, 128, "+P2", "+P2+ADown"),
)


def add_box(ax, box: np.ndarray, offset: tuple[int, int], color: str, width: float) -> None:
    ax.add_patch(Rectangle((box[0] - offset[0], box[1] - offset[1]),
                           box[2] - box[0], box[3] - box[1],
                           fill=False, edgecolor=color, linewidth=width,
                           joinstyle="miter"))


def draw_panel(ax, row: dict, model: str, detail: bool) -> None:
    image = row["image"]
    x0, y0, x1, y1 = row["crop"]
    if detail:
        ax.imshow(image[y0:y1, x0:x1], interpolation="nearest")
        offset = (x0, y0)
        width, height = x1-x0, y1-y0
    else:
        ax.imshow(image, interpolation="nearest")
        offset = (0, 0)
        height, width = image.shape[:2]
        ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False,
                               edgecolor=ROI, linewidth=1.05, joinstyle="miter"))
    for gt in row["gt"]:
        if not detail or (gt[2] > x0 and gt[0] < x1 and gt[3] > y0 and gt[1] < y1):
            add_box(ax, gt, offset, GT, 0.8 if detail else 0.6)
    pred = row["models"][model]["pred"]
    if detail:
        centers = (pred[:, :2] + pred[:, 2:4]) / 2
        pred = pred[(pred[:, 4] >= CONF) & (centers[:, 0] >= x0)
                    & (centers[:, 0] <= x1) & (centers[:, 1] >= y0)
                    & (centers[:, 1] <= y1)]
    else:
        pred = pred[pred[:, 4] >= CONF]
    for box in pred:
        add_box(ax, box[:4], offset, PRED, 0.9 if detail else 0.6)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.4)
        spine.set_edgecolor("#B7BEC4")
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_xticks([])
    ax.set_yticks([])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cache = json.loads((SOURCE / "raw_predictions.json").read_text(encoding="utf-8"))
    rows = [record(cache, "", *spec) for spec in ROWS]
    if not (rows[0]["models"]["YOLO11n"]["best_iou"] < 0.5
            and rows[0]["models"]["+P2"]["best_iou"] >= 0.5
            and rows[1]["models"]["+P2"]["best_iou"] < 0.75
            and rows[1]["models"]["+P2+ADown"]["best_iou"] >= 0.75):
        raise AssertionError("A source example no longer satisfies the intended direct contrast")

    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 7, "svg.fonttype": "none", "pdf.fonttype": 42,
                         "axes.linewidth": 0.5, "savefig.facecolor": "white"})
    fig = plt.figure(figsize=(7.2, 5.15), facecolor="white")
    grid = fig.add_gridspec(2, 4, left=0.05, right=0.95, bottom=0.17,
                            top=0.875, wspace=0.045, hspace=0.55)
    axes = [[fig.add_subplot(grid[i, j]) for j in range(4)] for i in range(2)]
    for i, row in enumerate(rows):
        for j in range(4):
            model = row["parent"] if j < 2 else row["child"]
            draw_panel(axes[i][j], row, model, detail=j % 2 == 1)
            axes[i][j].set_title("Full frame" if j % 2 == 0 else "Detail (5×)",
                                 fontsize=6.2, color="#4C5963", pad=2)

    fig.canvas.draw()
    for i, row in enumerate(rows):
        left = axes[i][0].get_position()
        right = axes[i][3].get_position()
        parent_center = (axes[i][0].get_position().x0 + axes[i][1].get_position().x1) / 2
        child_center = (axes[i][2].get_position().x0 + right.x1) / 2
        label_y = left.y1 + (0.055 if i == 0 else 0.052)
        fig.text(left.x0, label_y, f"{chr(97+i)}", fontsize=8, fontweight="bold", va="center")
        fig.text(parent_center, label_y, "YOLO11n" if i == 0 else "YOLO11n + P2",
                 fontsize=7.4, fontweight="bold", ha="center", va="center")
        fig.text(child_center, label_y, "YOLO11n + P2" if i == 0 else "YOLO11n + P2 + ADown",
                 fontsize=7.4, fontweight="bold", ha="center", va="center")
        status_y = left.y0 - 0.019
        if i == 0:
            fig.text(parent_center, status_y, "Target below 0.25 threshold",
                     fontsize=6.2, ha="center", va="top", color="#4C5963")
            fig.text(child_center, status_y, "Detected: IoU 0.75, score 0.39",
                     fontsize=6.2, ha="center", va="top", color="#4C5963")
        else:
            fig.text(parent_center, status_y, "IoU 0.61, score 0.57",
                     fontsize=6.2, ha="center", va="top", color="#4C5963")
            fig.text(child_center, status_y, "IoU 0.78, score 0.57",
                     fontsize=6.2, ha="center", va="top", color="#4C5963")

    fig.text(0.073, 0.075, "Ground truth", fontsize=6.3, va="center")
    fig.text(0.226, 0.075, "Prediction", fontsize=6.3, va="center")
    fig.text(0.398, 0.075, "Detail ROI", fontsize=6.3, va="center")
    fig.text(0.57, 0.075, "Display confidence ≥0.25; no image enhancement",
             fontsize=6.2, color="#4C5963", ha="left", va="center")
    from matplotlib.lines import Line2D
    fig.add_artist(Line2D([0.05, 0.066], [0.075, 0.075], transform=fig.transFigure,
                          color=GT, linewidth=1.5))
    fig.add_artist(Line2D([0.203, 0.219], [0.075, 0.075], transform=fig.transFigure,
                          color=PRED, linewidth=1.5))
    fig.add_artist(Rectangle((0.375, 0.068), 0.012, 0.014, transform=fig.transFigure,
                             fill=False, edgecolor=ROI, linewidth=1.0))

    prefix = OUT / "journal_qualitative_p2_adown"
    require_matplotlib_panel_alignment(fig, json_out=str(prefix)+".alignment.json",
                                       overlay_svg=str(prefix)+".alignment-overlay.svg",
                                       tolerance_pt=1.5, gutter_tolerance_pt=1.5,
                                       require_panel_labels=False, strict=True)
    fig.savefig(str(prefix)+".pdf", dpi=300)
    fig.savefig(str(prefix)+".svg", dpi=300)
    fig.savefig(str(prefix)+".png", dpi=300)
    fig.savefig(str(prefix)+".tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print(f"Saved {prefix}")


if __name__ == "__main__":
    main()
