#!/usr/bin/env python3
"""Create lossless audit overlays for annotation and split-leakage examples.

The stored YOLO boxes are drawn directly from the label files. Yellow dashed
boxes are explicitly marked as manual audit candidates and are never written
back to the dataset.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "DetectDataset"
OUTPUT = Path(__file__).resolve().parent / "results" / "audit_examples"
SPLITS = ("train", "valid", "test")
RF_SUFFIX = re.compile(r"\.rf\.[0-9a-f]+$")
VIDEO_FRAME = re.compile(r"^(video\d+)_(\d+)(?:_|$)", re.IGNORECASE)

COLORS = {
    "gt": "#D55E00",       # Okabe-Ito vermillion
    "second": "#0072B2",   # Okabe-Ito blue
    "candidate": "#F0E442",# Okabe-Ito yellow
    "text": "#111111",
    "panel": "#F5F5F5",
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    return ImageFont.truetype(str(path), size=size)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def base_name(path: Path) -> str:
    return RF_SUFFIX.sub("", path.stem)


def label_path(image_path: Path) -> Path:
    return image_path.parent.parent / "labels" / f"{image_path.stem}.txt"


def yolo_boxes(image_path: Path) -> list[tuple[float, float, float, float]]:
    with Image.open(image_path) as image:
        width, height = image.size
    path = label_path(image_path)
    boxes = []
    if not path.exists():
        return boxes
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) != 5:
            continue
        _, cx, cy, bw, bh = map(float, fields)
        boxes.append((
            (cx - bw / 2) * width,
            (cy - bh / 2) * height,
            (cx + bw / 2) * width,
            (cy + bh / 2) * height,
        ))
    return boxes


def label_text(image_path: Path) -> str:
    path = label_path(image_path)
    return path.read_text(encoding="utf-8").strip() if path.exists() else ""


def draw_dashed_rectangle(
    draw: ImageDraw.ImageDraw,
    box: tuple[float, float, float, float],
    fill: str,
    width: int = 4,
    dash: int = 12,
) -> None:
    x1, y1, x2, y2 = map(int, box)
    for start in range(x1, x2, dash * 2):
        draw.line((start, y1, min(start + dash, x2), y1), fill=fill, width=width)
        draw.line((start, y2, min(start + dash, x2), y2), fill=fill, width=width)
    for start in range(y1, y2, dash * 2):
        draw.line((x1, start, x1, min(start + dash, y2)), fill=fill, width=width)
        draw.line((x2, start, x2, min(start + dash, y2)), fill=fill, width=width)


def draw_caption(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, color: str) -> None:
    used_font = font(18, bold=True)
    x, y = xy
    bounds = draw.textbbox((x, y), text, font=used_font, stroke_width=0)
    pad = 4
    draw.rectangle((bounds[0] - pad, bounds[1] - pad, bounds[2] + pad, bounds[3] + pad), fill="#FFFFFFE8")
    draw.text((x, y), text, fill=color, font=used_font)


def annotated_image(
    path: Path,
    *,
    stored_label: str = "Stored GT",
    second_boxes: list[tuple[float, float, float, float]] | None = None,
    second_label: str = "Comparison",
    candidates: list[tuple[float, float, float, float]] | None = None,
) -> Image.Image:
    image = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for index, box in enumerate(yolo_boxes(path), 1):
        draw.rectangle(box, outline=COLORS["gt"], width=4)
        draw_caption(draw, (int(box[0]) + 3, max(3, int(box[1]) + 3)), f"{stored_label} {index}", COLORS["gt"])
    for index, box in enumerate(second_boxes or [], 1):
        draw.rectangle(box, outline=COLORS["second"], width=4)
        draw_caption(draw, (int(box[0]) + 3, max(3, int(box[1]) + 30)), f"{second_label} {index}", COLORS["second"])
    for index, box in enumerate(candidates or [], 1):
        draw_dashed_rectangle(draw, box, COLORS["candidate"], width=5)
        draw_caption(draw, (int(box[0]) + 3, max(3, int(box[1]) + 3)), f"Manual candidate {index} (not GT)", "#806D00")
    return image


def fit(image: Image.Image, size: tuple[int, int], background: str = "white") -> Image.Image:
    result = Image.new("RGB", size, background)
    copy = image.copy()
    copy.thumbnail(size, Image.Resampling.LANCZOS)
    result.paste(copy, ((size[0] - copy.width) // 2, (size[1] - copy.height) // 2))
    return result


def crop_union(image: Image.Image, boxes: list[tuple[float, float, float, float]]) -> Image.Image:
    if not boxes:
        return image.copy()
    x1 = min(box[0] for box in boxes)
    y1 = min(box[1] for box in boxes)
    x2 = max(box[2] for box in boxes)
    y2 = max(box[3] for box in boxes)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(150.0, x2 - x1, y2 - y1) * 1.6
    left = max(0, min(image.width - side, cx - side / 2))
    top = max(0, min(image.height - side, cy - side / 2))
    right = min(image.width, left + side)
    bottom = min(image.height, top + side)
    return image.crop((int(left), int(top), int(right), int(bottom)))


def issue_figure(
    output_name: str,
    path: Path,
    title: str,
    note: str,
    *,
    candidates: list[tuple[float, float, float, float]] | None = None,
    second_boxes: list[tuple[float, float, float, float]] | None = None,
    second_label: str = "Comparison",
    stored_label: str = "Stored GT",
    focus_boxes: list[tuple[float, float, float, float]] | None = None,
) -> None:
    annotated = annotated_image(
        path,
        stored_label=stored_label,
        second_boxes=second_boxes,
        second_label=second_label,
        candidates=candidates,
    )
    all_boxes = yolo_boxes(path) + (second_boxes or []) + (candidates or [])
    # For suspected omissions, zoom the unlabelled candidate itself instead of
    # spanning the (often distant) stored GT and making both objects tiny.
    crop_targets = focus_boxes or ((candidates or []) if candidates else all_boxes)
    crop = crop_union(annotated, crop_targets)
    canvas = Image.new("RGB", (1640, 930), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((40, 24), title, fill=COLORS["text"], font=font(30, bold=True))
    draw.text((40, 70), note, fill=COLORS["text"], font=font(20))
    draw.text((40, 104), str(path.relative_to(ROOT)), fill="#555555", font=font(17))
    canvas.paste(fit(annotated, (760, 720), COLORS["panel"]), (40, 150))
    canvas.paste(fit(crop, (760, 720), COLORS["panel"]), (840, 150))
    draw.text((40, 880), "Full image", fill=COLORS["text"], font=font(20, bold=True))
    draw.text((840, 880), "Audit zoom", fill=COLORS["text"], font=font(20, bold=True))
    canvas.save(OUTPUT / output_name, optimize=True)


def inventory() -> list[dict[str, object]]:
    rows = []
    for split in SPLITS:
        for path in sorted((DATASET / split / "images").iterdir()):
            if not path.is_file():
                continue
            base = base_name(path)
            match = VIDEO_FRAME.match(base)
            rows.append({
                "split": split,
                "path": path,
                "sha256": sha256(path),
                "base": base,
                "video": match.group(1).lower() if match else None,
                "frame": int(match.group(2)) if match else None,
                "label": label_text(path),
            })
    return rows


def pick_pair(
    rows: list[dict[str, object]],
    left_split: str,
    right_split: str,
    group_key: str,
    *,
    labels_equal: bool | None = None,
    hashes_equal: bool | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row[group_key])].append(row)
    for key in sorted(groups):
        left = [row for row in groups[key] if row["split"] == left_split]
        right = [row for row in groups[key] if row["split"] == right_split]
        for a in left:
            for b in right:
                if labels_equal is not None and ((a["label"] == b["label"]) != labels_equal):
                    continue
                if hashes_equal is not None and ((a["sha256"] == b["sha256"]) != hashes_equal):
                    continue
                return a, b
    raise RuntimeError(f"No pair found: {left_split}/{right_split}/{group_key}")


def pick_within_duplicate(rows: list[dict[str, object]], split: str) -> tuple[dict[str, object], dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row["split"] == split:
            groups[str(row["sha256"])].append(row)
    candidates = [group for group in groups.values() if len(group) >= 2]
    candidates.sort(key=lambda group: str(group[0]["path"]))
    return candidates[0][0], candidates[0][1]


def pick_adjacent_video(rows: list[dict[str, object]]) -> tuple[dict[str, object], dict[str, object]]:
    train = [row for row in rows if row["split"] == "train" and row["video"] is not None]
    test = [row for row in rows if row["split"] == "test" and row["video"] is not None]
    pairs = []
    for left in train:
        for right in test:
            if left["video"] == right["video"] and abs(int(left["frame"]) - int(right["frame"])) == 1:
                if left["sha256"] != right["sha256"]:
                    pairs.append((left, right))
    pairs.sort(key=lambda pair: (str(pair[0]["video"]), int(pair[0]["frame"]), str(pair[0]["path"])))
    if not pairs:
        raise RuntimeError("No adjacent train/test video-frame pair found")
    return pairs[0]


def pick_same_video_frame_different_export(
    rows: list[dict[str, object]],
) -> tuple[dict[str, object], dict[str, object]]:
    train = [row for row in rows if row["split"] == "train" and row["video"] is not None]
    test = [row for row in rows if row["split"] == "test" and row["video"] is not None]
    pairs = []
    for left in train:
        for right in test:
            same_source_frame = left["video"] == right["video"] and left["frame"] == right["frame"]
            if same_source_frame and left["sha256"] != right["sha256"]:
                pairs.append((left, right))
    pairs.sort(key=lambda pair: (str(pair[0]["video"]), int(pair[0]["frame"]), str(pair[0]["path"])))
    if not pairs:
        raise RuntimeError("No same-number train/test video frame with different bytes found")
    return pairs[0]


def pair_panel(pair: tuple[dict[str, object], dict[str, object]], title: str, note: str) -> Image.Image:
    left, right = pair
    panel = Image.new("RGB", (1740, 560), "white")
    draw = ImageDraw.Draw(panel)
    draw.text((30, 18), title, fill=COLORS["text"], font=font(25, bold=True))
    draw.text((30, 55), note, fill="#333333", font=font(17))
    for x, row in ((30, left), (885, right)):
        image = annotated_image(Path(row["path"]))
        panel.paste(fit(image, (825, 405), COLORS["panel"]), (x, 92))
        rel = str(Path(row["path"]).relative_to(ROOT))
        draw.text((x, 505), rel, fill=COLORS["text"], font=font(14))
        draw.text((x, 530), f"SHA256 {str(row['sha256'])[:16]}...", fill="#555555", font=font(14))
    return panel


def false_positive_candidate_sheet() -> None:
    """Render all high-confidence, zero-overlap baseline predictions for review."""
    source = Path(__file__).resolve().parent / "results" / "failure_analysis_test.json"
    report = json.loads(source.read_text(encoding="utf-8"))
    records = [
        row for row in report["models"]["YOLO11n"]["fixed_threshold"]["top_false_positives"]
        if row["error_type"] == "background_or_unlabeled"
    ]
    columns = 3
    cell_width, cell_height = 590, 510
    rows = (len(records) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_width, 70 + rows * cell_height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((25, 15), "YOLO11n unmatched predictions: manual false-positive vs missing-label review", fill=COLORS["text"], font=font(25, bold=True))
    for index, row in enumerate(records):
        path = ROOT / row["path"]
        predicted_box = tuple(float(value) for value in row["box"])
        image = annotated_image(
            path,
            second_boxes=[predicted_box],
            second_label=f"Pred conf={row['confidence']:.2f}",
        )
        panel = fit(image, (560, 420), COLORS["panel"])
        x = (index % columns) * cell_width + 15
        y = 70 + (index // columns) * cell_height
        sheet.paste(panel, (x, y))
        draw.text((x, y + 426), f"{index + 1}. {path.name}", fill=COLORS["text"], font=font(14, bold=True))
        draw.text((x, y + 450), f"pred={row['box']}  IoU={row['best_iou_to_any_gt']:.3f}", fill="#444444", font=font(12))
        draw.text((x, y + 472), "Blue=model prediction; red=stored GT", fill="#444444", font=font(13))
    sheet.save(OUTPUT / "false_positive_candidates_overview.png", optimize=True)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)

    issues = [
        (
            "annotation_01_loose_box_video16_673.png",
            DATASET / "test/images/video16_673_JPEG.rf.4bf0a8e50c2ed597ee11f8bba19bb29c.jpg",
            "Annotation audit 1: stored box appears substantially loose",
            "Red is the dataset GT. The visual drone occupies only part of the stored box.",
            [],
        ),
        (
            "annotation_02_partial_box_video14_204.png",
            DATASET / "test/images/video14_204_JPEG.rf.e0b2120015f89c44692af6b6f00f5fe4.jpg",
            "Annotation audit 2: stored box covers only part of a large drone",
            "The GT boundary is inconsistent with the full visible object extent.",
            [],
        ),
        (
            "annotation_05_ambiguous_region_pic1026.png",
            DATASET / "test/images/pic_1026_jpg.rf.155d6177f46e0e439462b8520176f77c.jpg",
            "Annotation audit 5: one very large region contains multiple visible structures",
            "The single stored GT is not a tight one-object box, making localization metrics ambiguous.",
            [],
        ),
        (
            "annotation_06_loose_box_video17_1060.png",
            DATASET / "test/images/video17_1060_JPEG.rf.7a071dc3aadcc1827863fe7d60260ae3.jpg",
            "Annotation audit 6: another loose-box example",
            "This kind of GT tightness variation can depress AP75 even when detection is visually plausible.",
            [],
        ),
    ]
    for name, path, title, note, candidates in issues:
        issue_figure(name, path, title, note, candidates=candidates)

    missing_1617 = (392.20, 282.48, 419.54, 313.33)
    issue_figure(
        "annotation_03_missing_label_video18_1617.png",
        DATASET / "test/images/video18_1617_JPEG.rf.24ae57f029ed398f60e4df69cee8c992.jpg",
        "Annotation audit 3: visible second drone has no stored GT",
        "Red is stored GT; blue is an unmatched YOLO11n prediction (confidence 0.57), not a label.",
        second_boxes=[missing_1617],
        second_label="Unmatched pred conf=0.57",
        focus_boxes=[missing_1617],
    )
    missing_553 = (348.05, 209.77, 379.70, 243.20)
    issue_figure(
        "annotation_04_missing_label_video18_553.png",
        DATASET / "test/images/video18_553_JPEG.rf.4e97ccd2ba4fe075be5f30ad83d6be57.jpg",
        "Annotation audit 4: one of several visible drones has no stored GT",
        "Red boxes are stored GT; blue is an unmatched YOLO11n prediction (confidence 0.39), not a label.",
        second_boxes=[missing_553],
        second_label="Unmatched pred conf=0.39",
        focus_boxes=[missing_553],
    )

    conflict_train = DATASET / "train/images/video14_212_JPEG.rf.af51a8b29dbb08ad839fd770d46d4270.jpg"
    conflict_test = DATASET / "test/images/video14_212_JPEG.rf.64e8c429d7a6e56881acb83bde412ada.jpg"
    issue_figure(
        "annotation_07_same_image_conflicting_train_test_gt.png",
        conflict_train,
        "Annotation audit 7: byte-identical image has different train/test boxes",
        "Red is the train GT; blue is the test GT copied onto the same pixels.",
        second_boxes=yolo_boxes(conflict_test),
        second_label="Test GT",
        stored_label="Train GT",
    )

    rows = inventory()
    pairs: list[tuple[str, str, tuple[dict[str, object], dict[str, object]]]] = []
    pairs.append((
        "Exact train-test duplicate, same labels",
        "Byte-identical pixels and identical YOLO label text occur in both model fitting and final testing.",
        pick_pair(rows, "train", "test", "sha256", labels_equal=True),
    ))
    pairs.append((
        "Exact train-test duplicate, conflicting labels",
        "Byte-identical pixels occur in both splits, but their YOLO label text differs.",
        pick_pair(rows, "train", "test", "sha256", labels_equal=False),
    ))
    pairs.append((
        "Exact train-valid duplicate",
        "Byte-identical pixels occur in both fitting and model-selection splits.",
        pick_pair(rows, "train", "valid", "sha256", labels_equal=True),
    ))
    pairs.append((
        "Exact valid-test duplicate",
        "Byte-identical pixels occur in both model-selection and final-test splits.",
        pick_pair(rows, "valid", "test", "sha256", labels_equal=True),
    ))
    pairs.append((
        "Within-train exact duplicate",
        "Repeated training pixels reweight this sample relative to unique samples.",
        pick_within_duplicate(rows, "train"),
    ))
    pairs.append((
        "Within-test exact duplicate",
        "Repeated test pixels make the reported sample count larger than the unique-image count.",
        pick_within_duplicate(rows, "test"),
    ))
    pairs.append((
        "Same video and frame ID, different train/test exports",
        "The source video/frame ID is identical but export names and bytes differ; this is source-level overlap.",
        pick_same_video_frame_different_export(rows),
    ))
    pairs.append((
        "Adjacent train/test video frames",
        "Consecutive frames from the same video source are separated across train and test.",
        pick_adjacent_video(rows),
    ))

    sheet = Image.new("RGB", (1800, 40 + len(pairs) * 580), "white")
    sheet_draw = ImageDraw.Draw(sheet)
    sheet_draw.text((30, 5), "Concrete split-leakage and duplication examples", fill=COLORS["text"], font=font(28, bold=True))
    manifest_rows = []
    for index, (title, note, pair) in enumerate(pairs):
        panel = pair_panel(pair, f"{index + 1}. {title}", note)
        sheet.paste(panel, (30, 40 + index * 580))
        left, right = pair
        manifest_rows.append({
            "type": title,
            "left": str(Path(left["path"]).relative_to(ROOT)),
            "right": str(Path(right["path"]).relative_to(ROOT)),
            "left_sha256": left["sha256"],
            "right_sha256": right["sha256"],
            "labels_equal": left["label"] == right["label"],
            "left_video": left["video"],
            "left_frame": left["frame"],
            "right_video": right["video"],
            "right_frame": right["frame"],
            "note": note,
        })
    sheet.save(OUTPUT / "leakage_examples_overview.png", optimize=True)
    false_positive_candidate_sheet()

    with (OUTPUT / "leakage_example_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    with (OUTPUT / "README.md").open("w", encoding="utf-8") as handle:
        handle.write("# Dataset audit examples\n\n")
        handle.write("Red boxes are stored YOLO GT. Dashed yellow boxes are manual audit candidates and were not added to the dataset.\n\n")
        handle.write("## Annotation examples\n\n")
        for name, _, title, note, _ in issues:
            handle.write(f"- `{name}` — {title}. {note}\n")
        handle.write("- `annotation_03_missing_label_video18_1617.png` — a visible second drone has no stored GT; blue is an unmatched baseline prediction.\n")
        handle.write("- `annotation_04_missing_label_video18_553.png` — one of several visible drones has no stored GT; blue is an unmatched baseline prediction.\n")
        handle.write("- `annotation_07_same_image_conflicting_train_test_gt.png` — byte-identical image with different train/test GT boxes.\n\n")
        handle.write("## Leakage examples\n\n")
        handle.write("See `leakage_examples_overview.png` and `leakage_example_manifest.csv` for exact paths and SHA-256 hashes.\n")

    print(f"Wrote audit figures to {OUTPUT}")


if __name__ == "__main__":
    main()
