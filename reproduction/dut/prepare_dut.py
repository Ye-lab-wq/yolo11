#!/usr/bin/env python3
"""Convert the official DUT Anti-UAV VOC release to an audited YOLO dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import statistics
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw


SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="Directory containing train/img, train/xml, ...")
    parser.add_argument("--output", type=Path, required=True, help="YOLO dataset output directory")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def materialize_image(source: Path, target: Path) -> str:
    if target.exists():
        if target.stat().st_size != source.stat().st_size:
            raise RuntimeError(f"Existing image differs in size: {target}")
        return "existing"
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def make_preview(records: dict[str, list[dict]], output: Path) -> None:
    chosen: list[dict] = []
    for split in SPLITS:
        ordered = sorted(records[split], key=lambda item: item["relative_area"])
        for index in (0, len(ordered) // 2, len(ordered) - 1):
            chosen.append(ordered[index])

    tile_w, tile_h = 480, 320
    canvas = Image.new("RGB", (tile_w * 3, tile_h * 3), "white")
    for position, record in enumerate(chosen):
        image = Image.open(record["source_image"]).convert("RGB")
        draw = ImageDraw.Draw(image)
        for xmin, ymin, xmax, ymax in record["boxes"]:
            width = max(2, round(max(image.size) / 500))
            draw.rectangle((xmin, ymin, xmax, ymax), outline=(255, 0, 0), width=width)
        image.thumbnail((tile_w, tile_h - 24), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (tile_w, tile_h), "white")
        tile.paste(image, ((tile_w - image.width) // 2, 24 + (tile_h - 24 - image.height) // 2))
        ImageDraw.Draw(tile).text((8, 5), f'{record["split"]}/{record["name"]}', fill=(0, 0, 0))
        canvas.paste(tile, ((position % 3) * tile_w, (position // 3) * tile_h))
    canvas.save(output, quality=92)


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    (output / "images").mkdir(parents=True, exist_ok=True)
    (output / "labels").mkdir(parents=True, exist_ok=True)

    records: dict[str, list[dict]] = {split: [] for split in SPLITS}
    names = Counter()
    link_modes = Counter()
    digest_locations: dict[str, list[tuple[str, str]]] = defaultdict(list)
    split_summary: dict[str, dict] = {}
    repaired_boxes: list[dict] = []
    rejected_boxes: list[dict] = []

    for split in SPLITS:
        image_source = source / split / "img"
        xml_source = source / split / "xml"
        image_output = output / "images" / split
        label_output = output / "labels" / split
        image_output.mkdir(parents=True, exist_ok=True)
        label_output.mkdir(parents=True, exist_ok=True)

        image_paths = sorted(image_source.glob("*.jpg"))
        xml_paths = sorted(xml_source.glob("*.xml"))
        if len(image_paths) != len(xml_paths):
            raise RuntimeError(f"{split}: {len(image_paths)} images != {len(xml_paths)} XML files")

        object_counts: list[int] = []
        relative_areas: list[float] = []
        resized_widths: list[float] = []
        resized_heights: list[float] = []
        invalid_boxes = 0

        for image_path in image_paths:
            xml_path = xml_source / f"{image_path.stem}.xml"
            if not xml_path.exists():
                raise RuntimeError(f"Missing XML for {image_path}")
            root = ET.parse(xml_path).getroot()
            xml_width = int(root.findtext("size/width", "0"))
            xml_height = int(root.findtext("size/height", "0"))
            with Image.open(image_path) as image:
                width, height = image.size
            if (width, height) != (xml_width, xml_height):
                raise RuntimeError(
                    f"Dimension mismatch {image_path}: image={(width, height)} XML={(xml_width, xml_height)}"
                )

            boxes: list[tuple[float, float, float, float]] = []
            yolo_lines: list[str] = []
            for obj in root.findall("object"):
                class_name = (obj.findtext("name") or "").strip()
                names[class_name] += 1
                bbox = obj.find("bndbox")
                if bbox is None:
                    invalid_boxes += 1
                    continue
                xmin = float(bbox.findtext("xmin", "nan"))
                ymin = float(bbox.findtext("ymin", "nan"))
                xmax = float(bbox.findtext("xmax", "nan"))
                ymax = float(bbox.findtext("ymax", "nan"))
                original_box = (xmin, ymin, xmax, ymax)
                # The release contains 32 edge-touching VOC boxes whose maximum
                # coordinate is exactly image_size + 1. This is an unambiguous
                # inclusive-coordinate convention, so clip only this exact case.
                xmax = float(width) if xmax == width + 1 else xmax
                ymax = float(height) if ymax == height + 1 else ymax
                if (xmin, ymin, xmax, ymax) != original_box:
                    repaired_boxes.append(
                        {
                            "split": split,
                            "xml": xml_path.name,
                            "image_size": [width, height],
                            "original": original_box,
                            "repaired": [xmin, ymin, xmax, ymax],
                        }
                    )
                if not (0 <= xmin < xmax <= width and 0 <= ymin < ymax <= height):
                    invalid_boxes += 1
                    rejected_boxes.append(
                        {
                            "split": split,
                            "xml": xml_path.name,
                            "image_size": [width, height],
                            "box": [xmin, ymin, xmax, ymax],
                        }
                    )
                    continue
                boxes.append((xmin, ymin, xmax, ymax))
                box_width = xmax - xmin
                box_height = ymax - ymin
                x_center = (xmin + xmax) / (2 * width)
                y_center = (ymin + ymax) / (2 * height)
                yolo_lines.append(
                    f"0 {x_center:.8f} {y_center:.8f} {box_width / width:.8f} {box_height / height:.8f}"
                )
                relative_areas.append(box_width * box_height / (width * height))
                scale = 640 / max(width, height)
                resized_widths.append(box_width * scale)
                resized_heights.append(box_height * scale)

            object_counts.append(len(boxes))
            label_path = label_output / f"{image_path.stem}.txt"
            label_path.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")
            link_modes[materialize_image(image_path, image_output / image_path.name)] += 1
            digest_locations[sha256(image_path)].append((split, image_path.name))
            record_area = max(
                ((xmax - xmin) * (ymax - ymin) / (width * height) for xmin, ymin, xmax, ymax in boxes),
                default=0.0,
            )
            records[split].append(
                {
                    "split": split,
                    "name": image_path.name,
                    "source_image": str(image_path),
                    "width": width,
                    "height": height,
                    "boxes": boxes,
                    "relative_area": record_area,
                }
            )

        tiny = sum(w < 8 and h < 8 for w, h in zip(resized_widths, resized_heights))
        coco_small = sum(w * h < 32**2 for w, h in zip(resized_widths, resized_heights))
        coco_medium = sum(32**2 <= w * h < 96**2 for w, h in zip(resized_widths, resized_heights))
        coco_large = sum(w * h >= 96**2 for w, h in zip(resized_widths, resized_heights))
        split_summary[split] = {
            "images": len(image_paths),
            "objects": sum(object_counts),
            "empty_images": sum(count == 0 for count in object_counts),
            "multi_object_images": sum(count > 1 for count in object_counts),
            "invalid_boxes": invalid_boxes,
            "relative_area": {
                "min": min(relative_areas, default=0.0),
                "median": statistics.median(relative_areas) if relative_areas else 0.0,
                "p90": percentile(relative_areas, 0.90),
                "max": max(relative_areas, default=0.0),
            },
            "at_letterbox_640": {
                "tiny_both_dimensions_lt_8": tiny,
                "coco_small_area_lt_32_sq": coco_small,
                "coco_medium_area_32_sq_to_96_sq": coco_medium,
                "coco_large_area_ge_96_sq": coco_large,
            },
        }

    cross_split_duplicates = []
    for digest, locations in digest_locations.items():
        if len({split for split, _ in locations}) > 1:
            cross_split_duplicates.append({"sha256": digest, "locations": locations})

    data_yaml = (
        f"path: {output}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "names:\n"
        "  0: UAV\n"
    )
    (output / "data.yaml").write_text(data_yaml, encoding="utf-8")
    make_preview(records, output / "label_preview.jpg")
    summary = {
        "source": str(source),
        "output": str(output),
        "class_names": dict(names),
        "materialization": dict(link_modes),
        "splits": split_summary,
        "repaired_boundary_boxes": repaired_boxes,
        "rejected_invalid_boxes": rejected_boxes,
        "cross_split_exact_duplicate_groups": len(cross_split_duplicates),
        "cross_split_exact_duplicates": cross_split_duplicates,
    }
    (output / "audit.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
