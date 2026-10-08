#!/usr/bin/env python3
"""Import the recovered self-collected video frames as an external test set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = ROOT / "DetectDataset_clean_v1"
RECOVERY_MANIFEST = ROOT / "reproduction" / "results" / "removed_extra_104_manifest.json"
SOURCE_VIDEO = Path("/home/b520/Downloads/yuheping/original_vedio.mp4")
SUBSETS = ("self_train", "self_test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_label(path: Path) -> tuple[int, str]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for line_number, line in enumerate(lines, 1):
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{path}:{line_number}: expected five YOLO fields")
        class_value, x, y, width, height = map(float, fields)
        valid = class_value == 0 and width > 0 and height > 0
        valid = valid and 0 <= x <= 1 and 0 <= y <= 1
        valid = valid and x - width / 2 >= -1e-6 and y - height / 2 >= -1e-6
        valid = valid and x + width / 2 <= 1 + 1e-6 and y + height / 2 <= 1 + 1e-6
        if not valid:
            raise ValueError(f"{path}:{line_number}: invalid class, box size, or bounds")
    return len(lines), "\n".join(sorted(lines))


def video_metadata(path: Path) -> dict[str, object]:
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,avg_frame_rate:format=duration",
        "-of", "json", str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def main() -> None:
    dataset = parse_args().dataset.resolve()
    existing = [dataset / subset for subset in SUBSETS if (dataset / subset).exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing subsets: {existing}")
    if not dataset.is_dir() or not RECOVERY_MANIFEST.is_file() or not SOURCE_VIDEO.is_file():
        raise FileNotFoundError("dataset, recovery manifest, or source video is missing")

    recovery = json.loads(RECOVERY_MANIFEST.read_text(encoding="utf-8"))
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for record in recovery["records"]:
        image = Path(record["recoverable_image"])
        label = Path(record["recoverable_label"])
        if not image.is_file() or not label.is_file():
            raise FileNotFoundError(f"missing recovered pair: {image}, {label}")
        if sha256(image) != record["image_sha256"] or sha256(label) != record["label_sha256"]:
            raise RuntimeError(f"recovered file no longer matches recorded hash: {image}")
        groups[record["image_sha256"]].append(record)

    clean_hashes: set[str] = set()
    with (dataset / "split_manifest.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            clean_hashes.add(row["image_sha256"])
    overlap = sorted(clean_hashes.intersection(groups))
    if overlap:
        raise RuntimeError(f"self-video frames overlap public clean dataset: {overlap}")

    prepared: list[dict[str, object]] = []
    for image_hash, records in sorted(groups.items()):
        label_hashes = {record["label_sha256"] for record in records}
        if len(label_hashes) != 1:
            raise RuntimeError(f"same self-video image has conflicting labels: {image_hash}")
        representative = min(records, key=lambda record: record["recoverable_image"])
        image = Path(representative["recoverable_image"])
        label = Path(representative["recoverable_label"])
        instances, canonical_label = validate_label(label)
        with Image.open(image) as opened:
            width, height = opened.size
            opened.verify()
        duplicate_paths = sorted(record["recoverable_image"] for record in records)
        prepared.append(
            {
                "image_hash": image_hash,
                "label_hash": next(iter(label_hashes)),
                "image": image,
                "label": label,
                "canonical_label": canonical_label,
                "instances": instances,
                "width": width,
                "height": height,
                "duplicate_paths": duplicate_paths,
            }
        )

    for subset in SUBSETS:
        (dataset / subset / "images").mkdir(parents=True)
        (dataset / subset / "labels").mkdir(parents=True)
    rows: list[dict[str, object]] = []
    for item in prepared:
        source_image = item["image"]
        source_label = item["label"]
        stem = source_image.stem.replace(" ", "_")
        if stem.startswith("1frame_"):
            subset = "self_train"
        elif stem.startswith("aa_"):
            subset = "self_test"
        else:
            raise RuntimeError(f"unrecognized self-video frame family: {source_image}")
        image_dir = dataset / subset / "images"
        label_dir = dataset / subset / "labels"
        filename = f"self_video__{stem}__{item['image_hash'][:16]}{source_image.suffix.lower()}"
        target_image = image_dir / filename
        target_label = label_dir / f"{target_image.stem}.txt"
        shutil.copy2(source_image, target_image)
        target_label.write_text(
            f"{item['canonical_label']}\n" if item["canonical_label"] else "", encoding="utf-8"
        )
        rows.append(
            {
                "new_split": subset,
                "new_image": str(target_image.relative_to(dataset)),
                "image_sha256": item["image_hash"],
                "label_sha256": sha256(target_label),
                "representative_source": str(source_image),
                "all_recovered_copies": "|".join(item["duplicate_paths"]),
                "copy_count": len(item["duplicate_paths"]),
                "instances": item["instances"],
                "width": item["width"],
                "height": item["height"],
                "source_group": "self_collected:original_vedio",
            }
        )

    manifest_path = dataset / "self_video_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    audit = {
        "subsets": {
            "self_train": "36 unique sequential frames used for training",
            "self_test": "16 unique timestamp-sampled frames used for final testing only",
        },
        "evaluation_scope": (
            "within-video unseen-frame evaluation; self_train and self_test come from the same video, "
            "so this is not unseen-video or source-disjoint evaluation"
        ),
        "provenance": {
            "status": "self-collected, confirmed by project owner on 2026-08-12",
            "source_video": str(SOURCE_VIDEO),
            "source_video_sha256": sha256(SOURCE_VIDEO),
            "video_metadata": video_metadata(SOURCE_VIDEO),
        },
        "recovered_paths": len(recovery["records"]),
        "unique_images_imported": len(rows),
        "split_counts": {
            subset: sum(row["new_split"] == subset for row in rows) for subset in SUBSETS
        },
        "duplicate_path_copies_removed": len(recovery["records"]) - len(rows),
        "duplicate_image_groups": sum(len(group) > 1 for group in groups.values()),
        "conflicting_label_groups": 0,
        "exact_overlap_with_public_clean_dataset": len(overlap),
        "instances": sum(int(row["instances"]) for row in rows),
        "dimensions": dict(
            sorted(
                {
                    f"{item['width']}x{item['height']}": sum(
                        candidate["width"] == item["width"] and candidate["height"] == item["height"]
                        for candidate in prepared
                    )
                    for item in prepared
                }.items()
            )
        ),
        "source_group": "self_collected:original_vedio",
        "manifest": str(manifest_path),
    }
    (dataset / "self_video_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
