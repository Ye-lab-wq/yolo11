#!/usr/bin/env python3
"""Audit the frozen UAV-Eagle external test before model inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
BENCH = Path(__file__).resolve().parent
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pixel_hash(path: Path) -> str:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        digest = hashlib.sha256()
        digest.update(f"{rgb.width}x{rgb.height}:".encode())
        digest.update(rgb.tobytes())
        return digest.hexdigest()


def images(directory: Path) -> list[Path]:
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=BENCH / "audit.json")
    args = parser.parse_args()
    archive = BENCH / "UAV-Eagle.zip"
    source = BENCH / "UAV-Eagle"
    external = images(source / "images")
    labels = sorted((source / "labels").glob("*.txt"))
    with zipfile.ZipFile(archive) as z:
        members = [entry for entry in z.infolist() if not entry.is_dir()]
        bad_paths = [entry.filename for entry in members if entry.filename.startswith("/") or ".." in Path(entry.filename).parts]
        bad_crc = z.testzip()
    if bad_paths or bad_crc:
        raise ValueError(f"Unsafe or corrupt archive: paths={bad_paths[:3]} crc={bad_crc}")

    image_stems = {p.stem for p in external}
    label_stems = {p.stem for p in labels}
    if len(external) != 510 or len(labels) != 510 or image_stems != label_stems:
        raise ValueError("Unexpected UAV-Eagle image/label inventory")
    classes: Counter[int] = Counter()
    box_count = 0
    target_negative_frames = 0
    for label in labels:
        lines = label.read_text(encoding="utf-8").splitlines()
        if not lines:
            target_negative_frames += 1
        for line in lines:
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(f"Malformed label: {label}: {line}")
            cls_float, cx, cy, w, h = map(float, fields)
            if cls_float != int(cls_float) or not (
                0 <= cx - w / 2 <= cx + w / 2 <= 1
                and 0 <= cy - h / 2 <= cy + h / 2 <= 1
                and w > 0 and h > 0
            ):
                raise ValueError(f"Invalid label: {label}: {line}")
            classes[int(cls_float)] += 1
            box_count += 1
    if set(classes) != {0}:
        raise ValueError(f"Class mapping must be checked before inference: {dict(classes)}")

    internal_dirs = [ROOT / "DetectDataset_clean_v2" / s / "images" for s in ("train", "valid", "test", "self_train", "self_test")]
    internal_paths = [p for directory in internal_dirs for p in images(directory)]
    internal_hashes = {pixel_hash(p) for p in internal_paths}
    external_hashes = [pixel_hash(p) for p in external]
    duplicates = [str(p.relative_to(source)) for p, h in zip(external, external_hashes) if h in internal_hashes]
    external_groups: dict[str, list[Path]] = {}
    for path, digest in zip(external, external_hashes):
        external_groups.setdefault(digest, []).append(path)
    duplicate_groups = [group for group in external_groups.values() if len(group) > 1]
    duplicate_labels_disagree = [
        [str(p.relative_to(source)) for p in group]
        for group in duplicate_groups
        if len({(source / "labels" / f"{p.stem}.txt").read_text(encoding="utf-8") for p in group}) > 1
    ]
    report = {
        "source_archive": str(archive),
        "source_archive_sha256": file_hash(archive),
        "images": len(external),
        "labels": len(labels),
        "instances": box_count,
        "target_negative_frames": target_negative_frames,
        "class_id_counts": dict(classes),
        "internal_images_compared": len(internal_paths),
        "decoded_rgb_overlap_with_internal": duplicates,
        "unique_external_decoded_images": len(set(external_hashes)),
        "external_duplicate_image_groups": [
            [str(p.relative_to(source)) for p in group] for group in duplicate_groups
        ],
        "external_duplicate_label_disagreement": duplicate_labels_disagree,
        "label_semantics": "YOLO class 0; preserved as supplied",
        "source_scope": "single UAV-Eagle source; not independent aircraft/video generalization",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if duplicates:
        raise ValueError("External test overlaps the internal dataset")


if __name__ == "__main__":
    main()
