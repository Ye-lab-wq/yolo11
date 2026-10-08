#!/usr/bin/env python3
"""Build a deterministic motion-blur derivative of the public test split."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "DetectDataset_clean_v2/test"
OUTPUT = ROOT / "DetectDataset_clean_v2_motion_blur"
SEVERITIES = {"light": 3, "moderate": 7, "strong": 11}
ANGLES = (0, 45, 90, 135)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def odd_scaled_kernel(kernel_at_640: int, width: int, height: int) -> int:
    value = max(3, round(kernel_at_640 * max(width, height) / 640))
    return value if value % 2 else value + 1


def motion_kernel(length: int, angle: int):
    center = (length - 1) / 2
    horizontal = np.zeros((length, length), dtype=np.float32)
    horizontal[int(center), :] = 1.0 / length
    rotation = cv2.getRotationMatrix2D((center, center), angle, 1.0)
    rotated = cv2.warpAffine(horizontal, rotation, (length, length), flags=cv2.INTER_LINEAR)
    total = float(rotated.sum())
    if total <= 0:
        raise RuntimeError(f"invalid motion kernel: length={length}, angle={angle}")
    return rotated / total


def write_yaml(severity: str) -> None:
    path = OUTPUT / f"data_{severity}.yaml"
    path.write_text(
        f"path: {OUTPUT / severity}\n"
        "train: images\n"
        "val: images\n"
        "test: images\n\n"
        "names:\n"
        "  0: drone\n",
        encoding="utf-8",
    )


def main() -> None:
    images = sorted(
        path.resolve()
        for path in (SOURCE / "images").iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if len(images) != 625:
        raise RuntimeError(f"expected 625 public-test images, found {len(images)}")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for severity, kernel_at_640 in SEVERITIES.items():
        image_dir = OUTPUT / severity / "images"
        label_dir = OUTPUT / severity / "labels"
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for index, source_image in enumerate(images, start=1):
            image = cv2.imread(str(source_image), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"failed to read {source_image}")
            height, width = image.shape[:2]
            angle = ANGLES[int(hashlib.sha256(source_image.name.encode()).hexdigest()[:8], 16) % len(ANGLES)]
            native_kernel = odd_scaled_kernel(kernel_at_640, width, height)
            blurred = cv2.filter2D(
                image,
                ddepth=-1,
                kernel=motion_kernel(native_kernel, angle),
                borderType=cv2.BORDER_REFLECT_101,
            )
            target_image = image_dir / f"{source_image.stem}.png"
            if not cv2.imwrite(str(target_image), blurred, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
                raise RuntimeError(f"failed to write {target_image}")

            source_label = SOURCE / "labels" / f"{source_image.stem}.txt"
            target_label = label_dir / source_label.name
            if target_label.exists():
                if target_label.read_bytes() != source_label.read_bytes():
                    raise RuntimeError(f"existing label differs: {target_label}")
            else:
                try:
                    os.link(source_label, target_label)
                except OSError:
                    shutil.copy2(source_label, target_label)
            manifest.append(
                {
                    "severity": severity,
                    "source": str(source_image.relative_to(ROOT)),
                    "generated": str(target_image.relative_to(ROOT)),
                    "width": width,
                    "height": height,
                    "kernel_at_640": kernel_at_640,
                    "native_kernel": native_kernel,
                    "angle_degrees": angle,
                    "source_sha256": sha256(source_image),
                    "generated_sha256": sha256(target_image),
                }
            )
            if index % 100 == 0:
                print(f"{severity}: {index}/{len(images)}", flush=True)
        write_yaml(severity)

    with (OUTPUT / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    metadata = {
        "source": str(SOURCE.resolve()),
        "source_split": "public test",
        "images_per_severity": len(images),
        "severities_kernel_at_640": SEVERITIES,
        "angles_degrees": ANGLES,
        "angle_assignment": "sha256(filename) modulo four; fixed across severity",
        "native_kernel_rule": "nearest odd integer to kernel_at_640 * max(width,height) / 640",
        "image_output": "lossless PNG at original dimensions",
        "labels": "unchanged YOLO labels, hard-linked when possible",
        "use": "secondary robustness benchmark only; not an independent dataset",
    }
    (OUTPUT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "README.md").write_text(
        "# DetectDataset clean-v2 motion-blur benchmark\n\n"
        "Deterministic derivative of the 625-image public test split. It is a secondary synthetic-corruption "
        "benchmark, not an independently collected dataset. Bounding boxes are unchanged. Light, moderate, and "
        "strong motion blur correspond to line-kernel lengths 3, 7, and 11 at a 640-pixel model input; native "
        "kernel sizes are resolution-scaled. Each source image receives one deterministic angle from 0/45/90/135 "
        "degrees, held fixed across severity. See `manifest.csv` for provenance and hashes.\n",
        encoding="utf-8",
    )
    print(OUTPUT, flush=True)


if __name__ == "__main__":
    main()
