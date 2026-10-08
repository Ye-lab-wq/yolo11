#!/usr/bin/env python3
"""Build a source-disjoint, exactly deduplicated DroneDetection dataset.

The input dataset is treated as read-only. Records that share exact image bytes,
the same pre-Roboflow source name, or an explicit video/capture sequence are
kept in one atomic group. Exact images with conflicting labels are excluded
instead of silently choosing one annotation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image


SPLITS = ("train", "valid", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
RF_SUFFIX = re.compile(r"\.rf\.[0-9a-f]+$", re.IGNORECASE)
EXPORT_SUFFIX = re.compile(r"_(?:jpeg|jpg|png)$", re.IGNORECASE)
VIDEO_FRAME = re.compile(r"^(video\d+)_(\d+)$", re.IGNORECASE)
FRAME_ONLY = re.compile(r"^frame[_-]?(\d+)$", re.IGNORECASE)
VIDEO_ONLY = re.compile(r"^video(\d+)$", re.IGNORECASE)
SAFE_NAME = re.compile(r"[^0-9A-Za-z._-]+")


@dataclass
class Record:
    index: int
    image: Path
    label: Path
    old_split: str
    image_sha256: str
    label_signature: str
    source_id: str
    capture_group: str | None
    width: int
    height: int
    instances: int
    size_counts: Counter[str]


@dataclass
class Component:
    component_id: str
    records: list[Record] = field(default_factory=list)
    images: int = 0
    instances: int = 0
    size_counts: Counter[str] = field(default_factory=Counter)
    capture_groups: set[str] = field(default_factory=set)


class DisjointSet:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent.parent
    parser.add_argument("--source", type=Path, default=root / "DetectDataset")
    parser.add_argument("--output", type=Path, default=root / "DetectDataset_clean_v1")
    parser.add_argument("--ratios", type=float, nargs=3, default=(0.70, 0.15, 0.15))
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--trials", type=int, default=512)
    parser.add_argument("--dry-run", action="store_true", help="Print the planned split without writing files.")
    parser.add_argument(
        "--materialize",
        choices=("copy", "hardlink"),
        default="copy",
        help="Copy is independent and is the publication-safe default.",
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_source_id(path: Path) -> str:
    stem = RF_SUFFIX.sub("", path.stem)
    previous = None
    while stem != previous:
        previous = stem
        stem = EXPORT_SUFFIX.sub("", stem)
    return stem.casefold()


def capture_group(source_id: str) -> str | None:
    match = VIDEO_FRAME.fullmatch(source_id)
    if match:
        return f"explicit_video:{match.group(1).casefold()}"
    if FRAME_ONLY.fullmatch(source_id):
        # No clip identifier exists in the export, so keep this visibly
        # sequential family together rather than claim frame independence.
        return "implicit_sequence:frame"
    if VIDEO_ONLY.fullmatch(source_id):
        # These names identify video-derived material but omit a reliable clip
        # delimiter. Conservatively keep the small family in one split.
        return "implicit_sequence:video"
    return None


def canonical_label(path: Path, width: int, height: int) -> tuple[str, Counter[str]]:
    if not path.exists():
        return "", Counter()
    canonical: list[str] = []
    sizes: Counter[str] = Counter()
    scale = min(640 / width, 640 / height)
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        fields = raw_line.split()
        if len(fields) != 5:
            raise ValueError(f"{path}:{line_number}: expected 5 fields")
        try:
            class_value, x, y, box_width, box_height = map(float, fields)
        except ValueError as error:
            raise ValueError(f"{path}:{line_number}: non-numeric label") from error
        class_id = int(class_value)
        valid = class_value == class_id == 0 and box_width > 0 and box_height > 0
        valid = valid and 0 <= x <= 1 and 0 <= y <= 1
        valid = valid and x - box_width / 2 >= -1e-6 and y - box_height / 2 >= -1e-6
        valid = valid and x + box_width / 2 <= 1 + 1e-6 and y + box_height / 2 <= 1 + 1e-6
        if not valid:
            raise ValueError(f"{path}:{line_number}: invalid class, size, or bounds")
        canonical.append(
            " ".join((str(class_id), *(format(value, ".10g") for value in (x, y, box_width, box_height))))
        )
        equivalent_side = math.sqrt(box_width * width * scale * box_height * height * scale)
        if equivalent_side < 16:
            sizes["tiny"] += 1
        elif equivalent_side < 32:
            sizes["small"] += 1
        elif equivalent_side < 96:
            sizes["medium"] += 1
        else:
            sizes["large"] += 1
    return "\n".join(sorted(canonical)), sizes


def scan_dataset(source: Path) -> tuple[list[Record], list[dict[str, str]]]:
    records: list[Record] = []
    invalid: list[dict[str, str]] = []
    for split in SPLITS:
        image_dir = source / split / "images"
        label_dir = source / split / "labels"
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(f"missing YOLO directories for split {split}: {source}")
        for image_path in sorted(image_dir.iterdir()):
            if not image_path.is_file() or image_path.suffix.casefold() not in IMAGE_SUFFIXES:
                continue
            label_path = label_dir / f"{image_path.stem}.txt"
            try:
                with Image.open(image_path) as image:
                    width, height = image.size
                    image.verify()
                label_signature, size_counts = canonical_label(label_path, width, height)
            except Exception as error:  # record corrupt images and invalid YOLO labels
                invalid.append(
                    {
                        "old_split": split,
                        "image": str(image_path),
                        "label": str(label_path),
                        "reason": str(error),
                    }
                )
                continue
            source_id = normalized_source_id(image_path)
            records.append(
                Record(
                    index=len(records),
                    image=image_path,
                    label=label_path,
                    old_split=split,
                    image_sha256=file_sha256(image_path),
                    label_signature=label_signature,
                    source_id=source_id,
                    capture_group=capture_group(source_id),
                    width=width,
                    height=height,
                    instances=len(label_signature.splitlines()) if label_signature else 0,
                    size_counts=size_counts,
                )
            )
    return records, invalid


def remove_conflicting_exact_images(
    records: list[Record],
) -> tuple[list[Record], list[dict[str, object]]]:
    by_hash: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        by_hash[record.image_sha256].append(record)
    conflicting_hashes = {
        image_hash
        for image_hash, group in by_hash.items()
        if len({record.label_signature for record in group}) > 1
    }
    conflicts: list[dict[str, object]] = []
    for image_hash in sorted(conflicting_hashes):
        group = by_hash[image_hash]
        conflicts.append(
            {
                "image_sha256": image_hash,
                "records": [
                    {
                        "old_split": record.old_split,
                        "image": str(record.image),
                        "label": str(record.label),
                        "label_signature": record.label_signature,
                    }
                    for record in group
                ],
            }
        )
    kept = [record for record in records if record.image_sha256 not in conflicting_hashes]
    for index, record in enumerate(kept):
        record.index = index
    return kept, conflicts


def build_components(records: list[Record]) -> tuple[list[Component], dict[str, int]]:
    dsu = DisjointSet(len(records))
    seen_hash: dict[str, int] = {}
    seen_source: dict[str, int] = {}
    seen_capture: dict[str, int] = {}
    for record in records:
        for key, table in (
            (record.image_sha256, seen_hash),
            (record.source_id, seen_source),
        ):
            if key in table:
                dsu.union(record.index, table[key])
            else:
                table[key] = record.index
        if record.capture_group:
            if record.capture_group in seen_capture:
                dsu.union(record.index, seen_capture[record.capture_group])
            else:
                seen_capture[record.capture_group] = record.index

    # Keep one deterministic copy of each exact image. Label signatures are
    # identical here because conflicting hashes were removed above.
    representative_by_hash: dict[str, Record] = {}
    for record in sorted(records, key=lambda item: (item.old_split, str(item.image))):
        representative_by_hash.setdefault(record.image_sha256, record)

    grouped: dict[int, Component] = {}
    for record in representative_by_hash.values():
        root = dsu.find(record.index)
        if root not in grouped:
            grouped[root] = Component(component_id=f"component_{len(grouped):05d}")
        component = grouped[root]
        component.records.append(record)
        component.images += 1
        component.instances += record.instances
        component.size_counts.update(record.size_counts)
        if record.capture_group:
            component.capture_groups.add(record.capture_group)
    components = sorted(grouped.values(), key=lambda item: item.component_id)
    stats = {
        "input_records_after_conflict_exclusion": len(records),
        "unique_exact_images": len(representative_by_hash),
        "duplicate_copies_removed": len(records) - len(representative_by_hash),
        "atomic_components": len(components),
    }
    return components, stats


def component_metrics(component: Component) -> list[int]:
    return [
        component.images,
        component.instances,
        component.size_counts["tiny"],
        component.size_counts["small"],
        component.size_counts["medium"],
        component.size_counts["large"],
    ]


def split_components(
    components: list[Component], ratios: tuple[float, float, float], seed: int, trials: int
) -> dict[str, str]:
    totals = [sum(values) for values in zip(*(component_metrics(c) for c in components))]
    targets = [[total * ratio for total in totals] for ratio in ratios]
    weights = (5.0, 2.0, 0.5, 1.0, 1.0, 1.0)
    sequence_components = {c.component_id for c in components if c.capture_groups}

    def objective(
        current: list[list[int]], allocation: dict[str, int], require_capture_coverage: bool = False
    ) -> float:
        value = 0.0
        for split_index in range(3):
            for metric_index, weight in enumerate(weights):
                denominator = max(targets[split_index][metric_index], 1.0)
                delta = (current[split_index][metric_index] - targets[split_index][metric_index]) / denominator
                value += weight * delta * delta
        # A source-disjoint test is useful only if validation and test contain
        # at least one capture sequence, not just unrelated still images.
        if require_capture_coverage:
            for split_index in (1, 2):
                if not any(allocation.get(component_id) == split_index for component_id in sequence_components):
                    value += 100.0
        return value

    best_score = float("inf")
    best_allocation: dict[str, int] | None = None
    for trial in range(trials):
        rng = random.Random(seed + trial)
        # Largest source groups are placed first; random tie/noise permits many
        # deterministic candidate packings without splitting any source.
        order = sorted(
            components,
            key=lambda item: (-(item.images + rng.random() * max(1.0, item.images * 0.08)), item.component_id),
        )
        current = [[0] * len(totals) for _ in range(3)]
        allocation: dict[str, int] = {}
        for component in order:
            values = component_metrics(component)
            candidates: list[tuple[float, float, int]] = []
            for split_index in range(3):
                for metric_index, metric_value in enumerate(values):
                    current[split_index][metric_index] += metric_value
                allocation[component.component_id] = split_index
                score = objective(current, allocation)
                fill = current[split_index][0] / max(targets[split_index][0], 1.0)
                candidates.append((score, fill, split_index))
                for metric_index, metric_value in enumerate(values):
                    current[split_index][metric_index] -= metric_value
                del allocation[component.component_id]
            _, _, chosen = min(candidates)
            allocation[component.component_id] = chosen
            for metric_index, metric_value in enumerate(values):
                current[chosen][metric_index] += metric_value
        score = objective(current, allocation, require_capture_coverage=True)
        if score < best_score:
            best_score = score
            best_allocation = allocation.copy()
    if best_allocation is None:
        raise RuntimeError("split optimizer produced no allocation")
    return {component_id: SPLITS[split_index] for component_id, split_index in best_allocation.items()}


def summarize_allocation(
    components: list[Component], allocation: dict[str, str]
) -> dict[str, dict[str, object]]:
    summary: dict[str, dict[str, object]] = {
        split: {"images": 0, "instances": 0, "size_counts": Counter(), "capture_groups": set()}
        for split in SPLITS
    }
    for component in components:
        split = allocation[component.component_id]
        summary[split]["images"] += component.images
        summary[split]["instances"] += component.instances
        summary[split]["size_counts"].update(component.size_counts)
        summary[split]["capture_groups"].update(component.capture_groups)
    return {
        split: {
            "images": values["images"],
            "instances": values["instances"],
            "size_counts": dict(values["size_counts"]),
            "capture_groups": sorted(values["capture_groups"]),
        }
        for split, values in summary.items()
    }


def safe_output_name(record: Record) -> str:
    base = SAFE_NAME.sub("_", record.source_id).strip("._-") or "image"
    base = base[:100]
    return f"{base}__{record.image_sha256[:16]}{record.image.suffix.casefold()}"


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def materialize_dataset(
    source: Path,
    output: Path,
    components: list[Component],
    allocation: dict[str, str],
    mode: str,
    ratios: tuple[float, float, float],
    seed: int,
    scan_count: int,
    invalid: list[dict[str, str]],
    conflicts: list[dict[str, object]],
    component_stats: dict[str, int],
) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    for split in SPLITS:
        (output / split / "images").mkdir(parents=True, exist_ok=False)
        (output / split / "labels").mkdir(parents=True, exist_ok=False)

    manifest: list[dict[str, object]] = []
    split_stats: dict[str, dict[str, object]] = {
        split: {"images": 0, "instances": 0, "size_counts": Counter(), "capture_groups": set()}
        for split in SPLITS
    }
    link = shutil.copy2 if mode == "copy" else lambda src, dst: Path(dst).hardlink_to(src)
    for component in components:
        new_split = allocation[component.component_id]
        for record in sorted(component.records, key=lambda item: str(item.image)):
            filename = safe_output_name(record)
            target_image = output / new_split / "images" / filename
            target_label = output / new_split / "labels" / f"{Path(filename).stem}.txt"
            link(record.image, target_image)
            target_label.write_text(
                f"{record.label_signature}\n" if record.label_signature else "", encoding="utf-8"
            )
            stats = split_stats[new_split]
            stats["images"] += 1
            stats["instances"] += record.instances
            stats["size_counts"].update(record.size_counts)
            if record.capture_group:
                stats["capture_groups"].add(record.capture_group)
            manifest.append(
                {
                    "new_split": new_split,
                    "new_image": str(target_image.relative_to(output)),
                    "old_split": record.old_split,
                    "old_image": str(record.image.relative_to(source)),
                    "image_sha256": record.image_sha256,
                    "source_id": record.source_id,
                    "capture_group": record.capture_group or "",
                    "component_id": component.component_id,
                    "instances": record.instances,
                    "width": record.width,
                    "height": record.height,
                }
            )

    data_yaml = (
        f"path: {output.resolve()}\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n\n"
        "nc: 1\n"
        "names: ['drone']\n"
    )
    (output / "data.yaml").write_text(data_yaml, encoding="utf-8")
    write_csv(
        output / "split_manifest.csv",
        manifest,
        [
            "new_split", "new_image", "old_split", "old_image", "image_sha256",
            "source_id", "capture_group", "component_id", "instances", "width", "height",
        ],
    )

    serializable_stats: dict[str, object] = {}
    for split, stats in split_stats.items():
        serializable_stats[split] = {
            "images": stats["images"],
            "instances": stats["instances"],
            "size_counts": dict(stats["size_counts"]),
            "capture_groups": sorted(stats["capture_groups"]),
        }
    audit = {
        "protocol_version": "clean_v1",
        "source_dataset": str(source.resolve()),
        "output_dataset": str(output.resolve()),
        "seed": seed,
        "requested_ratios": dict(zip(SPLITS, ratios)),
        "materialization": mode,
        "rules": {
            "deduplication": "one record per exact image SHA-256 across all legacy splits",
            "source_grouping": "same normalized pre-Roboflow filename is atomic",
            "capture_grouping": "explicit videoNN_frame, frameNNNN, and ambiguous videoNN families are atomic",
            "label_conflicts": "exclude all copies of an exact image when canonical labels disagree",
            "invalid_records": "exclude corrupt images and invalid YOLO labels",
            "object_size": "equivalent box side after 640 letterbox scaling",
        },
        "input_records_scanned": scan_count,
        "invalid_records_excluded": invalid,
        "conflicting_exact_images_excluded": conflicts,
        **component_stats,
        "splits": serializable_stats,
    }
    (output / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    readme = f"""# DroneDetection clean v1

This is a newly materialized dataset derived from `{source.resolve()}`. The source
directory was not modified.

Protocol: global exact-pixel deduplication, source/capture-disjoint grouping,
deterministic 70/15/15 allocation (seed {seed}), and exclusion of exact images
whose labels conflict. The legacy train/valid/test membership is not reused.

- `data.yaml`: Ultralytics dataset configuration
- `split_manifest.csv`: complete old-to-new provenance
- `audit.json`: protocol, exclusions, counts, sizes, and capture groups
- `train|valid|test/images|labels`: new YOLO dataset

Important: old checkpoints trained on the legacy split are not a fair comparison
on this dataset. Baseline and MEDA must both be retrained from scratch with the
same optimization settings, seed policy, epoch budget, and model-selection rule.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    return audit


def validate_output(output: Path) -> dict[str, object]:
    hash_to_splits: dict[str, set[str]] = defaultdict(set)
    source_to_splits: dict[str, set[str]] = defaultdict(set)
    capture_to_splits: dict[str, set[str]] = defaultdict(set)
    manifest_rows = list(csv.DictReader((output / "split_manifest.csv").open(encoding="utf-8")))
    for row in manifest_rows:
        split = row["new_split"]
        image = output / row["new_image"]
        label = output / split / "labels" / f"{image.stem}.txt"
        if not image.is_file() or not label.is_file():
            raise RuntimeError(f"missing materialized pair for {image}")
        actual_hash = file_sha256(image)
        if actual_hash != row["image_sha256"]:
            raise RuntimeError(f"hash mismatch after materialization: {image}")
        hash_to_splits[actual_hash].add(split)
        source_to_splits[row["source_id"]].add(split)
        if row["capture_group"]:
            capture_to_splits[row["capture_group"]].add(split)
    violations = {
        "exact_hash_cross_split": {key: sorted(value) for key, value in hash_to_splits.items() if len(value) > 1},
        "source_id_cross_split": {key: sorted(value) for key, value in source_to_splits.items() if len(value) > 1},
        "capture_group_cross_split": {key: sorted(value) for key, value in capture_to_splits.items() if len(value) > 1},
    }
    if any(violations.values()):
        raise RuntimeError(f"split leakage validation failed: {violations}")
    return {
        "manifest_records": len(manifest_rows),
        "unique_hashes": len(hash_to_splits),
        "unique_source_ids": len(source_to_splits),
        "capture_groups": {key: sorted(value) for key, value in sorted(capture_to_splits.items())},
        "leakage_violations": violations,
    }


def main() -> None:
    args = parse_args()
    ratios = tuple(args.ratios)
    if len(ratios) != 3 or any(value <= 0 for value in ratios) or not math.isclose(sum(ratios), 1.0):
        raise ValueError("--ratios must contain three positive values summing to 1")
    source = args.source.resolve()
    output = args.output.resolve()
    records, invalid = scan_dataset(source)
    scan_count = len(records) + len(invalid)
    records, conflicts = remove_conflicting_exact_images(records)
    components, component_stats = build_components(records)
    allocation = split_components(components, ratios, args.seed, args.trials)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "scan_count": scan_count,
                    "invalid_records": len(invalid),
                    "conflicting_exact_images": len(conflicts),
                    **component_stats,
                    "planned_splits": summarize_allocation(components, allocation),
                },
                indent=2,
            )
        )
        return
    audit = materialize_dataset(
        source=source,
        output=output,
        components=components,
        allocation=allocation,
        mode=args.materialize,
        ratios=ratios,
        seed=args.seed,
        scan_count=scan_count,
        invalid=invalid,
        conflicts=conflicts,
        component_stats=component_stats,
    )
    validation = validate_output(output)
    audit["validation"] = validation
    (output / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(output), "splits": audit["splits"], "validation": validation}, indent=2))


if __name__ == "__main__":
    main()
