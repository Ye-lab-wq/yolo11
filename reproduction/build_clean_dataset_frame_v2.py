#!/usr/bin/env python3
"""Build DroneDetection clean v2 with frame-level, format-aware deduplication.

Distinct video frames may cross splits. Only high-confidence identical-content
records and variants of the same original source image are prevented from
crossing splits. Near-duplicate matching is deliberately disabled between two
different explicit video-frame IDs to avoid deleting genuine adjacent frames.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

import build_clean_dataset as v1


EXPLICIT_VIDEO_FRAME = re.compile(r"^video\d+_\d+$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent.parent
    parser.add_argument("--source", type=Path, default=root / "DetectDataset")
    parser.add_argument("--output", type=Path, default=root / "DetectDataset_clean_v2")
    parser.add_argument("--ratios", type=float, nargs=3, default=(0.70, 0.15, 0.15))
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--trials", type=int, default=512)
    parser.add_argument("--materialize", choices=("copy", "hardlink"), default="copy")
    parser.add_argument("--near-mae", type=float, default=1.5)
    parser.add_argument("--near-p99", type=float, default=5.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def decoded_image_fingerprints(path: Path) -> tuple[str, str, int, int, tuple[str, str]]:
    with Image.open(path) as image:
        detected_format = image.format or "UNKNOWN"
        rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB"), dtype=np.uint8)
    digest = hashlib.sha256()
    digest.update(f"RGB:{rgb.shape[1]}x{rgb.shape[0]}:".encode())
    digest.update(rgb.tobytes())
    thumbnail_image = Image.fromarray(rgb).convert("L").resize((16, 16), Image.Resampling.LANCZOS)
    thumbnail = np.asarray(thumbnail_image, dtype=np.uint8)
    # Two staggered quantizers reduce boundary sensitivity while preventing one
    # pHash bucket of similar video frames from becoming an O(n^2) comparison.
    signatures = tuple(
        hashlib.sha256(((thumbnail.astype(np.uint16) + offset) // 8).astype(np.uint8).tobytes()).hexdigest()
        for offset in (0, 4)
    )
    return digest.hexdigest(), detected_format, rgb.shape[1], rgb.shape[0], signatures


@lru_cache(maxsize=32)
def rgb_array(path_text: str) -> np.ndarray:
    with Image.open(path_text) as image:
        return np.asarray(ImageOps.exif_transpose(image).convert("RGB"), dtype=np.int16)


def near_identical(left: v1.Record, right: v1.Record, mae_limit: float, p99_limit: float) -> tuple[bool, dict[str, float]]:
    # A different numbered video frame is a real temporal observation. Do not
    # collapse it merely because a stationary scene has a similar pHash.
    if (
        left.source_id != right.source_id
        and EXPLICIT_VIDEO_FRAME.fullmatch(left.source_id)
        and EXPLICIT_VIDEO_FRAME.fullmatch(right.source_id)
    ):
        return False, {}
    left_array = rgb_array(str(left.image))
    right_array = rgb_array(str(right.image))
    if left_array.shape != right_array.shape:
        return False, {}
    difference = np.abs(left_array - right_array)
    mae = float(difference.mean())
    p99 = float(np.percentile(difference, 99))
    p999 = float(np.percentile(difference, 99.9))
    fraction_gt10 = float(np.mean(difference > 10))
    squared = difference.astype(np.float64) ** 2
    psnr = float("inf") if mae == 0 else float(20 * math.log10(255 / math.sqrt(float(np.mean(squared)))))
    same_source = left.source_id == right.source_id
    effective_mae = mae_limit if same_source else min(mae_limit, 0.10)
    effective_p99 = p99_limit if same_source else min(p99_limit, 2.0)
    matched = mae <= effective_mae and p99 <= effective_p99
    return matched, {
        "same_normalized_source_id": same_source,
        "mae": mae,
        "p99_abs_diff": p99,
        "p999_abs_diff": p999,
        "fraction_abs_diff_gt10": fraction_gt10,
        "psnr_db": psnr,
    }


def format_aware_content_groups(
    records: list[v1.Record], mae_limit: float, p99_limit: float
) -> tuple[v1.DisjointSet, dict[str, object]]:
    content = v1.DisjointSet(len(records))
    file_hash_seen: dict[str, int] = {}
    pixel_hash_seen: dict[str, int] = {}
    robust_buckets: dict[tuple[int, int, str, str], list[int]] = defaultdict(list)
    format_counts: Counter[str] = Counter()
    extension_counts: Counter[str] = Counter()

    for position, record in enumerate(records, 1):
        pixel_hash, detected_format, decoded_width, decoded_height, robust_signatures = decoded_image_fingerprints(
            record.image
        )
        if (decoded_width, decoded_height) != (record.width, record.height):
            raise RuntimeError(f"orientation/dimension mismatch after decoding: {record.image}")
        record.pixel_sha256 = pixel_hash
        record.detected_format = detected_format
        record.robust_signatures = robust_signatures
        format_counts[detected_format] += 1
        extension_counts[record.image.suffix.casefold()] += 1
        if record.image_sha256 in file_hash_seen:
            content.union(record.index, file_hash_seen[record.image_sha256])
        else:
            file_hash_seen[record.image_sha256] = record.index
        if pixel_hash in pixel_hash_seen:
            content.union(record.index, pixel_hash_seen[pixel_hash])
        else:
            pixel_hash_seen[pixel_hash] = record.index
        for robust_signature in record.robust_signatures:
            # Different explicit frame IDs are genuine temporal observations and
            # never even enter the same near-duplicate candidate bucket.
            scope = record.source_id if EXPLICIT_VIDEO_FRAME.fullmatch(record.source_id) else "non_explicit_global"
            robust_buckets[(record.width, record.height, scope, robust_signature)].append(record.index)
        if position % 500 == 0 or position == len(records):
            print(f"format-aware decode: {position}/{len(records)}", file=sys.stderr, flush=True)

    near_edges: list[dict[str, object]] = []
    comparisons = 0
    candidate_pairs: set[tuple[int, int]] = set()
    print(
        f"robust buckets: {len(robust_buckets)}, max bucket: {max(map(len, robust_buckets.values()), default=0)}",
        file=sys.stderr,
        flush=True,
    )
    for indices in robust_buckets.values():
        if len(indices) < 2:
            continue
        ordered = sorted(set(indices))
        for left_position, left in enumerate(ordered):
            for right in ordered[left_position + 1 :]:
                candidate_pairs.add((left, right))
    for left, right in sorted(candidate_pairs):
        if content.find(left) == content.find(right):
            continue
        if (
            records[left].source_id != records[right].source_id
            and EXPLICIT_VIDEO_FRAME.fullmatch(records[left].source_id)
            and EXPLICIT_VIDEO_FRAME.fullmatch(records[right].source_id)
        ):
            continue
        comparisons += 1
        is_match, metrics = near_identical(records[left], records[right], mae_limit, p99_limit)
        if is_match:
            content.union(left, right)
            near_edges.append(
                {
                    "left": str(records[left].image),
                    "right": str(records[right].image),
                    **metrics,
                }
            )
    print(
        f"candidate pairs: {len(candidate_pairs)}, verified: {comparisons}, near edges: {len(near_edges)}",
        file=sys.stderr,
        flush=True,
    )

    return content, {
        "extension_counts": dict(extension_counts),
        "detected_format_counts": dict(format_counts),
        "unique_file_sha256": len(file_hash_seen),
        "unique_decoded_pixel_sha256": len(pixel_hash_seen),
        "robust_candidate_pairs": len(candidate_pairs),
        "verified_candidate_comparisons": comparisons,
        "near_reencoding_edges": near_edges,
        "near_thresholds": {
            "same_dimensions": True,
            "identical_one_of_two_staggered_quantized_16x16_thumbnails": True,
            "rgb_mean_absolute_error_max": mae_limit,
            "rgb_p99_absolute_difference_max": p99_limit,
            "cross_source_rgb_mean_absolute_error_max": min(mae_limit, 0.10),
            "cross_source_rgb_p99_absolute_difference_max": min(p99_limit, 2.0),
            "different_explicit_video_frame_ids_auto_merge": False,
        },
    }


def deduplicate_records(
    records: list[v1.Record], content: v1.DisjointSet
) -> tuple[list[v1.Record], list[dict[str, object]], list[dict[str, object]]]:
    groups: dict[int, list[v1.Record]] = defaultdict(list)
    for record in records:
        groups[content.find(record.index)].append(record)
    conflicts: list[dict[str, object]] = []
    duplicate_groups: list[dict[str, object]] = []
    kept: list[v1.Record] = []
    for group in sorted(groups.values(), key=lambda items: min(str(item.image) for item in items)):
        labels = {record.label_signature for record in group}
        methods: set[str] = set()
        if len({record.image_sha256 for record in group}) < len(group):
            methods.add("identical_file_sha256")
        if len({record.pixel_sha256 for record in group}) < len(group):
            methods.add("identical_decoded_rgb")
        if len({record.pixel_sha256 for record in group}) > 1:
            methods.add("verified_near_reencoding")
        record_rows = [
            {
                "old_split": record.old_split,
                "image": str(record.image),
                "file_sha256": record.image_sha256,
                "pixel_sha256": record.pixel_sha256,
                "source_id": record.source_id,
                "label_signature": record.label_signature,
            }
            for record in sorted(group, key=lambda item: str(item.image))
        ]
        if len(labels) > 1:
            conflicts.append({"methods": sorted(methods), "records": record_rows})
            continue
        representative = min(group, key=lambda item: (item.old_split, str(item.image)))
        kept.append(representative)
        if len(group) > 1:
            duplicate_groups.append(
                {"methods": sorted(methods), "copies": len(group), "representative": str(representative.image), "records": record_rows}
            )
    for index, record in enumerate(kept):
        record.index = index
        record.capture_group = None
    return kept, conflicts, duplicate_groups


def build_frame_components(records: list[v1.Record]) -> list[v1.Component]:
    dsu = v1.DisjointSet(len(records))
    source_seen: dict[str, int] = {}
    for record in records:
        if record.source_id in source_seen:
            dsu.union(record.index, source_seen[record.source_id])
        else:
            source_seen[record.source_id] = record.index
    grouped: dict[int, v1.Component] = {}
    for record in records:
        root = dsu.find(record.index)
        if root not in grouped:
            grouped[root] = v1.Component(component_id=f"component_{len(grouped):05d}")
        component = grouped[root]
        component.records.append(record)
        component.images += 1
        component.instances += record.instances
        component.size_counts.update(record.size_counts)
    return sorted(grouped.values(), key=lambda item: item.component_id)


def video_split_counts(components: list[v1.Component], allocation: dict[str, str]) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for component in components:
        split = allocation[component.component_id]
        for record in component.records:
            match = v1.VIDEO_FRAME.fullmatch(record.source_id)
            if match:
                counts[match.group(1).casefold()][split] += 1
    return {video: dict(values) for video, values in sorted(counts.items())}


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    ratios = tuple(args.ratios)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    if any(value <= 0 for value in ratios) or not math.isclose(sum(ratios), 1.0):
        raise ValueError("ratios must be positive and sum to one")

    records, invalid = v1.scan_dataset(source)
    scanned = len(records) + len(invalid)
    content, format_audit = format_aware_content_groups(records, args.near_mae, args.near_p99)
    unique_records, conflicts, duplicate_groups = deduplicate_records(records, content)
    components = build_frame_components(unique_records)
    allocation = v1.split_components(components, ratios, args.seed, args.trials)
    planned = v1.summarize_allocation(components, allocation)
    if args.dry_run:
        format_summary = {key: value for key, value in format_audit.items() if key != "near_reencoding_edges"}
        format_summary["near_reencoding_edges_count"] = len(format_audit["near_reencoding_edges"])
        format_summary["cross_source_near_edges_count"] = sum(
            not edge["same_normalized_source_id"] for edge in format_audit["near_reencoding_edges"]
        )
        print(
            json.dumps(
                {
                    "input_records": scanned,
                    "invalid_records": len(invalid),
                    "unique_records": len(unique_records),
                    "duplicate_groups": len(duplicate_groups),
                    "conflicting_content_groups": len(conflicts),
                    "format_audit": format_summary,
                    "planned_splits": planned,
                    "video_split_counts": video_split_counts(components, allocation),
                },
                indent=2,
            )
        )
        return

    audit = v1.materialize_dataset(
        source=source,
        output=output,
        components=components,
        allocation=allocation,
        mode=args.materialize,
        ratios=ratios,
        seed=args.seed,
        scan_count=scanned,
        invalid=invalid,
        conflicts=conflicts,
        component_stats={
            "records_after_content_conflict_exclusion": len(unique_records),
            "unique_content_images": len(unique_records),
            "duplicate_copies_removed": sum(group["copies"] - 1 for group in duplicate_groups),
            "atomic_source_components": len(components),
        },
    )
    validation = v1.validate_output(output)
    audit.update(
        {
            "protocol_version": "clean_frame_v2",
            "rules": {
                "split_unit": "individual frame; distinct frames from one video may cross splits",
                "content_deduplication": (
                    "file SHA-256, decoded RGB SHA-256, and conservative verified re-encoding match"
                ),
                "source_binding": "variants with the same normalized pre-Roboflow source ID remain in one split",
                "label_conflicts": "exclude high-confidence identical-content groups whose labels disagree",
                "adjacent_frames": "retained; different explicit video-frame IDs are never auto-merged by pHash",
            },
            "format_audit": format_audit,
            "duplicate_content_groups": duplicate_groups,
            "conflicting_content_groups_excluded": conflicts,
            "video_split_counts": video_split_counts(components, allocation),
            "validation": validation,
        }
    )
    (output / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "README.md").write_text(
        f"""# DroneDetection clean frame v2

This dataset is a frame-level, format-aware rebuild of `{source}`. Distinct
video frames may cross train/valid/test. High-confidence identical image content
cannot cross splits, including byte-identical files, identical decoded RGB pixels,
conservatively verified re-encodings, and variants sharing a normalized original
source ID. Different numbered video frames are not merged by perceptual similarity.

Public split sizes: train {planned['train']['images']}, valid {planned['valid']['images']},
test {planned['test']['images']}. See `audit.json` and `split_manifest.csv` for the
complete protocol and provenance.
""",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "unique_records": len(unique_records),
                "duplicates_removed": sum(group["copies"] - 1 for group in duplicate_groups),
                "conflicting_groups_excluded": len(conflicts),
                "splits": planned,
                "video_split_counts": audit["video_split_counts"],
                "validation": validation,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
