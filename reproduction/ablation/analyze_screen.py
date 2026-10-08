#!/usr/bin/env python3
"""Create direct-parent comparisons for the frozen validation-only screen."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "runs" / "ablation_screen_v1"
SUMMARY = RUN_ROOT / "screen_summary.csv"
EFFICIENCY = RUN_ROOT / "screen_efficiency.csv"

PARENTS = {
    "p2_nearest_ciou_s0": "yolo11n_ciou_s0",
    "p2_no_c2psa_ciou_s0": "p2_nearest_ciou_s0",
    "p2_adown_ciou_s0": "p2_nearest_ciou_s0",
    "p2_author_bilinear_ciou_s0": "p2_nearest_ciou_s0",
    "p2_official_dysample_ciou_s0": "p2_nearest_ciou_s0",
    "p2_rep_ciou_s0": "p2_nearest_ciou_s0",
    "p2_rep_official_ciou_s0": "p2_nearest_ciou_s0",
    "p2_msef_author_ciou_s0": "p2_nearest_ciou_s0",
    "p2_msef_paper_ciou_s0": "p2_nearest_ciou_s0",
    "p2_msef_author_ema_ciou_s0": "p2_msef_author_ciou_s0",
    "p2_p5_msef_author_ciou_s0": "p2_nearest_ciou_s0",
    "p2_nearest_author_focal_s0": "p2_nearest_ciou_s0",
    "p2_nearest_focaler_ciou_s0": "p2_nearest_ciou_s0",
}


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fmt(value: float | None, digits: int = 3) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def main() -> None:
    summaries = {row["experiment"]: row for row in load_csv(SUMMARY)}
    missing = (set(PARENTS) | set(PARENTS.values())) - set(summaries)
    if missing:
        raise RuntimeError(f"screen is incomplete; missing {sorted(missing)}")
    efficiency = {row["model"]: row for row in load_csv(EFFICIENCY)}
    comparisons = []
    for child_name, parent_name in PARENTS.items():
        child, parent = summaries[child_name], summaries[parent_name]
        delta = float(child["map50_95"]) - float(parent["map50_95"])
        child_latency = efficiency.get(child_name, {}).get("fp32_b1_latency_ms")
        parent_latency = efficiency.get(parent_name, {}).get("fp32_b1_latency_ms")
        latency_change = None
        if child_latency and parent_latency:
            latency_change = float(child_latency) / float(parent_latency) - 1.0
        accuracy_pass = delta >= 0.003
        efficiency_pass = delta >= -0.001 and latency_change is not None and latency_change <= -0.10
        comparisons.append(
            {
                "child": child_name,
                "parent": parent_name,
                "child_best_epoch": child["best_epoch"],
                "parent_best_epoch": parent["best_epoch"],
                "child_map50_95": float(child["map50_95"]),
                "parent_map50_95": float(parent["map50_95"]),
                "delta_map50_95_pp": delta * 100,
                "child_parameters": int(child["parameters"]),
                "parent_parameters": int(parent["parameters"]),
                "parameter_change_pct": (int(child["parameters"]) / int(parent["parameters"]) - 1) * 100,
                "b1_latency_change_pct": None if latency_change is None else latency_change * 100,
                "screen_decision": "advance" if accuracy_pass or efficiency_pass else "do_not_advance",
                "decision_basis": "accuracy" if accuracy_pass else "efficiency" if efficiency_pass else "threshold_not_met",
            }
        )

    csv_path = RUN_ROOT / "screen_comparisons.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)

    lines = [
        "# Validation-only component screen",
        "",
        "The table compares every factor to its direct parent. `Delta` is an absolute percentage-point change in",
        "best validation mAP50:95 within the frozen 60-epoch budget. No test result is used.",
        "",
        "| Child | Direct parent | mAP50:95/% | Delta/pp | Params/M | b1 latency delta/% | Decision |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in comparisons:
        lines.append(
            f"| {row['child']} | {row['parent']} | {float(row['child_map50_95']) * 100:.3f} | "
            f"{float(row['delta_map50_95_pp']):+.3f} | {int(row['child_parameters']) / 1e6:.3f} | "
            f"{fmt(row['b1_latency_change_pct'])} | {row['screen_decision']} ({row['decision_basis']}) |"
        )
    md_path = RUN_ROOT / "SCREEN_RESULTS.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(csv_path)
    print(md_path)


if __name__ == "__main__":
    main()
