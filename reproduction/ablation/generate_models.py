#!/usr/bin/env python3
"""Generate controlled YOLO11n-P2 ablation YAMLs from one auditable graph builder."""

from __future__ import annotations

from pathlib import Path

import yaml


HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "models"


def build_p2(
    *,
    upsample: str = "nearest",
    downsample: str = "conv",
    backbone_downsample: str | None = None,
    neck_downsample: str | None = None,
    fusion: str = "c3k2",
    p2_block: str = "c3k2",
    p2_ema: bool = False,
    p5_block: str = "c3k2",
    keep_c2psa: bool = True,
) -> dict:
    nodes: list[tuple[str, object, int, str, list]] = []

    def add(name: str, source: object, repeats: int, module: str, args: list) -> None:
        nodes.append((name, source, repeats, module, args))

    def down(name: str, output: int, region: str) -> None:
        if region == "backbone":
            selected = backbone_downsample or downsample
        elif region == "neck":
            selected = neck_downsample or downsample
        else:
            raise ValueError(region)
        if selected not in {"conv", "adown"}:
            raise ValueError(selected)
        module = "ADown" if selected == "adown" else "Conv"
        args = [output] if module == "ADown" else [output, 3, 2]
        add(name, -1, 1, module, args)

    def up(name: str) -> None:
        if upsample == "nearest":
            add(name, -1, 1, "nn.Upsample", [None, 2, "nearest"])
        elif upsample == "author_bilinear":
            add(name, -1, 1, "DySample", [2, "lp"])
        elif upsample == "official_dysample":
            add(name, -1, 1, "DySampleOfficial", [2, "lp", 4, False])
        else:
            raise ValueError(upsample)

    def block(name: str, output: int, kind: str, shortcut: bool = False) -> None:
        if kind == "c3k2":
            add(name, -1, 2, "C3k2", [output, shortcut])
        elif kind == "rep":
            add(name, -1, 1, "RepNCSPELAN4", [output, output, output // 2])
        elif kind == "rep_official":
            add(name, -1, 1, "RepNCSPELAN4Official", [output, output, output // 2])
        elif kind == "msef_author":
            add(name, -1, 1, "MSEF", [output, False, 1, 0.5])
        elif kind == "msef_paper":
            add(name, -1, 1, "MSEFPaper", [output, False, 1, 0.5])
        else:
            raise ValueError(kind)

    # Official YOLO11n backbone; only the requested downsampler is varied.
    add("p1", -1, 1, "Conv", [64, 3, 2])
    add("p2_stem", -1, 1, "Conv", [128, 3, 2])
    add("p2_backbone", -1, 2, "C3k2", [256, False, 0.25])
    down("p3_down", 256, "backbone")
    add("p3_backbone", -1, 2, "C3k2", [512, False, 0.25])
    down("p4_down", 512, "backbone")
    add("p4_backbone", -1, 2, "C3k2", [512, True])
    down("p5_down", 1024, "backbone")
    add("p5_backbone", -1, 2, "C3k2", [1024, True])
    add("sppf", -1, 1, "SPPF", [1024, 5])
    add("deep", -1, 2 if keep_c2psa else 1, "C2PSA" if keep_c2psa else "nn.Identity", [1024] if keep_c2psa else [])

    up("up_p4")
    add("cat_p4_top", [-1, "p4_backbone"], 1, "Concat", [1])
    block("p4_top", 512, fusion)
    up("up_p3")
    add("cat_p3_top", [-1, "p3_backbone"], 1, "Concat", [1])
    block("p3_top", 256, fusion)
    up("up_p2")
    add("cat_p2", [-1, "p2_backbone"], 1, "Concat", [1])
    block("p2_out", 128, p2_block)
    if p2_ema:
        add("p2_attention", -1, 1, "EMA", [16])
        p2_output = "p2_attention"
    else:
        p2_output = "p2_out"

    down("p3_head_down", 256, "neck")
    add("cat_p3_bottom", [-1, "p3_top"], 1, "Concat", [1])
    block("p3_out", 256, fusion)
    down("p4_head_down", 512, "neck")
    add("cat_p4_bottom", [-1, "p4_top"], 1, "Concat", [1])
    block("p4_out", 512, fusion)
    down("p5_head_down", 1024, "neck")
    add("cat_p5", [-1, "deep"], 1, "Concat", [1])
    block("p5_out", 1024, p5_block, shortcut=True)
    add("detect", [p2_output, "p3_out", "p4_out", "p5_out"], 1, "Detect", ["nc"])

    indices = {name: index for index, (name, *_rest) in enumerate(nodes)}

    def resolve(source: object) -> object:
        if isinstance(source, str):
            return indices[source]
        if isinstance(source, list):
            return [indices[item] if isinstance(item, str) else item for item in source]
        return source

    backbone_count = indices["deep"] + 1
    encoded = [[resolve(source), repeats, module, args] for _, source, repeats, module, args in nodes]
    return {
        "nc": 1,
        "scales": {"n": [0.50, 0.25, 1024]},
        "backbone": encoded[:backbone_count],
        "head": encoded[backbone_count:],
    }


VARIANTS = {
    "p2_nearest": {},
    "p2_author_bilinear": {"upsample": "author_bilinear"},
    "p2_official_dysample": {"upsample": "official_dysample"},
    "p2_adown": {"downsample": "adown"},
    "p2_rep": {"fusion": "rep"},
    "p2_rep_official": {"fusion": "rep_official"},
    "p2_msef_author": {"p2_block": "msef_author"},
    "p2_msef_paper": {"p2_block": "msef_paper"},
    "p2_msef_author_ema": {"p2_block": "msef_author", "p2_ema": True},
    "p2_p5_msef_author": {"p5_block": "msef_author"},
    "p2_no_c2psa": {"keep_c2psa": False},
    # Stage-2 one-factor ablations use P2+MSEFPaper as the direct parent.
    # Each entry below changes exactly one additional structural factor.
    "p2_msef_paper_adown": {"p2_block": "msef_paper", "downsample": "adown"},
    "p2_msef_paper_author_bilinear": {"p2_block": "msef_paper", "upsample": "author_bilinear"},
    "p2_msef_paper_official_dysample": {"p2_block": "msef_paper", "upsample": "official_dysample"},
    "p2_msef_paper_rep": {"p2_block": "msef_paper", "fusion": "rep"},
    "p2_msef_paper_rep_official": {"p2_block": "msef_paper", "fusion": "rep_official"},
    "p2_msef_paper_ema": {"p2_block": "msef_paper", "p2_ema": True},
    "p2_msef_paper_p5": {"p2_block": "msef_paper", "p5_block": "msef_paper"},
    "p2_msef_paper_no_c2psa": {"p2_block": "msef_paper", "keep_c2psa": False},
    "p2_msef_paper_adown_backbone": {"p2_block": "msef_paper", "backbone_downsample": "adown"},
    "p2_msef_paper_adown_neck": {"p2_block": "msef_paper", "neck_downsample": "adown"},
}


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for name, options in VARIANTS.items():
        path = MODEL_DIR / f"{name}.yaml"
        path.write_text(yaml.safe_dump(build_p2(**options), sort_keys=False), encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
