#!/usr/bin/env python
"""Select and render A3--A6 through one measured tree-layout path."""

import argparse
import json
from pathlib import Path

import numpy as np

from interaction_tree.renderer import STYLE, render_tree
from interaction_tree.schema import Interaction

HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
FIGURE_DIR = ROOT.parent
PACKAGE_ROOT = FIGURE_DIR.parent
INTERACTION_ROOT = PACKAGE_ROOT / "interactions" / "llm" / "examples" / "finance"
OUTPUT_DIR = FIGURE_DIR / "outputs"
TOP_K = 12
# Text block of the manuscript (Paper/main_812.pdf), measured from word
# bounding boxes: left 37.27, right 507.35, top 56.32, bottom 692.41 pt.
TEXT_BLOCK_POINTS = (470.09, 636.09)
MODELS = {
    "deepseek-ai__DeepSeek-R1-Distill-Llama-8B": {
        "title": "DeepSeekR1-Distill-Llama-8B",
        "figures": {0: "AIJ-FigA3-deepseek", 1: "AIJ-FigA4-deepseek"},
    },
    "Qwen__Qwen2.5-7B": {
        "title": "Qwen2.5-7B",
        "figures": {0: "AIJ-FigA5-qwen", 1: "AIJ-FigA6-qwen"},
    },
}
PROMPT_LINES = {
    0: [
        "New York Department of Health recommends that all people should wear N95, KN95, or KF94 masks in all public",
    ],
    1: [
        "On June 11, 2018, OpenAI researchers and engineers published a paper introducing the first generative pre-trained",
    ],
}


def select_interactions(model, sample):
    cache = INTERACTION_ROOT / model / f"Sample_{sample}/model1"
    I_and = np.load(cache / "I_and.npy")
    I_or = np.load(cache / "I_or.npy")
    masks = np.load(cache / "before_sparsify/player_masks.npy").astype(bool)
    candidates = []
    for kind, values in (("AND", I_and), ("OR", I_or)):
        for index, value in enumerate(values):
            candidates.append(
                Interaction(
                    kind=kind,
                    index=index,
                    value=float(value),
                    members=np.flatnonzero(masks[index]).tolist(),
                )
            )
    candidates.sort(key=lambda item: (-abs(item.value), item.kind, item.index))
    return candidates[:TOP_K], len(candidates)


def interaction_record(item):
    return {
        "kind": item.kind,
        "index": item.index,
        "value": item.value,
        "members": item.members,
    }


def run_synthetic_test():
    players = [
        "short",
        "two words",
        "the first generative pre-trained",
        "June 11, 2018,",
        "Department",
        "researchers",
        "KF94 masks",
        "public",
        "introducing",
        "all people",
    ]
    orders = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 5, 4, 3, 2, 1, 4, 3, 2, 1, 1]
    selected = [
        Interaction("AND" if i % 2 == 0 else "OR", i, (-1) ** i * (3.0 - i / 20), list(range(order)))
        for i, order in enumerate(orders)
    ]
    debug = ROOT / "debug-previews"
    debug.mkdir(exist_ok=True)
    metadata = render_tree(
        selected,
        players,
        "Synthetic",
        ["Measured layout stress test", "All row counts must remain visible"],
        "target",
        "Unified renderer synthetic test",
        debug / "synthetic_layout.pdf",
        debug / "synthetic_layout.png",
    )
    (debug / "synthetic_layout.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic-test", action="store_true")
    args = parser.parse_args()
    if args.synthetic_test:
        run_synthetic_test()
        return

    audit = json.loads((ROOT / "audit/a3_a6_experiment_audit.json").read_text())
    if audit["status"] != "PASS":
        raise RuntimeError("Experiment audit has not passed; refusing formal rendering")
    prompts = json.loads((ROOT / "data/prompts.json").read_text())
    players_by_prompt = json.loads((ROOT / "data/players.json").read_text())
    saved = OUTPUT_DIR
    saved.mkdir(exist_ok=True)
    metadata = {}

    for model, info in MODELS.items():
        for sample in [0, 1]:
            selected, candidate_count = select_interactions(model, sample)
            stem = info["figures"][sample]
            pdf = saved / f"{stem}.pdf"
            png = saved / f"{stem}.png"
            geometry = render_tree(
                selected,
                players_by_prompt[str(sample)],
                prompts[str(sample)]["label"],
                PROMPT_LINES[sample],
                prompts[str(sample)]["continuation"],
                info["title"],
                pdf,
                png,
            )
            metadata[stem] = {
                "figure": stem,
                "model": model,
                "prompt_id": sample,
                "prompt": prompts[str(sample)]["prompt"],
                "target": prompts[str(sample)]["continuation"],
                "output_pdf": str(pdf),
                "output_png": str(png),
                "threshold_applied": False,
                "candidate_interaction_count": candidate_count,
                "top_k_cap": TOP_K,
                "top_k_applied": True,
                "final_selected_count": len(selected),
                "selection_order": f"top {TOP_K} by descending absolute effect; ties by logic type then mask index",
                "selected_interactions": [interaction_record(item) for item in selected],
                "geometry": geometry,
            }
    panel_heights = {
        stem: meta["geometry"]["panel_points"][1] for stem, meta in metadata.items()
    }
    scale = STYLE["canvas_width_points"] / TEXT_BLOCK_POINTS[0]
    payload = {
        "style": STYLE,
        "figures": metadata,
        "page_fit": {
            "text_block_points": list(TEXT_BLOCK_POINTS),
            "text_block_source": "Paper/main_812.pdf word bounding boxes (left 37.27, right 507.35, top 56.32, bottom 692.41)",
            "grid": "4x1",
            "panel_width_points": TEXT_BLOCK_POINTS[0],
            "panel_heights_points": panel_heights,
            "panels_cropped_to_content": True,
            "combined_height_points": sum(panel_heights.values()),
            "canvas_to_panel_scale": scale,
            "effective_effect_font_points": STYLE["effect_font_points"] * scale,
            "effective_player_font_points": STYLE["player_font_points"] * scale,
        },
    }
    (saved / "render_metadata.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
