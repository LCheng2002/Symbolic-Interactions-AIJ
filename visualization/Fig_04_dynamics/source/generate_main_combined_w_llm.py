#!/usr/bin/env python3
"""Compose the original four Fig.04 models plus two Qwen LLM panels."""

from __future__ import annotations

import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def _load_notebook_code() -> str:
    nb = json.load(open(ROOT / "main.ipynb"))
    return "\n\n".join("".join(cell.get("source", [])) for cell in nb["cells"][:2])


def main() -> None:
    os.chdir(ROOT)
    namespace: dict = {"__name__": "__fig04_w_llm__"}
    exec(_load_notebook_code(), namespace)

    source_root = namespace["SOURCE_ROOT"]
    output_dir = namespace["OUTPUT_DIR"]
    output_dir.mkdir(parents=True, exist_ok=True)

    models = [
        ("Training VGG-11 on CIFAR-10", "vgg11_cifar10"),
        ("Training VGG-16 on CUB200-2011", "vgg16_cub2011"),
        ("Training AlexNet on Tiny ImageNet", "alexnet_tiny_imagenet"),
        ("Training ViT on CIFAR-10", "vit_cifar10"),
        ("Training Qwen2.5-3B on Emotion", "qwen25_3b_emotions_zjp"),
        ("Training Qwen2.5-7B on Emotion", "qwen25_7b_emotions_zjp"),
    ]

    namespace["composite_combined_figures"](
        source_root=source_root,
        output_dir=output_dir,
        models=models,
    )

    for ext in ["png", "pdf"]:
        src = output_dir / f"main_combined.{ext}"
        dst = output_dir / f"main_combined_w_LLM.{ext}"
        if dst.exists():
            dst.unlink()
        src.replace(dst)
        print(f"saved {dst}")


if __name__ == "__main__":
    main()
