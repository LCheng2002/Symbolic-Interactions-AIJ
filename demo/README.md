# Interaction-extraction demo: external SITKit

The extraction and sparsification implementations are not redistributed in this visualization package because of copyright and licensing restrictions. Use the official **SymTrustAI/SITKit** demonstration instead.

## Official demo and download

- [SITKit project and installation instructions](https://github.com/SymtrustAI/SITKit)
- [Chinese quickstart](https://github.com/SymtrustAI/SITKit/blob/main/README.zh-CN.md)
- [Official CLI example](https://github.com/SymtrustAI/SITKit/blob/main/examples/cli/quickstart.sh)
- [Official Jupyter notebook](https://github.com/SymtrustAI/SITKit/blob/main/examples/notebooks/sitkit_quickstart.ipynb)
- [SITKit 0.1.0 release: compiled wheel and SHA-256 file](https://github.com/SymtrustAI/SITKit/releases/tag/v0.1.0)

These are external links. This repository does not copy the toolkit, its compiled wheel, or its example dataset.

## Example workflow

The reviewed **SITKit 0.1.0** release supports Linux x86_64 and Python 3.11, with specified Qwen3-8B and Llama3-8B base-model revisions. Its public release supports at most 12 players; an NVIDIA GPU is recommended. Use the exact models and installation instructions published upstream.

After obtaining the official release, verifying its checksum, installing it in its own environment, and preparing the supported model, run the following from the official SITKit checkout:

```bash
sitkit --input-json examples/datasets/economics.json --check-only

sitkit \
  --model ./models/Qwen3-8B \
  --input-json examples/datasets/economics.json \
  --output-dir runs/economics
```

The upstream example produces per-sample interaction reports and an aggregate interaction-order distribution. Its published instructions also describe optional two-model comparison. Refer to the official README for the precise output paths and supported options.

This is an illustrative extraction demo, not a claim that SITKit 0.1.0 reproduces every historical AIJ experiment or exports arrays identical to the Zenodo archive. To reproduce the manuscript figures, use the archived arrays and this repository's visualization commands instead.

## Copyright and full authorization

SITKit has its own [PolyForm Strict License 1.0.0](https://github.com/SymtrustAI/SITKit/blob/main/LICENSE.md). Defined noncommercial uses are permitted under those terms. Modification, redistribution, and commercial use require separate written permission as stated by the publisher.

For access to the full toolkit, additional supported workflows, or authorization beyond the public license, contact **SymTrustAI: product@symtrustai.com** (product/toolkit) or **contact@symtrustai.com** (business/licensing). The scope and availability of any additional authorization must be confirmed by the company. See [ACCESS.md](../ACCESS.md).
