# Symbolic Interactions — AIJ visualization package

Interaction analysis and figure-reproduction code accompanying the manuscript
*Revisiting Generalization Power of a DNN in Terms of Symbolic Interactions*.

## What is public

This public snapshot starts from **precomputed interaction arrays** and provides the distribution analysis, fitting, and visualization used by the manuscript. It does not distribute model-training code, interaction-extraction implementations, or reward-table sparsification implementations.

```text
Zenodo interaction arrays -> analysis / distribution fitting -> manuscript figures
```

The interaction-extraction and sparsification components are subject to copyright and licensing restrictions and are not redistributed here. An illustrative public demonstration is available from [SymTrustAI/SITKit](https://github.com/SymtrustAI/SITKit). For access to the full toolkit or authorization beyond the public license, contact **product@symtrustai.com**. See [the demo entry](demo/README.md) and [access and licensing](ACCESS.md).

**The SITKit demo is optional. Figure reproduction from the archived arrays does not require SITKit or an extraction-toolkit activation.**

## Install and download the data

```bash
python -m pip install -r requirements.txt
bash download_data.sh
```

The interaction arrays are archived separately on [Zenodo, DOI 10.5281/zenodo.22052714](https://doi.org/10.5281/zenodo.22052714). The download script extracts them into `visualization/` and checks the SHA-256 manifest supplied inside the archive. See [docs/DATA.md](docs/DATA.md) for the data layout and prerequisites.

For notebook-based figures, additionally install Jupyter:

```bash
python -m pip install jupyterlab ipykernel
```

## Reproduce the figures

Run the following entries from `visualization/`:

```bash
cd visualization
python Fig_05_robustness/source/render_with_llm_cache.py
python Fig_05_robustness/source/compose_submission_figure.py
```

| Figure | Entry relative to `visualization/` | Output boundary |
|---|---|---|
| 1–2 | Editable PowerPoint under each `Fig_*/source/` | Archived manuscript PDFs under `final/` |
| 3 | `python Fig_03_distributions/source/render_with_llm_cache.py` | Direct panels; final composition in PowerPoint |
| 4 | `Fig_04_dynamics/source/main_w_LLM.ipynb` | Training-dynamics visualization from archived checkpoints' interactions |
| 5 | `python Fig_05_robustness/source/render_with_llm_cache.py`, then `compose_submission_figure.py` in the same directory | Code-rendered and composed figure |
| 6 | `Fig_06_fit/source/main_w_LLM.ipynb` | Distribution-fit panels; final composition in PowerPoint |
| 7 | `python Fig_07_noise/source/render_main_w_llm.py` | Noise-interaction panels; final composition in PowerPoint |
| 8 | `python Fig_08_noise/source/render_main.py` | Progressive-noise panels; final composition in PowerPoint |
| 9 | `Fig_09_correctness/source/main.ipynb` | Correct/incorrect-prediction panels; final composition in PowerPoint |
| A1–A2 | `python Fig_A1_A2_sparsity_matching/source/generate_sparsity.py`, then `generate_figures.py` in the same directory | Sparsity statistics and matching plots from existing interactions, not a sparsification algorithm |
| A3–A6 | `python Fig_A3_A6_llm_examples/source/scripts/render_a3_a6.py` | Interaction-tree panels from archived arrays |

Figure 9 also has an additional LLM output from `python Fig_09_correctness/source/render_llm_finance.py`; the figure README explains that it is not included in the archived manuscript PDF.

Each figure README identifies its inputs, direct exports, and final composition. Generated files are written under `outputs/`. The original manuscript PDFs remain under `final/`. Do not assume that PowerPoint-composed finals are byte-identical to direct code exports.

## External interaction-extraction demo

Use [the SITKit demo entry](demo/README.md), which links directly to the official CLI quickstart, Jupyter notebook, and compiled release. The reviewed SITKit 0.1.0 demo supports Linux x86_64 / Python 3.11 and specified Qwen3-8B and Llama3-8B base-model revisions. Follow the upstream instructions and license; this repository does not bundle or reimplement SITKit.

The public SITKit license permits defined noncommercial uses. Full-toolkit availability, modification, redistribution, commercial use, or other additional rights must be confirmed with SymTrustAI. See [ACCESS.md](ACCESS.md).

## Reproducibility and licensing boundary

Rebuilding a figure from archived interactions is not an independent reproduction of model training or interaction extraction. The SITKit demonstration is illustrative and is not claimed to regenerate all historical manuscript data. Scientific and data-provenance limitations, including the selected training-dynamics trajectories, are retained in [docs/PROVENANCE.md](docs/PROVENANCE.md) and the figure READMEs.

The existing [MIT license](LICENSE) covers the code distributed in this snapshot, subject to any file-specific notices; it does not license the external SITKit toolkit or excluded implementations. Numerical data follow the terms of their Zenodo record. See [AUTHORS.md](AUTHORS.md).
