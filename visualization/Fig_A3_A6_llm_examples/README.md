# Figures A3–A6: LLM interaction examples

Render the precomputed interaction arrays downloaded from Zenodo. This directory contains visualization code and the input metadata needed by that code; it does not contain model inference, interaction extraction, or sparsification implementations.

Run from `visualization/`:

```bash
python Fig_A3_A6_llm_examples/source/scripts/render_a3_a6.py
python Fig_A3_A6_llm_examples/source/scripts/make_combined_preview.py
```

Inputs are under `interactions/llm/examples/`. The renderer reads the retained `source/data/` prompt/player metadata and `source/audit/a3_a6_experiment_audit.json`, a record from the original experiments. Reading this historical audit record is not a new experimental verification.

Each panel shows the 12 largest-magnitude AND/OR interactions. Direct exports are written to `outputs/`; the original manuscript PDFs remain in `final/`.

For an external interaction-extraction demonstration and licensing information, see the repository-root `demo/README.md` and `ACCESS.md`.
