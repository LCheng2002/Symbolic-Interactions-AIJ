# Figure 8: progressive noise decomposition

Run from `visualization/`:

```bash
python Fig_08_noise/source/render_main.py
```

`source/main.ipynb` is a notebook entry for the same renderer. The renderer consumes precomputed arrays from `interactions/noise/clean/`, `interactions/noise/parameter_gaussian/`, and `interactions/noise/attack/`, then fits and plots the interaction distributions. It does not perform model inference, inject noise into a model, or sparsify rewards into interactions.

Direct exports are written under `outputs/`. The editable PowerPoint remains at `source/AIJ-Fig8-noise.pptx`; `final/AIJ-Fig8-noise.pdf` is the archived manuscript figure.

Historical model-evaluation drivers, extraction batch scripts, and their unused manual-segmentation links are not part of this visualization-only release.
