# Figure 7: newly emerged interactions

Run from `visualization/`:

```bash
python Fig_07_noise/source/render_main_w_llm.py
```

The scripts and notebooks read precomputed clean/noisy interaction arrays under `interactions/`, perform distribution analysis, and write direct exports under `outputs/`. They do not load neural-network checkpoints or generate new interactions.

`source/AIJ-Fig7-noise_LLM.pptx` retains the editable composition. `final/AIJ-Fig7-noise.pdf` is the archived manuscript figure. The direct code export and the PowerPoint-composed final PDF are different artifacts and are not expected to have identical hashes or page sizes.

Noise-injection, model-evaluation, and interaction-extraction drivers are not distributed in this public visualization package.
