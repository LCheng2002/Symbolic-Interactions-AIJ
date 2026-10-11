# Rendering from archived interactions

`data/` contains the prompt and player descriptions consumed by the renderer. `audit/` retains the original experiment records; these are read as archived metadata, not recomputed here.

The public scripts only read precomputed interactions and render figures:

- `scripts/render_a3_a6.py`: select the top-12 existing interactions and draw each panel.
- `scripts/interaction_tree/`: tree layout, drawing, and geometry validation.
- `scripts/make_combined_preview.py`: combine the rendered PNG panels.

Model loading, inference, target probing, and sparsification are outside this public package. Use the official SITKit demonstration linked in the repository-root `demo/README.md` for an illustrative external workflow; it is not a replacement for the historical manuscript extraction pipeline.
