# Figure 5: robustness to the player count

`source/render_with_llm_cache.py` renders the panel grid to `outputs/main_paper_robustness.{pdf,png}`; `source/compose_submission_figure.py` composes `final/AIJ-Fig5-robustness.{pdf,png}` (the exact manuscript figure) from that panel. Inputs are under `interactions/robustness_n/`.

Rebuild from the package root:

```bash
python Fig_05_robustness/source/render_with_llm_cache.py
python Fig_05_robustness/source/compose_submission_figure.py
```

Y-scaling convention (finalized 2026-08-20): the all/generalizable groups share one y-scale across `n` per row; the non-generalizable group is row-normalized and each panel fits its own bars independently, with bars filling ~2/3 of the panel height (`NON_GEN_FILL_INV = 1.5`). No panel shows y-axis tick labels — only bar shapes are compared.

See `INVESTIGATION.md` for the 2026-08-15 investigation of the LLM spindle-vs-n artifact and the subsequent migration: all LLM panels (Figures 3, 5, 6) were finalized on the unified dataset-level q_bound cache `interactions/robustness_n/llm_target_single_cache_globalq/`.
