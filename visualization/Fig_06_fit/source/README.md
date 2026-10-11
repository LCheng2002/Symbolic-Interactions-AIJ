# Fig. 06 Disentanglement Comparison

## Current Workflow

- `main_w_LLM.ipynb`: the only executable source. It generates the current
  nine-model comparison figure and writes the SIM summary.
- `saved-results/main.pdf` and `saved-results/main.png`: current figure.
- `saved-results/sim_results.csv`: current SIM summary.
- `saved-results/llm_sim_predict_tau_scan.csv`: retained record of the Finance
  LLM tau scan used to select the fixed current parameters.

The `*-compare_cache-predict_tau=*` directories cache reliable interaction
distributions for the current notebook configuration. They may be deleted and
will be regenerated on the next run.
