# Final `predict_tau` Selection

The final Fig. 06 configuration uses the threshold that maximizes the full
SIM, not the reliable-only `IoU(generalizable, decay)` diagnostic.

| Group | Models | Final `set_predict_tau` |
| --- | --- | --- |
| Existing configuration, unchanged | ResNet-20/MNIST, ResNet-20/CIFAR-10, VGG-11/CIFAR-10, BERT-Medium/SST-2, BERT-Base/SST-2 | `0.03` |
| Finance LLM | Qwen2.5-7B | `0.10` |
| Finance LLM | Llama-3-8B | `0.05` |
| Finance LLM | GLM-Z1-9B | `0.08` |
| Finance LLM | Phi-4 | `0.05` |

The four LLM values are the maxima in
`saved-results/llm_sim_predict_tau_scan.csv`, evaluated over
`{0.03, 0.05, 0.08, 0.10}`.  The values are already present in
`main_w_LLM.ipynb`.

The reliable-only diagnostic was archived in
`archive_predict_tau_reliable_20260713-085415/`; it is retained as analysis
evidence and is not used to select the final figure parameters.

## 2026-08-15 re-scan after the global-q_bound cache migration

The LLM rows were re-computed from the unified dataset-level q_bound cache
(`interactions/robustness_n/llm_target_single_cache_globalq/n=10`; see
`Fig_05_robustness/INVESTIGATION.md`). The scan was repeated on the same grid
`{0.03, 0.05, 0.08, 0.10}` with the same full-SIM argmax rule
(`archive_globalq_migration_20260815/Fig_06_outputs_globalq/llm_sim_predict_tau_scan.csv`):

| Model | Previous | Re-scanned argmax | SIM |
| --- | --- | --- | --- |
| Qwen2.5-7B | 0.10 | **0.08** | 0.805 |
| Llama-3-8B | 0.05 | 0.05 (unchanged) | 0.799 |
| GLM-Z1-9B | 0.08 | 0.08 (unchanged) | 0.874 |
| Phi-4 | 0.05 | 0.05 (unchanged) | 0.753 |

Only Qwen2.5-7B moved (0.10 -> 0.08, +0.003 SIM). The values are already
present in `main_w_LLM.ipynb`.
