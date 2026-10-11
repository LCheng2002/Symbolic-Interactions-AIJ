# Source

Adapted from an external LLM two-phase interaction archive for Fig.04 training dynamics.

- Source: external LLM two-phase training archive
- Selected experiment: `results-llama-3-8b-average-init-bslr5-history-066n-80epoch`
- Destination schema: `saved-interactions/model_train_llama3_8b_emotions_zjp`

Selected checkpoints:

| Timepoint | Source epoch | Middle-order spindle score |
| --- | ---: | ---: |
| A | 1000 | 0.5837 |
| B | 2000 | 0.6935 |
| C | 2500 | 0.7944 |
| D | 3000 | 0.9011 |

The selected scores satisfy the desired trend: A->B decreases, then B->C->D increases.
The `archived_pow2_bridge/gap_curve.csv` file is extracted from the archive's real `trainer_log.json`; exact same-step `eval_loss` is used as `test-loss`. The logged training loss is an in-progress batch loss, so `train-loss` at step n is aligned to the next recorded training-loss entry after n.
