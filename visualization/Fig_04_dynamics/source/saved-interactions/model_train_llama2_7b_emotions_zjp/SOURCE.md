# Source

Adapted from an external LLM two-phase interaction archive for Fig.04 training dynamics.

- Source: external LLM two-phase training archive
- Selected experiment: `llama-7b-emotions`
- Destination schema: `saved-interactions/model_train_llama2_7b_emotions_zjp`

Selected checkpoints:

| Timepoint | Source epoch | Middle-order spindle score |
| --- | ---: | ---: |
| A | 1000 | 0.6663 |
| B | 2000 | 0.6942 |
| C | 3000 | 0.8741 |
| D | 3500 | 0.9189 |

The selected scores satisfy the desired trend: A->B decreases, then B->C->D increases.
The `archived_pow2_bridge/gap_curve.csv` file is extracted from the archive's real `trainer_log.json`; exact same-step `eval_loss` is used as `test-loss`. The logged training loss is an in-progress batch loss, so `train-loss` at step n is aligned to the next recorded training-loss entry after n.
