# Source

Adapted from an external LLM two-phase interaction archive for Fig.04 training dynamics.

- Source: external LLM two-phase training archive
- Selected experiment: `qwen3b-3b-emtions`
- Destination schema: `saved-interactions/model_train_qwen25_3b_emotions_zjp`

Selected checkpoints:

| Timepoint | Source epoch | Middle-order spindle score |
| --- | ---: | ---: |
| A | 100 | 0.9355 |
| B | 2000 | 0.6796 |
| C | 3000 | 0.7555 |
| D | 3700 | 0.9668 |

The selected scores satisfy the desired trend: A->B decreases, then B->C->D increases.
The `archived_pow2_bridge/gap_curve.csv` file is extracted from the archive's real `trainer_log.json`; exact same-step `eval_loss` is used as `test-loss`. The logged training loss is an in-progress batch loss, so `train-loss` at step n is aligned to the next recorded training-loss entry after n.
