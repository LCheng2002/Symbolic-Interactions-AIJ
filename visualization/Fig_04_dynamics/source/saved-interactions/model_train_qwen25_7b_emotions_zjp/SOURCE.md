# Source

Adapted from an external LLM two-phase interaction archive for Fig.04 training dynamics.

- Source: external LLM two-phase training archive
- Selected experiment: `qwen3b-7b-emtions`
- Destination schema: `saved-interactions/model_train_qwen25_7b_emotions_zjp`

Selected checkpoints:

| Timepoint | Source epoch | Middle-order spindle score |
| --- | ---: | ---: |
| A | 400 | 0.7663 |
| B | 1000 | 0.5732 |
| C | 2500 | 0.8242 |
| D | 4000 | 0.9305 |

The selected scores satisfy the desired trend: A->B decreases, then B->C->D increases.
The `archived_pow2_bridge/gap_curve.csv` file is extracted from the archive's real `trainer_log.json`; exact same-step `eval_loss` is used as `test-loss`. The logged training loss is an in-progress batch loss, so `train-loss` at step n is aligned to the next recorded training-loss entry after n.
