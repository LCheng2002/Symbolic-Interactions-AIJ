# Data provenance and reproduction boundary

This public package reproduces analyses and figures from archived interactions. It does not distribute the original model-training, model-to-reward, interaction-extraction, or sparsification implementation. For external toolkit demonstrations and access requests, see `../demo/README.md` and `../ACCESS.md`.

## Archived inputs

The numerical inputs are supplied by [Zenodo record 22052714](https://doi.org/10.5281/zenodo.22052714). `visualization/PATH_MAPPING.tsv` preserves the historical mapping of input groups, and `visualization/MANIFEST.tsv` maps the archived manuscript figures. Historical filesystem paths in provenance records are not required installation locations.

| Input group | Provenance and interpretation |
|---|---|
| `generalization/` | Interactions from the original split-data model experiments, including standard and adversarially trained model groups. Used by Figures 3, 6, 9, A1, and A2. |
| `robustness_n/` | Original player-count experiments. The LLM `llm_target_single_cache_globalq/` arrays use the unified dataset-level bound convention finalized in the recorded Figure 5 investigation. Figures 3 and 6 use its n=10 inputs. |
| `training_dynamics/` | Historical checkpoint trajectories supplied as archived data. The original training/extraction pipeline is not distributed in this package. |
| `noise/` | Interactions measured in the original clean, parameter-noise, and attack-noise experiments. Rendering uses these arrays; it does not apply noise to a neural network. |
| `llm/` | Original noise, prediction-correctness, and illustrative interaction-tree inputs. Prompt/player metadata and historical audit records needed by A3–A6 remain with those renderers. |

## Scientific and data limitations retained from the original package

**Figure 4 checkpoint selection.** The two displayed Qwen series were selected to illustrate a decrease and subsequent increase in relative middle-order interaction strength. They are qualitative examples, not an unbiased summary of all available checkpoints. The Figure 4 README and its retained data-adaptation script document this selection.

**Training-dynamics data integrity.** Historical validation notes reported before/after-label inconsistencies for some ViT/Qwen training-dynamics sample families and unsuccessful elementwise re-computation for some other trajectories. Plotting the stored arrays does not resolve these issues. Those historical observations are not new validation results from this visualization-only release.

**Figure 5 scaling.** The recorded final convention gives the non-generalizable panels independent y-axis limits. Shape comparisons should not be interpreted as direct cross-panel magnitude comparisons. The Figure 5 README and `INVESTIGATION.md` retain the context and the unified LLM-bound migration record.

**Figure 6 threshold selection.** The recorded LLM thresholds were selected on a finite grid using full-SIM scores. See `Fig_06_fit/source/PREDICT_TAU_SELECTION.md` under `visualization/`, including its later global-bound re-scan. These are fitted/selected experimental settings, not an independent held-out validation claim.

**Figure 9 additional output.** The additional Finance LLM panels are not part of the archived manuscript PDF; see that figure's README.

## What reproduction means here

The supported public route is archived arrays -> analysis/fitting -> rendered panels. Some final PDFs were composed in PowerPoint; the original editable compositions and final PDFs are retained. A successful render or a checksum match is not evidence that the original extraction experiment has been re-run.

SITKit is linked as an illustrative public demonstration. Compatibility with every historical model, optimization setting, data layout, or numerical result in this manuscript has not been established. Exact re-extraction requires the appropriate original experiment setup and authorized software; coordinate with the paper's authors and SymTrustAI rather than assuming the public demonstration is a drop-in replacement.
