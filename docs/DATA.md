# Archived interaction data

## Download

The numerical inputs for the manuscript figures are distributed separately on [Zenodo](https://doi.org/10.5281/zenodo.22052714). The documented archive contains approximately 2.8 GB of interaction arrays and associated metadata; it is not stored in this Git repository.

From the repository root:

```bash
bash download_data.sh
```

The script requires `curl`, `tar`, and `sha256sum`. It downloads `interactions.tar.gz`, extracts it into `visualization/`, and verifies the included `CHECKSUMS.sha256`. On systems without `sha256sum`, install a compatible SHA-256 utility before using the script. The checksum manifest is delivered inside the archive, not tracked in Git.

To recheck a previously downloaded archive's extracted files:

```bash
cd visualization
sha256sum --check CHECKSUMS.sha256
```

A checksum match verifies the downloaded files against that archive's manifest. It does not establish mathematical self-consistency or independently reproduce the original experiment.

## Input groups

| Directory under `visualization/interactions/` | Used for |
|---|---|
| `generalization/` | Cross-model interaction comparisons, distribution fitting, and correct/incorrect-prediction analyses |
| `robustness_n/` | Player-count comparisons; the `llm_target_single_cache_globalq/` subdirectory also supplies the final LLM inputs for Figures 3 and 6 |
| `training_dynamics/` | Archived interaction trajectories used by Figure 4 |
| `noise/` | Precomputed clean, parameter-noise, and attack-noise interactions |
| `llm/` | LLM noise, prediction-correctness, and illustrative interaction-tree data |

Historical names such as `before_sparsify/`, `after_sparsify/`, `masks.npy`, and `rewards.npy` are part of the numerical-data layout expected by the plotting code. Keeping these input names does not redistribute a sparsification implementation. Do not rename them during download.

## Scope

The public rendering workflow uses archived interaction values and metadata. Model checkpoints, original training datasets, model inference, and interaction-extraction software are not prerequisites for that workflow. They are not included in this public code snapshot.

The SITKit demonstration is an independent optional workflow with its own model downloads, platform requirements, and license. See `../demo/README.md` and `../ACCESS.md`.

Data rights and usage conditions are those recorded with the Zenodo release. Preserve the provenance and limitations in `PROVENANCE.md` when interpreting figures.
