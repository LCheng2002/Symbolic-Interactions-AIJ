# AIJ interaction analysis and figure reproduction

This directory analyzes and visualizes precomputed symbolic interactions for *Revisiting Generalization Power of a DNN in Terms of Symbolic Interactions*.

Download the numerical inputs with the repository-root `download_data.sh`. They are archived separately on [Zenodo](https://doi.org/10.5281/zenodo.22052714) and are not tracked in Git.

## Scope

The public workflow is:

```text
archived interactions -> distribution analysis / fitting -> figure panels
```

`shared/tool.py` provides interaction loading, thresholding, cross-model comparisons, distribution fitting, and plotting utilities. These computations start from existing interaction values; they are not the model-to-interaction extraction or reward-table sparsification implementation.

The plotting workflow does not require SITKit, neural-network checkpoints, or an extraction-toolkit activation. For an optional external extraction demonstration, see `../demo/README.md`; authorization information is in `../ACCESS.md`.

## Layout and execution

Run the documented figure entry points from `visualization/`. Each figure README identifies its inputs and commands.

- `shared/`: analysis and plotting utilities.
- `interactions/`: downloaded numerical arrays and metadata.
- `Fig_*/source/`: retained visualization scripts/notebooks, metadata, and editable compositions.
- `Fig_*/outputs/`: generated direct exports; created when the renderer runs.
- `Fig_*/final/`: archived manuscript PDFs.

`MANIFEST.tsv` records the archived final-figure mapping; `PATH_MAPPING.tsv` records historical input locations. Historical paths are provenance, not required local directories. Numerical-data checksums are supplied by the Zenodo archive as `CHECKSUMS.sha256`.

## Reproduction boundary

A successful plot rebuild verifies a rendering workflow from archived inputs, not the original model training or interaction extraction. Some final figures were composed in PowerPoint; their direct code exports intentionally differ from the archived final PDFs. Preserve the data and scientific limitations described in `../docs/PROVENANCE.md` and the per-figure READMEs.
