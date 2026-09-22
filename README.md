# CID Reproducibility

Reproducibility artifacts and evaluation data for **Continuous Interaction Diffusion (CID)**.

The repository mirrors the paper's experiment organization:

- `experiments/main-results/` — Section 6.2 main results (placeholder until the final evaluation freeze)
- `experiments/efficiency/` — Section 6.3 efficiency (placeholder)
- `experiments/scaling/` — Section 6.4 scaling (placeholder)
- `experiments/stage0-necessity/` — appendix artifact for the Stage-0 training diagnostic, including preserved raw Stage-A metrics and a reproducible component-loss figure
- `experiments/provisional-state-decoding/` — appendix artifact for the preserved provisional-state decoding ablation
- `placeholders/datasets/` and `placeholders/models/` — manifests for large artifacts that will be copied/pinned at release time

Large model weights and full datasets are intentionally not duplicated here yet. Their final immutable identifiers/checksums will be filled in when the evaluation release is frozen.

## Reproducing the Stage-0 figure

```bash
python experiments/stage0-necessity/scripts/plot_stage0_components.py
```

The script regenerates the processed CSVs and both PDF/PNG versions of the figure from the preserved JSONL metrics.

## Source code

Training/runtime source: https://github.com/fwerkor/continuous-interaction-diffusion
