# Stage-0 necessity

This artifact compares two historical CID-v1 2B Stage-A runs:

1. **Direct Stage A** — initialized directly from `openbmb/MiniCPM5-2B-Base`.
2. **Stage 0 → Stage A** — initialized from the Stage-0 diffusion-adapted MiniCPM5-2B checkpoint.

To avoid mixing later restart/configuration changes into the comparison, the plotted interval is restricted to the common preserved segment **optimizer steps 100–8,200**. In that segment both runs use:

- 3 GPUs
- bf16
- learning rate `1e-4`
- effective global batch size 96
- teacher-forced epoch 1
- the same Stage-A training source and validation-v4 source
- `windows_total_in_epoch = 141,637`

The current Stage-0 route was later restarted with two GPUs and an updated materialization size, so those later records are deliberately excluded from this figure.

## Files

- `raw/no-stage0/train_metrics.jsonl` — complete preserved direct-Stage-A rank-0 aggregate metrics (through step 13,300)
- `raw/no-stage0/train.log` — original command/log proving direct initialization from the AR base
- `raw/with-stage0/train_metrics.common-3gpu.jsonl` — frozen pre-switch Stage-0→Stage-A metrics
- `raw/with-stage0/run-config.3gpu-before-switch.txt` — preserved pre-switch run configuration
- `processed/matched_component_losses.csv` — exact common-step data used by the figure
- `processed/endpoint_summary.csv` — step-8,200 endpoint values/ratios
- `figures/stage0_component_losses.{pdf,png}` — raw mean loss plus all 24 logged Stage-A loss components
- `scripts/plot_stage0_components.py` — deterministic regeneration script

This is a historical training diagnostic for Stage 0: direct Stage A approaches a much higher optimization plateau while the Stage-0-initialized run keeps descending. The paper keeps only a compact quantitative summary in the main ablation section and reports this full component-wise analysis in the appendix. It should not be presented as a core CID contribution or as a fully fresh one-variable rerun.
