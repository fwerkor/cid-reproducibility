# Provisional-state-aware interaction decoding

Preserved controlled CID-v1 0.4B Stage-A ablation. This implementation-level study is reported in the paper appendix rather than as a headline main-text ablation.

The two runs use the same v17 data materialization, v4 contract, four-GPU geometry, effective batch size 96, and fixed validation seed `1000003`. The mechanism difference is whether need/source/argument/refresh/routing decisions are decoded from the stale pre-update thought state or from the provisional post-update thought state.

- stale baseline code: `0ce1e73077cb6003998ba20e4ea63b1eb8dee39e`
- provisional-state code: `74b9eedda3975802f8b31e8bc1a42321eb3a7e9c`
- matched epoch-2 optimizer step: **62,645**

At the matched epoch-2 checkpoint, fixed-seed free-rollout validation reports:

| Metric | Stale | Provisional |
|---|---:|---:|
| Materialized display exact | 13.61% | 22.78% |
| Display-token accuracy | 20.40% | 27.75% |
| Need F1 | 1.92% | 9.10% |
| Rollout recovery failure | 8.14% | 5.81% |

Raw training/validation JSONL and original Stage-A logs are preserved under `raw/`. Model checkpoints are intentionally not duplicated here yet; model placeholders are tracked separately.
