# Model placeholder

Large model weights/checkpoints are intentionally omitted during active experimentation.

At the final evaluation freeze, this directory will contain a manifest with:

- model/checkpoint name
- source repository and immutable revision
- parameter count
- training stage
- SHA-256/checksum where feasible
- exact experiment(s) using that checkpoint

The small JSON/log artifacts needed to audit existing ablations are already committed under `experiments/`.
