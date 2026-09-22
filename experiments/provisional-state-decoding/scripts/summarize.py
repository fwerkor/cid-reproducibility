#!/usr/bin/env python3
"""Extract the matched epoch-2 provisional-state ablation summary."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STALE = ROOT / "raw/stale/validation_metrics.jsonl"
PROVISIONAL = ROOT / "raw/provisional/validation_metrics.jsonl"
OUT = ROOT / "processed/matched_epoch2_summary.csv"
TARGET_STEP = 62645


def row_at(path: Path, step: int) -> dict:
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            if int(row["optimizer_steps"]) == step:
                return row
    raise RuntimeError(f"No row at optimizer step {step}: {path}")


stale = row_at(STALE, TARGET_STEP)
provisional = row_at(PROVISIONAL, TARGET_STEP)

if stale["validation_seed"] != provisional["validation_seed"]:
    raise RuntimeError("Validation seeds differ")

metrics = {
    "materialized_display_exact_rate": "materialized_display_exact_rate",
    "display_token_accuracy": "display_token_accuracy",
    "need_f1": "need_f1",
    "rollout_recovery_failure_rate": "rollout_recovery_failure_rate",
}

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["optimizer_steps", "validation_seed", "metric", "stale", "provisional"])
    for name, key in metrics.items():
        a = stale["free_rollout"]["behavior_metrics"][key]
        b = provisional["free_rollout"]["behavior_metrics"][key]
        writer.writerow([TARGET_STEP, stale["validation_seed"], name, f"{a:.12g}", f"{b:.12g}"])

print(OUT)
