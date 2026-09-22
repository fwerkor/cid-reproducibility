#!/usr/bin/env python3
"""Build the Stage-0 necessity comparison from preserved Stage-A metrics."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[1]
DIRECT_PATH = ROOT / "raw/no-stage0/train_metrics.jsonl"
STAGE0_PATH = ROOT / "raw/with-stage0/train_metrics.common-3gpu.jsonl"
OUT_CSV = ROOT / "processed/matched_component_losses.csv"
OUT_SUMMARY = ROOT / "processed/endpoint_summary.csv"
OUT_PNG = ROOT / "figures/stage0_component_losses.png"
OUT_PDF = ROOT / "figures/stage0_component_losses.pdf"

MAX_STEP = 8200
ROLLING_WINDOW = 5


def load(path: Path) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            step = int(row["optimizer_steps"])
            if step <= MAX_STEP:
                rows[step] = row
    return rows


def moving_average(values: list[float], window: int = ROLLING_WINDOW) -> list[float]:
    out: list[float] = []
    radius = window // 2
    for i in range(len(values)):
        lo = max(0, i - radius)
        hi = min(len(values), i + radius + 1)
        out.append(sum(values[lo:hi]) / (hi - lo))
    return out


direct = load(DIRECT_PATH)
stage0 = load(STAGE0_PATH)
steps = sorted(set(direct) & set(stage0))
if not steps:
    raise SystemExit("No common optimizer steps found")

if steps[0] != 100 or steps[-1] != MAX_STEP:
    raise SystemExit(f"Unexpected common range: {steps[0]}..{steps[-1]}")

components = sorted(
    set(direct[steps[0]]["component_losses"])
    & set(stage0[steps[0]]["component_losses"])
)
metrics = ["raw_mean_loss", *components]

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
OUT_PNG.parent.mkdir(parents=True, exist_ok=True)

with OUT_CSV.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["optimizer_steps", "metric", "direct_stage_a", "stage0_then_stage_a"])
    for step in steps:
        for metric in metrics:
            if metric == "raw_mean_loss":
                a = direct[step][metric]
                b = stage0[step][metric]
            else:
                a = direct[step]["component_losses"][metric]
                b = stage0[step]["component_losses"][metric]
            writer.writerow([step, metric, f"{a:.12g}", f"{b:.12g}"])

with OUT_SUMMARY.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["metric", "direct_stage_a_at_8200", "stage0_then_stage_a_at_8200", "stage0_over_direct"])
    for metric in metrics:
        if metric == "raw_mean_loss":
            a = float(direct[MAX_STEP][metric])
            b = float(stage0[MAX_STEP][metric])
        else:
            a = float(direct[MAX_STEP]["component_losses"][metric])
            b = float(stage0[MAX_STEP]["component_losses"][metric])
        ratio = b / a if a else float("nan")
        writer.writerow([metric, f"{a:.12g}", f"{b:.12g}", f"{ratio:.12g}"])

plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "legend.fontsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

fig, axes = plt.subplots(5, 5, figsize=(12.2, 11.4), constrained_layout=True)
axes = axes.ravel()

label_map = {
    "raw_mean_loss": "Raw mean",
    "need_cell_route": "Need cell route",
    "need_display_route": "Need display route",
    "anchor_ground": "Anchor ground",
    "anchor_kind": "Anchor kind",
    "anchor_presence": "Anchor presence",
    "argument_ground": "Argument ground",
    "argument_presence": "Argument presence",
    "link_ground": "Link ground",
    "link_presence": "Link presence",
    "link_relation": "Link relation",
    "link_target_kind": "Link target kind",
}

for idx, metric in enumerate(metrics):
    ax = axes[idx]
    if metric == "raw_mean_loss":
        y_direct = [float(direct[s][metric]) for s in steps]
        y_stage0 = [float(stage0[s][metric]) for s in steps]
    else:
        y_direct = [float(direct[s]["component_losses"][metric]) for s in steps]
        y_stage0 = [float(stage0[s]["component_losses"][metric]) for s in steps]

    ax.plot(steps, y_direct, color="#D55E00", alpha=0.18, linewidth=0.8)
    ax.plot(steps, y_stage0, color="#0072B2", alpha=0.18, linewidth=0.8)
    ax.plot(steps, moving_average(y_direct), color="#D55E00", linewidth=1.6, label="Direct Stage A")
    ax.plot(steps, moving_average(y_stage0), color="#0072B2", linewidth=1.6, label="Stage 0 → Stage A")
    ax.set_title(label_map.get(metric, metric.replace("_", " ").title()))
    ax.grid(axis="y", alpha=0.16, linewidth=0.6)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    if idx // 5 == 4:
        ax.set_xlabel("Optimizer steps")
    else:
        ax.tick_params(labelbottom=False)

for ax in axes[len(metrics):]:
    ax.axis("off")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.015))
fig.suptitle(
    "Stage-A loss trajectories with and without Stage-0 initialization",
    fontsize=12,
    y=1.035,
)
fig.text(
    0.5,
    1.008,
    "Matched historical interval: steps 100–8,200; 3 GPUs; 141,637 Stage-A windows; moving average = 5 matched points",
    ha="center",
    va="bottom",
    fontsize=8,
)

fig.savefig(OUT_PNG, dpi=220, bbox_inches="tight")
fig.savefig(OUT_PDF, bbox_inches="tight")
print(f"common_steps={len(steps)} range={steps[0]}..{steps[-1]}")
print(f"components={len(components)}")
print(f"raw_loss@8200 direct={direct[MAX_STEP]['raw_mean_loss']:.6f} stage0={stage0[MAX_STEP]['raw_mean_loss']:.6f}")
print(OUT_PNG)
