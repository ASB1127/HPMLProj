#!/usr/bin/env python3
"""
Generate plots for LoRA+TopR sweeps for the SST-2 dataset from `graph/` outputs.

Expected directory layout (as produced by `lora_config/lora.py`):
  graph/sst2/r{rank}/topr{top_r}/epoch_loss.csv
  graph/sst2/r{rank}/topr{top_r}/flops_profiler_stats.csv

Plots produced:
  1) final training loss vs top-r
  2) effective FLOPs vs top-r
  3) training loss vs epoch (one line per top-r; optional eval loss)

This script lives under `graph/` and resolves input paths relative to its own
location, so it works whether you run it from the repo root or from inside
`graph/`.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


DATASET = "sst2"
TOPR_DIR_RE = re.compile(r"^topr(?P<topr>[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)$")


@dataclass(frozen=True)
class RunData:
    top_r: float
    dir_path: Path
    epoch_train_loss: List[Tuple[int, float]]
    epoch_eval_loss: List[Tuple[int, float]]
    flops_stats: Dict[str, float]


def _try_float(value: str | None) -> Optional[float]:
    if value is None:
        return None
    value = str(value).strip()
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _read_epoch_loss_csv(path: Path) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
    train: List[Tuple[int, float]] = []
    eval_: List[Tuple[int, float]] = []
    if not path.exists():
        return train, eval_

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epoch_raw = (row.get("epoch") or "").strip()
            if epoch_raw == "":
                continue
            try:
                epoch = int(float(epoch_raw))
            except ValueError:
                continue

            train_loss = _try_float(row.get("train_loss"))
            if train_loss is not None and math.isfinite(train_loss):
                train.append((epoch, train_loss))

            eval_loss = _try_float(row.get("eval_loss"))
            if eval_loss is not None and math.isfinite(eval_loss):
                eval_.append((epoch, eval_loss))

    train.sort(key=lambda x: x[0])
    eval_.sort(key=lambda x: x[0])
    return train, eval_


def _read_flops_stats(path: Path) -> Dict[str, float]:
    stats: Dict[str, float] = {}
    if not path.exists():
        return stats

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metric = (row.get("metric") or "").strip()
            value = _try_float(row.get("value"))
            if metric and value is not None and math.isfinite(value):
                stats[metric] = float(value)
    return stats


def _parse_topr_dir_name(name: str) -> Optional[float]:
    m = TOPR_DIR_RE.match(name)
    if not m:
        return None
    return _try_float(m.group("topr"))


def load_runs(graph_dir: Path, rank: int) -> List[RunData]:
    base = graph_dir / DATASET / f"r{rank}"
    if not base.exists():
        raise FileNotFoundError(f"Missing directory: {base}")

    runs: List[RunData] = []
    for child in sorted(base.iterdir()):
        if not child.is_dir():
            continue
        top_r = _parse_topr_dir_name(child.name)
        if top_r is None:
            continue

        train, eval_ = _read_epoch_loss_csv(child / "epoch_loss.csv")
        stats = _read_flops_stats(child / "flops_profiler_stats.csv")
        runs.append(
            RunData(
                top_r=float(top_r),
                dir_path=child,
                epoch_train_loss=train,
                epoch_eval_loss=eval_,
                flops_stats=stats,
            )
        )

    runs.sort(key=lambda r: r.top_r)
    return runs


def _final_train_loss(run: RunData) -> Optional[float]:
    return run.epoch_train_loss[-1][1] if run.epoch_train_loss else None


def _metric_by_topr(runs: Iterable[RunData], metric: str) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for r in runs:
        v = r.flops_stats.get(metric)
        if v is None:
            continue
        out.append((r.top_r, v))
    out.sort(key=lambda x: x[0])
    return out


def _maybe_set_style() -> None:
    try:
        import matplotlib.pyplot as plt

        plt.style.use("seaborn-v0_8")
    except Exception:
        return


def plot_all(
    runs: List[RunData],
    outdir: Path,
    flops_metric: str,
    include_eval: bool,
) -> None:
    _maybe_set_style()
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Final training loss vs top-r
    xs: List[float] = []
    ys: List[float] = []
    for r in runs:
        last = _final_train_loss(r)
        if last is None:
            continue
        xs.append(r.top_r)
        ys.append(last)

    if xs:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(xs, ys, marker="o", linewidth=2)
        ax.set_xlabel("top-r (keep fraction)")
        ax.set_ylabel("final training loss")
        ax.set_title("SST-2: Final Training Loss vs Top-r")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(outdir / "final_training_loss_vs_topr.png", dpi=200)
        plt.close(fig)

    # 2) Effective FLOPs vs top-r
    flops_pairs = _metric_by_topr(runs, flops_metric)
    if flops_pairs:
        fx = [p[0] for p in flops_pairs]
        fy = [p[1] for p in flops_pairs]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(fx, fy, marker="o", linewidth=2)
        ax.set_xlabel("top-r (keep fraction)")
        ax.set_ylabel(flops_metric)
        ax.set_title(f"SST-2: {flops_metric} vs Top-r")
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        fig.tight_layout()
        fig.savefig(outdir / "effective_flops_vs_topr.png", dpi=200)
        plt.close(fig)

    # 3) Training loss vs epoch (one line per top-r)
    if any(r.epoch_train_loss for r in runs):
        fig, ax = plt.subplots(figsize=(8, 5))
        for r in runs:
            if not r.epoch_train_loss:
                continue
            epochs = [e for e, _ in r.epoch_train_loss]
            losses = [v for _, v in r.epoch_train_loss]
            ax.plot(epochs, losses, marker="o", linewidth=2, label=f"topr={r.top_r:g} train")
            if include_eval and r.epoch_eval_loss:
                e_epochs = [e for e, _ in r.epoch_eval_loss]
                e_losses = [v for _, v in r.epoch_eval_loss]
                ax.plot(
                    e_epochs,
                    e_losses,
                    marker="x",
                    linestyle="--",
                    linewidth=1.8,
                    label=f"topr={r.top_r:g} eval",
                )

        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.set_title("SST-2: Loss vs Epoch")
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=2, fontsize=9)
        fig.tight_layout()
        fig.savefig(outdir / "training_loss_vs_epoch.png", dpi=200)
        plt.close(fig)


def main() -> None:
    script_dir = Path(__file__).resolve().parent  # .../graph

    p = argparse.ArgumentParser(description="Plot LoRA+TopR sweep results for SST-2 from graph/ outputs.")
    p.add_argument("--rank", required=True, type=int, help="LoRA rank (e.g., 4, 8, 16, 64).")
    p.add_argument(
        "--graph-dir",
        type=Path,
        default=script_dir,
        help="Directory containing `sst2/` (default: the directory containing this script).",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Where to save plots (default: graph/plots/sst2/r{rank}).",
    )
    p.add_argument(
        "--flops-metric",
        default="topr_effective_epoch_flops",
        choices=[
            "topr_effective_epoch_flops",
            "topr_effective_step_flops",
            "masked_epoch_flops",
            "masked_step_flops",
            "topr_keep_fraction",
        ],
        help="Which metric from flops_profiler_stats.csv to plot vs top-r.",
    )
    p.add_argument("--include-eval", action="store_true", help="Also plot eval loss vs epoch (dashed).")
    p.add_argument("--dry-run", action="store_true", help="Parse and print summary only (no plots written).")
    args = p.parse_args()

    runs = load_runs(args.graph_dir, args.rank)
    if not runs:
        raise SystemExit(f"No runs found under: {args.graph_dir / DATASET / f'r{args.rank}'}")

    outdir = args.outdir or (script_dir / "plots" / DATASET / f"r{args.rank}")

    if args.dry_run:
        print(f"Found {len(runs)} runs under {args.graph_dir / DATASET / f'r{args.rank}'}:")
        for r in runs:
            final_loss = _final_train_loss(r)
            flops = r.flops_stats.get(args.flops_metric)
            print(f"  topr={r.top_r:g} final_train_loss={final_loss} {args.flops_metric}={flops}")
        return

    plot_all(
        runs=runs,
        outdir=outdir,
        flops_metric=args.flops_metric,
        include_eval=bool(args.include_eval),
    )
    print(f"OK: wrote plots to {outdir}")


if __name__ == "__main__":
    main()
