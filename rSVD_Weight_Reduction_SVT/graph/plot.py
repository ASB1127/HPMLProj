#!/usr/bin/env python3
"""
Plotting script for rSVD Weight Reduction experiments with SVT.
Generates visualizations comparing theoretical FLOPs and training loss 
across different ranks and SVT truncation settings.
"""
from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class FlopsStats:
    """Dataclass to hold step-level and epoch-level FLOPs statistics."""
    step_flops: float
    epoch_flops: float


@dataclass(frozen=True)
class EpochLossRow:
    """Dataclass to hold training and evaluation loss for a specific epoch."""
    epoch: int
    train_loss: float
    eval_loss: Optional[float]


def _read_flops_stats(path: Path) -> FlopsStats:
    """Reads FLOPs profiling statistics from a CSV file into a FlopsStats object."""
    metrics: Dict[str, float] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metric = (row.get("metric") or "").strip()
            value = (row.get("value") or "").strip()
            if not metric or not value:
                continue
            metrics[metric] = float(value)

    missing = [k for k in ("step_flops", "epoch_flops") if k not in metrics]
    if missing:
        raise ValueError(f"Missing {missing} in {path}")
    return FlopsStats(step_flops=metrics["step_flops"], epoch_flops=metrics["epoch_flops"])


def _read_epoch_loss(path: Path) -> List[EpochLossRow]:
    """Reads epoch-wise loss data from a CSV file into a list of EpochLossRow objects."""
    rows: List[EpochLossRow] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epoch_s = (row.get("epoch") or "").strip()
            train_s = (row.get("train_loss") or "").strip()
            eval_s = (row.get("eval_loss") or "").strip()
            if not epoch_s or not train_s:
                continue
            rows.append(
                EpochLossRow(
                    epoch=int(float(epoch_s)),
                    train_loss=float(train_s),
                    eval_loss=(float(eval_s) if eval_s else None),
                )
            )
    if not rows:
        raise ValueError(f"No rows parsed from {path}")
    return rows


def _iter_rank_dirs(sst2_dir: Path) -> Iterable[Tuple[int, Path]]:
    """Iterates through and parses rank directories (e.g., 'r4') within a dataset directory."""
    pattern = re.compile(r"^r(\d+)$")
    for child in sorted(sst2_dir.iterdir()):
        if not child.is_dir():
            continue
        match = pattern.match(child.name)
        if not match:
            continue
        yield int(match.group(1)), child


def _safe_import_matplotlib():
    """Import matplotlib.pyplot safely, providing a helpful error if missing."""
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "matplotlib is required to generate plots. "
            "Install it (e.g. `pip install matplotlib`) and re-run."
        ) from e
    return plt


def _svt_keep_rank(start_rank: int) -> int:
    """Calculates the rank to keep after SVT truncation (defaulting to half of the start rank)."""
    return max(1, start_rank // 2)


def _steps_per_epoch_from_profiler(rank_to_flops: Dict[int, FlopsStats]) -> int:
    """Infers the number of steps per epoch based on profiled FLOPs statistics."""
    rank = sorted(rank_to_flops.keys())[0]
    stats = rank_to_flops[rank]
    if stats.step_flops <= 0:
        raise ValueError("Invalid step_flops in profiler stats.")
    steps = int(round(stats.epoch_flops / stats.step_flops))
    if steps <= 0:
        raise ValueError("Invalid epoch_flops/step_flops ratio in profiler stats.")
    return steps


def _theoretical_rsvd_linear_flops_per_step(
    *,
    in_features: int,
    out_features: int,
    rank: int,
    tokens_per_step: int,
) -> int:
    """Calculates the theoretical FLOPs for an rSVD factorized linear layer forward pass."""
    if rank <= 0:
        return 0
    # Efficient factorized compute for diagonal C:
    # (1) x @ B^T: 2*tokens*in*rank FLOPs (mul-add)
    # (2) scale by C: tokens*rank FLOPs (mul)
    # (3) @ A^T: 2*tokens*out*rank FLOPs (mul-add)
    return int(
        2 * tokens_per_step * (in_features * rank + out_features * rank)
        + tokens_per_step * rank
    )


def _plot_flops_saved_per_rank(
    *,
    rank_to_flops: Dict[int, FlopsStats],
    out_dir: Path,
    tokens_per_step: int,
    steps_per_epoch: int,
    num_transformer_layers: int,
    projections_per_layer: int,
    hidden_size: int,
) -> List[Path]:
    """Generates a bar chart showing theoretical FLOPs saved per rank removed by SVT."""
    plt = _safe_import_matplotlib()

    ranks = sorted(rank_to_flops.keys())
    x_labels: List[str] = []
    values: List[float] = []
    for start_rank in ranks:
        keep_rank = _svt_keep_rank(start_rank)
        if keep_rank >= start_rank:
            continue
        per_proj_baseline = _theoretical_rsvd_linear_flops_per_step(
            in_features=hidden_size,
            out_features=hidden_size,
            rank=start_rank,
            tokens_per_step=tokens_per_step,
        )
        per_proj_svt = _theoretical_rsvd_linear_flops_per_step(
            in_features=hidden_size,
            out_features=hidden_size,
            rank=keep_rank,
            tokens_per_step=tokens_per_step,
        )
        baseline_epoch_flops = per_proj_baseline * projections_per_layer * num_transformer_layers * steps_per_epoch
        svt_epoch_flops = per_proj_svt * projections_per_layer * num_transformer_layers * steps_per_epoch
        saved_per_removed_rank = (baseline_epoch_flops - svt_epoch_flops) / float(start_rank - keep_rank)
        x_labels.append(f"r={start_rank}→{keep_rank}")
        values.append(saved_per_removed_rank)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x_labels, values)
    ax.set_title("SST-2: Theoretical FLOPs saved per rank removed (SVT vs rSVD baseline)")
    ax.set_xlabel("SVT truncation (start→keep)")
    ax.set_ylabel("FLOPs saved / rank removed (per epoch, QKV only)")
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        out_dir / "sst2_theoretical_flops_saved_per_rank.png",
        out_dir / "sst2_theoretical_flops_saved_per_rank.pdf",
    ]
    for p in outputs:
        fig.savefig(p, dpi=200 if p.suffix == ".png" else None)
    plt.close(fig)
    return outputs


def _plot_flops_baseline_vs_svt(
    *,
    rank_to_flops: Dict[int, FlopsStats],
    out_dir: Path,
    tokens_per_step: int,
    steps_per_epoch: int,
    num_transformer_layers: int,
    projections_per_layer: int,
    hidden_size: int,
) -> List[Path]:
    """Generates a bar chart comparing baseline rSVD epoch FLOPs vs SVT theoretical FLOPs."""
    plt = _safe_import_matplotlib()

    ranks = sorted(rank_to_flops.keys())

    labels: List[str] = []
    baseline_vals: List[float] = []
    svt_vals: List[float] = []
    for start_rank in ranks:
        keep_rank = _svt_keep_rank(start_rank)
        labels.append(f"r={start_rank}")
        per_proj_baseline = _theoretical_rsvd_linear_flops_per_step(
            in_features=hidden_size,
            out_features=hidden_size,
            rank=start_rank,
            tokens_per_step=tokens_per_step,
        )
        per_proj_svt = _theoretical_rsvd_linear_flops_per_step(
            in_features=hidden_size,
            out_features=hidden_size,
            rank=keep_rank,
            tokens_per_step=tokens_per_step,
        )
        baseline_epoch_flops = per_proj_baseline * projections_per_layer * num_transformer_layers * steps_per_epoch
        svt_epoch_flops = per_proj_svt * projections_per_layer * num_transformer_layers * steps_per_epoch
        baseline_vals.append(baseline_epoch_flops)
        svt_vals.append(svt_epoch_flops)

    width = 0.42
    xs = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar([x - width / 2 for x in xs], baseline_vals, width=width, label="rSVD baseline (no SVT)")
    ax.bar([x + width / 2 for x in xs], svt_vals, width=width, label="Theoretical SVT (keep r//2)")

    ax.set_title("SST-2: Theoretical epoch FLOPs (SVT vs rSVD baseline)")
    ax.set_xlabel("Initial rSVD rank")
    ax.set_ylabel("FLOPs per epoch (QKV projections only)")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend(fontsize=9, ncol=2)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        out_dir / "sst2_theoretical_epoch_flops_baseline_vs_svt.png",
        out_dir / "sst2_theoretical_epoch_flops_baseline_vs_svt.pdf",
    ]
    for p in outputs:
        fig.savefig(p, dpi=200 if p.suffix == ".png" else None)
    plt.close(fig)
    return outputs


def _plot_training_loss_vs_epoch(
    *,
    rank_to_loss: Dict[int, List[EpochLossRow]],
    out_dir: Path,
) -> List[Path]:
    """Plots training loss vs epoch for multiple rSVD ranks."""
    plt = _safe_import_matplotlib()

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for rank in sorted(rank_to_loss.keys()):
        rows = sorted(rank_to_loss[rank], key=lambda r: r.epoch)
        epochs = [r.epoch for r in rows]
        train = [r.train_loss for r in rows]
        ax.plot(epochs, train, marker="o", linewidth=1.8, label=f"train (r={rank})")

    ax.set_title("SST-2: rSVD Weight Reduction + SVT training loss vs epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        out_dir / "sst2_training_loss_vs_epoch.png",
        out_dir / "sst2_training_loss_vs_epoch.pdf",
    ]
    for p in outputs:
        fig.savefig(p, dpi=200 if p.suffix == ".png" else None)
    plt.close(fig)
    return outputs


def _infer_baseline_rank(available_ranks: Iterable[int]) -> int:
    ranks = sorted(set(available_ranks))
    if not ranks:
        raise ValueError("No rank directories found.")
    return ranks[-1]


def main(argv: Optional[List[str]] = None) -> int:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Generate SST-2 plots from existing CSVs in ./graph/sst2/r*/ (or ./graph/sst2 when run inside graph/)."
    )
    parser.add_argument(
        "--sst2-dir",
        type=Path,
        default=script_dir / "sst2",
        help="Path to SST-2 metrics directory (default: <script_dir>/sst2)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=script_dir / "plots",
        help="Output directory for plots (default: <script_dir>/plots)",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size (default: 32)")
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length used for tokenization (default: 128)")
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=768,
        help="DistilBERT hidden size (default: 768)",
    )
    parser.add_argument(
        "--num-transformer-layers",
        type=int,
        default=6,
        help="Number of transformer layers (default: 6 for DistilBERT)",
    )
    parser.add_argument(
        "--projections-per-layer",
        type=int,
        default=3,
        help="Number of factorized projections per layer (default: 3 for Q/K/V)",
    )
    args = parser.parse_args(argv)

    sst2_dir: Path = args.sst2_dir
    if not sst2_dir.exists():
        raise FileNotFoundError(f"Missing SST-2 directory: {sst2_dir}")

    rank_to_flops: Dict[int, FlopsStats] = {}
    rank_to_loss: Dict[int, List[EpochLossRow]] = {}
    for rank, rank_dir in _iter_rank_dirs(sst2_dir):
        flops_path = rank_dir / "flops_profiler_stats.csv"
        loss_path = rank_dir / "epoch_loss.csv"
        if flops_path.exists():
            rank_to_flops[rank] = _read_flops_stats(flops_path)
        if loss_path.exists():
            rank_to_loss[rank] = _read_epoch_loss(loss_path)

    if not rank_to_flops:
        raise FileNotFoundError(f"No flops_profiler_stats.csv found under {sst2_dir}/r*/")
    if not rank_to_loss:
        raise FileNotFoundError(f"No epoch_loss.csv found under {sst2_dir}/r*/")

    tokens_per_step = int(args.batch_size) * int(args.seq_len)
    steps_per_epoch = _steps_per_epoch_from_profiler(rank_to_flops)

    outputs: List[Path] = []
    outputs += _plot_flops_baseline_vs_svt(
        rank_to_flops=rank_to_flops,
        out_dir=args.out_dir,
        tokens_per_step=tokens_per_step,
        steps_per_epoch=steps_per_epoch,
        num_transformer_layers=int(args.num_transformer_layers),
        projections_per_layer=int(args.projections_per_layer),
        hidden_size=int(args.hidden_size),
    )
    outputs += _plot_flops_saved_per_rank(
        rank_to_flops=rank_to_flops,
        out_dir=args.out_dir,
        tokens_per_step=tokens_per_step,
        steps_per_epoch=steps_per_epoch,
        num_transformer_layers=int(args.num_transformer_layers),
        projections_per_layer=int(args.projections_per_layer),
        hidden_size=int(args.hidden_size),
    )
    outputs += _plot_training_loss_vs_epoch(rank_to_loss=rank_to_loss, out_dir=args.out_dir)

    print("Wrote:")
    for p in outputs:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
