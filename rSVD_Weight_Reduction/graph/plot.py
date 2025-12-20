#!/usr/bin/env python3
"""
Plot scripts for rSVD_Weight_Reduction (SST-2 only).

Generates:
  1) Bar chart: peak memory vs rank
  2) Line chart: training loss vs epoch (one curve per rank)

Expected input layout (relative to this script by default):
  sst2/r{rank}/total_program_memory.csv
  sst2/r{rank}/epoch_peak_memory.csv
  sst2/r{rank}/epoch_loss.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal


MemorySource = Literal["auto", "total_program", "epoch_peak"]


def _require_matplotlib() -> None:
    try:
        import matplotlib  # noqa: F401
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: matplotlib.\n"
            "Install it (example): python3 -m pip install matplotlib\n"
            "Then re-run this script."
        ) from exc


def _bytes_to_gib(num_bytes: float) -> float:
    return num_bytes / (1024.0**3)


def _is_finite_number(value: float) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]


def _parse_rank_dirname(name: str) -> int | None:
    match = re.fullmatch(r"r(\d+)", name)
    if not match:
        return None
    return int(match.group(1))


def discover_ranks(dataset_dir: Path) -> list[int]:
    ranks: list[int] = []
    for child in dataset_dir.iterdir():
        if not child.is_dir():
            continue
        parsed = _parse_rank_dirname(child.name)
        if parsed is not None:
            ranks.append(parsed)
    return sorted(set(ranks))


def read_total_program_peak_bytes(rank_dir: Path) -> float | None:
    path = rank_dir / "total_program_memory.csv"
    if not path.exists():
        return None
    rows = _read_csv_rows(path)
    for row in rows:
        if row.get("metric") == "program_total_peak_memory":
            value = row.get("value_bytes", "").strip()
            if not value:
                return None
            try:
                parsed = float(value)
            except ValueError:
                return None
            return parsed
    return None


def read_epoch_peak_bytes(rank_dir: Path) -> float | None:
    path = rank_dir / "epoch_peak_memory.csv"
    if not path.exists():
        return None
    rows = _read_csv_rows(path)
    peaks: list[float] = []
    for row in rows:
        value = row.get("peak_memory_bytes", "").strip()
        if not value:
            continue
        try:
            peaks.append(float(value))
        except ValueError:
            continue
    if not peaks:
        return None
    return max(peaks)


def read_train_loss_curve(rank_dir: Path) -> tuple[list[int], list[float]]:
    path = rank_dir / "epoch_loss.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    rows = _read_csv_rows(path)
    epochs: list[int] = []
    train_losses: list[float] = []
    for row in rows:
        epoch_str = (row.get("epoch") or "").strip()
        train_loss_str = (row.get("train_loss") or "").strip()
        if not epoch_str or not train_loss_str:
            continue
        try:
            epoch = int(float(epoch_str))
            loss = float(train_loss_str)
        except ValueError:
            continue
        if _is_finite_number(loss):
            epochs.append(epoch)
            train_losses.append(loss)
    if not epochs:
        raise ValueError(f"No train_loss values found in {path}")
    paired = sorted(zip(epochs, train_losses), key=lambda p: p[0])
    epochs_sorted = [p[0] for p in paired]
    losses_sorted = [p[1] for p in paired]
    return epochs_sorted, losses_sorted


@dataclass(frozen=True)
class RankMemory:
    rank: int
    peak_bytes: float
    source: str


def collect_peak_memory(
    dataset_dir: Path, ranks: Iterable[int], source: MemorySource
) -> list[RankMemory]:
    results: list[RankMemory] = []
    for rank in ranks:
        rank_dir = dataset_dir / f"r{rank}"
        total_program = read_total_program_peak_bytes(rank_dir)
        epoch_peak = read_epoch_peak_bytes(rank_dir)

        if source == "total_program":
            peak = total_program
            src = "total_program_memory.csv"
        elif source == "epoch_peak":
            peak = epoch_peak
            src = "epoch_peak_memory.csv"
        else:
            if total_program is not None:
                peak = total_program
                src = "total_program_memory.csv"
            else:
                peak = epoch_peak
                src = "epoch_peak_memory.csv"

        if peak is None:
            raise SystemExit(
                f"Could not determine peak memory for rank {rank} under {rank_dir} "
                f"(looked for total_program_memory.csv and epoch_peak_memory.csv)."
            )

        results.append(RankMemory(rank=rank, peak_bytes=float(peak), source=src))

    return sorted(results, key=lambda r: r.rank)


def write_memory_summary_csv(rows: list[RankMemory], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "peak_bytes", "peak_gib", "source"])
        for row in rows:
            writer.writerow(
                [row.rank, int(row.peak_bytes), f"{_bytes_to_gib(row.peak_bytes):.6f}", row.source]
            )


def plot_peak_memory_bar(rows: list[RankMemory], out_path: Path) -> None:
    _require_matplotlib()
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)

    ranks = [r.rank for r in rows]
    peak_gib = [_bytes_to_gib(r.peak_bytes) for r in rows]
    sources = [r.source for r in rows]

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        pass

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar([str(r) for r in ranks], peak_gib)

    ax.set_title("SST-2 rSVD Weight Reduction: Peak GPU Memory vs Rank")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Peak GPU memory (GiB)")

    for bar, value in zip(bars, peak_gib):
        ax.annotate(
            f"{value:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    source_note = " (auto)" if len(set(sources)) > 1 else ""
    ax.text(
        0.99,
        0.01,
        f"Source: {', '.join(sorted(set(sources)))}{source_note}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        alpha=0.8,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_train_loss_curves(dataset_dir: Path, ranks: Iterable[int], out_path: Path) -> None:
    _require_matplotlib()
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        pass

    fig, ax = plt.subplots(figsize=(9, 5))

    for rank in sorted(ranks):
        epochs, losses = read_train_loss_curve(dataset_dir / f"r{rank}")
        ax.plot(epochs, losses, marker="o", linewidth=2, markersize=3, label=f"r{rank}")

    ax.set_title("SST-2 rSVD Weight Reduction: Training Loss vs Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss")
    ax.legend(title="Rank", ncol=3, fontsize=9, title_fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Generate SST-2 plots for rSVD Weight Reduction (peak memory vs rank; training loss vs epoch)."
    )
    parser.add_argument(
        "--graph-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Root folder containing dataset subfolders like sst2/ (default: this script's directory).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "plots",
        help="Directory for output plots (default: ./plots).",
    )
    parser.add_argument(
        "--memory-source",
        choices=["auto", "total_program", "epoch_peak"],
        default="auto",
        help="Which memory file to use per rank.",
    )
    parser.add_argument(
        "--ranks",
        type=str,
        default="",
        help="Comma-separated ranks to include (default: auto-discover from sst2/r*).",
    )
    args = parser.parse_args(argv)

    dataset_dir = args.graph_root / "sst2"
    if not dataset_dir.exists():
        raise SystemExit(f"Missing dataset directory: {dataset_dir}")

    if args.ranks.strip():
        ranks = [int(x) for x in args.ranks.split(",") if x.strip()]
    else:
        ranks = discover_ranks(dataset_dir)

    if not ranks:
        raise SystemExit(f"No ranks found under {dataset_dir} (expected folders like r4, r8, ...)")

    out_dir: Path = args.out_dir
    memory_rows = collect_peak_memory(dataset_dir, ranks, args.memory_source)

    write_memory_summary_csv(memory_rows, out_dir / "sst2_peak_memory_vs_rank.csv")
    plot_peak_memory_bar(memory_rows, out_dir / "sst2_peak_memory_vs_rank.png")
    plot_train_loss_curves(dataset_dir, ranks, out_dir / "sst2_train_loss_vs_epoch.png")

    print(f"Wrote: {(out_dir / 'sst2_peak_memory_vs_rank.csv').resolve()}")
    print(f"Wrote: {(out_dir / 'sst2_peak_memory_vs_rank.png').resolve()}")
    print(f"Wrote: {(out_dir / 'sst2_train_loss_vs_epoch.png').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
