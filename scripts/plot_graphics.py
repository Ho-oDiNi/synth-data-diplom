#!/usr/bin/env python3
# plot_graphics.py
"""
Рисует графики из metric_data/*.csv в graphics/*.png

Требования:
- Y: 0..100
- X: шаги от 0 до 2112 с шагом 192
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--points-dir", default="quality", help="Папка с CSV точками")
    p.add_argument("--out-dir", default="graphics", help="Папка для PNG графиков")
    p.add_argument("--metric", default=None, help="Какой metric.csv рисовать (без .csv)")
    p.add_argument("--absolute", default=False, action="store_true", help="Рисовать абсолютные значения (не проценты)")
    p.add_argument("--all", action="store_true", help="Рисовать все CSV из points-dir")
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


def plot_one(csv_path: Path, out_dir: Path, dpi: int, absolute: bool = False) -> Path:
    df = pd.read_csv(csv_path)

    required = {"metric", "method", "step", "score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path.name}: missing columns: {sorted(missing)}")

    metric_name = str(df["metric"].iloc[0]) if len(df) else csv_path.stem

    # step -> int
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df = df.dropna(subset=["step"]).copy()
    df["step"] = df["step"].astype(int)

    # score -> percent
    if not absolute:
        df["score"] = pd.to_numeric(df["score"], errors="coerce") * 100.0
    else:
        df["score"] = pd.to_numeric(df["score"], errors="coerce")

    # sort
    df = df.sort_values(["method", "step"])

    fig = plt.figure()
    ax = plt.gca()

    for method, g in df.groupby("method", sort=True):
        g = g.sort_values("step")
        ax.plot(g["step"], g["score"], marker="o", linewidth=1.5, label=str(method))

    # axis requirements
    ax.set_xlim(192, 2112)
    if not absolute:
        ax.set_ylim(0, 100)
    ax.set_xticks(list(range(192, 2112 + 1, 192)))

    ax.set_title(metric_name)
    ax.set_xlabel("step")
    if not absolute:
        ax.set_ylabel("score (%)")
    else:
        ax.set_ylabel("score")
    ax.grid(True, linewidth=0.5)
    ax.legend(title="method", fontsize=9)

    out_path = out_dir / f"{metric_name}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def main() -> int:
    args = parse_args()

    points_dir = Path("metric_data", args.points_dir)
    out_dir = Path(args.out_dir, args.points_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.all:
        csv_files = sorted(points_dir.glob("*.csv"))
        if not csv_files:
            raise SystemExit(f"No CSV files found in {points_dir}")
        for csv_path in csv_files:
            out_path = plot_one(csv_path, out_dir=out_dir, dpi=args.dpi, absolute=args.absolute)
            print(f"[OK] Saved: {out_path}")
        return 0

    if not args.metric:
        raise SystemExit("Use --metric <name> or --all")

    csv_path = points_dir / f"{args.metric}.csv"
    if not csv_path.exists():
        raise SystemExit(f"Not found: {csv_path}")

    out_path = plot_one(csv_path, out_dir=out_dir, dpi=args.dpi, absolute=args.absolute)
    print(f"[OK] Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
