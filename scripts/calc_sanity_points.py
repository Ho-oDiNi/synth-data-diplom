#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from _points_common import (
    project_root,
    iter_pairs,
    read_table,
    align_columns,
    build_loader,
    resolve_metric_class_in_module,
    instantiate_evaluator,
    reduce_metric_output,
    prepare_tabular_pair,
    _norm,
)

# Sanity metrics live here
MODULE = "synthcity.metrics.eval_sanity"

# минимальные алиасы (под твои кейсы)
ALIASES = {
    # нет алиасов — оставлено на будущее
}

# Явный белый список (чтобы не гадать, где метрика лежит)
ALLOWED = {
    "data_mismatch",
    "nearest_syn_neighbor_distance",
    "common_rows_proportion",
    "close_values_probability",
    "distant_values_probability",
}

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--metric", required=True)
    p.add_argument("--methods", nargs="*", default=None)
    return p.parse_args()

def main() -> int:
    args = parse_args()
    metric_in = _norm(args.metric)
    metric_canonical = ALIASES.get(metric_in, metric_in)

    if metric_canonical not in ALLOWED:
        raise SystemExit(f"Unsupported sanity metric '{args.metric}'. Allowed: {sorted(ALLOWED)}")

    metric_cls = resolve_metric_class_in_module(MODULE, metric_canonical)

    root = project_root()
    out_dir = root / "metric_data" / "sanity_checks"
    out_dir.mkdir(parents=True, exist_ok=True)

    workspace = root / ".synthcity_workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    evaluator = instantiate_evaluator(metric_cls, workspace=workspace, seed=0)

    rows = []
    for pair in iter_pairs(args.methods):
        try:
            real_df = read_table(pair.real_path)
            syn_df = read_table(pair.synth_path)
            real_df, syn_df = align_columns(real_df, syn_df)

            # data_mismatch важно считать на сырых dtypes
            if metric_canonical != "data_mismatch":
                real_df, syn_df = prepare_tabular_pair(real_df, syn_df, fillna=True)

            real_loader = build_loader(real_df)
            syn_loader = build_loader(syn_df)

            raw = evaluator.evaluate_default(real_loader, syn_loader) if hasattr(evaluator, "evaluate_default") else evaluator.evaluate(real_loader, syn_loader)
            score = reduce_metric_output(raw)

            rows.append({
                "metric": metric_cls.name(),
                "metric_requested": args.metric,
                "method": pair.method,
                "step": pair.step,
                "score": score,
                "n_real": len(real_df),
                "n_synth": len(syn_df),
                "real_file": pair.real_path.name,
                "synth_file": pair.synth_path.name,
                "error": "",
            })
        except Exception as e:
            rows.append({
                "metric": metric_cls.name(),
                "metric_requested": args.metric,
                "method": pair.method,
                "step": pair.step,
                "score": float("nan"),
                "n_real": None,
                "n_synth": None,
                "real_file": pair.real_path.name,
                "synth_file": pair.synth_path.name,
                "error": str(e),
            })

    df = pd.DataFrame(rows).sort_values(["method", "step"]).reset_index(drop=True)
    out_path = out_dir / f"{metric_in}.csv"  # файл = то, что ты ввёл
    df.to_csv(out_path, index=False, encoding="utf-8", na_rep="")
    print(f"[OK] Saved: {out_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
