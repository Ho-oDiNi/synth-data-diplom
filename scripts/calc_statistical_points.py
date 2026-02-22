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
    _norm,
)

MODULE = "synthcity.metrics.eval_statistical"

ALIASES = {
    "inverse_kl_divergence": "inv_kl_divergence",  # твой кейс
}

ALLOWED = {
    "inv_kl_divergence",
    "ks_test",
    "chi_squared_test",
    "jensenshannon_dist",
    "wasserstein_dist",
    "max_mean_discrepancy",
    "prdc",
    "alpha_precision",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--metric", required=True)
    p.add_argument("--methods", nargs="*", default=None)
    return p.parse_args()


def prepare_statistical_pair(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    *,
    min_numeric_ratio: float = 0.9,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Подготовка для eval_statistical: ВСЕ колонки -> numeric.
    - numeric: to_numeric + fillna median(real)
    - categorical: string + fillna 'MISSING' + one-hot (get_dummies) по объединению real+synth
    """
    real = real_df.copy()
    syn = synth_df.copy()

    numeric_cols: list[str] = []
    cat_cols: list[str] = []

    for c in real.columns:
        if pd.api.types.is_numeric_dtype(real[c].dtype):
            numeric_cols.append(c)
            continue

        parsed = pd.to_numeric(real[c], errors="coerce")
        ratio = float(parsed.notna().mean()) if len(parsed) else 0.0
        if ratio >= min_numeric_ratio:
            numeric_cols.append(c)
        else:
            cat_cols.append(c)

    # numeric -> float
    for c in numeric_cols:
        real[c] = pd.to_numeric(real[c], errors="coerce")
        syn[c] = pd.to_numeric(syn[c], errors="coerce")

    # fillna numeric (median from real)
    for c in numeric_cols:
        med = real[c].median(skipna=True)
        if pd.isna(med):
            med = 0.0
        real[c] = real[c].fillna(med)
        syn[c] = syn[c].fillna(med)

    if not cat_cols:
        # всё numeric уже
        real_out = real[numeric_cols].astype("float64")
        syn_out = syn[numeric_cols].astype("float64")
        return real_out, syn_out

    # categorical -> string + fillna + one-hot on (real+syn) to guarantee same columns
    for c in cat_cols:
        real[c] = real[c].astype("string").fillna("MISSING")
        syn[c] = syn[c].astype("string").fillna("MISSING")

    combined_cat = pd.concat([real[cat_cols], syn[cat_cols]], axis=0, ignore_index=True)
    dummies = pd.get_dummies(combined_cat, columns=cat_cols, dtype="float64")

    n_real = len(real)
    d_real = dummies.iloc[:n_real].reset_index(drop=True)
    d_syn = dummies.iloc[n_real:].reset_index(drop=True)

    # собрать итоговые таблицы (numeric + one-hot)
    real_num = real[numeric_cols].reset_index(drop=True).astype("float64")
    syn_num = syn[numeric_cols].reset_index(drop=True).astype("float64")

    real_out = pd.concat([real_num, d_real], axis=1)
    syn_out = pd.concat([syn_num, d_syn], axis=1)

    # гарантируем одинаковый порядок колонок
    syn_out = syn_out[real_out.columns]

    return real_out, syn_out


def main() -> int:
    args = parse_args()
    metric_in = _norm(args.metric)
    metric_canonical = ALIASES.get(metric_in, metric_in)

    if metric_canonical not in ALLOWED:
        raise SystemExit(f"Unsupported statistical metric '{args.metric}'. Allowed: {sorted(ALLOWED)}")

    metric_cls = resolve_metric_class_in_module(MODULE, metric_canonical)

    root = project_root()
    out_dir = root / "metric_data" / "statistical_tests"
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

            # Ключевое исправление: для statistical -> numeric-only (one-hot)
            real_df, syn_df = prepare_statistical_pair(real_df, syn_df)

            real_loader = build_loader(real_df)
            syn_loader = build_loader(syn_df)

            raw = (
                evaluator.evaluate_default(real_loader, syn_loader)
                if hasattr(evaluator, "evaluate_default")
                else evaluator.evaluate(real_loader, syn_loader)
            )
            score = reduce_metric_output(raw)

            rows.append(
                {
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
                }
            )
        except Exception as e:
            rows.append(
                {
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
                }
            )

    df = pd.DataFrame(rows).sort_values(["method", "step"]).reset_index(drop=True)
    out_path = out_dir / f"{metric_in}.csv"
    df.to_csv(out_path, index=False, encoding="utf-8", na_rep="")
    print(f"[OK] Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
