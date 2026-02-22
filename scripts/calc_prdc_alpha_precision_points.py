#!/usr/bin/env python3
from __future__ import annotations

import argparse
from typing import Any, Dict, Tuple

import pandas as pd

from _points_common import (
    project_root,
    iter_pairs,
    read_table,
    align_columns,
    build_loader,
    resolve_metric_class_in_module,
    instantiate_evaluator,
    _norm,
)

MODULE = "synthcity.metrics.eval_statistical"

ALLOWED = {"prdc", "alpha_precision"}

# friendly_name -> synthcity_key
SUBMETRICS: dict[str, dict[str, str]] = {
    "prdc": {
        "precision": "precision",
        "recall": "recall",
        "density": "density",
        "coverage": "coverage",
    },
    # SynthCity возвращает delta_* и authenticity_* (OC/naive). Берём OC как дефолт.
    "alpha_precision": {
        "alpha_precision": "delta_precision_alpha_OC",
        "beta_recall": "delta_coverage_beta_OC",
        "authenticity": "authenticity_OC",
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--metric", required=True, help="prdc | alpha_precision")
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

    for c in numeric_cols:
        real[c] = pd.to_numeric(real[c], errors="coerce")
        syn[c] = pd.to_numeric(syn[c], errors="coerce")

    for c in numeric_cols:
        med = real[c].median(skipna=True)
        if pd.isna(med):
            med = 0.0
        real[c] = real[c].fillna(med)
        syn[c] = syn[c].fillna(med)

    if not cat_cols:
        return real[numeric_cols].astype("float64"), syn[numeric_cols].astype("float64")

    for c in cat_cols:
        real[c] = real[c].astype("string").fillna("MISSING")
        syn[c] = syn[c].astype("string").fillna("MISSING")

    combined_cat = pd.concat([real[cat_cols], syn[cat_cols]], axis=0, ignore_index=True)
    dummies = pd.get_dummies(combined_cat, columns=cat_cols, dtype="float64")

    n_real = len(real)
    d_real = dummies.iloc[:n_real].reset_index(drop=True)
    d_syn = dummies.iloc[n_real:].reset_index(drop=True)

    real_num = real[numeric_cols].reset_index(drop=True).astype("float64")
    syn_num = syn[numeric_cols].reset_index(drop=True).astype("float64")

    real_out = pd.concat([real_num, d_real], axis=1)
    syn_out = pd.concat([syn_num, d_syn], axis=1)

    syn_out = syn_out[real_out.columns]
    return real_out, syn_out


def _flatten(obj: Any, prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    if isinstance(obj, pd.DataFrame):
        if len(obj) == 0:
            return out
        s = obj.mean(axis=0, numeric_only=True)
        return _flatten(s, prefix=prefix)

    if isinstance(obj, pd.Series):
        for k, v in obj.to_dict().items():
            kk = f"{prefix}.{k}" if prefix else str(k)
            out.update(_flatten(v, kk))
        return out

    if isinstance(obj, dict):
        for k, v in obj.items():
            kk = f"{prefix}.{k}" if prefix else str(k)
            out.update(_flatten(v, kk))
        return out

    if prefix:
        out[prefix] = obj
    return out


def _as_float(x: Any) -> float:
    try:
        if x is None:
            return float("nan")
        if isinstance(x, (int, float)):
            return float(x)
        return float(x)
    except Exception:
        return float("nan")


def extract_submetric_scores(raw: Any, metric_canonical: str) -> Dict[str, float]:
    """
    raw обычно dict от SynthCity evaluate().
    Достаём нужные ключи (с учётом алиасов и возможных префиксов).
    """
    mapping = SUBMETRICS[metric_canonical]  # friendly -> actual
    flat = _flatten(raw)

    norm_map: Dict[str, Tuple[str, Any]] = {}
    for k, v in flat.items():
        nk_full = _norm(k)
        nk_last = _norm(k.split(".")[-1])
        norm_map.setdefault(nk_full, (k, v))
        norm_map.setdefault(nk_last, (k, v))

    out: Dict[str, float] = {}
    missing: list[str] = []

    for friendly, actual_key in mapping.items():
        n_actual = _norm(actual_key)

        candidates = [
            n_actual,
            _norm(f"{metric_canonical}.{actual_key}"),
            _norm(f"{metric_canonical}_{actual_key}"),
            _norm(actual_key.split(".")[-1]),
        ]

        found = None
        for cand in candidates:
            if cand in norm_map:
                found = norm_map[cand]
                break

        if found is None:
            missing.append(friendly)
            out[friendly] = float("nan")
        else:
            _, val = found
            out[friendly] = _as_float(val)

    if missing:
        raise KeyError(
            f"Missing submetrics in raw output: {missing}. Raw keys: {sorted(flat.keys())[:50]}"
        )

    return out


def main() -> int:
    args = parse_args()
    metric_in = _norm(args.metric)
    metric_canonical = metric_in

    if metric_canonical not in ALLOWED:
        raise SystemExit(f"Unsupported metric '{args.metric}'. Allowed: {sorted(ALLOWED)}")

    metric_cls = resolve_metric_class_in_module(MODULE, metric_canonical)

    root = project_root()
    out_dir = root / "metric_data" / "statistical_tests"
    out_dir.mkdir(parents=True, exist_ok=True)

    workspace = root / ".synthcity_workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    evaluator = instantiate_evaluator(metric_cls, workspace=workspace, seed=0)

    # friendly names (как ты хочешь видеть в файлах)
    friendly_names = list(SUBMETRICS[metric_canonical].keys())
    rows_by_sub: dict[str, list[dict[str, Any]]] = {s: [] for s in friendly_names}

    for pair in iter_pairs(args.methods):
        real_df = None
        syn_df = None
        try:
            real_df = read_table(pair.real_path)
            syn_df = read_table(pair.synth_path)
            real_df, syn_df = align_columns(real_df, syn_df)

            real_df, syn_df = prepare_statistical_pair(real_df, syn_df)

            real_loader = build_loader(real_df)
            syn_loader = build_loader(syn_df)

            # ВАЖНО: для multi-output метрик берём dict через evaluate(), НЕ evaluate_default()
            raw = evaluator.evaluate(real_loader, syn_loader)

            scores = extract_submetric_scores(raw, metric_canonical)

            for sub_name, sub_score in scores.items():
                rows_by_sub[sub_name].append(
                    {
                        "metric_group": metric_cls.name(),
                        "metric": sub_name,
                        "metric_requested": args.metric,
                        "method": pair.method,
                        "step": pair.step,
                        "score": sub_score,
                        "n_real": len(real_df),
                        "n_synth": len(syn_df),
                        "real_file": pair.real_path.name,
                        "synth_file": pair.synth_path.name,
                        "error": "",
                    }
                )

        except Exception as e:
            n_real = len(real_df) if isinstance(real_df, pd.DataFrame) else None
            n_syn = len(syn_df) if isinstance(syn_df, pd.DataFrame) else None

            for sub_name in friendly_names:
                rows_by_sub[sub_name].append(
                    {
                        "metric_group": metric_cls.name(),
                        "metric": sub_name,
                        "metric_requested": args.metric,
                        "method": pair.method,
                        "step": pair.step,
                        "score": float("nan"),
                        "n_real": n_real,
                        "n_synth": n_syn,
                        "real_file": pair.real_path.name,
                        "synth_file": pair.synth_path.name,
                        "error": str(e),
                    }
                )

    for sub_name, rows in rows_by_sub.items():
        df = pd.DataFrame(rows).sort_values(["method", "step"]).reset_index(drop=True)
        out_path = out_dir / f"{metric_in}__{_norm(sub_name)}.csv"
        df.to_csv(out_path, index=False, encoding="utf-8", na_rep="")
        print(f"[OK] Saved: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
