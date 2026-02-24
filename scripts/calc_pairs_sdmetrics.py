#!/usr/bin/env python3
# scripts/calc_pairs_sdmetrics.py
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from sdmetrics.column_pairs import CorrelationSimilarity, ContingencySimilarity


# -----------------------------
# I/O helpers
# -----------------------------
@dataclass(frozen=True)
class Pair:
    method: str
    step: int
    real_path: Path
    synth_path: Path


def norm(x: str) -> str:
    return str(x).strip().lower()


def default_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def read_table(path: Path) -> pd.DataFrame:
    # extend if needed
    return pd.read_parquet(path)


def align_columns(real_df: pd.DataFrame, syn_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    common = [c for c in real_df.columns if c in syn_df.columns]
    return real_df[common].copy(), syn_df[common].copy()


def iter_pairs(root: Path, methods: Optional[List[str]] = None) -> Iterable[Pair]:
    real_dir = root / "real_data"
    synth_dir = root / "synth_data"

    real_files = sorted(real_dir.glob("step_*_real.parquet"))
    steps: List[Tuple[int, Path]] = []
    for rf in real_files:
        try:
            step = int(rf.stem.split("_")[1])
            steps.append((step, rf))
        except Exception:
            continue

    if methods is None:
        methods = sorted([p.name for p in synth_dir.iterdir() if p.is_dir()])

    for method in methods:
        for step, rf in steps:
            sf = synth_dir / method / f"step_{step}_synth.parquet"
            if sf.exists():
                yield Pair(method=method, step=step, real_path=rf, synth_path=sf)


# -----------------------------
# Schemas (optional)
# -----------------------------
SCHEMAS: Dict[str, Dict[str, List[str]]] = {
    "obesity": {
        "numeric": ["Age", "Height", "Weight"],
        "categorical": [
            "Gender",
            "family_history_with_overweight",
            "FAVC",
            "FCVC",
            "NCP",
            "CAEC",
            "SMOKE",
            "CH2O",
            "SCC",
            "FAF",
            "TUE",
            "CALC",
            "MTRANS",
            "NObeyesdad",
        ],
    }
}


def apply_schema(
    real_df: pd.DataFrame,
    syn_df: pd.DataFrame,
    schema_name: str,
    extra_num: Optional[List[str]] = None,
    extra_cat: Optional[List[str]] = None,
) -> Tuple[List[str], List[str]]:
    schema = SCHEMAS.get(schema_name)
    if not schema:
        raise ValueError(f"Unknown schema '{schema_name}'. Available: {sorted(SCHEMAS)}")

    cols = [c for c in real_df.columns if c in syn_df.columns]
    num = [c for c in schema["numeric"] if c in cols]
    cat = [c for c in schema["categorical"] if c in cols]

    # allow extending/overriding with explicit lists
    if extra_num:
        for c in extra_num:
            if c in cols and c not in num:
                num.append(c)
            if c in cat:
                cat.remove(c)
    if extra_cat:
        for c in extra_cat:
            if c in cols and c not in cat:
                cat.append(c)
            if c in num:
                num.remove(c)

    return num, cat


def infer_types_fallback(
    real_df: pd.DataFrame,
    syn_df: pd.DataFrame,
    num_cols: Optional[List[str]],
    cat_cols: Optional[List[str]],
    cat_unique_max: int,
    cat_unique_ratio: float,
) -> Tuple[List[str], List[str]]:
    cols = [c for c in real_df.columns if c in syn_df.columns]
    explicit_num = set(num_cols or [])
    explicit_cat = set(cat_cols or [])

    numeric: List[str] = []
    categorical: List[str] = []
    n = max(len(real_df), 1)

    for c in cols:
        if c in explicit_num:
            numeric.append(c)
            continue
        if c in explicit_cat:
            categorical.append(c)
            continue

        r = real_df[c]
        if (
            pd.api.types.is_bool_dtype(r.dtype)
            or pd.api.types.is_object_dtype(r.dtype)
            or pd.api.types.is_categorical_dtype(r.dtype)
            or pd.api.types.is_string_dtype(r.dtype)
        ):
            categorical.append(c)
            continue

        if pd.api.types.is_integer_dtype(r.dtype):
            nunique = int(r.nunique(dropna=True))
            if nunique <= cat_unique_max or (nunique / n) <= cat_unique_ratio:
                categorical.append(c)
            else:
                numeric.append(c)
            continue

        if pd.api.types.is_numeric_dtype(r.dtype):
            numeric.append(c)
            continue

        categorical.append(c)

    # stable order
    numeric = [c for c in cols if c in set(numeric)]
    categorical = [c for c in cols if c in set(categorical)]
    return numeric, categorical


# -----------------------------
# Coercions
# -----------------------------
def coerce_cat(s: pd.Series) -> pd.Series:
    out = s.astype("object")
    out = out.where(~out.isna(), other="__NA__")
    return out.astype(str)


def coerce_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def bin_numeric_on_real_quantiles(
    s_real: pd.Series,
    s_syn: pd.Series,
    bins: int,
) -> Tuple[pd.Series, pd.Series]:
    r = coerce_num(s_real)
    y = coerce_num(s_syn)
    r_valid = r.dropna()

    if len(r_valid) == 0:
        br = pd.Series(["__ALL_NA__"] * len(r), index=r.index)
        by = pd.Series(["__ALL_NA__"] * len(y), index=y.index)
        return br, by

    try:
        _, edges = pd.qcut(r_valid, q=bins, retbins=True, duplicates="drop")
        edges = np.unique(edges)
        if len(edges) < 3:
            raise ValueError("Not enough unique edges")
    except Exception:
        mn = float(np.nanmin(r_valid.to_numpy()))
        mx = float(np.nanmax(r_valid.to_numpy()))
        if not np.isfinite(mn) or not np.isfinite(mx) or mn == mx:
            edges = np.array([mn - 1.0, mn, mn + 1.0], dtype=float)
        else:
            edges = np.linspace(mn, mx, num=bins + 1)

    br = pd.cut(r, bins=edges, include_lowest=True)
    by = pd.cut(y, bins=edges, include_lowest=True)
    br = br.astype("object").where(~pd.isna(br), other="__NA__").astype(str)
    by = by.astype("object").where(~pd.isna(by), other="__NA__").astype(str)
    return br, by


# -----------------------------
# Pair iterators
# -----------------------------
def pairs(cols: List[str]) -> Iterable[Tuple[str, str]]:
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            yield cols[i], cols[j]


def pairs_mixed(num_cols: List[str], cat_cols: List[str]) -> Iterable[Tuple[str, str]]:
    for n in num_cols:
        for c in cat_cols:
            yield n, c


def mean_or_nan(values: List[float]) -> float:
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(np.mean(vals)) if vals else float("nan")


def drop_constant_cols(
    real_df: pd.DataFrame,
    syn_df: pd.DataFrame,
    cols: List[str],
    is_cat: bool,
) -> Tuple[List[str], List[str]]:
    good: List[str] = []
    dropped: List[str] = []
    for c in cols:
        r = coerce_cat(real_df[c]) if is_cat else coerce_num(real_df[c])
        s = coerce_cat(syn_df[c]) if is_cat else coerce_num(syn_df[c])

        if is_cat:
            nr = int(pd.Series(r).nunique(dropna=False))
            ns = int(pd.Series(s).nunique(dropna=False))
        else:
            # numeric const check: <2 unique finite values
            nr = int(pd.Series(r).dropna().nunique())
            ns = int(pd.Series(s).dropna().nunique())

        if nr < 2 or ns < 2:
            dropped.append(f"{c}(real={nr},synth={ns})")
        else:
            good.append(c)
    return good, dropped


# -----------------------------
# Metrics (correct sdmetrics signatures)
# -----------------------------
def compute_dpcm(real_df: pd.DataFrame, syn_df: pd.DataFrame, numeric_cols: List[str]) -> Tuple[float, str]:
    good_cols, dropped = drop_constant_cols(real_df, syn_df, numeric_cols, is_cat=False)
    if len(good_cols) < 2:
        return float("nan"), "No valid numeric pairs. Dropped constant cols: " + ", ".join(dropped)

    metric = CorrelationSimilarity()
    scores: List[float] = []
    first_err: Optional[str] = None

    for c1, c2 in pairs(good_cols):
        r1 = coerce_num(real_df[c1])
        r2 = coerce_num(real_df[c2])
        s1 = coerce_num(syn_df[c1])
        s2 = coerce_num(syn_df[c2])

        real_pair = pd.DataFrame({"x": r1, "y": r2})
        syn_pair = pd.DataFrame({"x": s1, "y": s2})

        try:
            scores.append(float(metric.compute(real_pair, syn_pair)))
        except Exception as e:
            if first_err is None:
                first_err = f"{type(e).__name__}: {e}"
            scores.append(float("nan"))

    score = mean_or_nan(scores)
    if math.isnan(score):
        msg = "All numeric-pair scores are NaN."
        if dropped:
            msg += " Dropped constant cols: " + ", ".join(dropped) + "."
        if first_err:
            msg += " First sdmetrics error: " + first_err
        return score, msg

    why = "Dropped constant cols: " + ", ".join(dropped) if dropped else ""
    return score, why


def compute_dcsm(real_df: pd.DataFrame, syn_df: pd.DataFrame, cat_cols: List[str]) -> Tuple[float, str]:
    good_cols, dropped = drop_constant_cols(real_df, syn_df, cat_cols, is_cat=True)
    if len(good_cols) < 2:
        return float("nan"), "No valid categorical pairs. Dropped constant cols: " + ", ".join(dropped)

    metric = ContingencySimilarity()
    scores: List[float] = []
    first_err: Optional[str] = None

    for c1, c2 in pairs(good_cols):
        r1 = coerce_cat(real_df[c1])
        r2 = coerce_cat(real_df[c2])
        s1 = coerce_cat(syn_df[c1])
        s2 = coerce_cat(syn_df[c2])

        real_pair = pd.DataFrame({"x": r1, "y": r2})
        syn_pair = pd.DataFrame({"x": s1, "y": s2})

        try:
            scores.append(float(metric.compute(real_pair, syn_pair)))
        except Exception as e:
            if first_err is None:
                first_err = f"{type(e).__name__}: {e}"
            scores.append(float("nan"))

    score = mean_or_nan(scores)
    if math.isnan(score):
        msg = "All categorical-pair scores are NaN."
        if dropped:
            msg += " Dropped constant cols: " + ", ".join(dropped) + "."
        if first_err:
            msg += " First sdmetrics error: " + first_err
        return score, msg

    why = "Dropped constant cols: " + ", ".join(dropped) if dropped else ""
    return score, why


def compute_mixed(
    real_df: pd.DataFrame,
    syn_df: pd.DataFrame,
    numeric_cols: List[str],
    cat_cols: List[str],
    bins: int,
) -> Tuple[float, str]:
    good_num, dropped_num = drop_constant_cols(real_df, syn_df, numeric_cols, is_cat=False)
    good_cat, dropped_cat = drop_constant_cols(real_df, syn_df, cat_cols, is_cat=True)

    if len(good_num) < 1 or len(good_cat) < 1:
        msg = "No valid mixed pairs."
        dropped = [*dropped_num, *dropped_cat]
        if dropped:
            msg += " Dropped constant cols: " + ", ".join(dropped)
        return float("nan"), msg

    metric = ContingencySimilarity()
    scores: List[float] = []
    first_err: Optional[str] = None

    for ncol, ccol in pairs_mixed(good_num, good_cat):
        br, bs = bin_numeric_on_real_quantiles(real_df[ncol], syn_df[ncol], bins=bins)
        rc = coerce_cat(real_df[ccol])
        sc = coerce_cat(syn_df[ccol])

        real_pair = pd.DataFrame({"x": br, "y": rc})
        syn_pair = pd.DataFrame({"x": bs, "y": sc})

        try:
            scores.append(float(metric.compute(real_pair, syn_pair)))
        except Exception as e:
            if first_err is None:
                first_err = f"{type(e).__name__}: {e}"
            scores.append(float("nan"))

    score = mean_or_nan(scores)
    if math.isnan(score):
        msg = "All mixed-pair scores are NaN."
        dropped = [*dropped_num, *dropped_cat]
        if dropped:
            msg += " Dropped constant cols: " + ", ".join(dropped) + "."
        if first_err:
            msg += " First sdmetrics error: " + first_err
        return score, msg

    dropped = [*dropped_num, *dropped_cat]
    why = "Dropped constant cols: " + ", ".join(dropped) if dropped else ""
    return score, why


# -----------------------------
# CLI
# -----------------------------
ALLOWED = {"dpcm", "dcsm", "mixed"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--metric", required=True, help="DPCM | DCSM | Mixed")
    p.add_argument("--methods", nargs="*", default=None, help="Optional synth method folder names.")
    p.add_argument("--bins", type=int, default=10, help="Bins for Mixed numeric->categorical (default 10).")

    # typing control
    p.add_argument("--schema", default=None, help=f"Optional schema name: {sorted(SCHEMAS)}")
    p.add_argument("--num-cols", nargs="*", default=None, help="Explicit numeric columns.")
    p.add_argument("--cat-cols", nargs="*", default=None, help="Explicit categorical columns.")

    # fallback inference knobs
    p.add_argument("--cat-unique-max", type=int, default=50)
    p.add_argument("--cat-unique-ratio", type=float, default=0.05)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    metric_in = norm(args.metric)
    if metric_in not in ALLOWED:
        raise SystemExit(f"Unsupported metric '{args.metric}'. Allowed: {sorted(ALLOWED)}")

    root = default_project_root()
    out_dir = root / "metric_data" / "pairs_metric"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{metric_in}.csv"

    rows: List[dict] = []

    for pair in iter_pairs(root, args.methods):
        try:
            real_df = read_table(pair.real_path)
            syn_df = read_table(pair.synth_path)
            real_df, syn_df = align_columns(real_df, syn_df)

            # resolve column types
            if args.schema:
                numeric_cols, cat_cols = apply_schema(
                    real_df,
                    syn_df,
                    args.schema,
                    extra_num=args.num_cols,
                    extra_cat=args.cat_cols,
                )
            else:
                numeric_cols, cat_cols = infer_types_fallback(
                    real_df,
                    syn_df,
                    num_cols=args.num_cols,
                    cat_cols=args.cat_cols,
                    cat_unique_max=int(args.cat_unique_max),
                    cat_unique_ratio=float(args.cat_unique_ratio),
                )

            if metric_in == "dpcm":
                score, why = compute_dpcm(real_df, syn_df, numeric_cols)
                metric_name = "DPCM"
            elif metric_in == "dcsm":
                score, why = compute_dcsm(real_df, syn_df, cat_cols)
                metric_name = "DCSM"
            else:
                score, why = compute_mixed(real_df, syn_df, numeric_cols, cat_cols, bins=int(args.bins))
                metric_name = "Mixed"

            rows.append(
                {
                    "metric": metric_name,
                    "metric_requested": args.metric,
                    "method": pair.method,
                    "step": pair.step,
                    "score": score,
                    "n_real": len(real_df),
                    "n_synth": len(syn_df),
                    "real_file": pair.real_path.name,
                    "synth_file": pair.synth_path.name,
                    "error": why,
                }
            )
        except Exception as e:
            rows.append(
                {
                    "metric": metric_in.upper(),
                    "metric_requested": args.metric,
                    "method": pair.method,
                    "step": pair.step,
                    "score": float("nan"),
                    "n_real": "",
                    "n_synth": "",
                    "real_file": pair.real_path.name,
                    "synth_file": pair.synth_path.name,
                    "error": f"{type(e).__name__}: {e}",
                }
            )

    df = pd.DataFrame(rows).sort_values(["method", "step"]).reset_index(drop=True)
    df.to_csv(out_path, index=False, encoding="utf-8", na_rep="")
    print(f"[OK] Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())