#!/usr/bin/env python3
# scripts/calc_consistency_points.py
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from _points_common import (
    project_root,
    iter_pairs,
    read_table,
    align_columns,
    prepare_tabular_pair,
    _norm,
)

ALLOWED = {"CVR", "CVC", "sCVC"}


# ----------------------------
# Constraints model
# ----------------------------
@dataclass(frozen=True)
class Constraint:
    """A single rule. Returns a boolean vector: True means violated."""
    name: str
    kind: str
    spec: Dict[str, Any]

    def violated(self, df: pd.DataFrame) -> np.ndarray:
        k = self.kind

        if k == "range":
            col = self.spec["col"]
            if col not in df.columns:
                raise KeyError(f"Missing column '{col}' for constraint '{self.name}'")
            s = df[col]
            mn = self.spec.get("min", None)
            mx = self.spec.get("max", None)
            inclusive = bool(self.spec.get("inclusive", True))

            # Treat NaNs as violations unless allow_na is true
            allow_na = bool(self.spec.get("allow_na", False))
            na_mask = s.isna().to_numpy()
            v = np.zeros(len(df), dtype=bool)

            if mn is not None:
                v |= (s < mn).to_numpy() if inclusive else (s <= mn).to_numpy()
            if mx is not None:
                v |= (s > mx).to_numpy() if inclusive else (s >= mx).to_numpy()

            if not allow_na:
                v |= na_mask
            return v

        if k == "in":
            col = self.spec["col"]
            if col not in df.columns:
                raise KeyError(f"Missing column '{col}' for constraint '{self.name}'")
            values = set(self.spec["values"])
            allow_na = bool(self.spec.get("allow_na", False))
            s = df[col]
            v = ~s.isin(values)
            if allow_na:
                v &= ~s.isna()
            return v.to_numpy()

        if k == "integer":
            col = self.spec["col"]
            if col not in df.columns:
                raise KeyError(f"Missing column '{col}' for constraint '{self.name}'")
            allow_na = bool(self.spec.get("allow_na", False))
            s = df[col]
            # numeric check; strings will become violations
            s_num = pd.to_numeric(s, errors="coerce")
            v = s_num.isna().to_numpy()  # non-numeric -> violation
            if allow_na:
                v &= ~s.isna().to_numpy()
            # integers: x == round(x)
            ok_int = np.isclose(s_num.to_numpy(), np.round(s_num.to_numpy()), equal_nan=False)
            if allow_na:
                ok_int |= s.isna().to_numpy()
            v |= ~ok_int
            return v

        if k == "gte":
            left = self.spec["left"]
            right = self.spec["right"]
            if left not in df.columns or right not in df.columns:
                raise KeyError(f"Missing '{left}' or '{right}' for constraint '{self.name}'")
            allow_na = bool(self.spec.get("allow_na", False))
            a = pd.to_numeric(df[left], errors="coerce")
            b = pd.to_numeric(df[right], errors="coerce")
            v = (a < b).to_numpy()
            if not allow_na:
                v |= a.isna().to_numpy() | b.isna().to_numpy()
            return v

        raise ValueError(f"Unsupported constraint kind '{k}' in '{self.name}'")


def load_constraints(path: Path) -> List[Constraint]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "constraints" not in raw:
        raise ValueError("Constraints file must be a JSON object with key 'constraints': [ ... ]")

    out: List[Constraint] = []
    for i, item in enumerate(raw["constraints"]):
        if not isinstance(item, dict) or "type" not in item:
            raise ValueError(f"Invalid constraint at index {i}: expected object with 'type'")
        kind = str(item["type"])
        name = str(item.get("name") or f"{kind}_{i}")
        spec = dict(item)
        spec.pop("type", None)
        spec.pop("name", None)
        out.append(Constraint(name=name, kind=kind, spec=spec))
    if not out:
        raise ValueError("No constraints found in file")
    return out


def infer_constraints_from_real(real_df: pd.DataFrame) -> List[Constraint]:
    """
    Fallback if no constraints file provided:
    - numeric columns: range [min, max]
    - categorical/object columns: in allowed set from real
    - integer-like numeric columns: integer constraint if dtype is int
    """
    constraints: List[Constraint] = []

    for col in real_df.columns:
        s = real_df[col]

        # Try to detect numeric
        if pd.api.types.is_numeric_dtype(s):
            # If integer dtype -> enforce integer + range
            if pd.api.types.is_integer_dtype(s):
                constraints.append(
                    Constraint(
                        name=f"{col}_integer",
                        kind="integer",
                        spec={"col": col, "allow_na": False},
                    )
                )
            mn = float(np.nanmin(s.to_numpy())) if s.notna().any() else None
            mx = float(np.nanmax(s.to_numpy())) if s.notna().any() else None
            constraints.append(
                Constraint(
                    name=f"{col}_range",
                    kind="range",
                    spec={"col": col, "min": mn, "max": mx, "inclusive": True, "allow_na": False},
                )
            )
            continue

        # Non-numeric: treat as categorical domain constraint
        vals = sorted(set(s.dropna().astype(str).tolist()))
        if vals:
            constraints.append(
                Constraint(
                    name=f"{col}_domain",
                    kind="in",
                    spec={"col": col, "values": vals, "allow_na": False},
                )
            )

    if not constraints:
        raise ValueError("Could not infer any constraints from real data (empty columns?)")
    return constraints


# ----------------------------
# Metrics: CVR / CVC / sCVC
# ----------------------------
def compute_violation_matrix(df: pd.DataFrame, constraints: List[Constraint]) -> np.ndarray:
    """
    Returns matrix V of shape (n_rows, n_constraints): True means violated.
    """
    n = len(df)
    m = len(constraints)
    V = np.zeros((n, m), dtype=bool)
    for j, c in enumerate(constraints):
        V[:, j] = c.violated(df)
    return V


def metric_cvr(V: np.ndarray) -> float:
    # share of rows that violate at least one constraint
    if V.size == 0:
        return float("nan")
    return float(np.mean(np.any(V, axis=1)))


def metric_cvc(V: np.ndarray) -> float:
    # share of constraints violated by at least one row
    if V.size == 0:
        return float("nan")
    return float(np.mean(np.any(V, axis=0)))


def metric_scvc(V: np.ndarray) -> float:
    # mean over constraints of per-constraint violation rate
    if V.size == 0:
        return float("nan")
    return float(np.mean(np.mean(V, axis=0)))


def compute_metric(metric_name: str, syn_df: pd.DataFrame, constraints: List[Constraint]) -> float:
    V = compute_violation_matrix(syn_df, constraints)
    
    if metric_name == "CVR":
        return metric_cvr(V)
    if metric_name == "CVC":
        return metric_cvc(V)
    if metric_name == "sCVC":
        return metric_scvc(V)
    raise ValueError(f"Unknown metric '{metric_name}'")


# ----------------------------
# CLI / main
# ----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--metric", required=True, help="One of: CVR, CVC, sCVC")
    p.add_argument("--methods", nargs="*", default=None, help="Subset of methods under synth_data/")
    p.add_argument(
        "--constraints",
        default=None,
        help="Path to JSON file with constraints. If omitted, constraints are inferred from real data.",
    )
    p.add_argument(
        "--no_prepare",
        action="store_true",
        help="Skip prepare_tabular_pair (only align columns). Useful if you want raw values only.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    metric_in = args.metric.strip()
    metric_norm = metric_in if metric_in in ALLOWED else metric_in.upper()
    if metric_norm not in ALLOWED:
        raise SystemExit(f"Unsupported consistency metric '{args.metric}'. Allowed: {sorted(ALLOWED)}")

    root = project_root()
    out_dir = root / "metric_data" / "consistency_metric"
    out_dir.mkdir(parents=True, exist_ok=True)

    constraints_path = Path(args.constraints) if args.constraints else None

    rows: List[Dict[str, Any]] = []
    for pair in iter_pairs(args.methods):
        # pair.real_path: real_data/step_{step}_real.parquet
        # pair.synth_path: synth_data/{method}/step_{step}_synth.parquet
        try:
            real_df = read_table(pair.real_path)
            syn_df = read_table(pair.synth_path)

            # Align schema
            real_df, syn_df = align_columns(real_df, syn_df)

            # Optional: handle missing values / type normalization like elsewhere in your pipeline
            if not args.no_prepare:
                real_df, syn_df = prepare_tabular_pair(real_df, syn_df, fillna=False)

            # Load/infer constraints per step from real data (so ranges/domains are tied to that real split)
            if constraints_path:
                constraints = load_constraints(constraints_path)
            else:
                constraints = infer_constraints_from_real(real_df)

            score = compute_metric(metric_norm, syn_df, constraints)

            rows.append(
                {
                    "metric": metric_norm,
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
                    "metric": metric_norm,
                    "metric_requested": args.metric,
                    "method": pair.method,
                    "step": pair.step,
                    "score": float("nan"),
                    "n_real": None,
                    "n_synth": None,
                    "real_file": getattr(pair.real_path, "name", ""),
                    "synth_file": getattr(pair.synth_path, "name", ""),
                    "error": str(e),
                }
            )

    df = pd.DataFrame(rows).sort_values(["method", "step"]).reset_index(drop=True)
    out_path = out_dir / f"{metric_norm}.csv"
    df.to_csv(out_path, index=False, encoding="utf-8", na_rep="")
    print(f"[OK] Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())