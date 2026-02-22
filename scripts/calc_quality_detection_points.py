#!/usr/bin/env python3
# scripts/calc_quality_points.py
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

# В SynthCity это обычно здесь:
MODULE = "synthcity.metrics.eval_detection"

# Поддерживаем метрики как в доке (через нормализацию/алиасы)
ALIASES = {
    # detection_*
    "detection_gmm": "detection_gmm",
    "detection_xgb": "detection_xgb",
    "detection_mlp": "detection_mlp",
    "detection_linear": "detection_linear",
}

ALLOWED = {
    "detection_gmm",
    "detection_xgb",
    "detection_mlp",
    "detection_linear",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # минимум аргументов: один обязательный metric, methods опционально
    p.add_argument("--metric", help="Напр: performance.xgb | detection_xgb | performance.feat_rank_distance")
    p.add_argument("--methods", nargs="*", default=None)
    return p.parse_args()


def find_metric_cls(metric_canonical: str) -> type:
    try:
        cls = resolve_metric_class_in_module(MODULE, metric_canonical)
    except Exception as e:     
        raise ValueError(f"Metric '{metric_canonical}' not found in {MODULE}. Last error: {e}")

    return cls

def main() -> int:
    args = parse_args()

    metric_in = args.metric.strip()
    metric_key = _norm(metric_in)

    # поддержка performance.xgb из доки
    canonical = ALIASES.get(metric_in, None) or ALIASES.get(metric_key, None) or metric_key
    canonical = _norm(canonical)

    if canonical not in ALLOWED:
        raise SystemExit(f"Unsupported quality metric '{metric_in}'. Allowed: {sorted(ALLOWED)}")

    metric_cls = find_metric_cls(canonical)

    root = project_root()
    out_dir = root / "metric_data" / "quality"
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

            # для performance/detection обязателен числовой формат
            # используем согласованную подготовку (категории -> числа, пропуски -> заполнение)
            real_df, syn_df = prepare_tabular_pair(real_df, syn_df, fillna=True)

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
                    "metric_requested": metric_in,
                    "module": MODULE,
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
                    "metric": metric_cls.name() if metric_cls else canonical,
                    "metric_requested": metric_in,
                    "module": MODULE,
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

    safe_name = metric_key.replace(".", "_")
    out_path = out_dir / f"{safe_name}.csv"
    df.to_csv(out_path, index=False, encoding="utf-8", na_rep="")
    print(f"[OK] Saved: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
