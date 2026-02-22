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
    load_privacy_config,
    _norm,
)

MODULE = "synthcity.metrics.eval_privacy"

ALIASES = {
    # частые пользовательские варианты
    "kmap": "k_map",
    "k-map": "k_map",
    "k_map": "k_map",
    
    "l_diversity": "distinct_l_diversity",
    "distinct l-diversity": "distinct_l_diversity",
    "distinct_l_diversity": "distinct_l_diversity",
    
    "k_anonymization": "k_anonymization",
    "l_diversity": "l_diversity_distinct",  # в synthcity часто именно так
    "delta_presence": "delta_presence",
    "identifiability_score": "identifiability_score",
    
    "domiasmia": "domiasmia",
    "domiasmia_bnaf": "domiasmia_bnaf",
    "domiasmia_kde": "domiasmia_kde",
    "domiasmia_prior": "domiasmia_prior",
}

ALLOWED = {
    "k_anonymization",
    "distinct_l_diversity",
    "k_map",
    "delta_presence",
    "identifiability_score",
    "domiasmia",
    "domiasmia_bnaf",
    "domiasmia_kde",
    "domiasmia_prior",
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
        raise SystemExit(f"Unsupported privacy metric '{args.metric}'. Allowed: {sorted(ALLOWED)}")

    metric_cls = resolve_metric_class_in_module(MODULE, metric_canonical)

    root = project_root()
    out_dir = root / "metric_data" / "privacy_metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    workspace = root / ".synthcity_workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    evaluator = instantiate_evaluator(metric_cls, workspace=workspace, seed=0)

    cfg = load_privacy_config()
    sensitive_columns = cfg.get("sensitive_columns", [])
    if sensitive_columns is None:
        sensitive_columns = []
    if not isinstance(sensitive_columns, list):
        sensitive_columns = []

    rows = []
    for pair in iter_pairs(args.methods):
        try:
            real_df = read_table(pair.real_path)
            syn_df = read_table(pair.synth_path)
            real_df, syn_df = align_columns(real_df, syn_df)

            # privacy метрики тоже плохо переносят object/NaN -> готовим
            real_df, syn_df = prepare_tabular_pair(real_df, syn_df, fillna=True)

            real_loader = build_loader(real_df, sensitive_columns=sensitive_columns)
            syn_loader = build_loader(syn_df, sensitive_columns=sensitive_columns)

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
