#!/usr/bin/env python3
import argparse
import json
import platform
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def write_table(df: pd.DataFrame, path_no_ext: Path, fmt: str) -> Path:
    fmt = fmt.lower()
    if fmt == "parquet":
        out = path_no_ext.with_suffix(".parquet")
        df.to_parquet(out, index=False)
        return out
    if fmt == "csv":
        out = path_no_ext.with_suffix(".csv")
        df.to_csv(out, index=False, encoding="utf-8")
        return out
    raise ValueError("fmt must be 'csv' or 'parquet'")


def list_real_files(real_dir: Path, pattern: str) -> list[Path]:
    files = sorted(real_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found: {real_dir / pattern}")
    return files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-dir", default="real_data", help="Папка с 11 реальными файлами (корень проекта)")
    parser.add_argument("--real-glob", default="step_*_real.parquet", help="Шаблон реальных файлов, например: real_*.csv")
    parser.add_argument("--algorithm", default="bayesian_network", help="Плагин synthcity (bayesian_network, ctgan, tvae, ...)")
    parser.add_argument("--target", default="", help="(опционально) target column для GenericDataLoader")
    parser.add_argument("--out-root", default="synth_data", help="Куда сохранять результаты")
    parser.add_argument("--format", default="parquet", choices=["csv", "parquet"], help="Формат сохранения синтетики")

    # NEW: --step 1..11 или all (по умолчанию)
    parser.add_argument(
        "--step",
        default="all",
        choices=["all"] + [str(i) for i in range(1, 12)],
        help="С какого шага начать обработку: 1..11 или all",
    )

    args = parser.parse_args()

    real_dir = Path(args.real_dir).resolve()
    out_root = Path(args.out_root).resolve()

    real_files = list_real_files(real_dir, args.real_glob)

    # (step_no, path) с привязкой к исходной нумерации
    pairs: list[tuple[int, Path]] = list(enumerate(real_files, start=1))

    if args.step != "all":
        step_no = int(args.step)
        if step_no < 1 or step_no > len(pairs):
            raise ValueError(f"--step должен быть в диапазоне 1..{len(pairs)}")
        pairs = pairs[step_no - 1 :]

    run_id = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = out_root / f"{args.algorithm}"
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_id": run_id,
        "timestamp_utc": ts,
        "algorithm": args.algorithm,
        "real_dir": str(real_dir),
        "real_glob": args.real_glob,
        "format": args.format,
        "target": args.target or None,
        "step": args.step,  # NEW
        "python": platform.python_version(),
    }
    try:
        import synthcity  # noqa
        manifest["synthcity_version"] = getattr(synthcity, "__version__", None)
    except Exception:
        manifest["synthcity_version"] = None

    with open(run_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    rows = []
    for step_no, real_path in pairs:
        real_df = read_table(real_path)
        n = len(real_df)

        plugin = Plugins().get(args.algorithm)

        loader = (
            GenericDataLoader(real_df, target_column=args.target)
            if args.target
            else GenericDataLoader(real_df)
        )

        plugin.fit(loader)
        synth_df = plugin.generate(count=n).dataframe()

        synth_path = write_table(
            synth_df,
            run_dir / f"step_{n+1}_synth",  # FIX: имя по номеру шага, а не по числу строк
            args.format,
        )

        rows.append(
            {
                "run_id": run_id,
                "algorithm": args.algorithm,
                "step": step_no,
                "rows": int(n),
                "real_file": str(real_path),
                "synth_file": str(synth_path),
            }
        )

        print(f"{args.algorithm} | step={step_no:02d} | rows={n:4d} | saved: {synth_path.name}")

    steps_df = pd.DataFrame(rows).sort_values(["step"])
    steps_df.to_csv(run_dir / "steps.csv", index=False, encoding="utf-8")

    print(f"\nSaved run: {run_dir}")


if __name__ == "__main__":
    main()
