from __future__ import annotations

from pathlib import Path
import pandas as pd


NUM_INT_COLS = ["Age"]
NUM_FLOAT_COLS = ["Height", "Weight"]


def convert_file(csv_path: Path, out_dir: Path) -> Path:
    # 1) читаем ВСЁ как строки, чтобы потом явно проставить типы
    df = pd.read_csv(csv_path, dtype="string")

    # 2) int колонки (nullable)
    for col in NUM_INT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("int32")

    # 3) float колонки
    for col in NUM_FLOAT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    # 4) остальные — категориальные
    cat_cols = [c for c in df.columns if c not in (NUM_INT_COLS + NUM_FLOAT_COLS)]
    for c in cat_cols:
        # оставляем пропуски как <NA>, остальное — category
        df[c] = df[c].astype("object")

    out_path = out_dir / (csv_path.stem + ".parquet")
    df.to_parquet(out_path, engine="pyarrow", index=False)
    return out_path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Convert step_*_real.csv to Parquet with fixed dtypes.")
    parser.add_argument("--in-dir", default="./real_data", help="Input directory with CSV files")
    parser.add_argument("--out-dir", default="./real_data", help="Output directory for Parquet files")
    args = parser.parse_args()

    in_dir = Path(args.in_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob("step_*_real.csv"))
    if not files:
        raise SystemExit(f"No files found: {in_dir / 'step_*_real.csv'}")

    for f in files:
        out = convert_file(f, out_dir)
        print(f"{f.name} -> {out.name}")


if __name__ == "__main__":
    main()
