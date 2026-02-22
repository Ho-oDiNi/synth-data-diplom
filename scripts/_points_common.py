from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from datetime import datetime, timezone

import pandas as pd
from synthcity.plugins.core.dataloader import GenericDataLoader

REAL_RE = re.compile(r"^step_(?P<step>\d+)_real\.(parquet|csv)$", re.IGNORECASE)
SYN_RE = re.compile(r"^step_(?P<step>\d+)_synth\.(parquet|csv)$", re.IGNORECASE)


@dataclass(frozen=True)
class StepPair:
    step: int
    method: str
    real_path: Path
    synth_path: Path


def project_root() -> Path:
    # scripts/<file>.py -> root = scripts/..
    return Path(__file__).resolve().parents[1]


def read_table(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == ".parquet":
        return pd.read_parquet(path)
    if suf == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file extension: {path.name}")


def extract_step(path: Path, kind: str) -> Optional[int]:
    m = REAL_RE.match(path.name) if kind == "real" else SYN_RE.match(path.name)
    if not m:
        return None
    return int(m.group("step"))


def list_steps(dir_path: Path, kind: str) -> dict[int, Path]:
    out: dict[int, Path] = {}
    if not dir_path.exists():
        return out
    for p in dir_path.iterdir():
        if not p.is_file():
            continue
        step = extract_step(p, kind=kind)
        if step is None:
            continue
        out[step] = p
    return dict(sorted(out.items(), key=lambda kv: kv[0]))


def list_methods(synth_dir: Path) -> list[str]:
    if not synth_dir.exists():
        return []
    return sorted([p.name for p in synth_dir.iterdir() if p.is_dir()])


def align_columns(real_df: pd.DataFrame, synth_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # строгий режим: набор колонок должен совпадать
    if set(real_df.columns) != set(synth_df.columns):
        missing_in_syn = sorted(set(real_df.columns) - set(synth_df.columns))
        missing_in_real = sorted(set(synth_df.columns) - set(real_df.columns))
        raise ValueError(
            "Column mismatch.\n"
            f"Missing in synth: {missing_in_syn}\n"
            f"Missing in real: {missing_in_real}"
        )
    # порядок как в real
    synth_df = synth_df[list(real_df.columns)]
    return real_df, synth_df


def iter_pairs(methods: Optional[list[str]] = None) -> list[StepPair]:
    root = project_root()
    real_dir = root / "real_data"
    synth_dir = root / "synth_data"

    real_steps = list_steps(real_dir, kind="real")
    if not real_steps:
        raise FileNotFoundError(f"No real files found in: {real_dir}")

    method_names = methods if methods else list_methods(synth_dir)
    if not method_names:
        raise FileNotFoundError(f"No method folders found in: {synth_dir}")

    pairs: list[StepPair] = []
    for method in method_names:
        method_dir = synth_dir / method
        synth_steps = list_steps(method_dir, kind="synth")
        if not synth_steps:
            continue

        common = sorted(set(real_steps.keys()) & set(synth_steps.keys()))
        for step in common:
            pairs.append(
                StepPair(
                    step=step,
                    method=method,
                    real_path=real_steps[step],
                    synth_path=synth_steps[step],
                )
            )

    if not pairs:
        raise FileNotFoundError("No matching (real, synth) step pairs found.")

    return pairs


def _norm(s: str) -> str:
    return str(s).strip().lower().replace("-", "_").replace(" ", "_").replace(".", "_")


def resolve_metric_class_in_module(module_name: str, wanted_metric_name: str) -> type:
    """
    wanted_metric_name — это то, что возвращает MetricEvaluator.name() (например 'inv_kl_divergence').
    """
    import importlib

    mod = importlib.import_module(module_name)

    wanted = _norm(wanted_metric_name)
    for obj in mod.__dict__.values():
        if not isinstance(obj, type):
            continue
        if not (hasattr(obj, "name") and callable(getattr(obj, "name"))):
            continue
        try:
            n = obj.name()
        except Exception:
            continue
        if isinstance(n, str) and _norm(n) == wanted:
            return obj

    raise ValueError(f"Metric '{wanted_metric_name}' not found in module {module_name}")


def instantiate_evaluator(cls: type, workspace: Path, seed: int = 0) -> Any:
    # минимум: workspace + random_state, дальше fallback
    for kwargs in (
        {"workspace": workspace, "random_state": seed},
        {"workspace": workspace},
        {},
    ):
        try:
            return cls(**kwargs)
        except Exception:
            continue
    return cls()  # последняя попытка, пусть упадёт наружу


def build_loader(df: pd.DataFrame, sensitive_columns: Optional[list[str]] = None) -> GenericDataLoader:
    sensitive_columns = sensitive_columns or []
    try:
        return GenericDataLoader(df, sensitive_columns=sensitive_columns)
    except TypeError:
        # fallback на другие сигнатуры
        try:
            return GenericDataLoader(df)
        except Exception as e:
            raise RuntimeError(f"Failed to create GenericDataLoader: {e}") from e


def reduce_metric_output(val: Any) -> float:
    if isinstance(val, (int, float)) and pd.notna(val):
        return float(val)

    if isinstance(val, dict):
        for k in ("mean", "score", "value"):
            if k in val and isinstance(val[k], (int, float)):
                return float(val[k])
        numeric = [v for v in val.values() if isinstance(v, (int, float))]
        if len(numeric) == 1:
            return float(numeric[0])

    raise ValueError(f"Cannot reduce metric output to float: {val}")


# --- Global category encoding (consistent across ALL files) ---

_CATEGORY_MAP_CACHE: Optional[dict[str, dict[str, int]]] = None
_CATEGORY_MAP_PATH_NAME = "category_maps.json"


def _category_map_path() -> Path:
    return project_root() / _CATEGORY_MAP_PATH_NAME


def _is_probably_numeric(series: pd.Series, threshold: float = 0.9) -> bool:
    if pd.api.types.is_numeric_dtype(series.dtype):
        return True
    parsed = pd.to_numeric(series, errors="coerce")
    ratio = float(parsed.notna().mean()) if len(parsed) else 0.0
    return ratio >= threshold


def _detect_categorical_columns_from_real_sample() -> list[str]:
    """
    Определяем, какие колонки считать категориальными, по одному real-файлу.
    (Схема real_data — источник истины.)
    """
    root = project_root()
    real_dir = root / "real_data"
    real_steps = list_steps(real_dir, kind="real")
    if not real_steps:
        return []

    first_path = next(iter(real_steps.values()))
    df = read_table(first_path)

    cat_cols: list[str] = []
    for c in df.columns:
        if not _is_probably_numeric(df[c], threshold=0.9):
            cat_cols.append(c)
    return cat_cols


def _read_only_columns(path: Path, cols: list[str]) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == ".parquet":
        # parquet умеет читать только нужные колонки
        return pd.read_parquet(path, columns=cols)
    if suf == ".csv":
        return pd.read_csv(path, usecols=cols)
    raise ValueError(f"Unsupported file extension: {path.name}")


def _collect_all_paths(methods: Optional[list[str]] = None) -> list[Path]:
    """
    Собираем все пути real + synth (по всем step и по выбранным методам).
    """
    root = project_root()
    real_dir = root / "real_data"
    synth_dir = root / "synth_data"

    paths: list[Path] = []

    # real
    for p in list_steps(real_dir, kind="real").values():
        paths.append(p)

    # synth
    method_names = methods if methods else list_methods(synth_dir)
    for m in method_names:
        md = synth_dir / m
        for p in list_steps(md, kind="synth").values():
            paths.append(p)

    return paths


def build_category_maps(methods: Optional[list[str]] = None) -> dict[str, dict[str, int]]:
    """
    Строит словари кодирования категорий для каждого категориального столбца:
      map[col][value] = code, где code начинается с 1.
    0 зарезервирован под NaN/неизвестные значения.
    """
    cat_cols = _detect_categorical_columns_from_real_sample()
    if not cat_cols:
        return {}

    value_sets: dict[str, set[str]] = {c: set() for c in cat_cols}
    all_paths = _collect_all_paths(methods=methods)

    for path in all_paths:
        try:
            df_part = _read_only_columns(path, cat_cols)
        except Exception:
            # если какой-то файл читается с ошибкой/колонок нет — пропускаем
            continue

        for c in cat_cols:
            if c not in df_part.columns:
                continue
            s = df_part[c].astype("string")
            # нормализуем строки: убираем пробелы
            s = s.str.strip()
            # собираем только непустые значения
            vals = s.dropna().unique().tolist()
            for v in vals:
                if v is None:
                    continue
                if v == "" or v.lower() == "nan":
                    continue
                value_sets[c].add(str(v))

    maps: dict[str, dict[str, int]] = {}
    for c, vals in value_sets.items():
        ordered = sorted(vals)  # детерминированно, одинаково на всех машинах
        maps[c] = {v: i + 1 for i, v in enumerate(ordered)}  # 1..N

    # сохраняем
    _category_map_path().write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "columns": maps,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return maps


def load_category_maps() -> dict[str, dict[str, int]]:
    global _CATEGORY_MAP_CACHE

    if _CATEGORY_MAP_CACHE is not None:
        return _CATEGORY_MAP_CACHE

    p = _category_map_path()
    if not p.exists():
        _CATEGORY_MAP_CACHE = {}
        return _CATEGORY_MAP_CACHE

    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        cols = data.get("columns", {})
        if isinstance(cols, dict):
            # гарантируем int
            fixed: dict[str, dict[str, int]] = {}
            for col, mp in cols.items():
                if not isinstance(mp, dict):
                    continue
                fixed[col] = {str(k): int(v) for k, v in mp.items()}
            _CATEGORY_MAP_CACHE = fixed
            return _CATEGORY_MAP_CACHE
    except Exception:
        pass

    _CATEGORY_MAP_CACHE = {}
    return _CATEGORY_MAP_CACHE


def ensure_category_maps(methods: Optional[list[str]] = None) -> dict[str, dict[str, int]]:
    """
    Если category_maps.json уже есть — грузим.
    Если нет — строим по всем файлам и сохраняем.
    """
    maps = load_category_maps()
    if maps:
        return maps
    maps = build_category_maps(methods=methods)
    global _CATEGORY_MAP_CACHE
    _CATEGORY_MAP_CACHE = maps
    return maps


def encode_categories_consistently(df: pd.DataFrame, cat_maps: dict[str, dict[str, int]]) -> pd.DataFrame:
    """
    Преобразует категориальные колонки в int-коды по глобальным словарям.
    Неизвестные/NaN -> 0.
    """
    out = df.copy()
    for col, mp in cat_maps.items():
        if col not in out.columns:
            continue
        s = out[col].astype("string").str.strip()
        encoded = s.map(mp).fillna(0).astype("int64")
        out[col] = encoded
    return out


def prepare_tabular_pair(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    *,
    fillna: bool = True,
    methods: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Единое кодирование категорий в числа для real+synth, согласованное по всем файлам проекта.

    - numeric -> to_numeric, NaN -> median(real)
    - categorical -> int-коды из category_maps.json (1..N), NaN/unknown -> 0
    """
    real = real_df.copy()
    syn = synth_df.copy()

    # 1) numeric cols определяем по real
    numeric_cols: list[str] = []
    for c in real.columns:
        if _is_probably_numeric(real[c], threshold=0.9):
            numeric_cols.append(c)

    # 2) numeric cast + fillna
    for c in numeric_cols:
        real[c] = pd.to_numeric(real[c], errors="coerce")
        syn[c] = pd.to_numeric(syn[c], errors="coerce")

    if fillna:
        for c in numeric_cols:
            med = real[c].median(skipna=True)
            if pd.isna(med):
                med = 0.0
            real[c] = real[c].fillna(med)
            syn[c] = syn[c].fillna(med)

    # 3) categorical -> global int codes
    cat_maps = ensure_category_maps(methods=methods)
    real = encode_categories_consistently(real, cat_maps)
    syn = encode_categories_consistently(syn, cat_maps)

    return real, syn


def load_privacy_config() -> dict[str, Any]:
    """
    Не требует аргументов.
    Если в корне проекта есть privacy_config.json — читаем.
    Формат:
      {"sensitive_columns": ["col1","col2"]}
    """
    cfg_path = project_root() / "privacy_config.json"
    if not cfg_path.exists():
        return {}
    try:
        return json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
