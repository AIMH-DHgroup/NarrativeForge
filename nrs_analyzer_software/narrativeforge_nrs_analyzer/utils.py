from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

TRUE_VALUES = {"true", "1", "yes", "y", "failed", "fail", "error", "timeout"}
FALSE_VALUES = {"false", "0", "no", "n", "success", "successful", "ok", "none", ""}


def parse_bool(value) -> bool:
    """Parse heterogeneous boolean values used in benchmark CSV files."""
    if pd.isna(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        if math.isnan(float(value)):
            return False
        return float(value) != 0.0
    text = str(value).strip().lower()
    if text in TRUE_VALUES:
        return True
    if text in FALSE_VALUES:
        return False
    # Conservative default: unknown strings are treated as non-failures, while
    # the original value remains available in the loaded data if needed.
    return False


def parse_parameter_count_b(parameters_value=None, model_name=None) -> float:
    """Return model size in billions of parameters when it can be inferred.

    Accepts values such as 4, 4.0, "4B", "4 b", "4000M", "7 billion",
    "gemma3:4b", "qwen3:1.7b", or raw counts such as 4000000000.
    """
    candidates = []
    if parameters_value is not None and not pd.isna(parameters_value):
        candidates.append(str(parameters_value).strip().lower())
    if model_name is not None and not pd.isna(model_name):
        candidates.append(str(model_name).strip().lower())

    for text in candidates:
        # Common textual forms: 4b, 4 b, 4 billion, 4000m, 4000 million.
        match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(b|bn|billion|m|mn|million)\b", text)
        if match:
            value = float(match.group(1))
            unit = match.group(2)
            return value if unit.startswith("b") else value / 1000.0

        # Ollama-style model tags: gemma3:4b, qwen3:1.7b-instruct.
        match = re.search(r"[:_\-/]([0-9]+(?:\.[0-9]+)?)b(?:\b|[_\-/])", text)
        if match:
            return float(match.group(1))

    # Numeric fallback after text-based parsing. Interpret small values as
    # billions, 1000-1e6 as millions, and very large values as raw parameters.
    try:
        if parameters_value is not None and not pd.isna(parameters_value):
            value = float(parameters_value)
            if value <= 0:
                return np.nan
            if value <= 1000:
                return value
            if value < 1_000_000:
                return value / 1000.0
            return value / 1_000_000_000.0
    except Exception:
        pass
    return np.nan


def parse_model_family(model_name=None) -> str:
    """Return a model-family label from common Ollama model names."""
    if model_name is None or pd.isna(model_name):
        return "unknown"
    text = str(model_name).strip().lower()
    base = text.split(":", 1)[0]
    base = re.sub(r"[^a-z0-9._-]+", "-", base).strip("-_")
    if not base:
        return "unknown"
    if base.startswith("llama"):
        return "llama"
    if base.startswith("gemma"):
        return base
    if base.startswith("qwen"):
        return base
    if base.startswith("deepseek"):
        return "deepseek"
    if base.startswith("mistral"):
        return "mistral"
    if base.startswith("phi"):
        return "phi"
    if base.startswith("granite"):
        return "granite"
    if base.startswith("cogito"):
        return "cogito"
    match = re.match(r"([a-z]+(?:[._-]?[0-9]+)?)", base)
    return match.group(1).replace("_", "-") if match else base


def ci95(series: Iterable[float]) -> float:
    values = pd.Series(series, dtype="float64").dropna()
    if len(values) <= 1:
        return np.nan
    return float(1.96 * values.std(ddof=1) / math.sqrt(len(values)))


def minmax(series: Iterable[float], invert: bool = False) -> pd.Series:
    s = pd.Series(series, dtype="float64")
    if s.dropna().empty:
        out = pd.Series(np.zeros(len(s)), index=s.index, dtype="float64")
    else:
        mn, mx = s.min(skipna=True), s.max(skipna=True)
        if pd.isna(mn) or pd.isna(mx) or mx == mn:
            out = pd.Series(np.ones(len(s)), index=s.index, dtype="float64")
        else:
            out = (s - mn) / (mx - mn)
            out = out.fillna(0.0)
    return 1.0 - out if invert else out


def ordered_values(values: Iterable, preferred_order: Sequence[str] | None = None) -> list[str]:
    """Return observed values, with preferred known categories first."""
    observed = [str(v) for v in pd.Series(list(values)).dropna().unique()]
    preferred_order = list(preferred_order or [])
    ordered = [v for v in preferred_order if v in observed]
    ordered.extend(sorted(v for v in observed if v not in ordered))
    return ordered


def safe_divide(numerator, denominator):
    numerator = pd.Series(numerator, dtype="float64")
    denominator = pd.Series(denominator, dtype="float64")
    out = numerator / denominator.replace({0: np.nan})
    return out.replace([np.inf, -np.inf], np.nan)


def make_output_dirs(output_dir: Path) -> tuple[Path, Path]:
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir, tables_dir


def compact_float(value, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def percentage(value, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{100 * float(value):.{digits}f}%"


def slug(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text).strip())
    return re.sub(r"_+", "_", text).strip("_").lower() or "item"
