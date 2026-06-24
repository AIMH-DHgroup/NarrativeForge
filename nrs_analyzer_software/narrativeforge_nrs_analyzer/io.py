from __future__ import annotations

import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .utils import parse_bool, parse_model_family, parse_parameter_count_b

REQUIRED_METRIC = "NRS"
IGNORED_MAIN_METRICS = {"CSV_NRS", "csv_nrs"}

COLUMN_ALIASES = {
    "nrs": "NRS",
    "runtime_seconds": "runtime_seconds",
    "runtime_sec": "runtime_seconds",
    "seconds": "runtime_seconds",
    "model": "model",
    "model_name": "model",
    "prompt_strategy": "prompt_strategy",
    "prompt": "prompt_strategy",
    "input_strategy": "input_strategy",
    "parameters": "parameters",
    "parameter_count": "parameters",
    "params": "parameters",
    "failed": "failed",
    "failure": "failed",
    "bertscore_f1": "bertscore_f1",
    "bertscore": "bertscore_f1",
    "semantic_similarity": "semantic_similarity",
    "semantic": "semantic_similarity",
    "case_id": "case_id",
    "case": "case_id",
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    existing = set(df.columns)
    for col in list(df.columns):
        key = str(col).strip().lower()
        target = COLUMN_ALIASES.get(key)
        if target and target not in existing:
            rename[col] = target
    if rename:
        df = df.rename(columns=rename)
    return df


def _derive_strategy_from_path(csv_path: Path) -> str | None:
    for part in [csv_path.parent.name, csv_path.parent.parent.name if csv_path.parent.parent else ""]:
        lower = str(part).lower()
        if lower.startswith("benchmark_all_"):
            return lower.replace("benchmark_all_", "", 1)
    return None


def _strategy_from_existing_column(df: pd.DataFrame) -> str | None:
    if "input_strategy" not in df.columns:
        return None
    values = df["input_strategy"].dropna().astype(str).str.strip()
    values = values[values != ""]
    unique = values.unique()
    if len(unique) == 1:
        return str(unique[0])
    return None


def _read_single_runs_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _normalize_columns(df)

    if REQUIRED_METRIC not in df.columns:
        raise ValueError(
            f"{path} does not contain an NRS column. CSV_NRS is intentionally not used as the main metric."
        )

    folder_strategy = _derive_strategy_from_path(path)
    existing_strategy = _strategy_from_existing_column(df)
    strategy = folder_strategy or existing_strategy or path.parent.name

    if "input_strategy" in df.columns:
        df["input_strategy_original"] = df["input_strategy"]
    else:
        df["input_strategy_original"] = np.nan
    df["input_strategy"] = str(strategy)

    if "model" not in df.columns:
        df["model"] = "unknown_model"
    if "prompt_strategy" not in df.columns:
        df["prompt_strategy"] = "unknown"
    if "runtime_seconds" not in df.columns:
        df["runtime_seconds"] = np.nan
    if "parameters" not in df.columns:
        df["parameters"] = np.nan
    if "failed" not in df.columns:
        df["failed"] = False
    if "case_id" not in df.columns:
        df["case_id"] = np.arange(len(df))

    df["NRS"] = pd.to_numeric(df["NRS"], errors="coerce")
    df["runtime_seconds"] = pd.to_numeric(df["runtime_seconds"], errors="coerce")
    df["model"] = df["model"].astype(str)
    df["prompt_strategy"] = df["prompt_strategy"].fillna("unknown").astype(str)
    df["input_strategy"] = df["input_strategy"].fillna("unknown").astype(str)
    df["failed"] = df["failed"].map(parse_bool).astype(bool)
    df["success"] = ~df["failed"]
    df["parameters_b"] = [
        parse_parameter_count_b(param, model)
        for param, model in zip(df["parameters"], df["model"])
    ]
    df["family"] = [parse_model_family(model) for model in df["model"]]
    if "source_file" in df.columns:
        df["benchmark_source_file"] = df["source_file"]
    df["nrs_runs_file"] = str(path)
    return df


def _find_runs_files(root: Path) -> list[Path]:
    files = sorted(root.rglob("nrs_runs.csv"))
    # Avoid duplicate files copied into previous output folders if a user points
    # the tool at a large project directory.
    files = [p for p in files if "tables" not in {part.lower() for part in p.parts}]
    return files


def load_runs(source: str | Path) -> tuple[pd.DataFrame, dict]:
    """Load and concatenate all nrs_runs.csv files from a ZIP or directory.

    Returns a combined DataFrame and a metadata dictionary describing file
    discovery and input-strategy overrides.
    """
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(source)

    metadata: dict = {
        "source": str(source),
        "source_type": "zip" if source.suffix.lower() == ".zip" else "directory",
        "files": [],
        "input_strategy_policy": "input_strategy is derived from benchmark_all_* parent folder when available; otherwise a single existing input_strategy value or the parent folder name is used.",
    }

    temp_dir_obj = None
    try:
        if source.suffix.lower() == ".zip":
            temp_dir_obj = tempfile.TemporaryDirectory(prefix="nrs_analyzer_")
            extract_root = Path(temp_dir_obj.name)
            with zipfile.ZipFile(source, "r") as zf:
                zf.extractall(extract_root)
            root = extract_root
        else:
            root = source

        files = _find_runs_files(root)
        if not files:
            raise FileNotFoundError(f"No nrs_runs.csv files were found under {source}")

        frames = []
        for path in files:
            frame = _read_single_runs_file(path)
            metadata["files"].append(
                {
                    "path": str(path.relative_to(root)) if path.is_relative_to(root) else str(path),
                    "rows": int(len(frame)),
                    "input_strategy": str(frame["input_strategy"].iloc[0]) if len(frame) else "unknown",
                    "original_input_strategy_values": sorted(
                        frame["input_strategy_original"].dropna().astype(str).unique().tolist()
                    ),
                }
            )
            frames.append(frame)

        combined = pd.concat(frames, ignore_index=True)
        combined = combined.dropna(subset=["NRS"], how="all")
        return combined, metadata
    finally:
        if temp_dir_obj is not None:
            temp_dir_obj.cleanup()


def write_dataframe_tables(tables: dict[str, pd.DataFrame], tables_dir: Path) -> dict[str, str]:
    written = {}
    tables_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        path = tables_dir / f"{name}.csv"
        table.to_csv(path, index=False)
        written[name] = str(path)
        if name in {
            "model_strategy_matrix",
            "model_family_summary",
            "top5_configurations",
            "case_study_summary",
            "best_configuration_by_case",
            "case_difficulty_ranking",
            "case_model_size_summary",
        }:
            tex_path = tables_dir / f"{name}.tex"
            table.to_latex(tex_path, index=False, float_format="%.3f", escape=True)
            written[f"{name}_tex"] = str(tex_path)
    return written


def make_bundle(output_dir: Path, bundle_name: str = "narrativeforge_nrs_analysis_output.zip") -> Path:
    bundle_path = output_dir / bundle_name
    if bundle_path.exists():
        bundle_path.unlink()
    with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(output_dir.rglob("*")):
            if path == bundle_path or path.is_dir():
                continue
            zf.write(path, arcname=str(path.relative_to(output_dir)))
    return bundle_path


def copy_input_to_output(source: str | Path, output_dir: Path) -> Path | None:
    """Optional helper for archival workflows. Not used by default."""
    source = Path(source)
    if not source.exists():
        return None
    target = output_dir / source.name
    if source.is_file():
        shutil.copy2(source, target)
    else:
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)
    return target
