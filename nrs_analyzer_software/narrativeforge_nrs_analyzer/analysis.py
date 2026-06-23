from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .coverage import CoverageOptions, apply_coverage_diagnostics, has_coverage_metrics
from .io import load_runs, make_bundle, write_dataframe_tables
from .plotting import create_all_figures
from .report import write_docx_report, write_html_report, write_latex_report, write_markdown_report
from .utils import ci95, make_output_dirs, minmax, ordered_values, safe_divide

DEFAULT_INPUT_ORDER = ["auto", "brief", "rag", "full"]
DEFAULT_PROMPT_ORDER = ["short", "standard", "detailed"]
DEFAULT_BALANCED_WEIGHTS = {
    "NRS": 0.50,
    "speed": 0.20,
    "size_efficiency": 0.15,
    "stability": 0.10,
    "reliability": 0.05,
}


@dataclass
class AnalysisOptions:
    """Configurable options for NarrativeForge NRS analysis."""

    input_order: list[str] = field(default_factory=lambda: DEFAULT_INPUT_ORDER.copy())
    prompt_order: list[str] = field(default_factory=lambda: DEFAULT_PROMPT_ORDER.copy())
    balanced_weights: dict[str, float] = field(default_factory=lambda: DEFAULT_BALANCED_WEIGHTS.copy())
    reliable_failure_rate_max: float = 0.05
    threshold_b: list[tuple[str, float]] = field(
        default_factory=lambda: [("<=2B", 2.0), ("<=4B", 4.0), ("<=8B", 8.0), ("<=16B", 16.0), ("all models", np.inf)]
    )
    dpi: int = 300
    figure_format: str = "png"
    coverage: bool = False
    coverage_thresholds: list[float] = field(default_factory=lambda: [0.70, 0.75, 0.80])
    coverage_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    skip_entity_coverage: bool = False
    skip_keyphrase_coverage: bool = False
    title: str = "NarrativeForge NRS Benchmark Analysis"
    hardware_note: str = (
        "Runtime values are benchmark runtimes on the machine used for the experiment. "
        "They should be interpreted comparatively unless the target deployment hardware is similar."
    )

    def normalized_weights(self) -> dict[str, float]:
        keys = ["NRS", "speed", "size_efficiency", "stability", "reliability"]
        weights = {k: float(self.balanced_weights.get(k, 0.0)) for k in keys}
        total = sum(weights.values())
        if total <= 0:
            return DEFAULT_BALANCED_WEIGHTS.copy()
        return {k: v / total for k, v in weights.items()}


@dataclass
class AnalysisResult:
    output_dir: Path
    figures_dir: Path
    tables_dir: Path
    bundle_path: Path
    summary: dict[str, Any]
    tables: dict[str, pd.DataFrame]
    figure_paths: dict[str, Path]


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if pd.isna(value) if not isinstance(value, (list, tuple, dict, str, bytes)) else False:
        return None
    raise TypeError(f"Object of type {type(value)} is not JSON serializable")


def aggregate_summary(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    grouped = df.groupby(group_cols, dropna=False).agg(
        attempts=("NRS", "size"),
        mean_NRS=("NRS", "mean"),
        sd_NRS=("NRS", "std"),
        ci95_NRS=("NRS", ci95),
        mean_runtime_seconds=("runtime_seconds", "mean"),
        sd_runtime_seconds=("runtime_seconds", "std"),
        failure_rate=("failed", "mean"),
        successes=("success", "sum"),
        parameter_median_b=("parameters_b", "median"),
        parameter_min_b=("parameters_b", "min"),
        parameter_max_b=("parameters_b", "max"),
    ).reset_index()
    successes = df[df["success"]].groupby(group_cols, dropna=False).agg(
        mean_NRS_success=("NRS", "mean"),
        sd_NRS_success=("NRS", "std"),
        mean_runtime_success=("runtime_seconds", "mean"),
        sd_runtime_success=("runtime_seconds", "std"),
    ).reset_index()
    return grouped.merge(successes, on=group_cols, how="left")


def _add_size_bins(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    bins = [0, 2, 4, 8, 16, np.inf]
    labels = ["<=2B", ">2-4B", ">4-8B", ">8-16B", ">16B"]
    out["size_bin"] = pd.cut(out["parameters_b"], bins=bins, labels=labels, include_lowest=True, right=True)
    return out


def _make_configuration_summary(df: pd.DataFrame, options: AnalysisOptions) -> pd.DataFrame:
    group_cols = ["model", "family", "input_strategy", "prompt_strategy", "parameters_b"]
    aggregations = {
        "attempts": ("NRS", "size"),
        "mean_NRS": ("NRS", "mean"),
        "sd_NRS": ("NRS", "std"),
        "mean_runtime_seconds": ("runtime_seconds", "mean"),
        "sd_runtime_seconds": ("runtime_seconds", "std"),
        "failure_rate": ("failed", "mean"),
        "successes": ("success", "sum"),
    }
    if "word_count" in df.columns:
        aggregations["mean_word_count"] = ("word_count", "mean")

    config = df.groupby(group_cols, dropna=False).agg(**aggregations).reset_index()
    success_config = df[df["success"]].groupby(group_cols, dropna=False).agg(
        mean_NRS_success=("NRS", "mean"),
        sd_NRS_success=("NRS", "std"),
        mean_runtime_success=("runtime_seconds", "mean"),
    ).reset_index()
    config = config.merge(success_config, on=group_cols, how="left")

    config["reliability"] = 1.0 - config["failure_rate"].fillna(0.0)
    config["NRS_per_second"] = safe_divide(config["mean_NRS"], config["mean_runtime_seconds"])
    config["NRS_per_billion_parameters"] = safe_divide(config["mean_NRS"], config["parameters_b"])
    config["NRS_per_parameter_second"] = safe_divide(
        config["mean_NRS"], config["parameters_b"] * config["mean_runtime_seconds"]
    )

    weights = options.normalized_weights()
    log_runtime = np.log1p(config["mean_runtime_seconds"])
    sd_for_stability = config["sd_NRS"].copy()
    if sd_for_stability.dropna().empty:
        sd_for_stability = pd.Series(np.zeros(len(config)), index=config.index)
    else:
        sd_for_stability = sd_for_stability.fillna(sd_for_stability.max())

    config["normalized_NRS"] = minmax(config["mean_NRS"])
    config["normalized_speed"] = minmax(log_runtime, invert=True)
    config["normalized_size_efficiency"] = minmax(config["NRS_per_billion_parameters"])
    config["normalized_stability"] = minmax(sd_for_stability, invert=True)
    config["normalized_reliability"] = config["reliability"].clip(0.0, 1.0)
    config["balanced_score"] = (
        weights["NRS"] * config["normalized_NRS"]
        + weights["speed"] * config["normalized_speed"]
        + weights["size_efficiency"] * config["normalized_size_efficiency"]
        + weights["stability"] * config["normalized_stability"]
        + weights["reliability"] * config["normalized_reliability"]
    )
    return config.sort_values(["mean_NRS", "failure_rate", "mean_runtime_seconds"], ascending=[False, True, True])


def _round_numeric(table: pd.DataFrame, digits: int = 3) -> pd.DataFrame:
    out = table.copy()
    for col in out.select_dtypes(include=["number"]).columns:
        out[col] = out[col].round(digits)
    return out


def _model_strategy_matrix(df: pd.DataFrame, options: AnalysisOptions) -> pd.DataFrame:
    success = df[df["success"]].copy()
    if success.empty:
        return pd.DataFrame()
    means = success.groupby(["model", "input_strategy"], dropna=False)["NRS"].mean().unstack("input_strategy")
    means = means.reindex(columns=options.input_order)
    means = means.rename(columns={strategy: f"{strategy}_NRS" for strategy in means.columns})
    meta = df.groupby("model", dropna=False).agg(
        parameters=("parameters_b", "median"),
        mean_runtime=("runtime_seconds", "mean"),
        failure_rate=("failed", "mean"),
    )
    out = meta.join(means, how="left").reset_index()
    nrs_cols = [f"{strategy}_NRS" for strategy in options.input_order if f"{strategy}_NRS" in out.columns]
    out["best_NRS"] = out[nrs_cols].max(axis=1, skipna=True)
    out["best_strategy"] = out[nrs_cols].idxmax(axis=1).str.replace("_NRS", "", regex=False)
    cols = ["model", "parameters", *nrs_cols, "best_strategy", "best_NRS", "mean_runtime", "failure_rate"]
    return _round_numeric(out[cols].sort_values("best_NRS", ascending=False), 3)


def _model_strategy_prompt_matrix(df: pd.DataFrame, options: AnalysisOptions) -> pd.DataFrame:
    success = df[df["success"]].copy()
    if success.empty:
        return pd.DataFrame()
    success["cell"] = success["input_strategy"].astype(str) + "_" + success["prompt_strategy"].astype(str)
    desired_cols = [f"{input_strategy}_{prompt}" for input_strategy in options.input_order for prompt in options.prompt_order]
    pivot = success.pivot_table(index="model", columns="cell", values="NRS", aggfunc="mean", observed=False)
    pivot = pivot.reindex(columns=desired_cols).reset_index()
    return _round_numeric(pivot, 3)


def _model_family_summary(df: pd.DataFrame) -> pd.DataFrame:
    success = df[df["success"]].copy()
    if success.empty:
        return pd.DataFrame()
    quality = success.groupby("family", dropna=False).agg(
        n_models=("model", "nunique"),
        mean_NRS=("NRS", "mean"),
        max_NRS=("NRS", "max"),
        mean_runtime=("runtime_seconds", "mean"),
    )
    failure = df.groupby("family", dropna=False).agg(mean_failure_rate=("failed", "mean"))
    out = quality.join(failure, how="left").reset_index().sort_values("max_NRS", ascending=False)
    return _round_numeric(out, 3)


def _top5_configurations(config: pd.DataFrame) -> pd.DataFrame:
    if config.empty:
        return pd.DataFrame()
    top = config.dropna(subset=["mean_NRS"]).sort_values("mean_NRS", ascending=False).head(5).copy()
    top.insert(0, "rank", range(1, len(top) + 1))
    out = top.rename(columns={"mean_runtime_seconds": "runtime_seconds", "parameters_b": "parameters"})[
        ["rank", "model", "input_strategy", "prompt_strategy", "mean_NRS", "runtime_seconds", "parameters", "failure_rate"]
    ]
    return _round_numeric(out, 3)


def _top_models_strategy_matrix(df: pd.DataFrame, options: AnalysisOptions) -> pd.DataFrame:
    success = df[df["success"]].copy()
    if success.empty:
        return pd.DataFrame()
    top_models = success.groupby("model", dropna=False)["NRS"].mean().sort_values(ascending=False).head(5).index.tolist()
    subset = success[success["model"].isin(top_models)]
    pivot = subset.pivot_table(index="model", columns="input_strategy", values="NRS", aggfunc="mean", observed=False)
    pivot = pivot.reindex(index=top_models, columns=options.input_order).reset_index()
    return _round_numeric(pivot, 3)


def _coverage_summary(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if not has_coverage_metrics(df):
        return pd.DataFrame()
    metric_cols = [
        "bertscore_f1",
        "semantic_similarity",
        "source_coverage_075",
        "generation_support_075",
        "compression_ratio",
        "entity_coverage",
        "keyphrase_coverage",
        "coverage_adjusted_bertscore_075",
        "coverage_adjusted_semantic_similarity_075",
        "omission_risk_075",
    ]
    for col in metric_cols:
        if col not in df.columns:
            df[col] = np.nan
    success = df[df["success"]].copy()
    means = success.groupby(group_cols, dropna=False).agg(
        **{f"mean_{col}": (col, "mean") for col in metric_cols},
        n_runs=("NRS", "size"),
    )
    failures = df.groupby(group_cols, dropna=False).agg(failure_rate=("failed", "mean"))
    return _round_numeric(means.join(failures, how="left").reset_index(), 3)


def _high_alignment_low_coverage_cases(df: pd.DataFrame) -> pd.DataFrame:
    required = ["bertscore_f1", "source_coverage_075"]
    if not all(col in df.columns for col in required):
        return pd.DataFrame()
    subset = df[(df["bertscore_f1"] >= 0.85) & (df["source_coverage_075"] <= 0.50)].copy()
    cols = [
        "model",
        "input_strategy",
        "prompt_strategy",
        "case_id",
        "bertscore_f1",
        "semantic_similarity",
        "source_coverage_075",
        "generation_support_075",
        "compression_ratio",
        "omission_risk_075",
    ]
    return _round_numeric(subset[[col for col in cols if col in subset.columns]].sort_values("omission_risk_075", ascending=False), 3)


def pareto_frontier(config: pd.DataFrame) -> pd.DataFrame:
    """Return configurations not dominated on NRS, runtime, parameters, failure."""
    cols = ["mean_NRS", "mean_runtime_seconds", "parameters_b", "failure_rate"]
    usable = config.dropna(subset=cols).copy().reset_index(drop=True)
    if usable.empty:
        return usable
    arr = usable[cols].to_numpy(dtype=float)
    dominated = np.zeros(len(usable), dtype=bool)
    for i in range(len(usable)):
        nrs_ge = arr[:, 0] >= arr[i, 0]
        runtime_le = arr[:, 1] <= arr[i, 1]
        params_le = arr[:, 2] <= arr[i, 2]
        failure_le = arr[:, 3] <= arr[i, 3]
        any_strict = (
            (arr[:, 0] > arr[i, 0])
            | (arr[:, 1] < arr[i, 1])
            | (arr[:, 2] < arr[i, 2])
            | (arr[:, 3] < arr[i, 3])
        )
        dominated[i] = bool(np.any(nrs_ge & runtime_le & params_le & failure_le & any_strict))
    return usable.loc[~dominated].sort_values(
        ["parameters_b", "mean_runtime_seconds", "mean_NRS"], ascending=[True, True, False]
    )


def _best_or_empty(config: pd.DataFrame, sort_cols: list[str], ascending: list[bool]) -> pd.Series:
    if config.empty:
        return pd.Series(dtype=object)
    return config.sort_values(sort_cols, ascending=ascending).iloc[0]


def _quality_loss_table(config: pd.DataFrame, options: AnalysisOptions) -> pd.DataFrame:
    tradeoff = config.dropna(subset=["parameters_b", "mean_NRS", "mean_runtime_seconds", "failure_rate"]).copy()
    if tradeoff.empty:
        return pd.DataFrame()

    global_best = tradeoff.sort_values(
        ["mean_NRS", "failure_rate", "mean_runtime_seconds"], ascending=[False, True, True]
    ).iloc[0]
    rows: list[dict[str, Any]] = []
    for label, threshold in options.threshold_b:
        subset = tradeoff if np.isinf(threshold) else tradeoff[tradeoff["parameters_b"] <= threshold]
        if subset.empty:
            rows.append({"threshold": label})
            continue
        best = subset.sort_values(["mean_NRS", "failure_rate", "mean_runtime_seconds"], ascending=[False, True, True]).iloc[0]
        reliable = subset[subset["failure_rate"] <= options.reliable_failure_rate_max]
        best_reliable = (
            reliable.sort_values(["mean_NRS", "failure_rate", "mean_runtime_seconds"], ascending=[False, True, True]).iloc[0]
            if not reliable.empty
            else pd.Series(dtype=object)
        )
        row = {
            "threshold": label,
            "best_model": best.get("model"),
            "best_input_strategy": best.get("input_strategy"),
            "best_prompt_strategy": best.get("prompt_strategy"),
            "parameters_b": best.get("parameters_b"),
            "mean_NRS": best.get("mean_NRS"),
            "NRS_loss_vs_global_best": global_best.get("mean_NRS") - best.get("mean_NRS"),
            "mean_runtime_seconds": best.get("mean_runtime_seconds"),
            "failure_rate": best.get("failure_rate"),
        }
        if not best_reliable.empty:
            row.update(
                {
                    "reliable_best_model": best_reliable.get("model"),
                    "reliable_best_input_strategy": best_reliable.get("input_strategy"),
                    "reliable_best_prompt_strategy": best_reliable.get("prompt_strategy"),
                    "reliable_best_parameters_b": best_reliable.get("parameters_b"),
                    "reliable_best_mean_NRS": best_reliable.get("mean_NRS"),
                    "reliable_NRS_loss_vs_global_best": global_best.get("mean_NRS") - best_reliable.get("mean_NRS"),
                    "reliable_best_runtime_seconds": best_reliable.get("mean_runtime_seconds"),
                    "reliable_best_failure_rate": best_reliable.get("failure_rate"),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _best_tradeoff_table(config: pd.DataFrame, options: AnalysisOptions) -> pd.DataFrame:
    tradeoff = config.dropna(subset=["parameters_b", "mean_NRS", "mean_runtime_seconds", "failure_rate"]).copy()
    if tradeoff.empty:
        return pd.DataFrame()
    criteria = [
        ("best raw quality", tradeoff, ["mean_NRS", "failure_rate", "mean_runtime_seconds"], [False, True, True]),
        ("best balanced score", tradeoff, ["balanced_score", "mean_NRS"], [False, False]),
        ("best reliable balanced score", tradeoff[tradeoff["failure_rate"] <= options.reliable_failure_rate_max], ["balanced_score", "mean_NRS"], [False, False]),
        ("best quality per second", tradeoff, ["NRS_per_second", "mean_NRS"], [False, False]),
        ("best quality per billion parameters", tradeoff, ["NRS_per_billion_parameters", "mean_NRS"], [False, False]),
        ("best quality per parameter-second", tradeoff, ["NRS_per_parameter_second", "mean_NRS"], [False, False]),
    ]
    rows = []
    for name, subset, cols, ascending in criteria:
        if subset.empty:
            rows.append({"criterion": name})
            continue
        row = _best_or_empty(subset, cols, ascending).to_dict()
        row = {"criterion": name, **row}
        rows.append(row)
    return pd.DataFrame(rows)


def _correlation_tables(df: pd.DataFrame, config: pd.DataFrame, options: AnalysisOptions) -> tuple[pd.DataFrame, dict[str, float]]:
    usable_config = config.dropna(subset=["parameters_b", "mean_NRS", "mean_runtime_seconds"])
    model_level = df.groupby(["model", "parameters_b"], dropna=False).agg(
        mean_NRS=("NRS", "mean"),
        mean_runtime_seconds=("runtime_seconds", "mean"),
        failure_rate=("failed", "mean"),
    ).reset_index().dropna(subset=["parameters_b", "mean_NRS", "mean_runtime_seconds"])

    def corr(a, b):
        if len(a) < 3:
            return np.nan
        return float(pd.Series(a).corr(pd.Series(b), method="spearman"))

    summary = {
        "spearman_parameters_NRS_config": corr(usable_config["parameters_b"], usable_config["mean_NRS"]),
        "spearman_parameters_runtime_config": corr(usable_config["parameters_b"], usable_config["mean_runtime_seconds"]),
        "spearman_parameters_NRS_model": corr(model_level["parameters_b"], model_level["mean_NRS"]),
        "spearman_parameters_runtime_model": corr(model_level["parameters_b"], model_level["mean_runtime_seconds"]),
    }

    rows = []
    for input_strategy in ordered_values(df["input_strategy"], options.input_order):
        subset = usable_config[usable_config["input_strategy"] == input_strategy]
        rows.append(
            {
                "input_strategy": input_strategy,
                "spearman_parameters_NRS": corr(subset["parameters_b"], subset["mean_NRS"]),
                "spearman_parameters_runtime": corr(subset["parameters_b"], subset["mean_runtime_seconds"]),
                "configurations": int(len(subset)),
            }
        )
    return pd.DataFrame(rows), summary


def make_tables(df: pd.DataFrame, options: AnalysisOptions) -> dict[str, pd.DataFrame]:
    df = _add_size_bins(df)
    success_df = df[df["success"]].copy()

    descriptive = pd.DataFrame(
        [
            {
                "scope": "all attempts",
                "total_runs": len(df),
                "models": df["model"].nunique(),
                "input_strategies": df["input_strategy"].nunique(),
                "prompt_strategies": df["prompt_strategy"].nunique(),
                "mean_NRS": df["NRS"].mean(),
                "sd_NRS": df["NRS"].std(),
                "mean_runtime_seconds": df["runtime_seconds"].mean(),
                "sd_runtime_seconds": df["runtime_seconds"].std(),
                "failure_rate": df["failed"].mean(),
                "parameter_min_b": df["parameters_b"].min(),
                "parameter_max_b": df["parameters_b"].max(),
            },
            {
                "scope": "successful runs only",
                "total_runs": len(success_df),
                "models": success_df["model"].nunique(),
                "input_strategies": success_df["input_strategy"].nunique(),
                "prompt_strategies": success_df["prompt_strategy"].nunique(),
                "mean_NRS": success_df["NRS"].mean(),
                "sd_NRS": success_df["NRS"].std(),
                "mean_runtime_seconds": success_df["runtime_seconds"].mean(),
                "sd_runtime_seconds": success_df["runtime_seconds"].std(),
                "failure_rate": 0.0,
                "parameter_min_b": success_df["parameters_b"].min(),
                "parameter_max_b": success_df["parameters_b"].max(),
            },
        ]
    )

    input_summary = aggregate_summary(df, ["input_strategy"]).sort_values("mean_NRS", ascending=False)
    prompt_summary = aggregate_summary(df, ["prompt_strategy"]).sort_values("mean_NRS", ascending=False)
    input_prompt_summary = aggregate_summary(df, ["input_strategy", "prompt_strategy"]).sort_values("mean_NRS", ascending=False)
    model_summary = aggregate_summary(df, ["model", "family", "parameters_b"]).sort_values("mean_NRS", ascending=False)
    model_input_summary = aggregate_summary(df, ["model", "family", "parameters_b", "input_strategy"])
    model_prompt_summary = aggregate_summary(df, ["model", "family", "parameters_b", "prompt_strategy"])

    model_order = {model: i for i, model in enumerate(model_summary["model"].tolist())}
    model_input_pivot = model_input_summary.pivot_table(
        index=["model", "parameters_b"], columns="input_strategy", values="mean_NRS", observed=False
    ).reset_index()
    model_input_pivot["rank"] = model_input_pivot["model"].map(model_order)
    model_input_pivot = model_input_pivot.sort_values("rank").drop(columns="rank")

    model_prompt_pivot = model_prompt_summary.pivot_table(
        index=["model", "parameters_b"], columns="prompt_strategy", values="mean_NRS", observed=False
    ).reset_index()
    model_prompt_pivot["rank"] = model_prompt_pivot["model"].map(model_order)
    model_prompt_pivot = model_prompt_pivot.sort_values("rank").drop(columns="rank")

    size_bin_summary = df.groupby("size_bin", observed=False, dropna=False).agg(
        models=("model", "nunique"),
        attempts=("NRS", "size"),
        mean_NRS=("NRS", "mean"),
        sd_NRS=("NRS", "std"),
        mean_runtime_seconds=("runtime_seconds", "mean"),
        sd_runtime_seconds=("runtime_seconds", "std"),
        failure_rate=("failed", "mean"),
        parameter_min_b=("parameters_b", "min"),
        parameter_max_b=("parameters_b", "max"),
    ).reset_index()

    configuration_summary = _make_configuration_summary(df, options)
    model_strategy_matrix = _model_strategy_matrix(df, options)
    model_strategy_prompt_matrix = _model_strategy_prompt_matrix(df, options)
    model_family_summary = _model_family_summary(df)
    top5_configurations = _top5_configurations(configuration_summary)
    top_models_strategy_matrix = _top_models_strategy_matrix(df, options)
    coverage_by_input_strategy = _coverage_summary(df.copy(), ["input_strategy"])
    coverage_by_prompt_strategy = _coverage_summary(df.copy(), ["prompt_strategy"])
    coverage_by_model = _coverage_summary(df.copy(), ["model", "family", "parameters_b"])
    coverage_by_model_strategy_prompt = _coverage_summary(df.copy(), ["model", "family", "input_strategy", "prompt_strategy", "parameters_b"])
    coverage_by_case = _coverage_summary(df.copy(), ["case_id"] if "case_id" in df.columns else ["source_file"])
    coverage_by_model_family = _coverage_summary(df.copy(), ["family"])
    high_alignment_low_coverage_cases = _high_alignment_low_coverage_cases(df)
    pareto = pareto_frontier(configuration_summary)
    quality_loss = _quality_loss_table(configuration_summary, options)
    best_tradeoff = _best_tradeoff_table(configuration_summary, options)
    parameter_correlations_by_input, correlation_summary = _correlation_tables(df, configuration_summary, options)

    correlation_summary_table = pd.DataFrame([correlation_summary])

    return {
        "descriptive_statistics": descriptive,
        "input_strategy_summary": input_summary,
        "prompt_strategy_summary": prompt_summary,
        "input_prompt_summary": input_prompt_summary,
        "model_summary": model_summary,
        "model_input_summary": model_input_summary,
        "model_prompt_summary": model_prompt_summary,
        "model_by_input_pivot_mean_NRS": model_input_pivot,
        "model_by_prompt_pivot_mean_NRS": model_prompt_pivot,
        "size_bin_summary": size_bin_summary,
        "configuration_summary_with_efficiency": configuration_summary,
        "model_strategy_matrix": model_strategy_matrix,
        "model_strategy_prompt_matrix": model_strategy_prompt_matrix,
        "model_family_summary": model_family_summary,
        "top5_configurations": top5_configurations,
        "top_models_strategy_matrix": top_models_strategy_matrix,
        "coverage_by_input_strategy": coverage_by_input_strategy,
        "coverage_by_prompt_strategy": coverage_by_prompt_strategy,
        "coverage_by_model": coverage_by_model,
        "coverage_by_model_strategy_prompt": coverage_by_model_strategy_prompt,
        "coverage_by_case": coverage_by_case,
        "coverage_by_model_family": coverage_by_model_family,
        "high_alignment_low_coverage_cases": high_alignment_low_coverage_cases,
        "quality_loss_by_parameter_threshold": quality_loss,
        "best_tradeoff_configurations": best_tradeoff,
        "pareto_optimal_configurations": pareto,
        "parameter_correlations_by_input_strategy": parameter_correlations_by_input,
        "parameter_correlation_summary": correlation_summary_table,
    }


def _make_summary(df: pd.DataFrame, tables: dict[str, pd.DataFrame], options: AnalysisOptions, metadata: dict) -> dict[str, Any]:
    config = tables["configuration_summary_with_efficiency"]
    tradeoff = config.dropna(subset=["parameters_b", "mean_NRS", "mean_runtime_seconds", "failure_rate"])

    def first_row(table_name: str) -> dict[str, Any]:
        table = tables.get(table_name, pd.DataFrame())
        return table.iloc[0].to_dict() if not table.empty else {}

    summary = {
        "title": options.title,
        "options": asdict(options),
        "source_metadata": metadata,
        "run_count": int(len(df)),
        "model_count": int(df["model"].nunique()),
        "input_strategy_count": int(df["input_strategy"].nunique()),
        "prompt_strategy_count": int(df["prompt_strategy"].nunique()),
        "case_count": int(df["case_id"].nunique()) if "case_id" in df.columns else None,
        "best_input_strategy": first_row("input_strategy_summary"),
        "best_prompt_strategy": first_row("prompt_strategy_summary"),
        "best_input_prompt_strategy": first_row("input_prompt_summary"),
        "best_model": first_row("model_summary"),
        "best_tradeoffs": tables.get("best_tradeoff_configurations", pd.DataFrame()).to_dict(orient="records"),
        "pareto_configuration_count": int(len(tables.get("pareto_optimal_configurations", pd.DataFrame()))),
        "configuration_count": int(len(config)),
    }
    if not tradeoff.empty:
        summary["global_best_configuration"] = tradeoff.sort_values(
            ["mean_NRS", "failure_rate", "mean_runtime_seconds"], ascending=[False, True, True]
        ).iloc[0].to_dict()
        summary["balanced_best_configuration"] = tradeoff.sort_values(
            ["balanced_score", "mean_NRS"], ascending=[False, False]
        ).iloc[0].to_dict()
    return summary


def run_analysis(source: str | Path, output_dir: str | Path, options: AnalysisOptions | None = None) -> AnalysisResult:
    """Run the complete NRS benchmark analysis pipeline."""
    options = options or AnalysisOptions()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir, tables_dir = make_output_dirs(output_dir)

    df, metadata = load_runs(source)
    # Ensure the metrics used by the benchmark are explicitly present and typed.
    df["NRS"] = pd.to_numeric(df["NRS"], errors="coerce")
    df["runtime_seconds"] = pd.to_numeric(df["runtime_seconds"], errors="coerce")
    df["failed"] = df["failed"].astype(bool)
    df["success"] = ~df["failed"]
    for metric in ("bertscore_f1", "semantic_similarity"):
        if metric in df.columns:
            df[metric] = pd.to_numeric(df[metric], errors="coerce")

    coverage_result = apply_coverage_diagnostics(
        df,
        CoverageOptions(
            enabled=options.coverage,
            thresholds=options.coverage_thresholds,
            model=options.coverage_model,
            skip_entity_coverage=options.skip_entity_coverage,
            skip_keyphrase_coverage=options.skip_keyphrase_coverage,
        ),
        tables_dir,
    )
    df = coverage_result.df

    combined_path = output_dir / "combined_nrs_runs.csv"
    df.to_csv(combined_path, index=False)

    tables = make_tables(df, options)
    written_tables = write_dataframe_tables(tables, tables_dir)
    figure_paths, figure_interpretations = create_all_figures(df, tables, figures_dir, options)

    summary = _make_summary(df, tables, options, metadata)
    summary["combined_runs_csv"] = str(combined_path)
    summary["tables"] = written_tables
    summary["figures"] = {k: str(v) for k, v in figure_paths.items()}
    summary["figure_interpretations"] = figure_interpretations
    summary["coverage_diagnostics"] = {"available": coverage_result.available, "warnings": coverage_result.warnings}

    summary_path = output_dir / "analysis_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=_json_default)

    interpretations_path = output_dir / "figure_interpretations.json"
    with open(interpretations_path, "w", encoding="utf-8") as f:
        json.dump(figure_interpretations, f, indent=2, default=_json_default)

    write_markdown_report(output_dir / "narrativeforge_nrs_analysis_report.md", df, tables, summary, options)
    write_html_report(output_dir / "narrativeforge_nrs_analysis_report.html", df, tables, summary, options)
    write_latex_report(output_dir / "narrativeforge_nrs_analysis_report.tex", df, tables, summary, options)
    docx_path = write_docx_report(output_dir / "narrativeforge_nrs_analysis_report.docx", df, tables, summary, options)
    if docx_path is not None:
        summary["docx_report"] = str(docx_path)

    bundle_path = make_bundle(output_dir)
    return AnalysisResult(
        output_dir=output_dir,
        figures_dir=figures_dir,
        tables_dir=tables_dir,
        bundle_path=bundle_path,
        summary=summary,
        tables=tables,
        figure_paths=figure_paths,
    )
