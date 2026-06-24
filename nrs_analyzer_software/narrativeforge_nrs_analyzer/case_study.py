from __future__ import annotations

import numpy as np
import pandas as pd

CASE_COLUMNS = ["case_id", "case_name", "case", "story_id", "scenario_id", "benchmark_case"]
INPUT_ORDER = ["auto", "brief", "rag", "full"]
SIZE_THRESHOLDS = [("<=2B", 2.0), ("<=4B", 4.0), ("<=8B", 8.0), ("<=16B", 16.0), ("all", np.inf)]


def detect_case_column(df: pd.DataFrame) -> str | None:
    lookup = {str(col).lower(): col for col in df.columns}
    return next((lookup[col] for col in CASE_COLUMNS if col in lookup), None)


def add_case_study_column(df: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    out = df.copy()
    col = detect_case_column(out)
    if col is None:
        return out, None
    out["case_study"] = out[col].fillna("unknown_case").astype(str).replace({"": "unknown_case"})
    return out, col


def _round(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    for col in out.select_dtypes(include=["number"]).columns:
        out[col] = out[col].round(3)
    return out


def _minmax(series: pd.Series, invert: bool = False) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(0.0)
    mn, mx = s.min(), s.max()
    if mx == mn:
        out = pd.Series(np.zeros(len(s)), index=s.index)
    else:
        out = (s - mn) / (mx - mn)
    return 1.0 - out if invert else out


def _bucket(parameters: float) -> str:
    if pd.isna(parameters):
        return "unknown"
    value = float(parameters)
    if value <= 2:
        return "<=2B"
    if value <= 4:
        return "<=4B"
    if value <= 8:
        return "<=8B"
    if value <= 16:
        return "<=16B"
    return ">16B"


def case_study_tables(df: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], str | None]:
    df, case_col = add_case_study_column(df)
    if case_col is None:
        return {}, None
    success = df[df["success"]].copy()

    counts = df.groupby("case_study", dropna=False).agg(
        n_runs=("NRS", "size"),
        n_successful_runs=("success", "sum"),
        failure_rate=("failed", "mean"),
        n_models=("model", "nunique"),
        n_input_strategies=("input_strategy", "nunique"),
        n_prompt_strategies=("prompt_strategy", "nunique"),
    )
    stats = success.groupby("case_study", dropna=False).agg(
        mean_NRS=("NRS", "mean"),
        std_NRS=("NRS", "std"),
        median_NRS=("NRS", "median"),
        q25_NRS=("NRS", lambda s: s.quantile(0.25)),
        q75_NRS=("NRS", lambda s: s.quantile(0.75)),
        min_NRS=("NRS", "min"),
        max_NRS=("NRS", "max"),
        mean_runtime_seconds=("runtime_seconds", "mean"),
        std_runtime_seconds=("runtime_seconds", "std"),
        median_runtime_seconds=("runtime_seconds", "median"),
        mean_parameters=("parameters_b", "mean"),
    )
    case_summary = _round(counts.join(stats, how="left").reset_index().sort_values("mean_NRS", ascending=False))

    config_group = ["case_study", "model", "input_strategy", "prompt_strategy"]
    config = df.groupby(config_group, dropna=False).agg(
        n_runs=("NRS", "size"),
        failure_rate=("failed", "mean"),
        parameters=("parameters_b", "median"),
    ).join(
        success.groupby(config_group, dropna=False).agg(
            mean_NRS=("NRS", "mean"),
            std_NRS=("NRS", "std"),
            mean_runtime_seconds=("runtime_seconds", "mean"),
        ),
        how="left",
    ).reset_index()
    top5 = _round(config.sort_values(["case_study", "mean_NRS"], ascending=[True, False]).groupby("case_study", dropna=False).head(5))
    best = config.sort_values(["case_study", "mean_NRS"], ascending=[True, False]).groupby("case_study", dropna=False).head(1).copy()
    best_configuration = _round(best.rename(columns={
        "model": "best_model",
        "input_strategy": "best_input_strategy",
        "prompt_strategy": "best_prompt_strategy",
        "mean_NRS": "best_mean_NRS",
        "std_NRS": "best_std_NRS",
        "mean_runtime_seconds": "best_runtime_seconds",
        "parameters": "best_parameters",
        "failure_rate": "best_failure_rate",
    })[["case_study", "best_model", "best_input_strategy", "best_prompt_strategy", "best_mean_NRS", "best_std_NRS", "best_runtime_seconds", "best_parameters", "best_failure_rate", "n_runs"]])

    difficulty = case_summary[["case_study", "mean_NRS", "std_NRS", "failure_rate"]].copy()
    difficulty["difficulty_score"] = 0.60 * _minmax(difficulty["mean_NRS"], invert=True) + 0.25 * _minmax(difficulty["failure_rate"]) + 0.15 * _minmax(difficulty["std_NRS"])
    if difficulty["difficulty_score"].nunique(dropna=True) < 3:
        difficulty["difficulty_label"] = "moderate"
    else:
        try:
            difficulty["difficulty_label"] = pd.qcut(difficulty["difficulty_score"], 3, labels=["easy", "moderate", "hard"]).astype(str)
        except ValueError:
            difficulty["difficulty_label"] = "moderate"
    difficulty = difficulty.sort_values("difficulty_score", ascending=False).reset_index(drop=True)
    difficulty.insert(0, "rank", range(1, len(difficulty) + 1))
    difficulty = _round(difficulty)

    input_nrs = _round(success.pivot_table(index="case_study", columns="input_strategy", values="NRS", aggfunc="mean", observed=False).reindex(columns=INPUT_ORDER).reset_index())
    input_runtime = _round(success.pivot_table(index="case_study", columns="input_strategy", values="runtime_seconds", aggfunc="mean", observed=False).reindex(columns=INPUT_ORDER).reset_index())
    input_delta_rows = []
    for _, row in input_nrs.iterrows():
        vals = row.drop(labels=["case_study"]).dropna()
        if vals.empty:
            continue
        input_delta_rows.append({
            "case_study": row["case_study"],
            "best_input_strategy": vals.idxmax(),
            "best_input_NRS": vals.max(),
            "worst_input_strategy": vals.idxmin(),
            "worst_input_NRS": vals.min(),
            "input_strategy_NRS_range": vals.max() - vals.min(),
            "full_minus_brief_NRS": row.get("full", np.nan) - row.get("brief", np.nan),
            "rag_minus_brief_NRS": row.get("rag", np.nan) - row.get("brief", np.nan),
            "full_minus_auto_NRS": row.get("full", np.nan) - row.get("auto", np.nan),
            "rag_minus_auto_NRS": row.get("rag", np.nan) - row.get("auto", np.nan),
        })
    input_delta = _round(pd.DataFrame(input_delta_rows).sort_values("input_strategy_NRS_range", ascending=False) if input_delta_rows else pd.DataFrame())

    prompt_order = sorted(success["prompt_strategy"].dropna().astype(str).unique())
    prompt_nrs = _round(success.pivot_table(index="case_study", columns="prompt_strategy", values="NRS", aggfunc="mean", observed=False).reindex(columns=prompt_order).reset_index())
    prompt_runtime = _round(success.pivot_table(index="case_study", columns="prompt_strategy", values="runtime_seconds", aggfunc="mean", observed=False).reindex(columns=prompt_order).reset_index())
    prompt_delta_rows = []
    for _, row in prompt_nrs.iterrows():
        vals = row.drop(labels=["case_study"]).dropna()
        if vals.empty:
            continue
        item = {
            "case_study": row["case_study"],
            "best_prompt_strategy": vals.idxmax(),
            "best_prompt_NRS": vals.max(),
            "worst_prompt_strategy": vals.idxmin(),
            "worst_prompt_NRS": vals.min(),
            "prompt_strategy_NRS_range": vals.max() - vals.min(),
        }
        if {"detailed", "short"}.issubset(row.index):
            item["detailed_minus_short_NRS"] = row.get("detailed", np.nan) - row.get("short", np.nan)
        if {"standard", "short"}.issubset(row.index):
            item["standard_minus_short_NRS"] = row.get("standard", np.nan) - row.get("short", np.nan)
        if {"detailed", "standard"}.issubset(row.index):
            item["detailed_minus_standard_NRS"] = row.get("detailed", np.nan) - row.get("standard", np.nan)
        prompt_delta_rows.append(item)
    prompt_delta = _round(pd.DataFrame(prompt_delta_rows).sort_values("prompt_strategy_NRS_range", ascending=False) if prompt_delta_rows else pd.DataFrame())

    df = df.copy()
    success = success.copy()
    df["parameter_bucket"] = df["parameters_b"].map(_bucket)
    success["parameter_bucket"] = success["parameters_b"].map(_bucket)
    bucket_counts = df.groupby(["case_study", "parameter_bucket"], dropna=False).agg(n_runs=("NRS", "size"), failure_rate=("failed", "mean"))
    bucket_stats = success.groupby(["case_study", "parameter_bucket"], dropna=False).agg(mean_NRS=("NRS", "mean"), std_NRS=("NRS", "std"), mean_runtime_seconds=("runtime_seconds", "mean"))
    bucket_best = config.copy()
    bucket_best["parameter_bucket"] = bucket_best["parameters"].map(_bucket)
    bucket_best = bucket_best.sort_values(["case_study", "parameter_bucket", "mean_NRS"], ascending=[True, True, False]).groupby(["case_study", "parameter_bucket"], dropna=False).head(1)[["case_study", "parameter_bucket", "model", "mean_NRS"]]
    bucket_best = bucket_best.rename(columns={"model": "best_model_in_bucket", "mean_NRS": "best_mean_NRS_in_bucket"}).set_index(["case_study", "parameter_bucket"])
    size_summary = _round(bucket_counts.join(bucket_stats, how="left").join(bucket_best, how="left").reset_index())

    loss_rows = []
    for case, case_configs in config.groupby("case_study", dropna=False):
        case_best = case_configs["mean_NRS"].max()
        for label, threshold in SIZE_THRESHOLDS:
            subset = case_configs if np.isinf(threshold) else case_configs[case_configs["parameters"] <= threshold]
            subset = subset.dropna(subset=["mean_NRS"])
            if subset.empty:
                loss_rows.append({"case_study": case, "threshold": label})
                continue
            best_row = subset.sort_values("mean_NRS", ascending=False).iloc[0]
            loss_rows.append({
                "case_study": case,
                "threshold": label,
                "best_model": best_row["model"],
                "best_input_strategy": best_row["input_strategy"],
                "best_prompt_strategy": best_row["prompt_strategy"],
                "best_mean_NRS": best_row["mean_NRS"],
                "runtime_seconds": best_row["mean_runtime_seconds"],
                "parameters": best_row["parameters"],
                "NRS_loss_vs_case_best": case_best - best_row["mean_NRS"],
            })
    size_loss = _round(pd.DataFrame(loss_rows))

    stability = case_summary[["case_study", "mean_NRS", "std_NRS", "mean_runtime_seconds", "std_runtime_seconds", "failure_rate"]].copy()
    stability["coefficient_of_variation_NRS"] = stability["std_NRS"] / stability["mean_NRS"].replace(0, np.nan)
    stability["coefficient_of_variation_runtime"] = stability["std_runtime_seconds"] / stability["mean_runtime_seconds"].replace(0, np.nan)
    stability["stability_score"] = 1 - _minmax(stability["std_NRS"])
    stability = _round(stability.sort_values("stability_score", ascending=False))

    return {
        "case_study_summary": case_summary,
        "best_configuration_by_case": best_configuration,
        "top5_configurations_by_case": top5,
        "case_difficulty_ranking": difficulty,
        "case_by_input_strategy_nrs": input_nrs,
        "case_by_input_strategy_runtime": input_runtime,
        "case_input_strategy_delta": input_delta,
        "case_by_prompt_strategy_nrs": prompt_nrs,
        "case_by_prompt_strategy_runtime": prompt_runtime,
        "case_prompt_strategy_delta": prompt_delta,
        "case_model_size_summary": size_summary,
        "case_model_size_loss": size_loss,
        "case_stability_summary": stability,
    }, case_col
