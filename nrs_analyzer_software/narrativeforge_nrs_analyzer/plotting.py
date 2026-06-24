from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import ordered_values


def _setup(dpi: int = 220):
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150, "savefig.dpi": dpi})


def _save(fig, figures_dir: Path, filename: str, figure_format: str) -> Path:
    figures_dir.mkdir(parents=True, exist_ok=True)
    if not filename.endswith(f".{figure_format}"):
        filename = f"{filename}.{figure_format}"
    path = figures_dir / filename
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _grid(ax, axis="y"):
    ax.grid(True, axis=axis, alpha=0.25)


def _placeholder(figures_dir: Path, filename: str, title: str, message: str, options) -> Path:
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.axis("off")
    ax.text(0.5, 0.55, title, ha="center", va="center", fontsize=13, weight="bold")
    ax.text(0.5, 0.42, message, ha="center", va="center", wrap=True)
    return _save(fig, figures_dir, filename, options.figure_format)


def _safe_ylim_for_means(ax, values: pd.Series, lower_padding=6, upper_padding=5):
    values = pd.Series(values).dropna()
    if values.empty:
        return
    ax.set_ylim(max(0, values.min() - lower_padding), min(100, values.max() + upper_padding))


def _bubble_sizes(values: pd.Series, min_size=25, max_size=220) -> pd.Series:
    s = pd.Series(values, dtype="float64").replace([np.inf, -np.inf], np.nan)
    if s.dropna().empty:
        return pd.Series(np.full(len(s), min_size), index=s.index)
    s = s.fillna(s.median())
    transformed = np.sqrt(np.clip(s, 0, None))
    mn, mx = transformed.min(), transformed.max()
    if mx == mn:
        return pd.Series(np.full(len(s), (min_size + max_size) / 2), index=s.index)
    return min_size + (max_size - min_size) * (transformed - mn) / (mx - mn)


def _clean_filename(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(text)).strip("_").lower() or "item"


def _plot_family_figures(tables: dict[str, pd.DataFrame], figures_dir: Path, options) -> dict[str, Path]:
    figures: dict[str, Path] = {}
    config = tables.get("configuration_summary_with_efficiency", pd.DataFrame()).copy()
    if config.empty or "family" not in config.columns:
        return figures
    family_dir = figures_dir / "family_scaling"
    model_level = config.dropna(subset=["family", "model", "parameters_b", "mean_NRS", "mean_runtime_seconds"]).groupby(
        ["family", "model", "parameters_b"], dropna=False
    ).agg(mean_NRS=("mean_NRS", "mean"), mean_runtime_seconds=("mean_runtime_seconds", "mean")).reset_index()
    for family, subset in model_level.groupby("family", dropna=False):
        subset = subset.sort_values("parameters_b")
        if subset.empty:
            continue
        stem = _clean_filename(family)

        fig, ax = plt.subplots(figsize=(6.8, 4.4))
        ax.plot(subset["parameters_b"], subset["mean_NRS"], marker="o", linewidth=1.8)
        ax.set_title(f"{family}: NRS scaling")
        ax.set_xlabel("Parameters (billions)")
        ax.set_ylabel("Mean NRS")
        _grid(ax)
        _safe_ylim_for_means(ax, subset["mean_NRS"])
        figures[f"family_{stem}_scaling_nrs"] = _save(fig, family_dir, f"{stem}_scaling_nrs", options.figure_format)

        fig, ax = plt.subplots(figsize=(6.8, 4.4))
        ax.plot(subset["parameters_b"], subset["mean_runtime_seconds"], marker="o", linewidth=1.8, color="#b45309")
        ax.set_title(f"{family}: runtime scaling")
        ax.set_xlabel("Parameters (billions)")
        ax.set_ylabel("Mean runtime (seconds)")
        _grid(ax)
        figures[f"family_{stem}_scaling_runtime"] = _save(fig, family_dir, f"{stem}_scaling_runtime", options.figure_format)

        fig, ax = plt.subplots(figsize=(6.8, 4.8))
        ax.scatter(subset["mean_runtime_seconds"], subset["mean_NRS"], s=70, alpha=0.8)
        for _, row in subset.iterrows():
            ax.annotate(str(row["model"]), (row["mean_runtime_seconds"], row["mean_NRS"]), fontsize=7, xytext=(4, 4), textcoords="offset points")
        ax.set_title(f"{family}: quality-efficiency")
        ax.set_xlabel("Mean runtime (seconds)")
        ax.set_ylabel("Mean NRS")
        _grid(ax)
        figures[f"family_{stem}_quality_efficiency"] = _save(fig, family_dir, f"{stem}_quality_efficiency", options.figure_format)
    return figures


def _plot_top5_figures(tables: dict[str, pd.DataFrame], figures_dir: Path, options) -> dict[str, Path]:
    figures: dict[str, Path] = {}
    top5 = tables.get("top5_configurations", pd.DataFrame()).copy()
    if top5.empty:
        return figures
    top_dir = figures_dir / "top5"
    top5["label"] = top5.apply(lambda r: f"{r['model']}\n{r['input_strategy']}/{r['prompt_strategy']}", axis=1)

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.bar(top5["label"], top5["mean_NRS"])
    ax.set_title("Top-5 configurations by NRS")
    ax.set_ylabel("Mean NRS")
    ax.tick_params(axis="x", labelrotation=35)
    _grid(ax)
    figures["top5_nrs_bar"] = _save(fig, top_dir, "top5_nrs_bar", options.figure_format)

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.bar(top5["label"], top5["runtime_seconds"], color="#0f766e")
    ax.set_title("Top-5 configurations: runtime")
    ax.set_ylabel("Runtime (seconds)")
    ax.tick_params(axis="x", labelrotation=35)
    _grid(ax)
    figures["top5_runtime_bar"] = _save(fig, top_dir, "top5_runtime_bar", options.figure_format)

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    sizes = _bubble_sizes(top5["parameters"], min_size=80, max_size=360)
    ax.scatter(top5["runtime_seconds"], top5["mean_NRS"], s=sizes, alpha=0.75)
    for _, row in top5.iterrows():
        ax.annotate(str(row["rank"]), (row["runtime_seconds"], row["mean_NRS"]), fontsize=9, weight="bold", xytext=(4, 4), textcoords="offset points")
    ax.set_title("Top-5 NRS vs runtime")
    ax.set_xlabel("Runtime (seconds)")
    ax.set_ylabel("Mean NRS")
    _grid(ax)
    figures["top5_nrs_runtime_scatter"] = _save(fig, top_dir, "top5_nrs_runtime_scatter", options.figure_format)

    radar = top5.copy()
    radar["speed"] = 1 - (radar["runtime_seconds"] - radar["runtime_seconds"].min()) / max(radar["runtime_seconds"].max() - radar["runtime_seconds"].min(), 1e-9)
    radar["reliability"] = 1 - radar["failure_rate"].fillna(0)
    radar["size_efficiency"] = 1 - (radar["parameters"] - radar["parameters"].min()) / max(radar["parameters"].max() - radar["parameters"].min(), 1e-9)
    radar["stability"] = 1.0
    radar["NRS_norm"] = (radar["mean_NRS"] - radar["mean_NRS"].min()) / max(radar["mean_NRS"].max() - radar["mean_NRS"].min(), 1e-9)
    metrics = ["NRS_norm", "speed", "reliability", "size_efficiency", "stability"]
    labels = ["NRS", "speed", "reliability", "size efficiency", "stability"]
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(6.4, 6.4), subplot_kw={"polar": True})
    for _, row in radar.iterrows():
        values = [float(row[m]) for m in metrics]
        values += values[:1]
        ax.plot(angles, values, linewidth=1.5, label=f"#{int(row['rank'])}")
        ax.fill(angles, values, alpha=0.08)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_title("Top-5 configuration radar")
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.10), fontsize=8)
    figures["top5_radar"] = _save(fig, top_dir, "top5_radar", options.figure_format)
    return figures


def _plot_top_model_figures(tables: dict[str, pd.DataFrame], figures_dir: Path, options) -> dict[str, Path]:
    figures: dict[str, Path] = {}
    model_summary = tables.get("model_summary", pd.DataFrame()).copy()
    heat = tables.get("top_models_strategy_matrix", pd.DataFrame()).copy()
    if model_summary.empty:
        return figures
    top_dir = figures_dir / "top_models"
    top = model_summary.sort_values("mean_NRS_success" if "mean_NRS_success" in model_summary.columns else "mean_NRS", ascending=False).head(5).copy()
    quality_col = "mean_NRS_success" if "mean_NRS_success" in top.columns else "mean_NRS"

    for key, col, title, ylabel in [
        ("top_models_nrs", quality_col, "Top models: NRS", "Mean NRS"),
        ("top_models_runtime", "mean_runtime_seconds", "Top models: runtime", "Runtime (seconds)"),
        ("top_models_failure", "failure_rate", "Top models: failure rate", "Failure rate"),
    ]:
        if col not in top.columns:
            continue
        fig, ax = plt.subplots(figsize=(7.4, 4.8))
        ax.bar(top["model"], top[col])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", labelrotation=35)
        _grid(ax)
        figures[key] = _save(fig, top_dir, key, options.figure_format)

    if not heat.empty:
        plot = heat.set_index("model")
        fig, ax = plt.subplots(figsize=(7.4, 4.8))
        image = ax.imshow(np.ma.masked_invalid(plot.to_numpy(dtype=float)), aspect="auto")
        ax.set_xticks(np.arange(len(plot.columns)))
        ax.set_xticklabels(plot.columns)
        ax.set_yticks(np.arange(len(plot.index)))
        ax.set_yticklabels(plot.index)
        for i in range(plot.shape[0]):
            for j in range(plot.shape[1]):
                value = plot.iloc[i, j]
                if not pd.isna(value):
                    ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=8)
        ax.set_title("Top models by input strategy")
        fig.colorbar(image, ax=ax, label="Mean NRS")
        figures["top_models_strategy_heatmap"] = _save(fig, top_dir, "top_models_strategy_heatmap", options.figure_format)
    return figures


def _plot_coverage_figures(df: pd.DataFrame, tables: dict[str, pd.DataFrame], figures_dir: Path, options) -> dict[str, Path]:
    figures: dict[str, Path] = {}
    if "source_coverage_075" not in df.columns or df["source_coverage_075"].dropna().empty:
        return figures
    coverage_dir = figures_dir / "coverage"

    def bar_from_table(key: str, x_col: str, y_col: str, filename: str, title: str, ylabel: str):
        table = tables.get(key, pd.DataFrame()).copy()
        if table.empty or x_col not in table.columns or y_col not in table.columns:
            return
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        ax.bar(table[x_col].astype(str), table[y_col])
        ax.set_title(title)
        ax.set_xlabel(x_col.replace("_", " "))
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", labelrotation=30)
        _grid(ax)
        figures[filename] = _save(fig, coverage_dir, filename, options.figure_format)

    bar_from_table("coverage_by_input_strategy", "input_strategy", "mean_source_coverage_075", "coverage_by_input_strategy", "Coverage by input strategy", "Mean source coverage @0.75")
    bar_from_table("coverage_by_prompt_strategy", "prompt_strategy", "mean_source_coverage_075", "coverage_by_prompt_strategy", "Coverage by prompt strategy", "Mean source coverage @0.75")

    scatter = df[df["success"]].copy()
    if {"bertscore_f1", "source_coverage_075", "compression_ratio"}.issubset(scatter.columns):
        plot = scatter.dropna(subset=["bertscore_f1", "source_coverage_075"])
        if not plot.empty:
            fig, ax = plt.subplots(figsize=(7.0, 5.0))
            sizes = _bubble_sizes(plot.get("compression_ratio", pd.Series(np.ones(len(plot)))), min_size=35, max_size=240)
            ax.scatter(plot["bertscore_f1"], plot["source_coverage_075"], s=sizes, alpha=0.65)
            ax.axhline(0.50, color="red", linestyle="--", linewidth=1, alpha=0.7)
            ax.axvline(0.85, color="red", linestyle="--", linewidth=1, alpha=0.7)
            ax.set_title("Coverage vs BERTScore")
            ax.set_xlabel("BERTScore F1")
            ax.set_ylabel("Source coverage @0.75")
            _grid(ax)
            figures["coverage_vs_bertscore"] = _save(fig, coverage_dir, "coverage_vs_bertscore", options.figure_format)

    model_cov = tables.get("coverage_by_model", pd.DataFrame()).copy()
    if not model_cov.empty and "mean_omission_risk_075" in model_cov.columns:
        plot = pd.concat([model_cov.nlargest(8, "mean_omission_risk_075"), model_cov.nsmallest(8, "mean_omission_risk_075")]).drop_duplicates("model")
        fig, ax = plt.subplots(figsize=(8.0, 5.0))
        ax.barh(plot["model"].astype(str), plot["mean_omission_risk_075"])
        ax.set_title("Omission risk by model")
        ax.set_xlabel("Mean omission risk @0.75")
        _grid(ax)
        figures["omission_risk_by_model"] = _save(fig, coverage_dir, "omission_risk_by_model", options.figure_format)

    case_cov = tables.get("coverage_by_case", pd.DataFrame()).copy()
    if not case_cov.empty and "mean_source_coverage_075" in case_cov.columns:
        x_col = "case_id" if "case_id" in case_cov.columns else case_cov.columns[0]
        plot = case_cov.sort_values("mean_source_coverage_075", ascending=False).head(30)
        fig, ax = plt.subplots(figsize=(9.0, 5.2))
        ax.bar(plot[x_col].astype(str), plot["mean_source_coverage_075"])
        ax.set_title("Coverage by case")
        ax.set_ylabel("Mean source coverage @0.75")
        ax.tick_params(axis="x", labelrotation=70)
        _grid(ax)
        figures["coverage_by_case"] = _save(fig, coverage_dir, "coverage_by_case", options.figure_format)

    config_cov = tables.get("coverage_by_model_strategy_prompt", pd.DataFrame()).copy()
    if not config_cov.empty and "mean_coverage_adjusted_bertscore_075" in config_cov.columns:
        plot = config_cov.sort_values("mean_coverage_adjusted_bertscore_075", ascending=False).head(15).copy()
        plot["label"] = plot.apply(lambda r: f"{r['model']}\n{r['input_strategy']}/{r['prompt_strategy']}", axis=1)
        fig, ax = plt.subplots(figsize=(9.0, 5.2))
        ax.bar(plot["label"], plot["mean_coverage_adjusted_bertscore_075"])
        ax.set_title("Top coverage-adjusted BERTScore configurations")
        ax.set_ylabel("Coverage-adjusted BERTScore @0.75")
        ax.tick_params(axis="x", labelrotation=45)
        _grid(ax)
        figures["coverage_adjusted_score_by_configuration"] = _save(fig, coverage_dir, "coverage_adjusted_score_by_configuration", options.figure_format)

    if {"source_coverage_075", "generation_support_075"}.issubset(scatter.columns):
        plot = scatter.dropna(subset=["source_coverage_075", "generation_support_075"])
        if not plot.empty:
            fig, ax = plt.subplots(figsize=(6.8, 5.2))
            ax.scatter(plot["source_coverage_075"], plot["generation_support_075"], alpha=0.65)
            ax.axhline(0.75, color="gray", linestyle="--", linewidth=1)
            ax.axvline(0.75, color="gray", linestyle="--", linewidth=1)
            ax.set_title("Generation support vs source coverage")
            ax.set_xlabel("Source coverage @0.75")
            ax.set_ylabel("Generation support @0.75")
            _grid(ax)
            figures["support_vs_coverage"] = _save(fig, coverage_dir, "support_vs_coverage", options.figure_format)
    return figures


def _plot_case_study_figures(tables: dict[str, pd.DataFrame], figures_dir: Path, options) -> dict[str, Path]:
    figures: dict[str, Path] = {}
    summary = tables.get("case_study_summary", pd.DataFrame()).copy()
    if summary.empty:
        return figures
    case_dir = figures_dir / "case_studies"

    def bar(table: pd.DataFrame, x: str, y: str, key: str, title: str, ylabel: str, ascending: bool = False):
        if table.empty or x not in table.columns or y not in table.columns:
            return
        plot = table.dropna(subset=[y]).sort_values(y, ascending=ascending)
        fig, ax = plt.subplots(figsize=(max(7.5, 0.32 * len(plot)), 5.0))
        ax.bar(plot[x].astype(str), plot[y])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", labelrotation=70)
        _grid(ax)
        figures[key] = _save(fig, case_dir, key, options.figure_format)

    bar(summary, "case_study", "mean_NRS", "case_mean_nrs", "Case-study mean NRS", "Mean NRS", ascending=False)
    bar(summary, "case_study", "failure_rate", "case_failure_rate", "Case-study failure rate", "Failure rate", ascending=False)
    bar(summary, "case_study", "mean_runtime_seconds", "case_runtime", "Case-study runtime", "Mean runtime (seconds)", ascending=False)

    difficulty = tables.get("case_difficulty_ranking", pd.DataFrame()).copy()
    if not difficulty.empty:
        bar(difficulty, "case_study", "difficulty_score", "case_difficulty_ranking", "Case-study difficulty ranking", "Difficulty score", ascending=False)

    def heatmap(table_key: str, key: str, title: str):
        table = tables.get(table_key, pd.DataFrame()).copy()
        if table.empty or "case_study" not in table.columns:
            return
        plot = table.set_index("case_study")
        fig, ax = plt.subplots(figsize=(7.5, max(4.8, 0.34 * len(plot))))
        image = ax.imshow(np.ma.masked_invalid(plot.to_numpy(dtype=float)), aspect="auto")
        ax.set_xticks(np.arange(len(plot.columns)))
        ax.set_xticklabels(plot.columns)
        ax.set_yticks(np.arange(len(plot.index)))
        ax.set_yticklabels(plot.index)
        ax.set_title(title)
        fig.colorbar(image, ax=ax, label="Mean NRS")
        figures[key] = _save(fig, case_dir, key, options.figure_format)

    heatmap("case_by_input_strategy_nrs", "case_input_strategy_heatmap", "Case x input strategy NRS")
    heatmap("case_by_prompt_strategy_nrs", "case_prompt_strategy_heatmap", "Case x prompt strategy NRS")
    bar(tables.get("case_input_strategy_delta", pd.DataFrame()), "case_study", "input_strategy_NRS_range", "case_input_strategy_gain", "Input-strategy NRS range by case", "NRS range", ascending=False)
    bar(tables.get("case_prompt_strategy_delta", pd.DataFrame()), "case_study", "prompt_strategy_NRS_range", "case_prompt_strategy_gain", "Prompt-strategy NRS range by case", "NRS range", ascending=False)

    loss = tables.get("case_model_size_loss", pd.DataFrame()).copy()
    if not loss.empty and {"case_study", "threshold", "NRS_loss_vs_case_best"}.issubset(loss.columns):
        pivot = loss.pivot_table(index="case_study", columns="threshold", values="NRS_loss_vs_case_best", aggfunc="mean", observed=False)
        fig, ax = plt.subplots(figsize=(max(8.0, 0.38 * len(pivot)), 5.2))
        pivot.plot(kind="bar", ax=ax)
        ax.set_title("Case quality loss under model-size thresholds")
        ax.set_ylabel("NRS loss vs case best")
        ax.tick_params(axis="x", labelrotation=70)
        _grid(ax)
        figures["case_size_threshold_loss"] = _save(fig, case_dir, "case_size_threshold_loss", options.figure_format)

    if {"mean_runtime_seconds", "mean_NRS", "failure_rate", "case_study"}.issubset(summary.columns):
        plot = summary.dropna(subset=["mean_runtime_seconds", "mean_NRS"])
        fig, ax = plt.subplots(figsize=(7.2, 5.2))
        sizes = _bubble_sizes(plot["failure_rate"], min_size=40, max_size=260)
        ax.scatter(plot["mean_runtime_seconds"], plot["mean_NRS"], s=sizes, alpha=0.7)
        for _, row in plot.iterrows():
            ax.annotate(str(row["case_study"]), (row["mean_runtime_seconds"], row["mean_NRS"]), fontsize=7, xytext=(4, 4), textcoords="offset points")
        ax.set_title("Case NRS vs runtime")
        ax.set_xlabel("Mean runtime (seconds)")
        ax.set_ylabel("Mean NRS")
        _grid(ax)
        figures["case_nrs_runtime_scatter"] = _save(fig, case_dir, "case_nrs_runtime_scatter", options.figure_format)

    stability = tables.get("case_stability_summary", pd.DataFrame()).copy()
    if not stability.empty and {"std_NRS", "mean_NRS", "failure_rate", "case_study"}.issubset(stability.columns):
        plot = stability.dropna(subset=["std_NRS", "mean_NRS"])
        fig, ax = plt.subplots(figsize=(7.2, 5.2))
        sizes = _bubble_sizes(plot["failure_rate"], min_size=40, max_size=260)
        ax.scatter(plot["std_NRS"], plot["mean_NRS"], s=sizes, alpha=0.7)
        for _, row in plot.iterrows():
            ax.annotate(str(row["case_study"]), (row["std_NRS"], row["mean_NRS"]), fontsize=7, xytext=(4, 4), textcoords="offset points")
        ax.set_title("Case stability: NRS variance vs quality")
        ax.set_xlabel("NRS standard deviation")
        ax.set_ylabel("Mean NRS")
        _grid(ax)
        figures["case_stability"] = _save(fig, case_dir, "case_stability", options.figure_format)

    if getattr(options, "case_detail_plots", False):
        _plot_per_case_details(tables, case_dir, options, figures)
    return figures


def _plot_per_case_details(tables: dict[str, pd.DataFrame], case_dir: Path, options, figures: dict[str, Path]) -> None:
    case_summary = tables.get("case_study_summary", pd.DataFrame())
    if case_summary.empty:
        return
    input_table = tables.get("case_by_input_strategy_nrs", pd.DataFrame())
    prompt_table = tables.get("case_by_prompt_strategy_nrs", pd.DataFrame())
    top5 = tables.get("top5_configurations_by_case", pd.DataFrame())
    config = tables.get("configuration_summary_with_efficiency", pd.DataFrame())
    for case in case_summary["case_study"].astype(str):
        outdir = case_dir / "per_case" / _clean_filename(case)
        row = input_table[input_table["case_study"].astype(str) == case] if not input_table.empty else pd.DataFrame()
        if not row.empty:
            plot = row.drop(columns=["case_study"]).T.reset_index()
            plot.columns = ["input_strategy", "mean_NRS"]
            fig, ax = plt.subplots(figsize=(5.8, 4.0))
            ax.bar(plot["input_strategy"], plot["mean_NRS"])
            ax.set_title(f"{case}: NRS by input")
            _grid(ax)
            figures[f"case_{_clean_filename(case)}_input_strategy_nrs"] = _save(fig, outdir, "input_strategy_nrs", options.figure_format)
        row = prompt_table[prompt_table["case_study"].astype(str) == case] if not prompt_table.empty else pd.DataFrame()
        if not row.empty:
            plot = row.drop(columns=["case_study"]).T.reset_index()
            plot.columns = ["prompt_strategy", "mean_NRS"]
            fig, ax = plt.subplots(figsize=(5.8, 4.0))
            ax.bar(plot["prompt_strategy"], plot["mean_NRS"])
            ax.set_title(f"{case}: NRS by prompt")
            _grid(ax)
            figures[f"case_{_clean_filename(case)}_prompt_strategy_nrs"] = _save(fig, outdir, "prompt_strategy_nrs", options.figure_format)
        subset = top5[top5["case_study"].astype(str) == case] if not top5.empty else pd.DataFrame()
        if not subset.empty:
            fig, ax = plt.subplots(figsize=(6.5, 4.2))
            ax.bar(subset["model"].astype(str), subset["mean_NRS"])
            ax.set_title(f"{case}: top models/configurations")
            ax.tick_params(axis="x", labelrotation=45)
            _grid(ax)
            figures[f"case_{_clean_filename(case)}_top_models_nrs"] = _save(fig, outdir, "top_models_nrs", options.figure_format)
        subset = config[config.get("case_study", pd.Series(dtype=str)).astype(str) == case] if "case_study" in config.columns else pd.DataFrame()
        if subset.empty:
            subset = top5[top5["case_study"].astype(str) == case] if not top5.empty else pd.DataFrame()
        if not subset.empty:
            fig, ax = plt.subplots(figsize=(6.5, 4.2))
            runtime_col = "mean_runtime_seconds" if "mean_runtime_seconds" in subset.columns else "runtime_seconds"
            ax.scatter(subset[runtime_col], subset["mean_NRS"], alpha=0.65)
            ax.set_title(f"{case}: NRS vs runtime")
            ax.set_xlabel("Runtime (seconds)")
            ax.set_ylabel("Mean NRS")
            _grid(ax)
            figures[f"case_{_clean_filename(case)}_nrs_runtime_by_configuration"] = _save(fig, outdir, "nrs_runtime_by_configuration", options.figure_format)


def create_all_figures(df: pd.DataFrame, tables: dict[str, pd.DataFrame], figures_dir: Path, options) -> tuple[dict[str, Path], list[dict[str, Any]]]:
    _setup(options.dpi)
    figures: dict[str, Path] = {}

    input_order = ordered_values(df["input_strategy"], options.input_order)
    prompt_order = ordered_values(df["prompt_strategy"], options.prompt_order)
    input_summary = tables["input_strategy_summary"].copy()
    prompt_summary = tables["prompt_strategy_summary"].copy()
    input_prompt_summary = tables["input_prompt_summary"].copy()
    config = tables["configuration_summary_with_efficiency"].copy()
    pareto = tables["pareto_optimal_configurations"].copy()
    quality_loss = tables["quality_loss_by_parameter_threshold"].copy()

    # 1. Mean NRS by input strategy.
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    plot = input_summary.set_index("input_strategy").reindex(input_order).reset_index()
    ax.bar(plot["input_strategy"].astype(str), plot["mean_NRS"], yerr=plot["ci95_NRS"], capsize=4)
    ax.set_title("Mean NRS by input strategy")
    ax.set_xlabel("Input strategy")
    ax.set_ylabel("Mean NRS (all attempts)")
    _grid(ax)
    _safe_ylim_for_means(ax, plot["mean_NRS"])
    figures["01_mean_nrs_by_input_strategy"] = _save(fig, figures_dir, "01_mean_nrs_by_input_strategy", options.figure_format)

    # 2. Mean NRS by prompt strategy.
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    plot = prompt_summary.set_index("prompt_strategy").reindex(prompt_order).reset_index()
    ax.bar(plot["prompt_strategy"].astype(str), plot["mean_NRS"], yerr=plot["ci95_NRS"], capsize=4)
    ax.set_title("Mean NRS by prompt strategy")
    ax.set_xlabel("Prompt strategy")
    ax.set_ylabel("Mean NRS (all attempts)")
    _grid(ax)
    _safe_ylim_for_means(ax, plot["mean_NRS"])
    figures["02_mean_nrs_by_prompt_strategy"] = _save(fig, figures_dir, "02_mean_nrs_by_prompt_strategy", options.figure_format)

    # 3. Input x prompt heatmap.
    heat = input_prompt_summary.pivot_table(index="input_strategy", columns="prompt_strategy", values="mean_NRS", observed=False)
    heat = heat.reindex(index=input_order, columns=prompt_order)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    masked = np.ma.masked_invalid(heat.to_numpy(dtype=float))
    image = ax.imshow(masked, aspect="auto")
    ax.set_xticks(np.arange(len(prompt_order)))
    ax.set_xticklabels(prompt_order)
    ax.set_yticks(np.arange(len(input_order)))
    ax.set_yticklabels(input_order)
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            value = heat.iloc[i, j]
            if not pd.isna(value):
                ax.text(j, i, f"{value:.1f}", ha="center", va="center")
    ax.set_title("Input strategy x prompt strategy: mean NRS")
    ax.set_xlabel("Prompt strategy")
    ax.set_ylabel("Input strategy")
    fig.colorbar(image, ax=ax, label="Mean NRS")
    figures["03_input_prompt_nrs_heatmap"] = _save(fig, figures_dir, "03_input_prompt_nrs_heatmap", options.figure_format)

    # 4. Mean runtime by input strategy.
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    plot = input_summary.set_index("input_strategy").reindex(input_order).reset_index()
    runtime_se = plot["sd_runtime_seconds"] / np.sqrt(plot["attempts"].replace(0, np.nan))
    ax.bar(plot["input_strategy"].astype(str), plot["mean_runtime_seconds"], yerr=runtime_se, capsize=4)
    ax.set_title("Mean runtime by input strategy")
    ax.set_xlabel("Input strategy")
    ax.set_ylabel("Mean runtime (seconds; all attempts)")
    _grid(ax)
    figures["04_mean_runtime_by_input_strategy"] = _save(fig, figures_dir, "04_mean_runtime_by_input_strategy", options.figure_format)

    # Shared filtered config for parameter/runtime scatterplots.
    scatter = config.dropna(subset=["parameters_b", "mean_runtime_seconds", "mean_NRS"]).copy()
    scatter = scatter[(scatter["parameters_b"] > 0) & (scatter["mean_runtime_seconds"] > 0)]

    if scatter.empty:
        figures["05_runtime_vs_parameters_by_input"] = _placeholder(figures_dir, "05_runtime_vs_parameters_by_input", "Runtime vs parameters", "No usable parameter/runtime values were available.", options)
    else:
        fig, ax = plt.subplots(figsize=(7.2, 5.0))
        for input_strategy in input_order:
            subset = scatter[scatter["input_strategy"] == input_strategy]
            if subset.empty:
                continue
            ax.scatter(subset["parameters_b"], subset["mean_runtime_seconds"], alpha=0.65, label=input_strategy)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title("Runtime vs parameters by input strategy")
        ax.set_xlabel("Parameters (billions, log scale)")
        ax.set_ylabel("Mean runtime (seconds, log scale)")
        _grid(ax, axis="both")
        ax.legend(title="Input", fontsize=8)
        figures["05_runtime_vs_parameters_by_input"] = _save(fig, figures_dir, "05_runtime_vs_parameters_by_input", options.figure_format)

    if scatter.empty:
        figures["06_runtime_vs_parameters_by_input_prompt"] = _placeholder(figures_dir, "06_runtime_vs_parameters_by_input_prompt", "Runtime vs parameters by input and prompt", "No usable parameter/runtime values were available.", options)
    else:
        n = max(1, len(input_order))
        cols = 2 if n > 1 else 1
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(10, max(4.5, 3.4 * rows)), squeeze=False, sharex=True, sharey=True)
        for ax in axes.flatten()[n:]:
            ax.axis("off")
        for ax, input_strategy in zip(axes.flatten(), input_order):
            for prompt_strategy in prompt_order:
                subset = scatter[(scatter["input_strategy"] == input_strategy) & (scatter["prompt_strategy"] == prompt_strategy)]
                if subset.empty:
                    continue
                ax.scatter(subset["parameters_b"], subset["mean_runtime_seconds"], alpha=0.65, label=prompt_strategy)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(str(input_strategy))
            ax.set_xlabel("Parameters (B, log)")
            ax.set_ylabel("Runtime (s, log)")
            _grid(ax, axis="both")
        axes.flatten()[0].legend(title="Prompt", fontsize=8)
        fig.suptitle("Runtime vs parameters by input strategy and prompt strategy")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        path = figures_dir / f"06_runtime_vs_parameters_by_input_prompt.{options.figure_format}"
        fig.savefig(path)
        plt.close(fig)
        figures["06_runtime_vs_parameters_by_input_prompt"] = path

    if scatter.empty:
        figures["07_nrs_vs_runtime_bubble_parameters"] = _placeholder(figures_dir, "07_nrs_vs_runtime_bubble_parameters", "NRS vs runtime", "No usable runtime values were available.", options)
    else:
        fig, ax = plt.subplots(figsize=(7.4, 5.0))
        for input_strategy in input_order:
            subset = scatter[scatter["input_strategy"] == input_strategy]
            if subset.empty:
                continue
            sizes = _bubble_sizes(subset["parameters_b"])
            ax.scatter(subset["mean_runtime_seconds"], subset["mean_NRS"], s=sizes, alpha=0.58, label=input_strategy)
        ax.set_xscale("log")
        ax.set_title("NRS vs runtime (bubble size = parameters)")
        ax.set_xlabel("Mean runtime (seconds, log scale)")
        ax.set_ylabel("Mean NRS")
        _grid(ax)
        ax.legend(title="Input", fontsize=8)
        figures["07_nrs_vs_runtime_bubble_parameters"] = _save(fig, figures_dir, "07_nrs_vs_runtime_bubble_parameters", options.figure_format)

    if scatter.empty:
        figures["08_nrs_vs_parameters_bubble_runtime"] = _placeholder(figures_dir, "08_nrs_vs_parameters_bubble_runtime", "NRS vs parameters", "No usable parameter values were available.", options)
    else:
        fig, ax = plt.subplots(figsize=(7.4, 5.0))
        for input_strategy in input_order:
            subset = scatter[scatter["input_strategy"] == input_strategy]
            if subset.empty:
                continue
            sizes = _bubble_sizes(np.log1p(subset["mean_runtime_seconds"]))
            ax.scatter(subset["parameters_b"], subset["mean_NRS"], s=sizes, alpha=0.58, label=input_strategy)
        ax.set_xscale("log")
        ax.set_title("NRS vs parameters (bubble size = runtime)")
        ax.set_xlabel("Parameters (billions, log scale)")
        ax.set_ylabel("Mean NRS")
        _grid(ax)
        ax.legend(title="Input", fontsize=8)
        figures["08_nrs_vs_parameters_bubble_runtime"] = _save(fig, figures_dir, "08_nrs_vs_parameters_bubble_runtime", options.figure_format)

    # 9. Top balanced configurations.
    balanced = config.dropna(subset=["balanced_score"]).sort_values("balanced_score", ascending=False).head(12).copy()
    if balanced.empty:
        figures["09_top_balanced_configurations"] = _placeholder(figures_dir, "09_top_balanced_configurations", "Top balanced configurations", "Balanced scores could not be computed.", options)
    else:
        balanced["label"] = balanced.apply(
            lambda row: f"{row['model']}\n{row['input_strategy']}/{row['prompt_strategy']}", axis=1
        )
        fig, ax = plt.subplots(figsize=(8.5, 5.8))
        ax.barh(np.arange(len(balanced)), balanced["balanced_score"])
        ax.set_yticks(np.arange(len(balanced)))
        ax.set_yticklabels(balanced["label"], fontsize=8)
        ax.invert_yaxis()
        ax.set_title("Top balanced configurations")
        ax.set_xlabel("Balanced score")
        _grid(ax)
        figures["09_top_balanced_configurations"] = _save(fig, figures_dir, "09_top_balanced_configurations", options.figure_format)

    # 10. Quality loss vs threshold.
    if quality_loss.empty or "NRS_loss_vs_global_best" not in quality_loss.columns:
        figures["10_quality_loss_vs_parameter_threshold"] = _placeholder(figures_dir, "10_quality_loss_vs_parameter_threshold", "Quality loss vs parameter threshold", "No quality-loss table could be computed.", options)
    else:
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        ax.plot(quality_loss["threshold"], quality_loss["NRS_loss_vs_global_best"], marker="o", label="Best within threshold")
        if "reliable_NRS_loss_vs_global_best" in quality_loss.columns:
            ax.plot(quality_loss["threshold"], quality_loss["reliable_NRS_loss_vs_global_best"], marker="o", label=f"Best with failure <= {options.reliable_failure_rate_max:.0%}")
        ax.set_title("Quality loss vs parameter threshold")
        ax.set_xlabel("Model-size threshold")
        ax.set_ylabel("NRS loss vs global best")
        _grid(ax)
        ax.legend(fontsize=8)
        figures["10_quality_loss_vs_parameter_threshold"] = _save(fig, figures_dir, "10_quality_loss_vs_parameter_threshold", options.figure_format)

    # 11. Pareto frontier.
    if scatter.empty or pareto.empty:
        figures["11_pareto_frontier"] = _placeholder(figures_dir, "11_pareto_frontier", "Pareto frontier", "No Pareto frontier could be computed.", options)
    else:
        fig, ax = plt.subplots(figsize=(7.4, 5.0))
        all_sizes = _bubble_sizes(scatter["parameters_b"], min_size=18, max_size=140)
        ax.scatter(scatter["mean_runtime_seconds"], scatter["mean_NRS"], s=all_sizes, alpha=0.22, label="Other configurations")
        pareto_plot = pareto.dropna(subset=["parameters_b", "mean_runtime_seconds", "mean_NRS"])
        pareto_sizes = _bubble_sizes(pareto_plot["parameters_b"], min_size=45, max_size=210)
        ax.scatter(pareto_plot["mean_runtime_seconds"], pareto_plot["mean_NRS"], s=pareto_sizes, alpha=0.85, label="Pareto-optimal")
        for _, row in pareto_plot.sort_values("mean_NRS", ascending=False).head(10).iterrows():
            label = str(row["model"]).split(":")[0]
            ax.annotate(label, (row["mean_runtime_seconds"], row["mean_NRS"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.set_xscale("log")
        ax.set_title("Pareto frontier: NRS vs runtime")
        ax.set_xlabel("Mean runtime (seconds, log scale)")
        ax.set_ylabel("Mean NRS")
        _grid(ax)
        ax.legend(fontsize=8)
        figures["11_pareto_frontier"] = _save(fig, figures_dir, "11_pareto_frontier", options.figure_format)

    figures.update(_plot_family_figures(tables, figures_dir, options))
    figures.update(_plot_top5_figures(tables, figures_dir, options))
    figures.update(_plot_top_model_figures(tables, figures_dir, options))
    figures.update(_plot_coverage_figures(df, tables, figures_dir, options))
    figures.update(_plot_case_study_figures(tables, figures_dir, options))

    interpretations = _figure_interpretations(tables)
    return figures, interpretations


def _top_value(table: pd.DataFrame, key: str, value: str) -> tuple[str, float] | tuple[str, None]:
    if table.empty or key not in table.columns or value not in table.columns:
        return "NA", None
    row = table.sort_values(value, ascending=False).iloc[0]
    return str(row[key]), float(row[value]) if not pd.isna(row[value]) else None


def _figure_interpretations(tables: dict[str, pd.DataFrame]) -> list[dict[str, str]]:
    input_name, input_nrs = _top_value(tables.get("input_strategy_summary", pd.DataFrame()), "input_strategy", "mean_NRS")
    prompt_name, prompt_nrs = _top_value(tables.get("prompt_strategy_summary", pd.DataFrame()), "prompt_strategy", "mean_NRS")
    config_table = tables.get("configuration_summary_with_efficiency", pd.DataFrame())
    pareto_table = tables.get("pareto_optimal_configurations", pd.DataFrame())
    best_balanced = "NA"
    if not config_table.empty and "balanced_score" in config_table.columns:
        row = config_table.sort_values("balanced_score", ascending=False).iloc[0]
        best_balanced = f"{row['model']} + {row['input_strategy']} + {row['prompt_strategy']}"

    return [
        {
            "figure": "01_mean_nrs_by_input_strategy",
            "title": "Mean NRS by input strategy",
            "interpretation": f"The figure compares expected NRS across input strategies. The highest observed aggregate input strategy is {input_name}"
            + (f" with mean NRS {input_nrs:.3f}." if input_nrs is not None else ".")
            + " This directly answers whether richer input context improves benchmark quality.",
        },
        {
            "figure": "02_mean_nrs_by_prompt_strategy",
            "title": "Mean NRS by prompt strategy",
            "interpretation": f"The figure compares prompt-level effects. The leading prompt strategy is {prompt_name}"
            + (f" with mean NRS {prompt_nrs:.3f}." if prompt_nrs is not None else ".")
            + " Prompt effects should be interpreted together with input strategy because the best prompt may change by input condition.",
        },
        {
            "figure": "03_input_prompt_nrs_heatmap",
            "title": "Input strategy x prompt strategy NRS heatmap",
            "interpretation": "The heatmap displays interaction effects between input strategy and prompt strategy. Cells with the largest values identify high-quality strategy pairs, while uneven row and column patterns indicate that prompt design does not affect all input conditions equally.",
        },
        {
            "figure": "04_mean_runtime_by_input_strategy",
            "title": "Mean runtime by input strategy",
            "interpretation": "The runtime chart shows the computational cost associated with each input strategy. It should be read as a comparative hardware-specific benchmark rather than an absolute prediction for commodity machines.",
        },
        {
            "figure": "05_runtime_vs_parameters_by_input",
            "title": "Runtime vs parameters by input strategy",
            "interpretation": "This plot shows whether runtime scales with model size and whether some input strategies shift the runtime curve upward. A log scale is used because model sizes and runtimes can differ by orders of magnitude.",
        },
        {
            "figure": "06_runtime_vs_parameters_by_input_prompt",
            "title": "Runtime vs parameters by input and prompt strategy",
            "interpretation": "The faceted runtime plot separates input and prompt effects. It helps identify whether runtime increases are driven primarily by model size, input length, prompt detail, or their combination.",
        },
        {
            "figure": "07_nrs_vs_runtime_bubble_parameters",
            "title": "NRS vs runtime, bubble size = parameters",
            "interpretation": "The tradeoff plot highlights configurations near the upper-left region: high quality and low runtime. Bubble size indicates whether the apparent efficiency requires a larger model.",
        },
        {
            "figure": "08_nrs_vs_parameters_bubble_runtime",
            "title": "NRS vs parameters, bubble size = runtime",
            "interpretation": "This figure shows whether quality improves proportionally with model size. Plateaus or small models near the top of the plot indicate that larger models are not always necessary for strong NRS.",
        },
        {
            "figure": "09_top_balanced_configurations",
            "title": "Top balanced configurations",
            "interpretation": f"The balanced ranking combines quality, speed, parameter efficiency, stability, and reliability. Under the current weights, the leading configuration is {best_balanced}. The weights are configurable and should be adjusted for deployment priorities.",
        },
        {
            "figure": "10_quality_loss_vs_parameter_threshold",
            "title": "Quality loss vs parameter threshold",
            "interpretation": "This plot quantifies the NRS cost of limiting deployment to smaller models. It shows the best available configuration under each model-size threshold and the quality loss relative to the global best configuration.",
        },
        {
            "figure": "11_pareto_frontier",
            "title": "Pareto frontier plot",
            "interpretation": f"The Pareto plot marks configurations that are not dominated on NRS, runtime, parameters, and failure rate. There are {len(pareto_table)} Pareto-optimal configurations. These are the most rational candidates for retesting and deployment evaluation.",
        },
    ]
