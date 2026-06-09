from __future__ import annotations

import math
from pathlib import Path

from .benchmark import compute_case_method_summary, compute_model_overall_summary, compute_model_prompt_summary, compute_prompt_strategy_summary, summarize_runs_csv
from .utils import read_csv, write_csv


def _safe_import_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        return None


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _as_float(value, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _markdown_table(rows: list[dict], columns: list[str], title: str) -> str:
    lines = [f"## {title}", ""]
    if not rows:
        return "\n".join(lines + ["No data."])
    header = " | ".join(columns)
    sep = " | ".join(["---"] * len(columns))
    lines.extend([header, sep])
    for row in rows[:50]:
        lines.append(" | ".join(str(row.get(c, "")) for c in columns))
    if len(rows) > 50:
        lines.append(f"_Showing first 50 of {len(rows)} rows._")
    return "\n".join(lines)


def _plot_mean_nrs(model_rows: list[dict], outpath: Path) -> Path:
    plt = _safe_import_matplotlib()
    if plt is None:
        svg_path = outpath.with_suffix(".svg")
        _write_svg_barh(model_rows, svg_path, value_key="mean_NRS", title="Mean NRS by method", xlabel="Mean NRS", max_value=100, color="#5b8ff9")
        return svg_path
    methods = [r["method"] for r in model_rows]
    scores = [(_as_float(r.get("mean_NRS"), math.nan) if r.get("mean_NRS") not in {None, ""} else math.nan) for r in model_rows]
    colors = ["#5b8ff9" if not math.isnan(v) else "#d9d9d9" for v in scores]
    fig, ax = plt.subplots(figsize=(10, max(4, 0.5 * len(methods))))
    ax.barh(methods, [0 if math.isnan(v) else v for v in scores], color=colors)
    ax.set_xlabel("Mean NRS")
    ax.set_title("Mean NRS by method")
    ax.set_xlim(0, 100)
    for idx, val in enumerate(scores):
        if not math.isnan(val):
            ax.text(val + 0.5, idx, f"{val:.1f}", va="center")
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    return outpath


def _plot_fcr(model_rows: list[dict], outpath: Path) -> Path:
    plt = _safe_import_matplotlib()
    if plt is None:
        svg_path = outpath.with_suffix(".svg")
        _write_svg_grouped_triple(model_rows, svg_path, title="Mean BERTScore, semantic similarity, and R by method")
        return svg_path
    methods = [r["method"] for r in model_rows]
    mean_f = [_as_float(r.get("mean_bertscore_f1")) for r in model_rows]
    mean_c = [_as_float(r.get("mean_semantic_similarity")) for r in model_rows]
    mean_r = [_as_float(r.get("mean_R"), 0.0) for r in model_rows]
    x = range(len(methods))
    fig, ax = plt.subplots(figsize=(12, max(4, 0.55 * len(methods))))
    ax.barh([i + 0.25 for i in x], mean_f, height=0.25, label="F", color="#5b8ff9")
    ax.barh(list(x), mean_c, height=0.25, label="C", color="#5ad8a6")
    ax.barh([i - 0.25 for i in x], mean_r, height=0.25, label="R", color="#f6bd16")
    ax.set_yticks(list(x))
    ax.set_yticklabels(methods)
    ax.set_xlim(0, 1)
    ax.set_title("Mean BERTScore, semantic similarity, and R by method")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    return outpath


def _plot_failed_rate(model_rows: list[dict], outpath: Path) -> Path:
    plt = _safe_import_matplotlib()
    if plt is None:
        svg_path = outpath.with_suffix(".svg")
        _write_svg_barh(model_rows, svg_path, value_key="failed_rate", title="Failure rate by method", xlabel="Failed rate (%)", max_value=100, color="#ff7875", scale=100)
        return svg_path
    methods = [r["method"] for r in model_rows]
    failed = [_as_float(r.get("failed_rate")) * 100 for r in model_rows]
    fig, ax = plt.subplots(figsize=(10, max(4, 0.5 * len(methods))))
    ax.barh(methods, failed, color="#ff7875")
    ax.set_xlabel("Failed rate (%)")
    ax.set_title("Failure rate by method")
    ax.set_xlim(0, 100)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    return outpath


def _plot_word_vs_nrs(run_rows: list[dict], outpath: Path) -> Path:
    plt = _safe_import_matplotlib()
    if plt is None:
        svg_path = outpath.with_suffix(".svg")
        _write_svg_scatter(run_rows, svg_path, title="NRS_no_R vs output length")
        return svg_path
    xs = [_as_float(r.get("word_count")) for r in run_rows]
    ys = [_as_float(r.get("NRS_no_R")) for r in run_rows]
    colors = ["#d62728" if str(r.get("failed", "")).lower() in {"true", "1", "yes"} else "#1f77b4" for r in run_rows]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(xs, ys, c=colors, alpha=0.75)
    ax.set_xlabel("Word count")
    ax.set_ylabel("NRS_no_R")
    ax.set_title("NRS_no_R vs output length")
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    return outpath


def _svg_escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _write_svg_barh(model_rows: list[dict], outpath: Path, *, value_key: str, title: str, xlabel: str, max_value: float, color: str, scale: float = 1.0, label_key: str = "method") -> None:
    width = 1000
    margin_left = 280
    margin_right = 60
    top = 50
    row_h = 42
    height = max(180, top + row_h * len(model_rows) + 40)
    scale_x = (width - margin_left - margin_right) / max_value
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">', f'<rect width="100%" height="100%" fill="white"/>', f'<text x="{margin_left}" y="28" font-size="20" font-family="Arial">{_svg_escape(title)}</text>']
    for i, row in enumerate(model_rows):
        y = top + i * row_h
        value = _as_float(row.get(value_key)) * scale
        bar_w = max(0, min(width - margin_left - margin_right, value * scale_x))
        parts.append(f'<text x="20" y="{y + 18}" font-size="14" font-family="Arial">{_svg_escape(row.get(label_key, ""))}</text>')
        parts.append(f'<rect x="{margin_left}" y="{y}" width="{bar_w}" height="24" fill="{color}" rx="4"/>')
        parts.append(f'<text x="{margin_left + bar_w + 8}" y="{y + 17}" font-size="12" font-family="Arial">{value:.1f}</text>')
    parts.append(f'<text x="{margin_left}" y="{height - 10}" font-size="12" font-family="Arial">{_svg_escape(xlabel)}</text>')
    parts.append("</svg>")
    outpath.write_text("\n".join(parts), encoding="utf-8")


def _write_svg_grouped_triple(model_rows: list[dict], outpath: Path, *, title: str) -> None:
    width = 1000
    margin_left = 280
    margin_right = 60
    top = 50
    row_h = 46
    height = max(180, top + row_h * len(model_rows) + 40)
    scale_x = (width - margin_left - margin_right)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">', f'<rect width="100%" height="100%" fill="white"/>', f'<text x="{margin_left}" y="28" font-size="20" font-family="Arial">{_svg_escape(title)}</text>']
    colors = [("BERTScore", "#5b8ff9", "mean_bertscore_f1"), ("Semantic", "#5ad8a6", "mean_semantic_similarity"), ("R", "#f6bd16", "mean_R")]
    for i, row in enumerate(model_rows):
        y = top + i * row_h
        parts.append(f'<text x="20" y="{y + 18}" font-size="14" font-family="Arial">{_svg_escape(row.get("method", ""))}</text>')
        for j, (label, color, field) in enumerate(colors):
            value = _as_float(row.get(field))
            x = margin_left + j * 220
            parts.append(f'<rect x="{x}" y="{y}" width="{value * scale_x / 3:.0f}" height="10" fill="{color}" rx="2"/>')
            parts.append(f'<text x="{x}" y="{y + 24}" font-size="11" font-family="Arial">{label}: {value:.2f}</text>')
    parts.append("</svg>")
    outpath.write_text("\n".join(parts), encoding="utf-8")


def _write_svg_heatmap(rows: list[dict], outpath: Path, *, x_key: str, y_key: str, value_key: str, title: str, x_label: str, y_label: str) -> Path:
    xs = sorted({str(row.get(x_key, "")) for row in rows if str(row.get(x_key, "")).strip()})
    ys = sorted({str(row.get(y_key, "")) for row in rows if str(row.get(y_key, "")).strip()})
    if not xs or not ys:
        outpath.write_text("", encoding="utf-8")
        return outpath
    matrix = [[None for _ in xs] for _ in ys]
    for row in rows:
        x = str(row.get(x_key, "")).strip()
        y = str(row.get(y_key, "")).strip()
        if not x or not y:
            continue
        if x not in xs or y not in ys:
            continue
        value = row.get(value_key)
        if value in {None, ""}:
            continue
        matrix[ys.index(y)][xs.index(x)] = _as_float(value, 0.0)

    plt = _safe_import_matplotlib()
    if plt is not None:
        import numpy as np

        fig, ax = plt.subplots(figsize=(max(8, 0.8 * len(xs) + 4), max(4, 0.5 * len(ys) + 3)))
        data = np.array([[0.0 if cell is None else float(cell) for cell in row] for row in matrix], dtype=float)
        im = ax.imshow(data, aspect="auto", cmap="viridis", vmin=0, vmax=100)
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels(xs, rotation=45, ha="right")
        ax.set_yticks(range(len(ys)))
        ax.set_yticklabels(ys)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        for i, y in enumerate(ys):
            for j, x in enumerate(xs):
                cell = matrix[i][j]
                if cell is None:
                    continue
                ax.text(j, i, f"{cell:.1f}", ha="center", va="center", color="white" if cell > 50 else "black", fontsize=9)
        fig.colorbar(im, ax=ax, label=value_key)
        fig.tight_layout()
        fig.savefig(outpath, dpi=200)
        plt.close(fig)
        return outpath

    cell_w = 120
    cell_h = 34
    left = 220
    top = 80
    width = left + cell_w * len(xs) + 40
    height = top + cell_h * len(ys) + 70
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">', '<rect width="100%" height="100%" fill="white"/>', f'<text x="{left}" y="28" font-size="20" font-family="Arial">{_svg_escape(title)}</text>']
    parts.append(f'<text x="20" y="{top - 18}" font-size="12" font-family="Arial">{_svg_escape(y_label)}</text>')
    for i, y in enumerate(ys):
        yy = top + i * cell_h
        parts.append(f'<text x="20" y="{yy + 20}" font-size="12" font-family="Arial">{_svg_escape(y)}</text>')
        for j, x in enumerate(xs):
            xx = left + j * cell_w
            value = matrix[i][j]
            if value is None:
                fill = "#f0f0f0"
                label = ""
            else:
                alpha = max(0.15, min(1.0, value / 100.0))
                fill = f"rgba(91,143,249,{alpha:.2f})"
                label = f"{value:.1f}"
            parts.append(f'<rect x="{xx}" y="{yy}" width="{cell_w - 2}" height="{cell_h - 2}" fill="{fill}" stroke="#ddd"/>')
            if label:
                parts.append(f'<text x="{xx + cell_w/2}" y="{yy + 21}" text-anchor="middle" font-size="11" font-family="Arial">{label}</text>')
    for j, x in enumerate(xs):
        xx = left + j * cell_w + cell_w / 2
        parts.append(f'<text x="{xx}" y="{top + cell_h * len(ys) + 24}" text-anchor="middle" font-size="11" font-family="Arial">{_svg_escape(x)}</text>')
    parts.append(f'<text x="{left}" y="{height - 10}" font-size="12" font-family="Arial">{_svg_escape(x_label)}</text>')
    parts.append("</svg>")
    svg_path = outpath.with_suffix(".svg") if outpath.suffix.lower() != ".svg" else outpath
    svg_path.write_text("\n".join(parts), encoding="utf-8")
    return svg_path


def _prompt_score(row: dict) -> float:
    value = row.get("rank_score")
    if value in {None, ""}:
        value = row.get("mean_CSV_NRS") if row.get("source_type") == "csv" else row.get("mean_NRS")
    return _as_float(value, 0.0)


def _plot_prompt_strategy_mean(rows: list[dict], outpath: Path) -> Path:
    plt = _safe_import_matplotlib()
    if plt is None:
        svg_path = outpath.with_suffix(".svg")
        _write_svg_barh(rows, svg_path, value_key="rank_score", title="Mean score by prompt strategy", xlabel="Mean score", max_value=100, color="#5b8ff9", label_key="prompt_strategy")
        return svg_path
    labels = [f"{r.get('source_type', '')}:{r.get('prompt_strategy', '')}" for r in rows]
    scores = [_prompt_score(r) for r in rows]
    fig, ax = plt.subplots(figsize=(10, max(4, 0.5 * len(labels))))
    ax.barh(labels, scores, color="#5b8ff9")
    ax.set_xlabel("Mean score")
    ax.set_title("Mean score by prompt strategy")
    ax.set_xlim(0, 100)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    return outpath


def _plot_prompt_strategy_failed_rate(rows: list[dict], outpath: Path) -> Path:
    plt = _safe_import_matplotlib()
    if plt is None:
        svg_path = outpath.with_suffix(".svg")
        _write_svg_barh(rows, svg_path, value_key="failed_rate", title="Failure rate by prompt strategy", xlabel="Failed rate (%)", max_value=100, color="#ff7875", scale=100, label_key="prompt_strategy")
        return svg_path
    labels = [f"{r.get('source_type', '')}:{r.get('prompt_strategy', '')}" for r in rows]
    failed = [_as_float(r.get("failed_rate")) * 100 for r in rows]
    fig, ax = plt.subplots(figsize=(10, max(4, 0.5 * len(labels))))
    ax.barh(labels, failed, color="#ff7875")
    ax.set_xlabel("Failed rate (%)")
    ax.set_title("Failure rate by prompt strategy")
    ax.set_xlim(0, 100)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    return outpath


def _plot_prompt_strategy_runtime(rows: list[dict], outpath: Path) -> Path:
    plt = _safe_import_matplotlib()
    if plt is None:
        svg_path = outpath.with_suffix(".svg")
        _write_svg_barh(rows, svg_path, value_key="mean_runtime_seconds", title="Runtime by prompt strategy", xlabel="Runtime (s)", max_value=max(1.0, max((_as_float(r.get('mean_runtime_seconds')) for r in rows), default=1.0)), color="#5ad8a6", label_key="prompt_strategy")
        return svg_path
    labels = [f"{r.get('source_type', '')}:{r.get('prompt_strategy', '')}" for r in rows]
    runtime = [_as_float(r.get("mean_runtime_seconds")) for r in rows]
    fig, ax = plt.subplots(figsize=(10, max(4, 0.5 * len(labels))))
    ax.barh(labels, runtime, color="#5ad8a6")
    ax.set_xlabel("Runtime (s)")
    ax.set_title("Runtime by prompt strategy")
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    return outpath


def _plot_prompt_strategy_word_count(rows: list[dict], outpath: Path) -> Path:
    plt = _safe_import_matplotlib()
    if plt is None:
        svg_path = outpath.with_suffix(".svg")
        _write_svg_barh(rows, svg_path, value_key="mean_word_count", title="Word count by prompt strategy", xlabel="Mean word count", max_value=max(1.0, max((_as_float(r.get('mean_word_count')) for r in rows), default=1.0)), color="#f6bd16", label_key="prompt_strategy")
        return svg_path
    labels = [f"{r.get('source_type', '')}:{r.get('prompt_strategy', '')}" for r in rows]
    wc = [_as_float(r.get("mean_word_count")) for r in rows]
    fig, ax = plt.subplots(figsize=(10, max(4, 0.5 * len(labels))))
    ax.barh(labels, wc, color="#f6bd16")
    ax.set_xlabel("Mean word count")
    ax.set_title("Word count by prompt strategy")
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    return outpath


def _plot_model_prompt_heatmap(rows: list[dict], outpath: Path) -> Path:
    return _write_svg_heatmap(rows, outpath, x_key="prompt_strategy", y_key="model", value_key="rank_score", title="Model x prompt strategy heatmap", x_label="Prompt strategy", y_label="Model")


def _plot_input_prompt_heatmap(rows: list[dict], outpath: Path) -> Path:
    return _write_svg_heatmap(rows, outpath, x_key="prompt_strategy", y_key="input_strategy", value_key="rank_score", title="Input strategy x prompt strategy heatmap", x_label="Prompt strategy", y_label="Input strategy")


def _write_svg_scatter(run_rows: list[dict], outpath: Path, *, title: str) -> None:
    width = 1000
    height = 700
    margin_left = 80
    margin_right = 40
    margin_bottom = 70
    margin_top = 50
    xs = [_as_float(r.get("word_count")) for r in run_rows]
    ys = [max(0.0, min(100.0, _as_float(r.get("F")) * 100)) for r in run_rows]
    if not xs or not ys:
        outpath.write_text("", encoding="utf-8")
        return
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = 0, 100
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    def px(x: float) -> float:
        return margin_left + (x - x_min) / max(x_max - x_min, 1) * plot_w
    def py(y: float) -> float:
        return margin_top + (1 - (y - y_min) / max(y_max - y_min, 1)) * plot_h
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">', f'<rect width="100%" height="100%" fill="white"/>', f'<text x="{margin_left}" y="28" font-size="20" font-family="Arial">{_svg_escape(title)}</text>']
    parts.append(f'<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{width - margin_right}" y2="{margin_top + plot_h}" stroke="#222"/>')
    parts.append(f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" stroke="#222"/>')
    for x_tick in range(0, 101, 20):
        y = py(x_tick)
        parts.append(f'<line x1="{margin_left - 5}" y1="{y}" x2="{margin_left}" y2="{y}" stroke="#222"/>')
        parts.append(f'<text x="20" y="{y + 4}" font-size="11" font-family="Arial">{x_tick}</text>')
    step = max(1, len(run_rows) // 6)
    for i, row in enumerate(run_rows):
        color = "#d62728" if str(row.get("failed", "")).lower() in {"true", "1", "yes"} else "#1f77b4"
        parts.append(f'<circle cx="{px(_as_float(row.get("word_count")))}" cy="{py(max(0.0, min(100.0, _as_float(row.get("F")) * 100)))}" r="5" fill="{color}" opacity="0.8"/>')
        if i % step == 0:
            parts.append(f'<text x="{px(_as_float(row.get("word_count"))) + 6}" y="{py(max(0.0, min(100.0, _as_float(row.get("F")) * 100))) - 6}" font-size="10" font-family="Arial">{_svg_escape(row.get("method", ""))}</text>')
    parts.append(f'<text x="{width/2 - 80}" y="{height - 20}" font-size="12" font-family="Arial">Word count</text>')
    parts.append(f'<text x="10" y="{height/2}" font-size="12" font-family="Arial" transform="rotate(-90 10,{height/2})">Faithfulness F (x100)</text>')
    parts.append("</svg>")
    outpath.write_text("\n".join(parts), encoding="utf-8")


def visualize_results(runs_csv: Path, outdir: Path, consider_runtime: bool = False) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    viz_dir = outdir / "visuals"
    charts_dir = viz_dir / "charts"
    tables_dir = viz_dir / "tables"
    _ensure_dir(charts_dir)
    _ensure_dir(tables_dir)

    rows = read_csv(runs_csv)
    summarize_runs_csv(runs_csv, outdir)
    case_rows = compute_case_method_summary(rows)
    model_rows = compute_model_overall_summary(case_rows)
    prompt_rows = compute_prompt_strategy_summary(case_rows)
    model_prompt_rows = compute_model_prompt_summary(case_rows)
    docx_model_rows = [row for row in model_rows if str(row.get("source_type", "docx")).lower() != "csv"]
    docx_rows = [row for row in rows if str(row.get("source_type", "docx")).lower() != "csv"]

    write_csv(tables_dir / "nrs_runs_table.csv", rows)
    write_csv(tables_dir / "nrs_case_method_summary_table.csv", case_rows)
    write_csv(tables_dir / "nrs_model_overall_summary_table.csv", model_rows)
    write_csv(tables_dir / "nrs_prompt_strategy_summary_table.csv", prompt_rows)
    write_csv(tables_dir / "nrs_model_prompt_summary_table.csv", model_prompt_rows)

    chart_paths = [
        _plot_mean_nrs(docx_model_rows, charts_dir / "mean_nrs_by_method.png"),
        _plot_fcr(docx_model_rows, charts_dir / "mean_bertscore_semantic_r_by_method.png"),
        _plot_failed_rate(docx_model_rows, charts_dir / "failed_rate_by_method.png"),
        _plot_word_vs_nrs(docx_rows, charts_dir / "nrs_no_r_vs_word_count.png"),
        _plot_prompt_strategy_mean(prompt_rows, charts_dir / "mean_score_by_prompt_strategy.png"),
        _plot_prompt_strategy_failed_rate(prompt_rows, charts_dir / "failed_rate_by_prompt_strategy.png"),
        _plot_prompt_strategy_runtime(prompt_rows, charts_dir / "runtime_by_prompt_strategy.png"),
        _plot_prompt_strategy_word_count(prompt_rows, charts_dir / "word_count_by_prompt_strategy.png"),
        _plot_model_prompt_heatmap(model_prompt_rows, charts_dir / "model_prompt_strategy_heatmap.png"),
        _plot_input_prompt_heatmap(prompt_rows, charts_dir / "input_prompt_strategy_heatmap.png"),
    ]

    report = viz_dir / "visual_report.md"
    report.write_text(
        "\n\n".join([
            "# EventWeaver Visual Report",
            _markdown_table(model_rows, ["source_type", "model", "dimension", "arch", "parameters", "quantization", "context_length", "prompt_strategy", "input_strategy", "method", "cases_count", "total_runs", "mean_NRS", "mean_CSV_NRS", "mean_Q", "std_NRS", "std_CSV_NRS", "mean_runtime_seconds", "mean_word_count", "mean_paragraph_count", "failed_rate", "mean_bertscore_f1", "mean_semantic_similarity", "mean_R", "rank"], "Model Overall Summary"),
            _markdown_table(case_rows, ["source_type", "case_id", "source_file", "row_index", "row_id", "row_title", "model", "dimension", "arch", "parameters", "quantization", "context_length", "prompt_kind", "prompt_strategy", "input_strategy", "method", "number_of_runs", "mean_runtime_seconds", "mean_word_count", "mean_paragraph_count", "total_broken_sentences", "total_forbidden_formatting", "failed_runs", "mean_bertscore_f1", "mean_semantic_similarity", "robustness_available", "R_stab", "R_struct", "R_fail", "R", "mean_Q", "mean_field_coverage", "mean_format_score", "CSV_NRS", "CSV_NRS_no_R", "NRS", "NRS_no_R"], "Case-Method Summary"),
            _markdown_table(prompt_rows, ["source_type", "prompt_kind", "prompt_strategy", "input_strategy", "number_of_outputs", "mean_NRS", "mean_CSV_NRS", "std_NRS", "std_CSV_NRS", "mean_NRS_no_R", "mean_bertscore_f1", "mean_semantic_similarity", "mean_field_coverage", "mean_format_score", "mean_R", "mean_runtime_seconds", "mean_word_count", "mean_paragraph_count", "failed_rate", "rank"], "Prompt Strategy Summary"),
            _markdown_table(model_prompt_rows, ["source_type", "model", "dimension", "arch", "parameters", "quantization", "context_length", "prompt_kind", "prompt_strategy", "input_strategy", "number_of_outputs", "mean_NRS", "mean_CSV_NRS", "std_NRS", "std_CSV_NRS", "mean_runtime_seconds", "failed_rate", "rank"], "Model Prompt Summary"),
            "## Charts",
            *[f"- `{path.relative_to(viz_dir)}`" for path in chart_paths],
        ]),
        encoding="utf-8",
    )

    return {"report": str(report), "charts_dir": str(charts_dir), "tables_dir": str(tables_dir)}
