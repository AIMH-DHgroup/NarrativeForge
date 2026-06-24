from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .analysis import AnalysisOptions, DEFAULT_BALANCED_WEIGHTS, run_analysis
from .version import __version__


def _parse_csv_list(text: str | None) -> list[str] | None:
    if text is None:
        return None
    values = [item.strip() for item in text.split(",") if item.strip()]
    return values or None


def _parse_weights(text: str | None) -> dict[str, float] | None:
    if not text:
        return None
    # Accept either JSON or comma-separated values in the default order.
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return {str(k): float(v) for k, v in parsed.items()}
    except json.JSONDecodeError:
        pass
    parts = [float(part.strip()) for part in text.split(",") if part.strip()]
    keys = ["NRS", "speed", "size_efficiency", "stability", "reliability"]
    if len(parts) != len(keys):
        raise argparse.ArgumentTypeError(
            "Weights must be JSON or five comma-separated numbers: NRS,speed,size_efficiency,stability,reliability"
        )
    return dict(zip(keys, parts))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nrs-analyze",
        description="Analyze NarrativeForge NRS benchmark ZIPs or extracted folders and create CSV tables, charts, and Markdown/HTML reports.",
    )
    parser.add_argument("source", help="Path to experiments.zip or an extracted experiment directory containing nrs_runs.csv files.")
    parser.add_argument("-o", "--output", default="nrs_analysis_output", help="Output directory. Default: nrs_analysis_output")
    parser.add_argument("--input-order", default=None, help="Comma-separated preferred input-strategy order, e.g. auto,brief,rag,full")
    parser.add_argument("--prompt-order", default=None, help="Comma-separated preferred prompt-strategy order, e.g. short,standard,detailed")
    parser.add_argument(
        "--weights",
        default=None,
        help=(
            "Balanced-score weights as JSON or five comma-separated numbers in order "
            "NRS,speed,size_efficiency,stability,reliability. Default: "
            + json.dumps(DEFAULT_BALANCED_WEIGHTS)
        ),
    )
    parser.add_argument("--reliable-failure-max", type=float, default=0.05, help="Failure-rate cutoff for reliable threshold rows. Default: 0.05")
    parser.add_argument("--title", default="NarrativeForge NRS Benchmark Analysis", help="Report title.")
    parser.add_argument("--hardware-note", default=None, help="Runtime interpretation note to include in generated reports.")
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI. Default: 300")
    parser.add_argument("--figure-format", default="png", choices=["png", "pdf", "svg"], help="Figure format. Default: png")
    parser.add_argument("--coverage", action="store_true", help="Force coverage diagnostics and print stronger warnings if optional dependencies are missing.")
    parser.add_argument("--coverage-thresholds", nargs="+", type=float, default=[0.70, 0.75, 0.80], help="Sentence coverage thresholds. Default: 0.70 0.75 0.80")
    parser.add_argument("--coverage-model", default="sentence-transformers/all-MiniLM-L6-v2", help="Sentence-transformers model for coverage diagnostics.")
    parser.add_argument("--recompute-coverage", action="store_true", help="Ignore coverage_metrics.csv and recompute all coverage diagnostics.")
    parser.add_argument("--skip-entity-coverage", action="store_true", help="Disable spaCy entity coverage.")
    parser.add_argument("--skip-keyphrase-coverage", action="store_true", help="Disable KeyBERT keyphrase coverage.")
    parser.add_argument("--case-detail-plots", action="store_true", help="Generate per-case detailed plots under figures/case_studies/per_case/.")
    parser.add_argument("--version", action="version", version=f"narrativeforge-nrs-analyzer {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    weights = _parse_weights(args.weights) or DEFAULT_BALANCED_WEIGHTS.copy()
    input_order = _parse_csv_list(args.input_order)
    prompt_order = _parse_csv_list(args.prompt_order)

    options = AnalysisOptions(
        input_order=input_order or AnalysisOptions().input_order,
        prompt_order=prompt_order or AnalysisOptions().prompt_order,
        balanced_weights=weights,
        reliable_failure_rate_max=args.reliable_failure_max,
        dpi=args.dpi,
        figure_format=args.figure_format,
        title=args.title,
        coverage=args.coverage,
        coverage_thresholds=args.coverage_thresholds,
        coverage_model=args.coverage_model,
        recompute_coverage=args.recompute_coverage,
        skip_entity_coverage=args.skip_entity_coverage,
        skip_keyphrase_coverage=args.skip_keyphrase_coverage,
        case_detail_plots=args.case_detail_plots,
    )
    if args.hardware_note:
        options.hardware_note = args.hardware_note

    try:
        result = run_analysis(Path(args.source), Path(args.output), options)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("NarrativeForge NRS analysis complete")
    print(f"Output directory: {result.output_dir}")
    print(f"HTML report: {result.output_dir / 'narrativeforge_nrs_analysis_report.html'}")
    print(f"Markdown report: {result.output_dir / 'narrativeforge_nrs_analysis_report.md'}")
    print(f"Bundle ZIP: {result.bundle_path}")
    best_input = result.summary.get("best_input_strategy", {})
    best_model = result.summary.get("best_model", {})
    if best_input:
        print(f"Best input strategy: {best_input.get('input_strategy')} (mean NRS={best_input.get('mean_NRS'):.3f})")
    if best_model:
        print(f"Best model: {best_model.get('model')} (mean NRS={best_model.get('mean_NRS'):.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
