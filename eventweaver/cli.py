from __future__ import annotations

import argparse
from pathlib import Path

from .benchmark import evaluate_folder, summarize_runs_csv, write_manifest_template
from .generation import generate_narratives
from .models import MODEL_PRESETS, resolve_models
from .prompts import CSV_PROMPT_STRATEGIES, DOCX_PROMPT_STRATEGIES
from .visualize import visualize_results


PROMPT_STRATEGY_CHOICES = sorted(set(DOCX_PROMPT_STRATEGIES + CSV_PROMPT_STRATEGIES))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="eventweaver", description="EventWeaver local narrative generation and NRS benchmark")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("generate", help="Generate narratives from docx case-study forms")
    p.add_argument("input")
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--model-preset", choices=sorted(MODEL_PRESETS.keys()), default=None)
    p.add_argument("--input-strategy", choices=["auto", "full", "brief", "rag"], default="auto")
    p.add_argument("--max-source-words", type=int, default=3500)
    p.add_argument("--chunk-words", type=int, default=350)
    p.add_argument("--chunk-overlap", type=int, default=80)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--csv-id-column", default="Card ID")
    p.add_argument("--csv-title-column", default="Descriptor of the value chain")
    p.add_argument("--csv-text-columns", nargs="+", default=None)
    p.add_argument("--csv-all-columns", action="store_true", default=True)
    p.add_argument("--csv-max-rows", type=int, default=0)
    p.add_argument("--prompt-kind", choices=["auto", "cultural-heritage", "value-chain"], default="auto")
    p.add_argument("--prompt-strategy", choices=PROMPT_STRATEGY_CHOICES, default="standard")
    p.add_argument("--prompt-strategies", nargs="+", default=None)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--num-ctx", type=int, default=None)

    p = sub.add_parser("evaluate", help="Compute NRS benchmark reports")
    p.add_argument("--sources-dir", default=None)
    p.add_argument("--outputs-dir", default=None)
    p.add_argument("--manifest", default=None)
    p.add_argument("--outdir", default="benchmark_results")
    p.add_argument("--semantic-method", default="sentence-transformers", choices=["tfidf", "auto", "sbert", "sentence-transformers"])
    p.add_argument("--consider-runtime", action="store_true")
    p.add_argument("--excel", action="store_true")

    p = sub.add_parser("summarize", help="Summarize an existing nrs_runs.csv")
    p.add_argument("csv")
    p.add_argument("--outdir", default="benchmark_results")

    p = sub.add_parser("visualize", help="Create tables and graphics from nrs_runs.csv")
    p.add_argument("csv")
    p.add_argument("--outdir", default="benchmark_results")
    p.add_argument("--consider-runtime", action="store_true")

    p = sub.add_parser("write-manifest-template", help="Write a sample manifest CSV")
    p.add_argument("path")

    p = sub.add_parser("all", help="Run generation and evaluation")
    p.add_argument("input")
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--model-preset", choices=sorted(MODEL_PRESETS.keys()), default=None)
    p.add_argument("--input-strategy", choices=["auto", "full", "brief", "rag"], default="auto")
    p.add_argument("--max-source-words", type=int, default=3500)
    p.add_argument("--chunk-words", type=int, default=350)
    p.add_argument("--chunk-overlap", type=int, default=80)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--csv-id-column", default="Card ID")
    p.add_argument("--csv-title-column", default="Descriptor of the value chain")
    p.add_argument("--csv-text-columns", nargs="+", default=None)
    p.add_argument("--csv-all-columns", action="store_true", default=True)
    p.add_argument("--csv-max-rows", type=int, default=0)
    p.add_argument("--prompt-kind", choices=["auto", "cultural-heritage", "value-chain"], default="auto")
    p.add_argument("--prompt-strategy", choices=PROMPT_STRATEGY_CHOICES, default="standard")
    p.add_argument("--prompt-strategies", nargs="+", default=None)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--outdir", default="benchmark_results")
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--num-ctx", type=int, default=None)
    p.add_argument("--excel", action="store_true")

    return parser


def _write_summary(csv_path: Path, outdir: Path) -> None:
    summarize_runs_csv(csv_path, outdir)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    models = resolve_models(getattr(args, "models", None), getattr(args, "model_preset", None))

    if args.command == "generate":
        generate_narratives(
            Path(args.input),
            models,
            args.runs,
            Path(args.output_dir),
            args.temperature,
            args.num_ctx,
            input_strategy=args.input_strategy,
            max_source_words=args.max_source_words,
            chunk_words=args.chunk_words,
            chunk_overlap=args.chunk_overlap,
            top_k=args.top_k,
            csv_id_column=args.csv_id_column,
            csv_title_column=args.csv_title_column,
            csv_text_columns=args.csv_text_columns,
            csv_all_columns=args.csv_all_columns,
            csv_max_rows=args.csv_max_rows,
            prompt_kind=args.prompt_kind,
            prompt_strategy=args.prompt_strategy,
            prompt_strategies=args.prompt_strategies,
        )
        return 0
    if args.command == "evaluate":
        evaluate_folder(
            sources_dir=Path(args.sources_dir) if args.sources_dir else None,
            outputs_dir=Path(args.outputs_dir) if args.outputs_dir else None,
            manifest=Path(args.manifest) if args.manifest else None,
            outdir=Path(args.outdir),
            semantic_method=args.semantic_method,
            excel=args.excel,
        )
        return 0
    if args.command == "summarize":
        _write_summary(Path(args.csv), Path(args.outdir))
        return 0
    if args.command == "visualize":
        visualize_results(Path(args.csv), Path(args.outdir))
        return 0
    if args.command == "write-manifest-template":
        write_manifest_template(Path(args.path))
        return 0
    if args.command == "all":
        input_path = Path(args.input)
        generate_narratives(
            input_path,
            models,
            args.runs,
            Path(args.output_dir),
            args.temperature,
            args.num_ctx,
            input_strategy=args.input_strategy,
            max_source_words=args.max_source_words,
            chunk_words=args.chunk_words,
            chunk_overlap=args.chunk_overlap,
            top_k=args.top_k,
            csv_id_column=args.csv_id_column,
            csv_title_column=args.csv_title_column,
            csv_text_columns=args.csv_text_columns,
            csv_all_columns=args.csv_all_columns,
            csv_max_rows=args.csv_max_rows,
            prompt_kind=args.prompt_kind,
            prompt_strategy=args.prompt_strategy,
            prompt_strategies=args.prompt_strategies,
        )
        evaluate_folder(sources_dir=input_path if input_path.is_dir() else input_path.parent, outputs_dir=Path(args.output_dir), outdir=Path(args.outdir), excel=args.excel)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
