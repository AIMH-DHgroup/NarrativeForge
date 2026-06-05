from __future__ import annotations

from pathlib import Path

from .docx_reader import read_docx_text
from .ollama_client import generate_ollama
from .prompts import GENERAL_PROMPT_TEMPLATE
from .source_preparation import prepare_source_context
from .utils import write_csv
from .utils import case_id_from_path, iter_input_docs, safe_model_name


def _save_run_manifest(output_dir: Path, records: list[dict]) -> None:
    manifest = output_dir / "generation_runs.csv"
    fieldnames = [
        "source",
        "output",
        "method",
        "run",
        "input_strategy",
        "resolved_input_strategy",
        "original_source_word_count",
        "prepared_source_word_count",
        "selected_chunk_count",
        "max_source_words",
        "chunk_words",
        "chunk_overlap",
        "top_k",
        "runtime_seconds",
        "error",
    ]
    write_csv(manifest, records, fieldnames=fieldnames)


def generate_narratives(
    input_path: Path,
    models: list[str],
    runs: int,
    output_dir: Path,
    temperature: float,
    num_ctx: int,
    input_strategy: str = "brief",
    max_source_words: int = 3500,
    chunk_words: int = 350,
    chunk_overlap: int = 80,
    top_k: int = 8,
) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    for source in iter_input_docs(input_path):
        source_text = read_docx_text(source)
        if not source_text.strip():
            records.append({
                "source": str(source),
                "output": "",
                "method": "",
                "run": "",
                "input_strategy": input_strategy,
                "resolved_input_strategy": "",
                "original_source_word_count": 0,
                "prepared_source_word_count": 0,
                "selected_chunk_count": 0,
                "max_source_words": max_source_words,
                "chunk_words": chunk_words,
                "chunk_overlap": chunk_overlap,
                "top_k": top_k,
                "runtime_seconds": "",
                "error": "empty source",
            })
            continue
        case_id = case_id_from_path(source)
        prepared = prepare_source_context(source_text, input_strategy, max_source_words, chunk_words, chunk_overlap, top_k)
        prompt = GENERAL_PROMPT_TEMPLATE.format(case_study_text=prepared.text)
        for model in models:
            method = safe_model_name(model)
            for run in range(1, runs + 1):
                result = generate_ollama(prompt, model, temperature=temperature, num_ctx=num_ctx)
                out = output_dir / f"{case_id}_narrative_{method}_{prepared.strategy_used}_run{run}.txt"
                out.write_text(result.text, encoding="utf-8")
                records.append({
                    "source": str(source),
                    "output": str(out),
                    "method": model,
                    "run": str(run),
                    "input_strategy": input_strategy,
                    "resolved_input_strategy": prepared.strategy_used,
                    "original_source_word_count": prepared.original_source_word_count,
                    "prepared_source_word_count": prepared.prepared_source_word_count,
                    "selected_chunk_count": prepared.selected_chunk_count,
                    "max_source_words": max_source_words,
                    "chunk_words": chunk_words,
                    "chunk_overlap": chunk_overlap,
                    "top_k": top_k,
                    "runtime_seconds": f"{result.runtime_seconds:.3f}",
                    "error": result.error or "",
                })
                if result.error:
                    print(f"Generation failed for {source.name} with {model} run {run}: {result.error}")
                else:
                    print(f"Saved {out}")
    _save_run_manifest(output_dir, records)
    return records
