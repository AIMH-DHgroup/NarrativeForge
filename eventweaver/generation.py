from __future__ import annotations

from pathlib import Path

from .ollama_client import generate_ollama
from .prompts import prompt_template_for
from .source_preparation import prepare_source_context
from .source_record import SourceRecord, build_output_filename, iter_source_records, source_text_word_count
from .text_metrics import split_paragraphs
from .utils import read_csv, write_csv


def _save_run_manifest(output_dir: Path, records: list[dict]) -> None:
    manifest = output_dir / "generation_metadata.csv"
    fieldnames = [
        "source_type",
        "source",
        "row_index",
        "row_id",
        "row_title",
        "output",
        "method",
        "run",
        "prompt_kind",
        "resolved_prompt_kind",
        "input_strategy",
        "resolved_input_strategy",
        "original_source_word_count",
        "prepared_source_word_count",
        "selected_chunk_count",
        "max_source_words",
        "chunk_words",
        "chunk_overlap",
        "top_k",
        "source_word_count",
        "output_word_count",
        "paragraph_count",
        "runtime_seconds",
        "error",
    ]
    existing: list[dict] = []
    if manifest.exists():
        existing = read_csv(manifest)
    combined = [row for row in existing if str(row.get("output", "")) not in {str(r.get("output", "")) for r in records if r.get("output")}]
    combined.extend(records)
    write_csv(manifest, combined, fieldnames=fieldnames)
    write_csv(output_dir / "generation_runs.csv", combined, fieldnames=fieldnames)


def generate_narratives(
    input_path: Path,
    models: list[str],
    runs: int,
    output_dir: Path,
    temperature: float,
    num_ctx: int,
    input_strategy: str = "auto",
    max_source_words: int = 3500,
    chunk_words: int = 350,
    chunk_overlap: int = 80,
    top_k: int = 8,
    csv_id_column: str = "Card ID",
    csv_title_column: str = "Descriptor of the value chain",
    csv_text_columns: list[str] | None = None,
    csv_all_columns: bool = True,
    csv_max_rows: int = 0,
    prompt_kind: str = "auto",
) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    if csv_text_columns:
        csv_all_columns = False
    source_records = iter_source_records(
        input_path,
        csv_id_column=csv_id_column,
        csv_title_column=csv_title_column,
        csv_text_columns=csv_text_columns,
        csv_all_columns=csv_all_columns,
        csv_max_rows=csv_max_rows,
        prompt_kind=prompt_kind,
    )

    for source in source_records:
        if not source.source_text.strip():
            records.append({
                "source_type": source.source_type,
                "source": str(source.source_path),
                "row_index": source.row_index or "",
                "row_id": source.row_id,
                "row_title": source.row_title,
                "output": "",
                "method": "",
                "run": "",
                "prompt_kind": prompt_kind,
                "resolved_prompt_kind": source.prompt_kind,
                "input_strategy": input_strategy,
                "resolved_input_strategy": "",
                "original_source_word_count": 0,
                "prepared_source_word_count": 0,
                "selected_chunk_count": 0,
                "max_source_words": max_source_words,
                "chunk_words": chunk_words,
                "chunk_overlap": chunk_overlap,
                "top_k": top_k,
                "source_word_count": 0,
                "output_word_count": 0,
                "paragraph_count": 0,
                "runtime_seconds": "",
                "error": "empty source",
            })
            continue
        prompt_text = source.source_text
        prepared_source = source.source_text
        prepared_word_count = source_text_word_count(source)
        resolved_input_strategy = input_strategy
        if source.source_type == "docx":
            prepared = prepare_source_context(source.source_text, input_strategy, max_source_words, chunk_words, chunk_overlap, top_k)
            prompt_text = prepared.text
            prepared_source = prepared.text
            prepared_word_count = prepared.prepared_source_word_count
            resolved_input_strategy = prepared.strategy_used

        prompt = prompt_template_for(source.prompt_kind).format(source_text=prompt_text, case_study_text=prompt_text)
        for model in models:
            for run in range(1, runs + 1):
                result = generate_ollama(prompt, model, temperature=temperature, num_ctx=num_ctx)
                out = output_dir / build_output_filename(source, model, run)
                out.write_text(result.text, encoding="utf-8")
                output_word_count = source_text_word_count(SourceRecord(source.source_type, out, result.text, source.prompt_kind))
                records.append({
                    "source_type": source.source_type,
                    "source": str(source.source_path),
                    "row_index": source.row_index or "",
                    "row_id": source.row_id,
                    "row_title": source.row_title,
                    "output": str(out),
                    "method": model,
                    "run": str(run),
                    "prompt_kind": prompt_kind,
                    "resolved_prompt_kind": source.prompt_kind,
                    "input_strategy": input_strategy if source.source_type == "docx" else "full",
                    "resolved_input_strategy": resolved_input_strategy,
                    "original_source_word_count": source_text_word_count(source),
                    "prepared_source_word_count": prepared_word_count,
                    "selected_chunk_count": 0 if source.source_type == "csv" else prepared.selected_chunk_count,
                    "max_source_words": max_source_words,
                    "chunk_words": chunk_words,
                    "chunk_overlap": chunk_overlap,
                    "top_k": top_k,
                    "source_word_count": source_text_word_count(source),
                    "output_word_count": output_word_count,
                    "paragraph_count": len(split_paragraphs(result.text)),
                    "runtime_seconds": f"{result.runtime_seconds:.3f}",
                    "error": result.error or "",
                })
                if result.error:
                    print(f"Generation failed for {source.source_path.name} with {model} run {run}: {result.error}")
                else:
                    print(f"Saved {out}")
    _save_run_manifest(output_dir, records)
    return records
