from __future__ import annotations

from pathlib import Path

from .ollama_client import generate_ollama, list_ollama_models, resolve_ollama_host
from .prompts import prompt_template_for, validate_prompt_strategy
from .models import resolve_num_ctx_for_model
from .source_preparation import prepare_source_context
from .source_record import SourceRecord, build_output_filename, iter_source_records, source_text_word_count
from .text_metrics import split_paragraphs
from .utils import read_csv, safe_model_name, write_csv


def _save_run_manifest(output_dir: Path, records: list[dict]) -> None:
    manifest = output_dir / "generation_metadata.csv"
    fieldnames = [
        "source_type",
        "source",
        "source_file",
        "case_id",
        "row_index",
        "row_id",
        "row_title",
        "output",
        "model",
        "method",
        "run",
        "prompt_kind",
        "prompt_strategy",
        "input_strategy",
        "resolved_input_strategy",
        "ollama_host",
        "num_ctx",
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
    num_ctx: int | None,
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
    prompt_strategy: str = "standard",
    prompt_strategies: list[str] | None = None,
    ollama_host: str | None = None,
    ollama_timeout: int = 900,
    ollama_retries: int = 1,
    skip_ollama_preflight: bool = False,
) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    resolved_ollama_host = resolve_ollama_host(ollama_host)
    if csv_text_columns:
        csv_all_columns = False
    selected_prompt_strategies = prompt_strategies if prompt_strategies else [prompt_strategy]
    selected_prompt_strategies = [str(s).strip().lower() for s in selected_prompt_strategies if str(s).strip()]
    if not selected_prompt_strategies:
        selected_prompt_strategies = ["standard"]
    selected_prompt_strategies = list(dict.fromkeys(selected_prompt_strategies))
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
        expected_prompt_kind = "value-chain" if source.source_type == "csv" else "cultural-heritage"
        if source.prompt_kind != expected_prompt_kind:
            raise ValueError(
                f"Prompt kind '{source.prompt_kind}' is not valid for {source.source_type} input at {source.source_path}. "
                f"Use '{expected_prompt_kind}' or leave --prompt-kind as auto."
            )

    for source in source_records:
        for strategy in selected_prompt_strategies:
            validate_prompt_strategy(source.prompt_kind, strategy)

    print("Ollama startup check", flush=True)
    print(f"  Host: {resolved_ollama_host}", flush=True)
    print(f"  Models selected: {len(models)}", flush=True)
    print(f"  Input strategy: {input_strategy}", flush=True)
    print(f"  Prompt strategies: {', '.join(selected_prompt_strategies)}", flush=True)
    if num_ctx is None:
        print("  num_ctx policy: per-model context_length for full input; 8192 otherwise", flush=True)
    else:
        print(f"  num_ctx policy: explicit override {num_ctx}", flush=True)
    print("  Concurrency: sequential local Ollama requests", flush=True)

    if skip_ollama_preflight:
        print("  Reachable: skipped by --skip-ollama-preflight", flush=True)
    else:
        try:
            available_models = list_ollama_models(resolved_ollama_host)
        except RuntimeError as exc:
            print("  Reachable: no", flush=True)
            raise RuntimeError(
                f"Ollama is not reachable at {resolved_ollama_host}.\n"
                "Start it with:\n"
                "    ollama serve\n"
                "Then verify with:\n"
                "    ollama list\n"
                f"    Invoke-WebRequest {resolved_ollama_host}/api/tags\n"
                f"Details: {exc}"
            ) from exc
        print("  Reachable: yes", flush=True)
        for model in models:
            print(f"  Model available: {model}: {'yes' if model in available_models else 'no'}", flush=True)
        missing_models = sorted(set(models) - set(available_models))
        if missing_models:
            raise RuntimeError(
                "The following requested Ollama model(s) are not installed locally: "
                + ", ".join(missing_models)
                + ". Install them with 'ollama pull <model>' or choose a different --models/--model-preset value."
            )

    for source in source_records:
        if not source.source_text.strip():
            records.append({
                "source_type": source.source_type,
                "source": str(source.source_path),
                "source_file": str(source.source_path),
                "case_id": "" if source.source_type == "csv" else source.source_path.stem,
                "row_index": source.row_index or "",
                "row_id": source.row_id,
                "row_title": source.row_title,
                "output": "",
                "model": "",
                "method": "",
                "run": "",
                "prompt_kind": source.prompt_kind,
                "prompt_strategy": "",
                "input_strategy": input_strategy,
                "resolved_input_strategy": "",
                "ollama_host": resolved_ollama_host,
                "num_ctx": "",
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
        resolved_input_strategy = input_strategy if source.source_type == "docx" else "csv-row"
        if source.source_type == "docx":
            prepared = prepare_source_context(source.source_text, input_strategy, max_source_words, chunk_words, chunk_overlap, top_k)
            prompt_text = prepared.text
            prepared_source = prepared.text
            prepared_word_count = prepared.prepared_source_word_count
            resolved_input_strategy = prepared.strategy_used

        for strategy in selected_prompt_strategies:
            prompt = prompt_template_for(source.prompt_kind, strategy).format(source_text=prompt_text, case_study_text=prompt_text)
            for model in models:
                model_slug = safe_model_name(model)
                input_slug = safe_model_name(input_strategy if source.source_type == "docx" else "csv-row")
                method = f"{model_slug}__{input_slug}__{safe_model_name(strategy)}" if source.source_type == "docx" else f"{model_slug}__{safe_model_name(strategy)}"
                resolved_num_ctx = resolve_num_ctx_for_model(model, input_strategy, num_ctx)
                for run in range(1, runs + 1):
                    print(
                        "Starting generation | "
                        f"host={resolved_ollama_host} | "
                        f"source={source.source_path.name} | "
                        f"model={model} | "
                        f"prompt_strategy={strategy} | "
                        f"run={run} | "
                        f"num_ctx={resolved_num_ctx}",
                        flush=True,
                    )
                    result = generate_ollama(
                        prompt,
                        model,
                        temperature=temperature,
                        num_ctx=resolved_num_ctx,
                        timeout=ollama_timeout,
                        ollama_host=resolved_ollama_host,
                        retries=ollama_retries,
                    )
                    out = output_dir / build_output_filename(source, model, run, prompt_strategy=strategy, input_strategy=input_strategy if source.source_type == "docx" else None)
                    out.write_text(result.text, encoding="utf-8")
                    output_word_count = source_text_word_count(SourceRecord(source.source_type, out, result.text, source.prompt_kind))
                    records.append({
                        "source_type": source.source_type,
                        "source": str(source.source_path),
                        "source_file": str(source.source_path),
                        "case_id": source.row_id if source.source_type == "csv" else source.source_path.stem,
                        "row_index": source.row_index or "",
                        "row_id": source.row_id,
                        "row_title": source.row_title,
                        "output": str(out),
                        "model": model,
                        "method": method,
                        "run": str(run),
                        "prompt_kind": source.prompt_kind,
                        "prompt_strategy": strategy,
                        "input_strategy": input_strategy if source.source_type == "docx" else "csv-row",
                        "resolved_input_strategy": resolved_input_strategy,
                        "ollama_host": resolved_ollama_host,
                        "num_ctx": resolved_num_ctx,
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
                        print(
                            "Generation failed | "
                            f"host={resolved_ollama_host} | "
                            f"source={source.source_path.name} | "
                            f"model={model} | "
                            f"prompt_strategy={strategy} | "
                            f"run={run} | "
                            f"num_ctx={resolved_num_ctx} | "
                            f"error={result.error}",
                            flush=True,
                        )
                    else:
                        print(f"Saved {out}", flush=True)
    _save_run_manifest(output_dir, records)
    return records
