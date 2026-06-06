# NarrativeForge

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey)
![Ollama](https://img.shields.io/badge/Ollama-local%20LLMs-black)
![Status](https://img.shields.io/badge/status-research%20prototype-orange)
![License](https://img.shields.io/github/license/AIMH-DHgroup/NarrativeForge)
![Repo](https://img.shields.io/badge/GitHub-AIMH--DHgroup%2FNarrativeForge-blue)

Local LLM pipeline for transforming cultural-heritage documents and tabular case-study records into event-ready narratives.

> NarrativeForge is the public project name. The Python module exposed by the CLI is `eventweaver`, so command examples use `python -m eventweaver ...`.

## Short Project Summary

NarrativeForge is a local Python pipeline for transforming `.docx` cultural-heritage case-study forms and `.csv` row-based datasets into short event-ready narratives using local Ollama models, then benchmarking the outputs with NRS and CSV-NRS reliability scores.

The project is designed for research workflows that need grounded narrative generation, repeatable benchmarking, and Windows-friendly local execution.

## Why NarrativeForge?

- It keeps generation local by using Ollama models on your own machine.
- It supports both narrative case-study forms and row-based tabular datasets.
- It includes long-document handling so large `.docx` forms do not have to be processed as a single raw prompt.
- It produces benchmarkable outputs with diagnostics, summaries, and visual reports.
- It is suitable for reproducible research experiments on cultural heritage and structured case-study data.

## Key Features

- Read `.docx` case-study forms.
- Read `.csv` datasets where each row is one narrative source.
- Generate one narrative per `.docx` file.
- Generate one narrative per CSV row.
- Use local Ollama models only.
- Support multiple models and multiple runs.
- Support model presets.
- Support long-document strategies: `auto`, `full`, `brief`, and `rag`.
- Support a cultural-heritage prompt for `.docx`.
- Support a value-chain prompt for `.csv`.
- Record execution time for every model run.
- Compute NRS / CleanScore-style metrics for `.docx`.
- Compute CSV-NRS / Row-NRS for `.csv` rows.
- Report diagnostics such as word count, paragraph count, broken sentence count, forbidden formatting count, failed runs, and runtime.
- Produce CSV reports.
- Produce optional Excel reports.
- Produce visualizations.
- Work locally on Windows, including PowerShell.

## Repository Structure

| Path | Purpose |
| --- | --- |
| `eventweaver/` | Python package and CLI implementation |
| `case_studies/` | Example `.docx` case-study inputs |
| `data/` | Example `.csv` tabular inputs |
| `outputs/` | Generated narratives for `.docx` experiments |
| `outputs_csv/` | Generated narratives for CSV experiments |
| `benchmark_results/` | Benchmark outputs for `.docx` runs |
| `benchmark_csv/` | Benchmark outputs for CSV runs |
| `run_all_preset_experiments.ps1` | PowerShell launcher for preset-based experiments |
| `run_csv_experiments.ps1` | PowerShell launcher for CSV experiments |

## Installation

### Windows PowerShell

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

Optional metrics and report dependencies:

```powershell
pip install bert-score sentence-transformers pandas openpyxl matplotlib
```

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Optional metrics and report dependencies:

```bash
pip install bert-score sentence-transformers pandas openpyxl matplotlib
```

## Ollama Setup

Install and start Ollama locally, then pull the models you want to use.

```powershell
ollama list
ollama pull qwen3:8b
ollama pull gemma3:12b
ollama pull llama3.1:8b
```

Models must be installed locally before generation.

## Quick Start

### `.docx`

```powershell
python -m eventweaver all case_studies `
  --models qwen3:8b gemma3:12b `
  --runs 3 `
  --input-strategy auto `
  --output-dir outputs `
  --outdir benchmark_results `
  --excel
```

### `.csv`

```powershell
python -m eventweaver all data\MOVING_VCs_DATASET_FINAL_V2.csv `
  --models qwen3:8b gemma3:12b `
  --runs 3 `
  --prompt-kind value-chain `
  --csv-id-column "Card ID" `
  --csv-title-column "Descriptor of the value chain" `
  --csv-all-columns `
  --output-dir outputs_csv `
  --outdir benchmark_csv `
  --excel
```

## Input Formats

### `.docx`

- One document produces one generated narrative.
- Text is extracted from paragraphs and tables.
- Long documents can be handled through `auto`, `brief`, `full`, or `rag`.
- The cultural-heritage prompt is used for `.docx` sources.

### `.csv`

- One row produces one generated narrative.
- Empty rows are skipped.
- Row ID and title columns can be configured.
- CSV outputs include the row index, row ID, title slug, model name, and run number in filenames.
- CSV is useful for tabular case-study datasets such as value-chain records.
- The value-chain prompt is used for `.csv` sources.

## Generating Narratives from `.docx`

```powershell
python -m eventweaver generate case_studies --runs 3
```

```powershell
python -m eventweaver generate case_studies `
  --models qwen3:8b gemma3:12b llama3.1:8b `
  --runs 3 `
  --input-strategy auto
```

## Generating Narratives from `.csv`

```powershell
python -m eventweaver generate data\MOVING_VCs_DATASET_FINAL_V2.csv `
  --models qwen3:8b gemma3:12b `
  --runs 3 `
  --prompt-kind value-chain `
  --csv-id-column "Card ID" `
  --csv-title-column "Descriptor of the value chain" `
  --csv-all-columns
```

Smoke test with a row limit:

```powershell
python -m eventweaver generate data\MOVING_VCs_DATASET_FINAL_V2.csv `
  --models qwen3:8b `
  --runs 1 `
  --prompt-kind value-chain `
  --csv-max-rows 5
```

## Long-Document Strategies

- `auto`: default; chooses full text for short sources and brief mode for longer sources.
- `full`: sends the complete extracted source to the model; may hit context limits.
- `brief`: creates a compact structured source brief before generation.
- `rag`: retrieves relevant sections/chunks before generation.

Increasing `--num-ctx` can help, but it uses more memory and may be slower.
RAG can be more stable than sending too much context.
Evaluation should compare outputs against the original source where applicable.

Examples:

```powershell
python -m eventweaver generate case_studies `
  --input-strategy brief `
  --models qwen3:8b gemma3:12b `
  --runs 3
```

```powershell
python -m eventweaver generate case_studies `
  --input-strategy rag `
  --top-k 8 `
  --chunk-words 350 `
  --chunk-overlap 80 `
  --models qwen3:8b `
  --runs 3
```

```powershell
python -m eventweaver generate case_studies `
  --input-strategy full `
  --num-ctx 16384 `
  --models llama3.1:8b `
  --runs 3
```

## Model Presets

The current code defines these presets:

| Preset | Purpose |
| --- | --- |
| `default` | Current default model set |
| `tiny` | Very light smoke tests |
| `small` | Fast models for quick experiments |
| `balanced` | Good routine set for most machines |
| `medium` | Mid-size models |
| `large` | High-capability local models under the current budget |
| `long-context` | Models chosen for longer prompts and document tasks |
| `narrative` | Models suitable for narrative rewriting and summarization |
| `reasoning` | Reasoning-style models |
| `high` | Strong models under the current budget |
| `all` | Full feasible benchmark set |
| `mixed` | Same model set as `all` |

Example:

```powershell
python -m eventweaver generate case_studies --model-preset all --runs 1
```

Custom model names still work through `--models`, even if they are not listed in a preset.

## Evaluation Metrics

### NRS for `.docx`

Preferred NRS:

`NRS = 100 * (0.35 * mean_bertscore_f1 + 0.35 * mean_semantic_similarity + 0.30 * R)`

Fallback if BERTScore is unavailable:

`NRS = 100 * (0.70 * mean_semantic_similarity + 0.30 * R)`

For a single output:

`NRS_no_R = 100 * (0.50 * bertscore_f1 + 0.50 * semantic_similarity)`

Fallback for a single output if BERTScore is unavailable:

`NRS_no_R = 100 * semantic_similarity`

Robustness:

`R = 0.50 * R_stab + 0.25 * R_struct + 0.25 * R_fail`

- `R_stab`: stability of semantic quality across repeated runs.
- `R_struct`: stability of paragraph/event structure across repeated runs.
- `R_fail`: penalty for failed outputs.

### CSV-NRS / Row-NRS for `.csv`

CSV row evaluation is implemented and exposed in the benchmark outputs as `CSV_NRS`.

Per-row quality score:

`Q = 0.30 * bertscore_f1 + 0.25 * semantic_similarity + 0.30 * field_coverage + 0.15 * format_score`

Fallback without BERTScore:

`Q = 0.45 * semantic_similarity + 0.35 * field_coverage + 0.20 * format_score`

Repeated-run CSV score:

`CSV_NRS = 100 * (0.70 * mean_Q + 0.30 * R)`

The CSV benchmark uses the same robustness term `R` as the `.docx` benchmark.

## Diagnostics

The generated reports include diagnostics such as:

- `runtime_seconds`
- `word_count`
- `paragraph_count`
- `broken_sentence_count`
- `forbidden_formatting_count`
- `failed`
- `std_NRS`

For CSV rows, the per-run reports also include `field_coverage`, `format_score`, `Q`, and `CSV_NRS`.

## Reports and Visualizations

Generation writes metadata to:

- `generation_metadata.csv`
- `generation_runs.csv`

Benchmark evaluation writes:

- `nrs_runs.csv`
- `nrs_case_method_summary.csv`
- `nrs_model_overall_summary.csv`

CSV row runs are reported in the same benchmark files using `source_type=csv` and the `CSV_NRS` columns.

If `--excel` is used, an Excel report is written as `nrs_report.xlsx`.

Visualization outputs are written under the benchmark folder, typically in:

- `benchmark_results/visuals/`
- `benchmark_csv/visuals/`

This includes summary tables, charts, and a `visual_report.md` file.

## Evaluation Commands

Evaluate generated outputs:

```powershell
python -m eventweaver evaluate `
  --sources-dir case_studies `
  --outputs-dir outputs `
  --outdir benchmark_results `
  --excel
```

Visualize benchmark outputs:

```powershell
python -m eventweaver visualize benchmark_results\nrs_runs.csv --outdir benchmark_results
```

Summarize an existing runs CSV:

```powershell
python -m eventweaver summarize benchmark_results\nrs_runs.csv --outdir benchmark_results
```

Manifest support:

```powershell
python -m eventweaver write-manifest-template manifest.csv
python -m eventweaver evaluate --manifest manifest.csv --outdir benchmark_results
```

## PowerShell Experiment Scripts

The repository includes convenience launchers for Windows PowerShell:

```powershell
.\run_all_preset_experiments.ps1
.\run_csv_experiments.ps1
```

Warning: all-model experiments can take a very long time. The number of generations is approximately:

`number_of_sources x number_of_models x runs`

Start with smoke tests before running full benchmarks.

## Examples

### Quick Smoke Test

```powershell
python -m eventweaver all case_studies `
  --models qwen3:8b `
  --runs 1 `
  --output-dir outputs_smoke `
  --outdir benchmark_smoke
```

### Balanced `.docx` Benchmark

```powershell
python -m eventweaver all case_studies `
  --models gemma3:12b qwen3:8b qwen3:14b gemma3:4b `
  --runs 3 `
  --input-strategy auto `
  --output-dir outputs_best `
  --outdir benchmark_best `
  --excel
```

### Long-Document RAG Benchmark

```powershell
python -m eventweaver all case_studies `
  --models gemma3:12b qwen3:8b qwen3:14b `
  --runs 3 `
  --input-strategy rag `
  --top-k 8 `
  --chunk-words 350 `
  --chunk-overlap 80 `
  --output-dir outputs_rag `
  --outdir benchmark_rag `
  --excel
```

### CSV Value-Chain Benchmark

```powershell
python -m eventweaver all data\MOVING_VCs_DATASET_FINAL_V2.csv `
  --models gemma3:12b qwen3:8b qwen3:14b `
  --runs 3 `
  --prompt-kind value-chain `
  --csv-id-column "Card ID" `
  --csv-title-column "Descriptor of the value chain" `
  --csv-all-columns `
  --output-dir outputs_csv `
  --outdir benchmark_csv `
  --excel
```

## Recommended Workflows

- Quick smoke test first, then larger runs.
- For `.docx`, start with `--input-strategy auto` and only switch to `rag` if the source is very long.
- For `.csv`, start with a 1-run or 5-row smoke test before scaling to the full dataset.
- Use `--excel` when you want a portable benchmark report.

## Troubleshooting

- PowerShell blocks activation:

  ```powershell
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  .\.venv\Scripts\Activate.ps1
  ```

- Ollama not running:

  ```powershell
  ollama list
  ```

- Model not installed:

  ```powershell
  ollama pull qwen3:8b
  ```

- Context window too small:
  - use `--input-strategy brief`
  - use `--input-strategy rag`
  - increase `--num-ctx` if memory allows

- BERTScore / transformer warnings:
  - warnings about missing or unexpected weights are usually harmless for metric computation

- Missing optional dependencies:

  ```powershell
  pip install bert-score sentence-transformers pandas openpyxl matplotlib
  ```

- Long CSV experiments take too long:
  - reduce `--runs`
  - use fewer models
  - use `--csv-max-rows` for a smoke test
  - start with `smoke_5rows`

- DeepSeek `<think>` blocks:
  - these are removed automatically in generation output

## Research Context

NarrativeForge is intended for experiments on grounded narrative generation, cultural-heritage digital workflows, narrative-event preparation, local LLM evaluation, long-document prompting, RAG, and reliability benchmarking. It is especially useful when expert-produced documentary sources need to be transformed into concise event-ready narratives while preserving traceability and processability.

## Citation

Suggested citation:

```bibtex
@software{narrativeforge2026,
  title = {NarrativeForge: Local LLM Pipeline for Event-Ready Narrative Generation},
  author = {Pratelli, Nicolo},
  year = {2026},
  url = {https://github.com/AIMH-DHgroup/NarrativeForge}
}
```

If you add a `CITATION.cff` file later, keep this section aligned with it.

## License

No license file is currently included. Add a license before public release.

## Contributing

- Open an issue to discuss the change.
- Create a branch for the work.
- Run the smoke tests before submitting.
- Open a pull request when ready.

## Roadmap

- Improved section-aware RAG.
- Human expert evaluation interface.
- Richer CSV field-coverage scoring.
- Multilingual prompts.
- More visual dashboards.
- Citation-ready benchmark exports.
- Integration with semantic story maps or narrative-event ontologies.
- Support for additional structured formats.
