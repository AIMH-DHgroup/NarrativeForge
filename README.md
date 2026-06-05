# EventWeaver

Local Python pipeline for generating cultural-heritage narratives from `.docx` case-study forms and computing NRS benchmark reports.

## Install on Windows

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Ollama

Start Ollama locally and pull the models you want:

```powershell
ollama pull llama3.1:8b
ollama pull qwen2.5:7b
ollama pull gemma3:4b
ollama pull deepseek-r1:7b
```

## Models

Default models:

```powershell
llama3.1:8b
qwen3:8b
gemma3:12b
mistral:7b
phi3.5:3.8b
phi4-mini:3.8b
deepseek-r1:7b
```

Complete model list:

```powershell
llama3.2:3b
llama3.1:8b
llama3.1:13b
qwen2.5:7b
qwen2.5:14b
qwen2.5:32b
qwen3:8b
qwen3:14b
qwen3:30b
qwen3:32b
gemma3:4b
gemma3:12b
gemma2:9b
gemma2:27b
mistral:7b
mistral-nemo:12b
mistral-small3.2:24b
phi3.5:3.8b
phi4-mini:3.8b
deepseek-r1:7b
deepseek-r1:14b
deepseek-r1:32b
```

## Commands

Generate narratives:

```powershell
python -m eventweaver generate case_studies --runs 3
```

Generate narratives with selected models:

```powershell
python -m eventweaver generate case_studies --models qwen3:8b gemma3:12b deepseek-r1:7b --runs 3
```

Long-document handling:

- `auto` is the default: it uses full text for short forms and brief mode when the source exceeds 3500 words.
- `full` sends the complete extracted source text and can hit context limits on long forms.
- `brief` forces the compact brief format for every source.
- `rag` retrieves the most relevant source sections and chunks before generation.
- Increasing `--num-ctx` can help, but it uses more memory and may be slower.

Examples:

```powershell
python -m eventweaver generate case_studies --input-strategy auto --models qwen3:8b llama3.1:8b --runs 3
python -m eventweaver generate case_studies --input-strategy rag --top-k 8 --chunk-words 350 --chunk-overlap 80 --models qwen3:8b
python -m eventweaver generate case_studies --input-strategy full --num-ctx 16384 --models llama3.1:8b
```

For larger local models, use a preset:

```powershell
python -m eventweaver generate case_studies --model-preset larger --runs 3 --output-dir outputs
```

To test all available preset models:

```powershell
python -m eventweaver generate case_studies --model-preset all --runs 3 --output-dir outputs
```

Intermediate presets are also available:

```powershell
python -m eventweaver generate case_studies --model-preset small --runs 3 --output-dir outputs
python -m eventweaver generate case_studies --model-preset medium --runs 3 --output-dir outputs
python -m eventweaver generate case_studies --model-preset large --runs 3 --output-dir outputs
python -m eventweaver generate case_studies --model-preset reasoning --runs 3 --output-dir outputs
```

Evaluate:

```powershell
python -m eventweaver evaluate --sources-dir case_studies --outputs-dir outputs --outdir benchmark_results
```

Visualize tables and graphics:

```powershell
python -m eventweaver visualize benchmark_results\nrs_runs.csv --outdir benchmark_results
```

Full pipeline:

```powershell
python -m eventweaver all case_studies --runs 3 --output-dir outputs --outdir benchmark_results
```

Or with larger models:

```powershell
python -m eventweaver all case_studies --model-preset larger --runs 3 --output-dir outputs --outdir benchmark_results
```

Other useful presets:

```powershell
python -m eventweaver all case_studies --model-preset balanced --runs 3 --output-dir outputs --outdir benchmark_results
python -m eventweaver all case_studies --model-preset medium-large --runs 3 --output-dir outputs --outdir benchmark_results
python -m eventweaver all case_studies --model-preset high --runs 3 --output-dir outputs --outdir benchmark_results
```

Summarize only:

```powershell
python -m eventweaver summarize benchmark_results\nrs_runs.csv --outdir benchmark_results
```

Manifest template:

```powershell
python -m eventweaver write-manifest-template manifest.csv
```

Manifest evaluation:

```powershell
python -m eventweaver evaluate --manifest manifest.csv --outdir benchmark_results
```

## NRS

Preferred benchmark score:

`NRS = 100 * (0.35 * mean_bertscore_f1 + 0.35 * mean_semantic_similarity + 0.30 * R)`

If BERTScore is unavailable:

`NRS = 100 * (0.70 * mean_semantic_similarity + 0.30 * R)`

For a single output:

`NRS_no_R = 100 * (0.50 * bertscore_f1 + 0.50 * semantic_similarity)`

If BERTScore is unavailable:

`NRS_no_R = 100 * semantic_similarity`

Robustness:

`R = 0.50 * R_stab + 0.25 * R_struct + 0.25 * R_fail`

Score interpretation:

- 90-100 Excellent
- 80-89 Very good
- 70-79 Acceptable
- 60-69 Weak
- <60 Not suitable

- BERTScore-F1 is optional and uses the `bert-score` package when available.
- Semantic similarity uses `sentence-transformers` when available, otherwise TF-IDF cosine similarity.

## Notes

- `deepseek-r1` outputs have `<think>...</think>` blocks removed automatically.
- Output files use UTF-8 `.txt`.
- Excel reports are optional and require `pandas` and `openpyxl`.
- Visual charts are written as SVG fallback graphics when `matplotlib` is not installed.
- Optional metrics can be enabled with `pip install bert-score sentence-transformers`.
