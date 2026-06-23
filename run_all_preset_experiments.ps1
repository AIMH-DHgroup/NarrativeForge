# EventWeaver full experiment runner using --model-preset all
# Run from the EventWeaver project root.

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
$ErrorActionPreference = "Stop"

function python {
  & python.exe @args
  if ($LASTEXITCODE -ne 0) {
    Write-Host "Command failed with exit code ${LASTEXITCODE}: python $($args -join ' ')" -ForegroundColor Red
    exit $LASTEXITCODE
  }
}

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Optional metric/report dependencies
pip install bert-score sentence-transformers pandas openpyxl matplotlib

$CASE_STUDIES = "case_studies"
$RUNS = 3

Write-Host "Experiment 1.1: auto strategy, prompt standard strategy"
# Experiment 1.1: auto strategy, prompt standard strategy
python -m eventweaver all $CASE_STUDIES `
  --model-preset all `
  --runs $RUNS `
  --input-strategy auto `
  --prompt-strategy standard `
  --num-ctx 8192 `
  --output-dir outputs_all_auto_standard `
  --outdir benchmark_all_auto_standard `
  --excel

python -m eventweaver visualize benchmark_all_auto_standard\nrs_runs.csv --outdir benchmark_all_auto_standard
python -m eventweaver summarize benchmark_all_auto_standard\nrs_runs.csv --outdir benchmark_all_auto_standard

# Experiment 1.2: auto strategy, prompt short strategy
python -m eventweaver all $CASE_STUDIES `
  --model-preset all `
  --runs $RUNS `
  --input-strategy auto `
  --prompt-strategy short `
  --num-ctx 8192 `
  --output-dir outputs_all_auto_short `
  --outdir benchmark_all_auto_short `
  --excel

python -m eventweaver visualize benchmark_all_auto_short\nrs_runs.csv --outdir benchmark_all_auto_short
python -m eventweaver summarize benchmark_all_auto_short\nrs_runs.csv --outdir benchmark_all_auto_short

# Experiment 1.3: auto strategy, prompt detailed strategy
python -m eventweaver all $CASE_STUDIES `
  --model-preset all `
  --runs $RUNS `
  --input-strategy auto `
  --prompt-strategy detailed `
  --num-ctx 8192 `
  --output-dir outputs_all_auto_detailed `
  --outdir benchmark_all_auto_detailed `
  --excel

python -m eventweaver visualize benchmark_all_auto_detailed\nrs_runs.csv --outdir benchmark_all_auto_detailed
python -m eventweaver summarize benchmark_all_auto_detailed\nrs_runs.csv --outdir benchmark_all_auto_detailed


# Experiment 2.1: brief strategy, prompt standard strategy
python -m eventweaver all $CASE_STUDIES `
  --model-preset all `
  --runs $RUNS `
  --input-strategy brief `
  --prompt-strategy standard `
  --num-ctx 8192 `
  --output-dir outputs_all_brief_standard `
  --outdir benchmark_all_brief_standard `
  --excel

python -m eventweaver visualize benchmark_all_brief_standard\nrs_runs.csv --outdir benchmark_all_brief_standard
python -m eventweaver summarize benchmark_all_brief_standard\nrs_runs.csv --outdir benchmark_all_brief_standard

# Experiment 2.2: brief strategy, prompt short strategy
python -m eventweaver all $CASE_STUDIES `
  --model-preset all `
  --runs $RUNS `
  --input-strategy brief `
  --prompt-strategy short `
  --num-ctx 8192 `
  --output-dir outputs_all_brief_short `
  --outdir benchmark_all_brief_short `
  --excel

python -m eventweaver visualize benchmark_all_brief_short\nrs_runs.csv --outdir benchmark_all_brief_short
python -m eventweaver summarize benchmark_all_brief_short\nrs_runs.csv --outdir benchmark_all_brief_short

# Experiment 2.3: brief strategy, prompt detailed strategy
python -m eventweaver all $CASE_STUDIES `
  --model-preset all `
  --runs $RUNS `
  --input-strategy brief `
  --prompt-strategy detailed `
  --num-ctx 8192 `
  --output-dir outputs_all_brief_detailed `
  --outdir benchmark_all_brief_detailed `
  --excel

python -m eventweaver visualize benchmark_all_brief_detailed\nrs_runs.csv --outdir benchmark_all_brief_detailed
python -m eventweaver summarize benchmark_all_brief_detailed\nrs_runs.csv --outdir benchmark_all_brief_detailed


# Experiment 3.1: RAG strategy, compact retrieval, prompt standard strategy
python -m eventweaver all $CASE_STUDIES `
  --model-preset all `
  --runs $RUNS `
  --input-strategy rag `
  --prompt-strategy standard `
  --num-ctx 8192 `
  --top-k 8 `
  --chunk-words 350 `
  --chunk-overlap 80 `
  --output-dir outputs_all_rag_standard `
  --outdir benchmark_all_rag_standard `
  --excel

python -m eventweaver visualize benchmark_all_rag_standard\nrs_runs.csv --outdir benchmark_all_rag_standard
python -m eventweaver summarize benchmark_all_rag_standard\nrs_runs.csv --outdir benchmark_all_rag_standard

# Experiment 3.2: RAG strategy, compact retrieval, prompt short strategy
python -m eventweaver all $CASE_STUDIES `
  --model-preset all `
  --runs $RUNS `
  --input-strategy rag `
  --prompt-strategy short `
  --num-ctx 8192 `
  --top-k 8 `
  --chunk-words 350 `
  --chunk-overlap 80 `
  --output-dir outputs_all_rag_short `
  --outdir benchmark_all_rag_short `
  --excel

python -m eventweaver visualize benchmark_all_rag_short\nrs_runs.csv --outdir benchmark_all_rag_short
python -m eventweaver summarize benchmark_all_rag_short\nrs_runs.csv --outdir benchmark_all_rag_short

# Experiment 3.3: RAG strategy, compact retrieval, prompt detailed strategy
python -m eventweaver all $CASE_STUDIES `
  --model-preset all `
  --runs $RUNS `
  --input-strategy rag `
  --prompt-strategy detailed `
  --num-ctx 8192 `
  --top-k 8 `
  --chunk-words 350 `
  --chunk-overlap 80 `
  --output-dir outputs_all_rag_detailed `
  --outdir benchmark_all_rag_detailed `
  --excel

python -m eventweaver visualize benchmark_all_rag_detailed\nrs_runs.csv --outdir benchmark_all_rag_detailed
python -m eventweaver summarize benchmark_all_rag_detailed\nrs_runs.csv --outdir benchmark_all_rag_detailed


# Experiment 4.1: full strategy, prompt standard strategy
python -m eventweaver all $CASE_STUDIES `
  --model-preset all `
  --runs $RUNS `
  --input-strategy full `
  --prompt-strategy standard `
  --output-dir outputs_all_full_standard `
  --outdir benchmark_all_full_standard `
  --excel

python -m eventweaver visualize benchmark_all_full_standard\nrs_runs.csv --outdir benchmark_all_full_standard
python -m eventweaver summarize benchmark_all_full_standard\nrs_runs.csv --outdir benchmark_all_full_standard

# Experiment 4.2: full strategy, prompt short strategy
python -m eventweaver all $CASE_STUDIES `
  --model-preset all `
  --runs $RUNS `
  --input-strategy full `
  --prompt-strategy short `
  --output-dir outputs_all_full_short `
  --outdir benchmark_all_full_short `
  --excel

python -m eventweaver visualize benchmark_all_full_short\nrs_runs.csv --outdir benchmark_all_full_short
python -m eventweaver summarize benchmark_all_full_short\nrs_runs.csv --outdir benchmark_all_full_short

# Experiment 4.3: full strategy, prompt detailed strategy
python -m eventweaver all $CASE_STUDIES `
  --model-preset all `
  --runs $RUNS `
  --input-strategy full `
  --prompt-strategy detailed `
  --output-dir outputs_all_full_detailed `
  --outdir benchmark_all_full_detailed `
  --excel

python -m eventweaver visualize benchmark_all_full_detailed\nrs_runs.csv --outdir benchmark_all_full_detailed
python -m eventweaver summarize benchmark_all_full_detailed\nrs_runs.csv --outdir benchmark_all_full_detailed


# Experiment 5.1: RAG strategy, larger retrieval packet, prompt standard strategy
python -m eventweaver all $CASE_STUDIES `
  --model-preset all `
  --runs $RUNS `
  --input-strategy rag `
  --prompt-strategy standard `
  --num-ctx 8192 `
  --top-k 12 `
  --chunk-words 500 `
  --chunk-overlap 100 `
  --output-dir outputs_all_rag_large_standard `
  --outdir benchmark_all_rag_large_standard `
  --excel

python -m eventweaver visualize benchmark_all_rag_large_standard\nrs_runs.csv --outdir benchmark_all_rag_large_standard
python -m eventweaver summarize benchmark_all_rag_large_standard\nrs_runs.csv --outdir benchmark_all_rag_large_standard

# Experiment 5.2: RAG strategy, larger retrieval packet, prompt short strategy
python -m eventweaver all $CASE_STUDIES `
  --model-preset all `
  --runs $RUNS `
  --input-strategy rag `
  --prompt-strategy short `
  --num-ctx 8192 `
  --top-k 12 `
  --chunk-words 500 `
  --chunk-overlap 100 `
  --output-dir outputs_all_rag_large_short `
  --outdir benchmark_all_rag_large_short `
  --excel

python -m eventweaver visualize benchmark_all_rag_large_short\nrs_runs.csv --outdir benchmark_all_rag_large_short
python -m eventweaver summarize benchmark_all_rag_large_short\nrs_runs.csv --outdir benchmark_all_rag_large_short

# Experiment 5.3: RAG strategy, larger retrieval packet, prompt detailed strategy
python -m eventweaver all $CASE_STUDIES `
  --model-preset all `
  --runs $RUNS `
  --input-strategy rag `
  --prompt-strategy detailed `
  --num-ctx 8192 `
  --top-k 12 `
  --chunk-words 500 `
  --chunk-overlap 100 `
  --output-dir outputs_all_rag_large_detailed `
  --outdir benchmark_all_rag_large_detailed `
  --excel

python -m eventweaver visualize benchmark_all_rag_large_detailed\nrs_runs.csv --outdir benchmark_all_rag_large_detailed
python -m eventweaver summarize benchmark_all_rag_large_detailed\nrs_runs.csv --outdir benchmark_all_rag_large_detailed

Write-Host "All EventWeaver experiments with --model-preset all completed."
