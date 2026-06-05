# EventWeaver full experiment runner using --model-preset all
# Run from the EventWeaver project root.

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Optional metric/report dependencies
pip install bert-score sentence-transformers pandas openpyxl matplotlib

$CASE_STUDIES = "case_studies"
$RUNS = 3

# Experiment 1: auto strategy
python -m eventweaver all $CASE_STUDIES `
  --model-preset all `
  --runs $RUNS `
  --input-strategy auto `
  --output-dir outputs_all_auto `
  --outdir benchmark_all_auto `
  --excel

python -m eventweaver visualize benchmark_all_auto\nrs_runs.csv --outdir benchmark_all_auto
python -m eventweaver summarize benchmark_all_auto\nrs_runs.csv --outdir benchmark_all_auto


# Experiment 2: brief strategy
python -m eventweaver all $CASE_STUDIES `
  --model-preset all `
  --runs $RUNS `
  --input-strategy brief `
  --output-dir outputs_all_brief `
  --outdir benchmark_all_brief `
  --excel

python -m eventweaver visualize benchmark_all_brief\nrs_runs.csv --outdir benchmark_all_brief
python -m eventweaver summarize benchmark_all_brief\nrs_runs.csv --outdir benchmark_all_brief


# Experiment 3: RAG strategy, compact retrieval
python -m eventweaver all $CASE_STUDIES `
  --model-preset all `
  --runs $RUNS `
  --input-strategy rag `
  --top-k 8 `
  --chunk-words 350 `
  --chunk-overlap 80 `
  --output-dir outputs_all_rag `
  --outdir benchmark_all_rag `
  --excel

python -m eventweaver visualize benchmark_all_rag\nrs_runs.csv --outdir benchmark_all_rag
python -m eventweaver summarize benchmark_all_rag\nrs_runs.csv --outdir benchmark_all_rag


# Experiment 4: full strategy with larger context
python -m eventweaver all $CASE_STUDIES `
  --model-preset all `
  --runs $RUNS `
  --input-strategy full `
  --num-ctx 16384 `
  --output-dir outputs_all_full_16k `
  --outdir benchmark_all_full_16k `
  --excel

python -m eventweaver visualize benchmark_all_full_16k\nrs_runs.csv --outdir benchmark_all_full_16k
python -m eventweaver summarize benchmark_all_full_16k\nrs_runs.csv --outdir benchmark_all_full_16k


# Experiment 5: RAG strategy, larger retrieval packet
python -m eventweaver all $CASE_STUDIES `
  --model-preset all `
  --runs $RUNS `
  --input-strategy rag `
  --top-k 12 `
  --chunk-words 500 `
  --chunk-overlap 100 `
  --output-dir outputs_all_rag_large `
  --outdir benchmark_all_rag_large `
  --excel

python -m eventweaver visualize benchmark_all_rag_large\nrs_runs.csv --outdir benchmark_all_rag_large
python -m eventweaver summarize benchmark_all_rag_large\nrs_runs.csv --outdir benchmark_all_rag_large

Write-Host "All EventWeaver experiments with --model-preset all completed."