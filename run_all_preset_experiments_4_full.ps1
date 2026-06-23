# EventWeaver all-preset experiments: full input strategy.
# Run from the EventWeaver project root.

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

$CASE_STUDIES = "case_studies"
$RUNS = 3
$INPUT_STRATEGY = "full"
$PROMPT_STRATEGIES = @("standard", "short", "detailed")
$OUTPUT_DIR = "outputs_all_full"
$BENCH_DIR = "benchmark_all_full"

function Stop-If-Failed {
    param([string]$StepName)
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Step failed: $StepName" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

function Run-Report {
    param(
        [string]$RunsCsv,
        [string]$BenchDir
    )
    python -m eventweaver summarize $RunsCsv --outdir $BenchDir
    Stop-If-Failed "summarize"
    python -m eventweaver visualize $RunsCsv --outdir $BenchDir
    Stop-If-Failed "visualize"
}

if (-not (Test-Path ".\.venv\Scripts\Activate.ps1")) {
    Write-Host "ERROR: Virtual environment not found at .\.venv" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $CASE_STUDIES)) {
    Write-Host "ERROR: case_studies folder not found: $CASE_STUDIES" -ForegroundColor Red
    exit 1
}

.\.venv\Scripts\Activate.ps1

Write-Host "Running all-preset experiment for input strategy: $INPUT_STRATEGY" -ForegroundColor Green
Write-Host "Using prompt strategies: $($PROMPT_STRATEGIES -join ', ')" -ForegroundColor Cyan
Write-Host "Using num_ctx policy: per-model context_length" -ForegroundColor Cyan

python -m eventweaver all $CASE_STUDIES `
  --model-preset all `
  --runs $RUNS `
  --input-strategy $INPUT_STRATEGY `
  --prompt-kind cultural-heritage `
  --prompt-strategies $PROMPT_STRATEGIES `
  --output-dir $OUTPUT_DIR `
  --outdir $BENCH_DIR `
  --excel
Stop-If-Failed "all-preset experiment ($INPUT_STRATEGY)"

Run-Report -RunsCsv "$BENCH_DIR\nrs_runs.csv" -BenchDir $BENCH_DIR

Write-Host "Outputs: $OUTPUT_DIR" -ForegroundColor Cyan
Write-Host "Benchmark: $BENCH_DIR" -ForegroundColor Cyan
Write-Host "Full input all-preset experiment completed." -ForegroundColor Green
