$ErrorActionPreference = "Stop"

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

$ProjectRoot = $PSScriptRoot
Set-Location $ProjectRoot

# Edit this if your CSV lives elsewhere.
$CSV_FILE = "moving\MOVING_VCs_DATASET_FINAL_V2.csv"
$RUNS = 3
$MAX_ROWS_SMOKE = 5
$OUTPUT_ROOT = "outputs_csv"
$BENCHMARK_ROOT = "benchmark_csv"

# Optional dependencies for metrics and reports.
# Uncomment if you want to refresh the local environment first.
# pip install bert-score sentence-transformers pandas openpyxl matplotlib

function Invoke-PostProcessing {
    param (
        [string]$BenchmarkDir
    )

    $possibleFiles = @(
        "$BenchmarkDir\nrs_runs.csv",
        "$BenchmarkDir\csv_nrs_runs.csv",
        "$BenchmarkDir\row_nrs_runs.csv"
    )

    foreach ($file in $possibleFiles) {
        if (Test-Path -LiteralPath $file) {
            python -m eventweaver visualize "$file" --outdir "$BenchmarkDir"
            python -m eventweaver summarize "$file" --outdir "$BenchmarkDir"
            return
        }
    }

    Write-Host "No runs CSV found in $BenchmarkDir"
}

function Invoke-Experiment {
    param (
        [string]$Header,
        [string[]]$Arguments,
        [string]$BenchmarkDir
    )

    Write-Host $Header
    python -m eventweaver @Arguments
    Invoke-PostProcessing -BenchmarkDir $BenchmarkDir
}

if (-not (Test-Path -LiteralPath $CSV_FILE)) {
    Write-Host "CSV file not found: $CSV_FILE"
    exit 1
}

if (-not (Test-Path -LiteralPath ".\.venv\Scripts\Activate.ps1")) {
    Write-Host "Warning: .venv\Scripts\Activate.ps1 not found. Continuing with the current Python environment."
} else {
    .\.venv\Scripts\Activate.ps1
}

if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    Write-Host "Ollama CLI not found in PATH. Install Ollama or add it to PATH before running experiments."
    exit 1
}

try {
    ollama list | Out-Host
} catch {
    Write-Host "Ollama is not reachable. Start the Ollama service and try again."
    exit 1
}

# Safety notes:
# - `--model-preset all` with all CSV rows and 3 runs can take a very long time.
# - Total generations = number_of_rows × number_of_models × runs.
# - Start with smoke_5rows or all_smoke before all_full.

New-Item -ItemType Directory -Force -Path $OUTPUT_ROOT | Out-Null
New-Item -ItemType Directory -Force -Path $BENCHMARK_ROOT | Out-Null

Write-Host "Starting CSV smoke test..."
# If your installed EventWeaver version does not support `--csv-max-rows`, comment out the next block.
Invoke-Experiment `
    -Header "Starting CSV smoke test..." `
    -Arguments @(
        "-m", "eventweaver", "all", $CSV_FILE,
        "--models", "qwen3:8b", "gemma3:4b",
        "--runs", "1",
        "--prompt-kind", "value-chain",
        "--csv-id-column", "Card ID",
        "--csv-title-column", "Descriptor of the value chain",
        "--csv-all-columns",
        "--csv-max-rows", "$MAX_ROWS_SMOKE",
        "--output-dir", "$OUTPUT_ROOT\smoke_5rows",
        "--outdir", "$BENCHMARK_ROOT\smoke_5rows",
        "--excel"
    ) `
    -BenchmarkDir "$BENCHMARK_ROOT\smoke_5rows"

Write-Host "Starting best-model CSV experiment..."
Invoke-Experiment `
    -Header "Starting best-model CSV experiment..." `
    -Arguments @(
        "-m", "eventweaver", "all", $CSV_FILE,
        "--models", "gemma3:12b", "qwen3:8b", "qwen3:14b", "gemma3:4b",
        "--runs", "$RUNS",
        "--prompt-kind", "value-chain",
        "--csv-id-column", "Card ID",
        "--csv-title-column", "Descriptor of the value chain",
        "--csv-all-columns",
        "--output-dir", "$OUTPUT_ROOT\best_auto",
        "--outdir", "$BENCHMARK_ROOT\best_auto",
        "--excel"
    ) `
    -BenchmarkDir "$BENCHMARK_ROOT\best_auto"

Write-Host "Starting all-model smoke test..."
Invoke-Experiment `
    -Header "Starting all-model smoke test..." `
    -Arguments @(
        "-m", "eventweaver", "all", $CSV_FILE,
        "--model-preset", "all",
        "--runs", "1",
        "--prompt-kind", "value-chain",
        "--csv-id-column", "Card ID",
        "--csv-title-column", "Descriptor of the value chain",
        "--csv-all-columns",
        "--output-dir", "$OUTPUT_ROOT\all_smoke",
        "--outdir", "$BENCHMARK_ROOT\all_smoke",
        "--excel"
    ) `
    -BenchmarkDir "$BENCHMARK_ROOT\all_smoke"

Write-Host "Starting all-model full CSV benchmark..."
Invoke-Experiment `
    -Header "Starting all-model full CSV benchmark..." `
    -Arguments @(
        "-m", "eventweaver", "all", $CSV_FILE,
        "--model-preset", "all",
        "--runs", "$RUNS",
        "--prompt-kind", "value-chain",
        "--csv-id-column", "Card ID",
        "--csv-title-column", "Descriptor of the value chain",
        "--csv-all-columns",
        "--output-dir", "$OUTPUT_ROOT\all_full",
        "--outdir", "$BENCHMARK_ROOT\all_full",
        "--excel"
    ) `
    -BenchmarkDir "$BENCHMARK_ROOT\all_full"

Write-Host "Starting fast-model comparison..."
Invoke-Experiment `
    -Header "Starting fast-model comparison..." `
    -Arguments @(
        "-m", "eventweaver", "all", $CSV_FILE,
        "--models", "gemma3:4b", "qwen3:8b", "mistral:7b", "phi4-mini:3.8b",
        "--runs", "$RUNS",
        "--prompt-kind", "value-chain",
        "--csv-id-column", "Card ID",
        "--csv-title-column", "Descriptor of the value chain",
        "--csv-all-columns",
        "--output-dir", "$OUTPUT_ROOT\fast",
        "--outdir", "$BENCHMARK_ROOT\fast",
        "--excel"
    ) `
    -BenchmarkDir "$BENCHMARK_ROOT\fast"

Write-Host "Starting larger-model comparison..."
Invoke-Experiment `
    -Header "Starting larger-model comparison..." `
    -Arguments @(
        "-m", "eventweaver", "all", $CSV_FILE,
        "--models", "gemma3:12b", "qwen3:14b", "qwen3:32b", "mistral-small3.2:24b", "phi4:14b",
        "--runs", "$RUNS",
        "--prompt-kind", "value-chain",
        "--csv-id-column", "Card ID",
        "--csv-title-column", "Descriptor of the value chain",
        "--csv-all-columns",
        "--output-dir", "$OUTPUT_ROOT\large",
        "--outdir", "$BENCHMARK_ROOT\large",
        "--excel"
    ) `
    -BenchmarkDir "$BENCHMARK_ROOT\large"

Write-Host "CSV experiments completed."
