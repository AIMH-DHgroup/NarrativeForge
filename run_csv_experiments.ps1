# NarrativeForge CSV-only experiment runner
# Purpose:
# - Generate one narrative for each CSV row
# - Use all installed models through --model-preset all
# - Evaluate the generated narratives
# - Print the benchmark metrics in the PowerShell terminal
#
# Run from the NarrativeForge project root:
# .\run_csv_all_models_experiment.ps1

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# -----------------------------
# Configuration
# -----------------------------

$CSV_FILE = "data\MOVING_VCs_DATASET_FINAL_V2.csv"
$RUNS = 3

$OUTPUT_DIR = "outputs_csv_all_models"
$BENCHMARK_DIR = "benchmark_csv_all_models"

$CSV_ID_COLUMN = "Card ID"
$CSV_TITLE_COLUMN = "Descriptor of the value chain"

# Optional: set to a positive number for a quick test.
# Use 0 or comment out --csv-max-rows for all rows.
$CSV_MAX_ROWS = 5

# -----------------------------
# Helper functions
# -----------------------------

function Stop-If-Failed {
    param (
        [string]$StepName
    )

    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "ERROR: Step failed: $StepName" -ForegroundColor Red
        Write-Host "Exit code: $LASTEXITCODE" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

function Find-RunsCsv {
    param (
        [string]$BenchmarkDir
    )

    $possibleFiles = @(
        "$BenchmarkDir\nrs_runs.csv",
        "$BenchmarkDir\csv_nrs_runs.csv",
        "$BenchmarkDir\row_nrs_runs.csv"
    )

    foreach ($file in $possibleFiles) {
        if (Test-Path $file) {
            return $file
        }
    }

    return $null
}

function Find-CaseSummaryCsv {
    param (
        [string]$BenchmarkDir
    )

    $possibleFiles = @(
        "$BenchmarkDir\nrs_case_method_summary.csv",
        "$BenchmarkDir\csv_nrs_case_method_summary.csv",
        "$BenchmarkDir\row_nrs_case_method_summary.csv"
    )

    foreach ($file in $possibleFiles) {
        if (Test-Path $file) {
            return $file
        }
    }

    return $null
}

function Find-ModelSummaryCsv {
    param (
        [string]$BenchmarkDir
    )

    $possibleFiles = @(
        "$BenchmarkDir\nrs_model_overall_summary.csv",
        "$BenchmarkDir\csv_nrs_model_overall_summary.csv",
        "$BenchmarkDir\row_nrs_model_overall_summary.csv"
    )

    foreach ($file in $possibleFiles) {
        if (Test-Path $file) {
            return $file
        }
    }

    return $null
}

function Print-CsvPreview {
    param (
        [string]$Path,
        [string]$Title,
        [int]$Rows = 20
    )

    if (-not (Test-Path $Path)) {
        Write-Host "Missing file: $Path" -ForegroundColor Yellow
        return
    }

    Write-Host ""
    Write-Host "============================================================"
    Write-Host $Title -ForegroundColor Cyan
    Write-Host "File: $Path"
    Write-Host "============================================================"

    try {
        Import-Csv $Path | Select-Object -First $Rows | Format-Table -AutoSize
    }
    catch {
        Write-Host "Could not format CSV as table. Printing first lines instead." -ForegroundColor Yellow
        Get-Content $Path -TotalCount ($Rows + 1)
    }
}

function Print-KeyMetrics {
    param (
        [string]$BenchmarkDir
    )

    $runsCsv = Find-RunsCsv $BenchmarkDir
    $caseSummaryCsv = Find-CaseSummaryCsv $BenchmarkDir
    $modelSummaryCsv = Find-ModelSummaryCsv $BenchmarkDir

    if ($null -eq $runsCsv) {
        Write-Host "No runs CSV found in $BenchmarkDir" -ForegroundColor Yellow
        return
    }

    Write-Host ""
    Write-Host "Generating summary and visualizations..." -ForegroundColor Green

    python -m eventweaver summarize $runsCsv --outdir $BenchmarkDir
    Stop-If-Failed "summarize"

    python -m eventweaver visualize $runsCsv --outdir $BenchmarkDir
    Stop-If-Failed "visualize"

    # Re-detect summaries after summarize command
    $caseSummaryCsv = Find-CaseSummaryCsv $BenchmarkDir
    $modelSummaryCsv = Find-ModelSummaryCsv $BenchmarkDir

    Print-CsvPreview -Path $runsCsv -Title "PER-RUN METRICS PREVIEW" -Rows 10

    if ($null -ne $caseSummaryCsv) {
        Print-CsvPreview -Path $caseSummaryCsv -Title "CASE / ROW + MODEL SUMMARY" -Rows 20
    }

    if ($null -ne $modelSummaryCsv) {
        Print-CsvPreview -Path $modelSummaryCsv -Title "OVERALL MODEL SUMMARY" -Rows 30
    }

    Write-Host ""
    Write-Host "Main metrics to inspect:" -ForegroundColor Green
    Write-Host "- NRS or CSV_NRS / Row_NRS"
    Write-Host "- NRS_no_R or CSV_NRS_no_R if only one run is available"
    Write-Host "- mean_bertscore_f1"
    Write-Host "- mean_semantic_similarity"
    Write-Host "- field_coverage, if CSV-specific scoring is implemented"
    Write-Host "- format_score, if CSV-specific scoring is implemented"
    Write-Host "- R, R_stab, R_struct, R_fail"
    Write-Host "- mean_runtime_seconds"
    Write-Host "- mean_word_count"
    Write-Host "- mean_paragraph_count"
    Write-Host "- failed_rate"
    Write-Host "- broken_sentence_count"
    Write-Host "- forbidden_formatting_count"
}

# -----------------------------
# Pre-flight checks
# -----------------------------

Write-Host "NarrativeForge CSV all-model experiment" -ForegroundColor Green
Write-Host "CSV file: $CSV_FILE"
Write-Host "Runs per model: $RUNS"
Write-Host "Output directory: $OUTPUT_DIR"
Write-Host "Benchmark directory: $BENCHMARK_DIR"

if (-not (Test-Path $CSV_FILE)) {
    Write-Host ""
    Write-Host "ERROR: CSV file not found: $CSV_FILE" -ForegroundColor Red
    Write-Host "Edit `$CSV_FILE at the top of this script."
    exit 1
}

if (-not (Test-Path ".\.venv\Scripts\Activate.ps1")) {
    Write-Host ""
    Write-Host "ERROR: Virtual environment not found at .\.venv" -ForegroundColor Red
    Write-Host "Create it first:"
    Write-Host "python -m venv .venv"
    Write-Host ".\.venv\Scripts\Activate.ps1"
    Write-Host "pip install -r requirements.txt"
    exit 1
}

Write-Host ""
Write-Host "Activating virtual environment..." -ForegroundColor Green
.\.venv\Scripts\Activate.ps1

Write-Host ""
Write-Host "Checking Ollama..." -ForegroundColor Green
ollama list
Stop-If-Failed "ollama list"

Write-Host ""
Write-Host "Optional dependency check..." -ForegroundColor Green
Write-Host "Installing optional metric/report packages. Comment this line if you do not want it."
pip install bert-score sentence-transformers pandas openpyxl matplotlib
Stop-If-Failed "optional dependency installation"

# -----------------------------
# Experiment
# -----------------------------

Write-Host ""
Write-Host "============================================================"
Write-Host "STARTING CSV EXPERIMENT WITH ALL MODELS" -ForegroundColor Cyan
Write-Host "============================================================"
Write-Host "Warning: this can take a long time."
Write-Host "Number of generations = CSV rows x models in preset 'all' x runs."
Write-Host ""

if ($CSV_MAX_ROWS -gt 0) {
    Write-Host "Processing only first $CSV_MAX_ROWS rows." -ForegroundColor Yellow

    python -m eventweaver all $CSV_FILE `
      --model-preset all `
      --runs $RUNS `
      --prompt-kind value-chain `
      --csv-id-column $CSV_ID_COLUMN `
      --csv-title-column $CSV_TITLE_COLUMN `
      --csv-all-columns `
      --csv-max-rows $CSV_MAX_ROWS `
      --output-dir $OUTPUT_DIR `
      --outdir $BENCHMARK_DIR `
      --excel
}
else {
    Write-Host "Processing all CSV rows." -ForegroundColor Yellow

    python -m eventweaver all $CSV_FILE `
      --model-preset all `
      --runs $RUNS `
      --prompt-kind value-chain `
      --csv-id-column $CSV_ID_COLUMN `
      --csv-title-column $CSV_TITLE_COLUMN `
      --csv-all-columns `
      --output-dir $OUTPUT_DIR `
      --outdir $BENCHMARK_DIR `
      --excel
}

Stop-If-Failed "CSV all-model generation and evaluation"

# -----------------------------
# Print metrics
# -----------------------------

Print-KeyMetrics -BenchmarkDir $BENCHMARK_DIR

Write-Host ""
Write-Host "============================================================"
Write-Host "CSV EXPERIMENT COMPLETED" -ForegroundColor Green
Write-Host "============================================================"
Write-Host "Outputs: $OUTPUT_DIR"
Write-Host "Benchmark results: $BENCHMARK_DIR"