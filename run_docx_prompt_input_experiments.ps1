# NarrativeForge DOCX prompt/input strategy experiment runner

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

$CASE_STUDIES = "case_studies"
$RUNS = 3

$INPUT_STRATEGIES = @("auto", "full", "brief", "rag")
$PROMPT_STRATEGIES = @("standard", "short", "detailed")

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

foreach ($inputStrategy in $INPUT_STRATEGIES) {
    $outputDir = "outputs_docx_prompt_input_$inputStrategy"
    $benchDir = "benchmark_docx_prompt_input_$inputStrategy"

    Write-Host "Running DOCX experiment for input strategy: $inputStrategy" -ForegroundColor Green

    python -m eventweaver all $CASE_STUDIES `
      --model-preset all `
      --runs $RUNS `
      --input-strategy $inputStrategy `
      --prompt-kind cultural-heritage `
      --prompt-strategies $PROMPT_STRATEGIES `
      --output-dir $outputDir `
      --outdir $benchDir `
      --excel
    Stop-If-Failed "DOCX experiment ($inputStrategy)"

    Run-Report -RunsCsv "$benchDir\nrs_runs.csv" -BenchDir $benchDir

    Write-Host "DOCX outputs: $outputDir" -ForegroundColor Cyan
    Write-Host "DOCX benchmark: $benchDir" -ForegroundColor Cyan
}

Write-Host "DOCX prompt/input strategy experiments completed." -ForegroundColor Green
