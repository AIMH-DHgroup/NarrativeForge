# NarrativeForge prompt-strategy experiment runner

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

$CASE_STUDIES = "case_studies"
$CSV_FILE = "data\MOVING_VCs_DATASET_FINAL_V2.csv"
$RUNS = 3

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

.\.venv\Scripts\Activate.ps1

Write-Host "Running DOCX prompt-strategy experiment..." -ForegroundColor Green
python -m eventweaver all $CASE_STUDIES `
  --models qwen3:8b gemma3:12b qwen3:14b `
  --runs $RUNS `
  --input-strategy auto `
  --prompt-kind cultural-heritage `
  --prompt-strategies standard short detailed strict event_focused faithfulness_first digital_heritage_focused `
  --output-dir outputs_docx_prompt_exp `
  --outdir benchmark_docx_prompt_exp `
  --excel
Stop-If-Failed "DOCX experiment"
Run-Report -RunsCsv "benchmark_docx_prompt_exp\nrs_runs.csv" -BenchDir "benchmark_docx_prompt_exp"
Write-Host "DOCX outputs: outputs_docx_prompt_exp" -ForegroundColor Cyan
Write-Host "DOCX benchmark: benchmark_docx_prompt_exp" -ForegroundColor Cyan

Write-Host "Running CSV prompt-strategy experiment..." -ForegroundColor Green
python -m eventweaver all $CSV_FILE `
  --models qwen3:8b gemma3:12b qwen3:14b `
  --runs $RUNS `
  --prompt-kind value-chain `
  --prompt-strategies standard numeric_aware field_coverage concise territorial_context innovation_focused `
  --csv-id-column "Card ID" `
  --csv-title-column "Descriptor of the value chain" `
  --csv-all-columns `
  --output-dir outputs_csv_prompt_exp `
  --outdir benchmark_csv_prompt_exp `
  --excel
Stop-If-Failed "CSV experiment"
Run-Report -RunsCsv "benchmark_csv_prompt_exp\nrs_runs.csv" -BenchDir "benchmark_csv_prompt_exp"
Write-Host "CSV outputs: outputs_csv_prompt_exp" -ForegroundColor Cyan
Write-Host "CSV benchmark: benchmark_csv_prompt_exp" -ForegroundColor Cyan

Write-Host "Prompt-strategy experiments completed." -ForegroundColor Green
