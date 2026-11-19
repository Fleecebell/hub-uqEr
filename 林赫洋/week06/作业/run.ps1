<#
Run this assignment with the py312 interpreter.

Usage:
    .\run.ps1 tfidf\classify_tfidf.py
    .\run.ps1 bert\finetune.py --model macbert --pool max

NOTE: keep this file ASCII-only -- Windows PowerShell 5.1 reads .ps1 as GBK
      unless it has a UTF-8 BOM, and Chinese text would be garbled.
#>
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Rest
)

$py = "D:\Miniconda3\envs\py312\python.exe"
if (-not (Test-Path $py)) {
    Write-Error "py312 not found: $py  (edit `$py in run.ps1)"
    exit 1
}
& $py @Rest
exit $LASTEXITCODE
