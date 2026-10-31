# Chạy qua đêm: A1 (6 cấu hình) rồi A2 (U-Net + DeepLabV3), ghi log vào _overnight_log.txt
#   .\run_overnight.ps1
#   .\run_overnight.ps1 -Only A2      # chỉ chạy A2
param(
    [ValidateSet("A1", "A2", "ALL")]
    [string]$Only = "ALL"
)

$ErrorActionPreference = "Continue"
$py = ".\.venv\Scripts\python.exe"
$log = "_overnight_log.txt"
$env:PYTHONIOENCODING = "utf-8"

function Log($msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $msg
    Write-Host $line
    Add-Content -Path $log -Value $line -Encoding utf8
}

if (-not (Test-Path $py)) { Write-Host "Chưa có .venv"; exit 1 }

if ($Only -in @("A1", "ALL") -and -not (Test-Path "A1\results\summary.json")) {
    Log "bắt đầu A1 (6 cấu hình, ~3 h)"
    & $py "A1\run_all.py" --limit 20000 *>&1 | Tee-Object -FilePath "_a1_log.txt" | Out-Null
    Log "A1 xong (exit $LASTEXITCODE)"
} else {
    Log "bỏ qua A1 (đã có summary.json hoặc -Only A2)"
}

if ($Only -in @("A2", "ALL") -and -not (Test-Path "A2\results\summary.json")) {
    Log "bắt đầu A2 (U-Net + DeepLabV3, ~2 h)"
    & $py "A2\run_all.py" *>&1 | Tee-Object -FilePath "_a2_log.txt" | Out-Null
    Log "A2 xong (exit $LASTEXITCODE)"
} else {
    Log "bỏ qua A2 (đã có summary.json)"
}

Log "hoàn tất. Kết quả: A1\results\summary.md, A2\results\summary.md"
