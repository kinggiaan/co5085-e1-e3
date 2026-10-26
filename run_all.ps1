# Chạy toàn bộ E1–E3 rồi in bảng tổng hợp.
#   .\run_all.ps1            # chạy hết (khoảng 15–20 phút trên GTX 1650)
#   .\run_all.ps1 -Only E1   # chỉ một bài
param(
    [ValidateSet("E1", "E2", "E3", "ALL")]
    [string]$Only = "ALL"
)

$ErrorActionPreference = "Stop"
$py = ".\.venv\Scripts\python.exe"

if (-not (Test-Path $py)) {
    Write-Host "Chưa có .venv. Chạy trước:" -ForegroundColor Yellow
    Write-Host "  python -m venv .venv; .\.venv\Scripts\activate; pip install -r requirements.txt"
    exit 1
}

Write-Host "== Kiểm tra môi trường ==" -ForegroundColor Cyan
& $py "scripts\smoke_test.py"

foreach ($ex in @("E1", "E2", "E3")) {
    if ($Only -ne "ALL" -and $Only -ne $ex) { continue }
    Write-Host "`n== $ex ==" -ForegroundColor Cyan
    & $py "$ex\run_all.py"
}

Write-Host "`n== Bảng tổng hợp ==" -ForegroundColor Cyan
foreach ($ex in @("E1", "E2", "E3")) {
    $p = "$ex\results\summary.md"
    if (Test-Path $p) {
        Write-Host "`n--- $ex ---"
        Get-Content $p -Encoding utf8
    } else {
        Write-Host "`n--- $ex --- (chưa có kết quả)"
    }
}
