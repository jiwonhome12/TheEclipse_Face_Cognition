# GitHub에서 최신 버전을 받아온다 (git clone 으로 받은 경우). 학생 DB·사진은 git에 없으므로 그대로 남는다.
$AppDir = Split-Path -Parent $PSScriptRoot

if (-not (Test-Path (Join-Path $AppDir ".git"))) {
    Write-Host "ZIP으로 받은 폴더라 자동 업데이트를 할 수 없습니다."
    Write-Host "GitHub에서 새 ZIP을 받아 이 폴더에 덮어쓰세요 (faces.db 와 attendance_photos 폴더는 지우지 마세요)."
    exit 1
}
git -C $AppDir pull --ff-only
if ($LASTEXITCODE -ne 0) { Write-Host "업데이트에 실패했습니다. 위 메시지를 확인하세요." -ForegroundColor Red; exit 1 }

$pythonFile = Join-Path $PSScriptRoot "python_path.txt"
if (Test-Path $pythonFile) {
    $python = (Get-Content $pythonFile -Raw).Trim()
    & $python -m pip install -r (Join-Path $AppDir "requirements.txt")
}
Write-Host "업데이트 완료. 프로그램을 껐다가 다시 켜면 새 버전이 적용됩니다." -ForegroundColor Green
