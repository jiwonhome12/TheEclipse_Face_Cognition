# 창의공간 얼굴인식 출석체크 — 윈도우 설치 스크립트 (install.bat 을 더블클릭하면 실행된다)
#  1) 파이썬 확인 (없으면 winget으로 설치 제안)
#  2) 필요한 패키지 설치 + 얼굴 인식 모델 미리 받기
#  3) 매일 06:00 자동 실행 작업 등록 + 로그인할 때 자동 실행 + 바탕화면 바로가기
#  자정(00:00)이 되면 프로그램이 스스로 종료된다 (attendance_app.py --kiosk)

$ErrorActionPreference = "Stop"
$AppDir = Split-Path -Parent $PSScriptRoot
$StartScript = Join-Path $PSScriptRoot "start.ps1"
$PythonPathFile = Join-Path $PSScriptRoot "python_path.txt"
$TaskName = "TheEclipseAttendance"
$ShortcutName = "창의공간 출석체크.lnk"

function Step($text) { Write-Host ""; Write-Host "▶ $text" -ForegroundColor Cyan }

function Get-PythonExe {
    # py 런처를 먼저 쓰고, 없으면 PATH의 python을 쓴다
    $tries = @(
        { py -3 -c "import sys; print(sys.executable)" },
        { python -c "import sys; print(sys.executable)" }
    )
    foreach ($try in $tries) {
        try {
            $out = & $try 2>$null
            if ($LASTEXITCODE -eq 0 -and $out -and (Test-Path $out.Trim())) { return $out.Trim() }
        } catch {}
    }
    # winget으로 방금 설치했다면 PATH가 아직 갱신되지 않았을 수 있다
    $candidates = Get-ChildItem "$env:LOCALAPPDATA\Programs\Python\Python3*\python.exe" -ErrorAction SilentlyContinue |
        Sort-Object FullName -Descending
    if ($candidates) { return $candidates[0].FullName }
    return $null
}

Write-Host "=== 창의공간 얼굴인식 출석체크 설치 ===" -ForegroundColor Green
Write-Host "설치 위치: $AppDir"

# ---------- 1. 파이썬 ----------
Step "파이썬 확인"
$Python = Get-PythonExe
if (-not $Python) {
    Write-Host "파이썬이 설치되어 있지 않습니다."
    $answer = Read-Host "winget으로 Python 3.12를 설치할까요? (Y/N)"
    if ($answer -notmatch '^[Yy]') {
        Write-Host "https://www.python.org/downloads/ 에서 파이썬을 설치한 뒤 (설치 화면에서 'Add python.exe to PATH' 체크) 다시 실행하세요."
        exit 1
    }
    winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements
    $Python = Get-PythonExe
    if (-not $Python) { Write-Host "파이썬 설치를 확인하지 못했습니다. 창을 닫고 install.bat 을 다시 실행하세요." -ForegroundColor Red; exit 1 }
}
& $Python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)"
if ($LASTEXITCODE -ne 0) { Write-Host "파이썬 3.10 이상이 필요합니다: $Python" -ForegroundColor Red; exit 1 }
Write-Host "사용할 파이썬: $Python"
Set-Content -Path $PythonPathFile -Value $Python -Encoding UTF8

# ---------- 2. 패키지 + 모델 ----------
Step "필요한 패키지 설치 (처음이면 몇 분 걸립니다)"
& $Python -m pip install --upgrade pip
& $Python -m pip install -r (Join-Path $AppDir "requirements.txt")
if ($LASTEXITCODE -ne 0) { Write-Host "패키지 설치에 실패했습니다. 위 오류를 확인하세요." -ForegroundColor Red; exit 1 }

Step "얼굴 인식 모델 내려받기 (처음 한 번, 약 300MB)"
Push-Location $AppDir
try { & $Python attendance_app.py --prepare } finally { Pop-Location }
if ($LASTEXITCODE -ne 0) { Write-Host "모델 준비에 실패했습니다. 인터넷 연결을 확인하고 다시 실행하세요." -ForegroundColor Red; exit 1 }

# ---------- 3. 자동 실행 ----------
Step "매일 06:00 자동 실행 작업 등록"
$launch = "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$StartScript`""
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $launch -WorkingDirectory $AppDir
$trigger = New-ScheduledTaskTrigger -Daily -At "06:00"
# StartWhenAvailable: 06:00에 컴퓨터가 꺼져 있었으면 켜진 뒤 바로 실행한다
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings `
    -Description "창의공간 얼굴인식 출석체크 — 06:00 자동 실행 (00:00 자동 종료)" -Force | Out-Null
Write-Host "작업 스케줄러에 '$TaskName' 등록 완료"

Step "로그인할 때 자동 실행 + 바탕화면 바로가기"
$shell = New-Object -ComObject WScript.Shell
# 시작프로그램: 컴퓨터를 켜고 로그인하면 실행 (00:00~06:00이면 실행하지 않는다)
$startup = $shell.CreateShortcut((Join-Path ([Environment]::GetFolderPath("Startup")) $ShortcutName))
$startup.TargetPath = "powershell.exe"
$startup.Arguments = $launch
$startup.WorkingDirectory = $AppDir
$startup.WindowStyle = 7
$startup.Save()
# 바탕화면: 언제든 직접 실행
$desktop = $shell.CreateShortcut((Join-Path ([Environment]::GetFolderPath("Desktop")) $ShortcutName))
$desktop.TargetPath = "powershell.exe"
$desktop.Arguments = "$launch -Manual"
$desktop.WorkingDirectory = $AppDir
$desktop.WindowStyle = 7
$desktop.IconLocation = "$env:SystemRoot\System32\shell32.dll,138"
$desktop.Save()
Write-Host "바탕화면에 '창의공간 출석체크' 바로가기를 만들었습니다."

Write-Host ""
Write-Host "=== 설치 완료 ===" -ForegroundColor Green
Write-Host "· 매일 06:00에 자동으로 켜지고, 00:00에 자동으로 꺼집니다."
Write-Host "· 지금 바로 실행하려면 바탕화면의 '창의공간 출석체크'를 더블클릭하세요."
Write-Host "· 기본 관리자 계정(1234 / 1234)은 운영 전에 꼭 바꾸세요."
