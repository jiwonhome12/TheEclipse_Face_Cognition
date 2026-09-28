# 창의공간 얼굴인식 출석체크 실행 — 작업 스케줄러(06:00)·시작프로그램·바탕화면 바로가기가 이 파일을 부른다
#  -Manual : 바탕화면 바로가기로 직접 실행 (00:00~06:00에도 켜진다)
param([switch]$Manual)

$AppDir = Split-Path -Parent $PSScriptRoot

# 06:00~24:00이 운영 시간 (attendance_app.py 의 VALID_ATTENDANCE_START 와 같은 기준)
$inHours = (Get-Date).Hour -ge 6
if (-not $Manual -and -not $inHours) { exit 0 }

# 이미 켜져 있으면 하나 더 띄우지 않는다 (카메라를 두 프로그램이 동시에 못 쓴다)
$running = Get-CimInstance Win32_Process -Filter "Name LIKE 'python%'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -like "*attendance_app.py*" }
if ($running) { exit 0 }

# install.ps1 이 저장한 파이썬을 쓰고, 콘솔 창이 뜨지 않는 pythonw 로 실행한다
$pythonFile = Join-Path $PSScriptRoot "python_path.txt"
$python = if (Test-Path $pythonFile) { (Get-Content $pythonFile -Raw).Trim() } else { "python.exe" }
$pythonw = Join-Path (Split-Path $python -Parent) "pythonw.exe"
if (-not (Test-Path $pythonw)) { $pythonw = $python }

$arguments = @("`"$(Join-Path $AppDir 'attendance_app.py')`"")
if ($inHours) { $arguments += "--kiosk" }   # 운영 시간에 켰으면 자정에 스스로 종료
Start-Process -FilePath $pythonw -ArgumentList $arguments -WorkingDirectory $AppDir
