# 창의공간 얼굴인식 출석체크 (exe 배포판) 설치 — install.bat 을 더블클릭하면 실행된다
#  파이썬 설치가 필요 없다. 이 폴더를 원하는 위치(예: C:\TheEclipseAttendance)에 둔 뒤 실행한다.
#  1) 인터넷에서 받은 파일 차단 해제
#  2) 얼굴 인식 모델 내려받기 (처음 한 번, 약 300MB)
#  3) 자가진단 (카메라 · 얼굴 인식 확인)
#  4) 매일 06:00 자동 실행 + 로그인 시 자동 실행 + 바탕화면 바로가기 (00:00 자동 종료)

$ErrorActionPreference = "Stop"
$AppDir = $PSScriptRoot
$Exe = Join-Path $AppDir "TheEclipseAttendance.exe"
$TaskName = "TheEclipseAttendance"
$ShortcutName = "창의공간 출석체크.lnk"

function Step($text) { Write-Host ""; Write-Host "▶ $text" -ForegroundColor Cyan }

Write-Host "=== 창의공간 얼굴인식 출석체크 설치 ===" -ForegroundColor Green
Write-Host "설치 위치: $AppDir"
if (-not (Test-Path $Exe)) { Write-Host "TheEclipseAttendance.exe 를 찾을 수 없습니다. zip 압축을 먼저 푸세요." -ForegroundColor Red; exit 1 }
if ($AppDir -like "*OneDrive*") {
    Write-Host "※ OneDrive 폴더 안에 설치하면 학생 DB가 동기화되면서 충돌할 수 있습니다. C:\TheEclipseAttendance 같은 곳을 권장합니다." -ForegroundColor Yellow
}

Step "인터넷에서 받은 파일 차단 해제"
Get-ChildItem $AppDir -Recurse -File | Unblock-File -ErrorAction SilentlyContinue

Step "얼굴 인식 모델 내려받기 (처음 한 번, 몇 분 걸릴 수 있습니다)"
Start-Process -FilePath $Exe -ArgumentList "--prepare" -WorkingDirectory $AppDir -Wait
Write-Host "완료"

Step "자가진단 — 카메라 앞에 앉아 10초 정도 기다려 주세요"
$selftest = Join-Path $AppDir "logs\selftest.txt"
Remove-Item $selftest -ErrorAction SilentlyContinue
Start-Process -FilePath $Exe -ArgumentList "--selftest" -WorkingDirectory $AppDir -Wait
if (Test-Path $selftest) {
    Get-Content $selftest -Encoding UTF8 | ForEach-Object { Write-Host "  $_" }
} else {
    Write-Host "  자가진단 결과가 없습니다. logs\app.log 를 확인하세요." -ForegroundColor Yellow
}

Step "매일 06:00 자동 실행 작업 등록"
$action = New-ScheduledTaskAction -Execute $Exe -Argument "--scheduled" -WorkingDirectory $AppDir
$trigger = New-ScheduledTaskTrigger -Daily -At "06:00"
# StartWhenAvailable: 06:00에 컴퓨터가 꺼져 있었으면 켜진 뒤 바로 실행한다
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit ([TimeSpan]::Zero)
Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings `
    -Description "창의공간 얼굴인식 출석체크 — 06:00 자동 실행 (00:00 자동 종료)" -Force | Out-Null
Write-Host "작업 스케줄러에 '$TaskName' 등록 완료"

Step "로그인할 때 자동 실행 + 바탕화면 바로가기"
$shell = New-Object -ComObject WScript.Shell
# 시작프로그램: 로그인하면 실행 (00:00~06:00이면 켜지지 않는다)
$startup = $shell.CreateShortcut((Join-Path ([Environment]::GetFolderPath("Startup")) $ShortcutName))
$startup.TargetPath = $Exe
$startup.Arguments = "--scheduled"
$startup.WorkingDirectory = $AppDir
$startup.Save()
# 바탕화면: 언제든 직접 실행 (운영 시간에 켰으면 자정에 종료)
$desktop = $shell.CreateShortcut((Join-Path ([Environment]::GetFolderPath("Desktop")) $ShortcutName))
$desktop.TargetPath = $Exe
$desktop.Arguments = "--kiosk"
$desktop.WorkingDirectory = $AppDir
$desktop.Save()
Write-Host "바탕화면에 '창의공간 출석체크' 바로가기를 만들었습니다."

Write-Host ""
Write-Host "=== 설치 완료 ===" -ForegroundColor Green
Write-Host "· 매일 06:00에 자동으로 켜지고, 00:00에 자동으로 꺼집니다."
Write-Host "· 지금 바로 실행하려면 바탕화면의 '창의공간 출석체크'를 더블클릭하세요."
Write-Host "· 기본 관리자 계정(1234 / 1234)은 운영 전에 꼭 바꾸세요."
