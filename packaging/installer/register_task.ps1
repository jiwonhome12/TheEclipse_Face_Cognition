# 설치 프로그램이 부르는 스크립트 — 작업 스케줄러에 매일 06:00 자동 실행 + 15분마다 생존 확인(watchdog)을 등록한다.
# 이미 켜져 있으면 프로그램이 알아서 바로 종료(--scheduled)하므로 평소엔 아무 일도 하지 않고,
# 윈도우 업데이트 재부팅 등으로 꺼졌을 때만 15분 안에 다시 켠다. 00:00~06:00에는 켜지지 않는다.
param(
    [Parameter(Mandatory = $true)][string]$Exe,
    [Parameter(Mandatory = $true)][string]$WorkDir
)
$ErrorActionPreference = "Stop"
$TaskName = "TheEclipseAttendance"

$action = New-ScheduledTaskAction -Execute $Exe -Argument "--scheduled" -WorkingDirectory $WorkDir
$trigger = New-ScheduledTaskTrigger -Daily -At "06:00"
$trigger.Repetition = (New-ScheduledTaskTrigger -Once -At "06:00" `
    -RepetitionInterval (New-TimeSpan -Minutes 15) `
    -RepetitionDuration (New-TimeSpan -Hours 18)).Repetition
# StartWhenAvailable: 06:00에 컴퓨터가 꺼져 있었으면 켜진 뒤 바로 실행한다
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit ([TimeSpan]::Zero)
Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings `
    -Description "창의공간 얼굴인식 출석체크 - 06:00 자동 실행 + 15분마다 생존 확인 (00:00 자동 종료)" -Force | Out-Null
