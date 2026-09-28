# 자동 실행(작업 스케줄러 · 시작프로그램)과 바탕화면 바로가기를 지운다.
# 이 폴더, 학생 DB(faces.db), 출퇴근 사진(attendance_photos)은 그대로 둔다.
$TaskName = "TheEclipseAttendance"
$ShortcutName = "창의공간 출석체크.lnk"

Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
foreach ($folder in @([Environment]::GetFolderPath("Startup"), [Environment]::GetFolderPath("Desktop"))) {
    Remove-Item (Join-Path $folder $ShortcutName) -ErrorAction SilentlyContinue
}
Write-Host "자동 실행과 바로가기를 지웠습니다. (프로그램 폴더와 출석 데이터는 그대로입니다)"
