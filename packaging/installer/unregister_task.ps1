# 제거 프로그램이 부르는 스크립트 — 자동 실행 작업을 지운다 (학생 DB·사진은 건드리지 않는다)
Unregister-ScheduledTask -TaskName "TheEclipseAttendance" -Confirm:$false -ErrorAction SilentlyContinue
