#!/usr/bin/env bash
# 창의공간 얼굴인식 출석체크 — 라즈베리파이 자동 실행·바로가기 해제
#   bash raspberrypi/uninstall.sh
# 출결 DB(faces.db)와 출퇴근 사진(attendance_photos)은 지우지 않는다.
set -u

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_ID="the-eclipse-attendance"

# 실행 중이면 끈다 (자동 실행 루프 먼저 — 안 그러면 프로그램을 끄자마자 다시 켠다)
pkill -f "$APP_DIR/raspberrypi/start.sh" 2>/dev/null || true
pkill -f "$APP_DIR/.venv/bin/python attendance_app.py" 2>/dev/null || true

rm -f "$HOME/.config/autostart/$APP_ID.desktop" \
      "$HOME/.local/share/applications/$APP_ID.desktop"
DESKTOP_DIR="$(xdg-user-dir DESKTOP 2>/dev/null || echo "$HOME/Desktop")"
rm -f "$DESKTOP_DIR/$APP_ID.desktop"
echo "✔ 자동 실행과 바로가기를 지웠습니다."

read -r -p "설치한 파이썬 패키지(.venv, 약 1~2GB)도 지울까요? [y/N] " answer || answer=""
case "$answer" in
    y|Y) rm -rf "$APP_DIR/.venv"; echo "✔ .venv를 지웠습니다." ;;
esac

echo "출결 기록(faces.db)과 출퇴근 사진(attendance_photos)은 그대로 두었습니다."
