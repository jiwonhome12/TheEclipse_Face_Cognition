#!/usr/bin/env bash
# 창의공간 얼굴인식 출석체크 — 라즈베리파이 실행 스크립트 (install.sh가 바로가기·자동 실행에 등록한다)
#
#   bash raspberrypi/start.sh            지금 한 번 실행 (메뉴·바탕화면 바로가기)
#   bash raspberrypi/start.sh --daemon   로그인할 때 자동 실행용
#        - 06:00~24:00이면 바로 켜고, 프로그램은 자정에 스스로 종료한다
#        - 정상 종료(자정 또는 직접 닫음)하면 다음 날 06:00까지 기다렸다가 다시 켠다
#        - 비정상 종료(오류)하면 잠시 뒤 다시 켠다
#
#   카메라: USB 웹캠이 있으면 그대로 쓰고, 없고 카메라 모듈(CSI)만 있으면 libcamerify로 감싸서 실행한다.
#           강제로 정하려면 THE_ECLIPSE_CAMERA=usb 또는 THE_ECLIPSE_CAMERA=csi 를 붙여 실행한다.
set -u

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$APP_DIR/.venv/bin/python"
LOG="$APP_DIR/logs/app.log"
mkdir -p "$APP_DIR/logs"
cd "$APP_DIR" || exit 1

if [ ! -x "$PY" ]; then
    echo "설치가 되어 있지 않습니다. 먼저 bash raspberrypi/install.sh 를 실행하세요." >&2
    exit 1
fi

# 라즈베리파이 카메라 모듈은 libcamera 전용이라 OpenCV(V4L2)로 바로 열리지 않는다 — libcamerify가 이어준다
camera_wrapper() {
    local mode="${THE_ECLIPSE_CAMERA:-auto}"
    if [ "$mode" = "auto" ]; then
        if v4l2-ctl --list-devices 2>/dev/null | grep -qi "usb"; then
            mode="usb"
        elif command -v libcamerify >/dev/null 2>&1; then
            mode="csi"
        else
            mode="usb"
        fi
    fi
    if [ "$mode" = "csi" ] && command -v libcamerify >/dev/null 2>&1; then
        echo "libcamerify"
    fi
}
WRAP="$(camera_wrapper)"

run_app() {  # 인자는 attendance_app.py에 그대로 넘긴다
    echo "[$(date '+%F %T')] 시작 $*${WRAP:+ (카메라 모듈: libcamerify)}" >> "$LOG"
    # $WRAP은 비어 있을 수 있어서 일부러 따옴표 없이 쓴다
    $WRAP "$PY" attendance_app.py "$@" >> "$LOG" 2>&1
    local code=$?
    echo "[$(date '+%F %T')] 종료 (코드 $code)" >> "$LOG"
    return "$code"
}

seconds_until_6am() {
    local now target
    now=$(date +%s)
    target=$(date -d "today 06:00" +%s)
    [ "$now" -lt "$target" ] || target=$(date -d "tomorrow 06:00" +%s)
    echo $(( target - now ))
}

if [ "${1:-}" != "--daemon" ]; then
    run_app "$@"
    exit $?
fi

crashes=0
while true; do
    if [ "$(date +%H)" -ge 6 ]; then
        run_app --scheduled
        if [ $? -ne 0 ]; then
            # 오류로 꺼졌다 — 잠시 뒤 다시 켠다. 계속 실패하면 간격을 늘려서 로그가 넘치지 않게 한다
            crashes=$(( crashes + 1 ))
            if [ "$crashes" -ge 5 ]; then sleep 300; else sleep 10; fi
            continue
        fi
    fi
    crashes=0
    # 정상 종료(자정 또는 직접 닫음) 또는 새벽 — 다음 06:00까지 기다린다
    sleep "$(seconds_until_6am)"
done
