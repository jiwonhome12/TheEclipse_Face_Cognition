#!/usr/bin/env bash
# 창의공간 얼굴인식 출석체크 — 라즈베리파이 설치 스크립트
#
#   사용법 (라즈베리파이 터미널에서, 프로젝트 폴더 안에서):
#       bash raspberrypi/install.sh
#
#   1) 시스템 패키지 설치 — 파이썬, Tk(화면), 한글 글꼴, 빌드 도구
#   2) 가상환경(.venv)에 파이썬 패키지 설치 + 얼굴 인식 모델 미리 받기
#   3) 로그인하면 자동 실행 (매일 06:00에 켜지고 00:00에 스스로 종료) + 메뉴·바탕화면 바로가기
#
#   준비물: 64비트 라즈베리파이 OS(데스크톱 버전), 라즈베리파이 4/5 (메모리 4GB 이상 권장),
#           USB 웹캠 또는 라즈베리파이 카메라 모듈, 인터넷 연결
#   다시 실행해도 안전하다 (이미 설치된 것은 건너뛰거나 최신으로 맞춘다).
#   설치 도중 재부팅·전원 차단으로 끊겼어도 같은 명령으로 다시 실행하면 망가진 부분을 고치고 이어서 진행한다:
#     패키지 관리자 중단 상태 · 덜 만들어진 가상환경 · 덜 설치된 패키지 · 덜 받은 얼굴 모델 · 덜 복사된 출결 데이터
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$APP_DIR/.venv"
PY="$VENV/bin/python"
START="$APP_DIR/raspberrypi/start.sh"
APP_ID="the-eclipse-attendance"
APP_NAME="창의공간 출석체크"

step() { printf '\n\033[1;36m▶ %s\033[0m\n' "$*"; }
ok()   { printf '\033[1;32m✔ %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m⚠ %s\033[0m\n' "$*"; }
fail() { printf '\033[1;31m✖ %s\033[0m\n' "$*" >&2; exit 1; }

# 부팅 직후에는 시스템 자동 업데이트가 패키지 관리자를 잡고 있을 수 있어서, 풀릴 때까지 최대 10분 기다린다
apt_get() { sudo apt-get -o DPkg::Lock::Timeout=600 "$@"; }

# 배포판마다 이름이 다른 패키지(예: 데비안 13의 libglib2.0-0t64)는 후보 중 있는 것 하나를 설치한다
apt_install_any() {
    local pkg
    for pkg in "$@"; do
        if apt-cache show "$pkg" >/dev/null 2>&1; then
            apt_get install -y "$pkg" && return 0
        fi
    done
    warn "패키지를 찾지 못했습니다: $* (없어도 동작할 수 있어서 계속 진행합니다)"
}

echo "=== $APP_NAME 설치 (라즈베리파이) ==="
echo "설치 위치: $APP_DIR"

# ---------- 0. 환경 확인 ----------
step "환경 확인"
[ "$(uname -s)" = "Linux" ] || fail "라즈베리파이 OS(리눅스)에서 실행해 주세요."
[ "$(id -u)" -ne 0 ] || fail "sudo 없이 실행해 주세요: bash raspberrypi/install.sh (필요할 때만 비밀번호를 묻습니다)"
[ -f "$APP_DIR/attendance_app.py" ] || fail "attendance_app.py를 찾을 수 없습니다. 프로젝트 폴더 안의 raspberrypi/install.sh를 실행해 주세요."

# ---------- USB에서 실행한 경우: 홈 폴더로 복사한 뒤 거기서 설치를 이어간다 ----------
# USB 메모리(FAT32/exFAT/NTFS)는 실행 권한·심볼릭 링크를 지원하지 않아 가상환경(.venv)을 만들 수 없고,
# USB를 뽑으면 프로그램도 사라진다. 그래서 홈 폴더에 자리를 잡는다.
FS_TYPE="$(stat -f -c %T "$APP_DIR" 2>/dev/null || echo unknown)"
ON_USB=0
case "$APP_DIR" in /media/*|/mnt/*|/run/media/*) ON_USB=1 ;; esac
case "$FS_TYPE" in msdos|vfat|fat*|exfat|fuseblk|ntfs*) ON_USB=1 ;; esac
TARGET="$HOME/TheEclipse_Face_Cognition"
if [ "$ON_USB" = 1 ] && [ "$APP_DIR" != "$TARGET" ]; then
    step "USB에서 실행했습니다 — 홈 폴더로 복사합니다: $TARGET"
    mkdir -p "$TARGET/raspberrypi"
    # 프로그램 파일은 새 것으로 덮어쓴다
    for item in attendance_app.py requirements.txt README.md assets face_landmarker.task; do
        if [ -e "$APP_DIR/$item" ]; then cp -r "$APP_DIR/$item" "$TARGET/"; fi
    done
    cp "$APP_DIR"/raspberrypi/*.sh "$TARGET/raspberrypi/"
    # 윈도우에서 담는 과정에서 줄바꿈이 CRLF가 됐어도 실행되게 LF로 바꾼다
    sed -i 's/\r$//' "$TARGET"/raspberrypi/*.sh
    # 출결 데이터는 라즈베리파이에 이미 있으면 덮어쓰지 않는다 (업데이트할 때 여기서 쌓인 기록이 날아가지 않게)
    # 임시 이름으로 다 복사한 뒤에 이름을 바꾼다 — 복사 도중 재부팅돼도 반쪽짜리 faces.db가 '이미 있음'으로
    # 남아서 다시는 복사되지 않는 일이 없게 한다
    for item in faces.db attendance_photos; do
        rm -rf "$TARGET/.$item.copying"
        if [ -e "$APP_DIR/$item" ] && [ ! -e "$TARGET/$item" ]; then
            cp -r "$APP_DIR/$item" "$TARGET/.$item.copying"
            mv "$TARGET/.$item.copying" "$TARGET/$item"
            ok "출결 데이터 복사: $item"
        elif [ -e "$APP_DIR/$item" ]; then
            warn "라즈베리파이에 이미 $item 이(가) 있어서 USB의 것으로 덮어쓰지 않았습니다."
        fi
    done
    ok "복사 완료 — 설치를 이어갑니다 (USB는 설치가 끝날 때까지 꽂아 두세요)."
    # USB 위치를 넘겨서, USB에 담겨 온 얼굴 인식 모델을 인터넷 없이 바로 설치할 수 있게 한다
    exec env THE_ECLIPSE_SOURCE="$APP_DIR" bash "$TARGET/raspberrypi/install.sh" "$@"
fi

ARCH="$(uname -m)"
if [ "$ARCH" != "aarch64" ]; then
    warn "64비트 OS가 아닙니다 (현재: $ARCH). 얼굴 인식 엔진(onnxruntime)은 64비트 라즈베리파이 OS에서만 설치됩니다."
    read -r -p "그래도 계속할까요? [y/N] " answer || answer=""
    case "$answer" in y|Y) ;; *) exit 1 ;; esac
fi

MEM_MB=$(awk '/MemTotal/ {print int($2 / 1024)}' /proc/meminfo)
if [ "$MEM_MB" -lt 3000 ]; then
    warn "메모리가 ${MEM_MB}MB입니다. 얼굴 인식 모델이 커서 4GB 이상을 권장합니다 (2GB 이하는 매우 느리거나 멈출 수 있음)."
fi
FREE_MB=$(df -Pm "$APP_DIR" | awk 'NR == 2 {print $4}')
if [ "$FREE_MB" -lt 3000 ]; then
    fail "저장 공간이 부족합니다 (남은 공간 ${FREE_MB}MB). 패키지와 모델에 약 3GB가 필요합니다."
fi
OS_NAME="$( (. /etc/os-release 2>/dev/null && echo "${PRETTY_NAME:-리눅스}") || echo 리눅스)"
ok "$OS_NAME · $ARCH · 메모리 ${MEM_MB}MB · 남은 공간 ${FREE_MB}MB"

# ---------- 1. 시스템 패키지 ----------
step "시스템 패키지 설치 (sudo 비밀번호를 물어볼 수 있습니다)"
# 이전 설치가 재부팅으로 끊겼으면 패키지 관리자가 '중단됨' 상태로 남아 이후 설치가 전부 막힌다 — 먼저 마무리한다
sudo dpkg --configure -a
apt_get update
# python3-dev·build-essential·cmake: insightface를 라즈베리파이에서 직접 빌드할 때 필요
# python3-tk: 화면(Tkinter) · fonts-nanum: 화면과 카메라 위 이름 표시용 한글 글꼴
apt_get install -y python3 python3-venv python3-dev python3-tk \
    build-essential cmake pkg-config fonts-nanum v4l-utils
apt_install_any libopenblas-dev libopenblas0
apt_install_any libglib2.0-0t64 libglib2.0-0
apt_install_any libgl1 libgl1-mesa-glx
# 라즈베리파이 카메라 모듈(CSI)을 OpenCV로 열기 위한 libcamerify (USB 웹캠만 쓰면 없어도 된다)
apt_install_any libcamera-v4l2
fc-cache -f >/dev/null 2>&1 || true

python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' \
    || fail "파이썬 3.9 이상이 필요합니다 (현재 $(python3 --version 2>&1))."
ok "시스템 패키지 준비 완료 ($(python3 --version 2>&1))"

# 카메라 장치(/dev/video*)를 열려면 video 그룹이어야 한다 (라즈베리파이 기본 사용자는 이미 들어 있다)
ME="$(id -un)"
if ! id -nG "$ME" | grep -qw video; then
    sudo usermod -aG video "$ME"
    warn "카메라 사용 권한(video 그룹)을 추가했습니다. 설치가 끝나면 한 번 재부팅해 주세요."
fi

# ---------- 2. 파이썬 패키지 ----------
step "가상환경 준비 (.venv)"
# 가상환경을 만드는 도중에 끊기면 python은 있는데 pip가 없는 반쪽 상태가 된다 — 그러면 새로 만든다
if [ -x "$PY" ] && ! "$PY" -m pip --version >/dev/null 2>&1; then
    warn "이전 설치가 중간에 끊겨 가상환경이 망가져 있습니다. 새로 만듭니다."
    rm -rf "$VENV"
fi
[ -x "$PY" ] || python3 -m venv "$VENV"
"$PY" -m pip install --upgrade pip setuptools wheel

step "파이썬 패키지 설치 — insightface는 라즈베리파이에서 직접 빌드해서 10~30분 걸릴 수 있습니다"
# 화면은 Tk가 그리므로 OpenCV는 GUI 없는(headless) 버전으로 충분하고 더 가볍다
"$PY" -m pip install numpy Cython
"$PY" -m pip install opencv-python-headless pillow openpyxl onnxruntime
# 관리자 [학생 등록]의 구글폼 학생 명단(Sheets API, 서비스 계정) 연동
"$PY" -m pip install google-auth requests
"$PY" -m pip install insightface
ok "얼굴 인식 패키지 설치 완료"

step "눈 깜빡임 검사용 mediapipe 설치 (실패해도 출석체크는 동작합니다)"
# mediapipe를 그냥 설치하면 opencv-contrib-python이 같이 깔려 위의 OpenCV를 덮어쓸 수 있다.
# 그래서 --no-deps로 설치하고, 나머지 의존 패키지만 골라서 설치한다 (OpenCV·jax는 뺀다 — 쓰지 않는다).
MP_REQS="$(mktemp)"
trap 'rm -f "$MP_REQS"' EXIT
if "$PY" -m pip install --no-deps mediapipe \
    && "$PY" - "$MP_REQS" <<'EOF'
import re
import sys
from importlib.metadata import requires

skip = ("opencv", "jax", "jaxlib", "sentencepiece")
with open(sys.argv[1], "w", encoding="utf-8") as f:
    for req in requires("mediapipe") or []:
        spec, _, marker = req.partition(";")
        if "extra" in marker:
            continue
        name = re.split(r"[\s<>=!~\[]", spec.strip(), maxsplit=1)[0].lower()
        if not name.startswith(skip):
            f.write(req + "\n")
EOF
then
    if "$PY" -m pip install -r "$MP_REQS" && "$PY" -c "from mediapipe.tasks.python import vision" >/dev/null 2>&1; then
        ok "mediapipe 설치 완료"
    else
        warn "mediapipe를 불러오지 못했습니다. 눈 깜빡임 검사 없이 동작합니다 (위조판별은 그대로 사용)."
    fi
else
    warn "이 라즈베리파이용 mediapipe가 없습니다. 눈 깜빡임 검사 없이 동작합니다 (위조판별은 그대로 사용)."
fi

step "설치 확인"
check_packages() {
    "$PY" - <<'EOF'
import cv2, numpy, onnxruntime, insightface, PIL, tkinter, openpyxl  # noqa: F401
import google.auth, requests  # noqa: F401  (구글폼 명단 연동)
from insightface.app import FaceAnalysis  # noqa: F401  (의존 패키지까지 다 들어왔는지)
print(f"  OpenCV {cv2.__version__} · onnxruntime {onnxruntime.__version__} · insightface {insightface.__version__}")
EOF
}
if check_packages; then
    ok "파이썬 패키지 확인 완료"
else
    CHECK_CODE=$?
    if [ "$CHECK_CODE" -ge 128 ]; then
        # 불러오다 프로그램 자체가 죽었다(Illegal instruction 등) — 덜 설치된 게 아니라 CPU 호환성 문제다
        warn "패키지를 불러오다 프로그램이 멈췄습니다 (코드 $CHECK_CODE). 아래 'CPU 호환성 시험'에서 원인을 찾아 고칩니다."
    else
        # 설치 도중 끊기면 pip는 '설치됨'으로 기록했는데 파일이 덜 풀린 패키지가 남을 수 있다 — 강제로 다시 설치한다
        warn "덜 설치된 패키지가 있습니다 (이전 설치가 중간에 끊긴 경우). 다시 설치합니다 — 시간이 더 걸릴 수 있습니다."
        "$PY" -m pip install --force-reinstall numpy opencv-python-headless pillow openpyxl onnxruntime insightface \
            google-auth requests
    fi
fi

# ---------- 3. 모델 ----------
step "얼굴 인식 모델 확인"
# 윈도우 PC에서 USB에 모델을 같이 담아 왔으면(make_usb_bundle) 인터넷 없이 그걸 설치한다.
# 학교·회사 와이파이는 모델이 있는 GitHub 다운로드 서버를 막는 경우가 있다.
MODEL_HOME="$HOME/.insightface/models"
for src in "$APP_DIR/models/buffalo_l" ${THE_ECLIPSE_SOURCE:+"$THE_ECLIPSE_SOURCE/models/buffalo_l"}; do
    if [ -d "$src" ] && [ ! -d "$MODEL_HOME/buffalo_l" ]; then
        mkdir -p "$MODEL_HOME"
        rm -rf "$MODEL_HOME/.buffalo_l.copying"
        cp -r "$src" "$MODEL_HOME/.buffalo_l.copying"   # 다 복사한 뒤 이름을 바꿔서 반쪽 모델이 남지 않게
        mv "$MODEL_HOME/.buffalo_l.copying" "$MODEL_HOME/buffalo_l"
        ok "USB에 담겨 온 얼굴 인식 모델을 설치했습니다 (다운로드 생략)"
    fi
done
# insightface는 모델 폴더가 '있기만 하면' 다시 받지 않는다. 받거나 압축을 푸는 도중에 끊겼다면
# 반쪽짜리 모델이 영영 남아 얼굴 인식이 안 되므로, 파일이 빠졌거나 덜 받아졌으면 지우고 새로 받게 한다.
"$PY" - <<'EOF'
import os
import shutil

models = os.path.expanduser("~/.insightface/models")
model_dir = os.path.join(models, "buffalo_l")
expected = {  # 정상 파일 크기(바이트) — 이보다 작으면 덜 받아진 것
    "1k3d68.onnx": 143607619,
    "2d106det.onnx": 5030888,
    "det_10g.onnx": 16923827,
    "genderage.onnx": 1322532,
    "w600k_r50.onnx": 174383860,
}
if os.path.isdir(model_dir):
    broken = [name for name, size in expected.items()
              if not os.path.isfile(os.path.join(model_dir, name))
              or os.path.getsize(os.path.join(model_dir, name)) < size]
    if broken:
        shutil.rmtree(model_dir, ignore_errors=True)
        zip_path = model_dir + ".zip"
        if os.path.exists(zip_path):
            os.remove(zip_path)
        print(f"  ⚠ 덜 받아진 모델을 지웠습니다 ({', '.join(broken)}) — 새로 받습니다.")
    else:
        print("  모델이 이미 온전하게 있습니다.")
EOF

# ---------- 3-1. CPU 호환성 시험 ----------
# pip가 받아오는 최신 라이브러리는 더 새 CPU(ARMv8.2 이상) 기준으로 빌드된 경우가 있어서, 라즈베리파이에 따라
# 실행하는 순간 'Illegal instruction'으로 죽는다. 라이브러리를 하나씩 별도 프로세스에서 실제로 돌려 보고,
# 죽는 것이 있으면 그 라이브러리만 이전 버전으로 바꿔 가며 이 라즈베리파이에서 되는 버전을 찾는다.
step "이 라즈베리파이에서 실행되는지 라이브러리별로 시험 (CPU 호환성)"
mkdir -p "$APP_DIR/logs"
COMPAT_LOG="$APP_DIR/logs/compat.log"
BOARD="$(tr -d '\0' < /proc/device-tree/model 2>/dev/null || echo "알 수 없는 보드")"
{
    echo "=== $(date '+%F %T') 호환성 시험"
    echo "보드: $BOARD"
    grep -m1 -i '^Features' /proc/cpuinfo || true
} > "$COMPAT_LOG"
echo "  보드: $BOARD"

probe() {  # $1=파이썬 코드 — 별도 프로세스라서 CPU 오류로 죽어도 설치 스크립트는 계속된다
    (cd "$APP_DIR" && "$PY" -c "$1") >> "$COMPAT_LOG" 2>&1
}

older_versions() {  # $1=패키지, $2=series(기본)|all — 지금보다 오래된 버전들 (최신부터)
    # 1.25.1, 1.25.0, 1.24.4 ... 처럼 패치가 많아서 하나씩 내려가면 너무 오래 걸린다.
    # CPU 호환성은 보통 계열 단위로 바뀌므로 계열(1.25, 1.24 ...)마다 하나만 시험한다.
    # 버전이 몇 개 없는 패키지는 all로 전부 시험한다.
    "$PY" - "$1" "${2:-series}" <<'EOF' 2>> "$COMPAT_LOG"
import subprocess
import sys
from importlib.metadata import version
from pip._vendor.packaging.version import Version

pkg, mode = sys.argv[1], sys.argv[2]
current = Version(version(pkg))
out = subprocess.run([sys.executable, "-m", "pip", "index", "versions", pkg],
                     capture_output=True, text=True).stdout
line = next((l for l in out.splitlines() if l.startswith("Available versions:")), "")
best = {}
for text in line.split(":", 1)[-1].split(","):
    if not text.strip():
        continue
    v = Version(text.strip())
    series = (v.major, v.minor, v.micro) if mode == "all" else (v.major, v.minor)
    current_series = (current.major, current.minor, current.micro) if mode == "all" else (current.major, current.minor)
    if v < current and not v.is_prerelease and series != current_series:
        if series not in best or v > best[series]:
            best[series] = v
for v in sorted(best.values(), reverse=True):
    print(v)
EOF
}

repair_by_downgrade() {  # $1=패키지, $2=시험 코드, $3=series(기본)|all — 되는 버전을 찾으면 0
    local v tries=0
    for v in $(older_versions "$1" "${3:-series}"); do
        tries=$((tries + 1))
        [ "$tries" -le 15 ] || break
        echo "    → $1 $v 로 바꿔서 시험..."
        echo "--- $1 $v 시험" >> "$COMPAT_LOG"
        "$PY" -m pip install -q --no-deps "$1==$v" >> "$COMPAT_LOG" 2>&1 || continue
        if probe "$2"; then
            ok "$1 $v 에서 정상 동작합니다 — 이 버전을 씁니다"
            return 0
        fi
    done
    return 1
}

check_lib() {  # $1=보여줄 이름, $2=패키지, $3=시험 코드
    echo "--- $1" >> "$COMPAT_LOG"
    if probe "$3"; then
        echo "  ✔ $1"
        return 0
    fi
    warn "$1 이(가) 이 라즈베리파이에서 실행되지 않습니다 ($2 $("$PY" -c "from importlib.metadata import version; print(version('$2'))" 2>/dev/null)). 이전 버전으로 바꿔 봅니다."
    repair_by_downgrade "$2" "$3" \
        || fail "$1 을(를) 실행할 수 있는 버전을 찾지 못했습니다 ($BOARD). $COMPAT_LOG 내용을 알려 주세요."
}

check_lib "numpy (계산)" numpy \
    'import numpy as np; a = np.random.rand(256, 256); (a @ a).sum(); np.linalg.svd(a[:64, :64])'
check_lib "OpenCV (카메라·이미지)" opencv-python-headless \
    'import cv2, numpy as np; im = np.random.randint(0, 255, (480, 640, 3), np.uint8); cv2.resize(im, (320, 240)); cv2.cvtColor(im, cv2.COLOR_BGR2RGB); cv2.GaussianBlur(im, (5, 5), 0); cv2.warpAffine(im, np.float32([[1, 0, 5], [0, 1, 5]]), (112, 112))'
check_lib "scipy (얼굴 정렬 계산)" scipy \
    'import numpy as np, scipy.linalg; scipy.linalg.lstsq(np.random.rand(20, 5), np.random.rand(20))'
check_lib "scikit-image (얼굴 정렬)" scikit-image \
    'import numpy as np; from skimage import transform; transform.SimilarityTransform().estimate(np.random.rand(5, 2), np.random.rand(5, 2))'
# 얼굴 인식 엔진: 작은 시험 모델을 만들어 돌리고, 실제 얼굴 모델이 있으면 그것도 한 번씩 돌려 본다
check_lib "onnxruntime (얼굴 인식 엔진)" onnxruntime "$(cat <<'EOF'
import glob, os
import numpy as np
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

w = numpy_helper.from_array(np.random.rand(8, 3, 3, 3).astype(np.float32), "w")
g = numpy_helper.from_array(np.random.rand(8, 16).astype(np.float32), "g")
nodes = [helper.make_node("Conv", ["x", "w"], ["c"], pads=[1, 1, 1, 1]),
         helper.make_node("Relu", ["c"], ["r"]),
         helper.make_node("GlobalAveragePool", ["r"], ["p"]),
         helper.make_node("Flatten", ["p"], ["f"]),
         helper.make_node("Gemm", ["f", "g"], ["y"])]
graph = helper.make_graph(nodes, "probe",
                          [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 64, 64])],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 16])], [w, g])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 8  # 예전 버전 엔진도 읽을 수 있게
sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
sess.run(None, {"x": np.random.rand(1, 3, 64, 64).astype(np.float32)})

for path in sorted(glob.glob(os.path.expanduser("~/.insightface/models/buffalo_l/*.onnx"))):
    if not os.path.basename(path).startswith(("det_", "w600k")):
        continue  # 실제로 쓰는 검출·인식 모델만
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    inp = s.get_inputs()[0]
    shape = [d if isinstance(d, int) else 256 for d in inp.shape]
    shape[0] = 1
    s.run(None, {inp.name: np.random.rand(*shape).astype(np.float32)})
    print("ok", os.path.basename(path))
EOF
)"
# insightface 자체는 계산을 거의 안 하므로, 불러오다 죽으면 함께 쓰는 라이브러리 중 하나가 원인이다 — 그걸 찾아 바꾼다
echo "--- insightface" >> "$COMPAT_LOG"
if probe 'from insightface.app import FaceAnalysis'; then
    echo "  ✔ insightface (얼굴 인식)"
else
    warn "insightface를 불러오다 멈춥니다 — 함께 쓰는 라이브러리 중 원인을 찾습니다."
    for pair in sklearn:scikit-learn albumentations:albumentations albucore:albucore simsimd:simsimd \
                stringzilla:stringzilla matplotlib:matplotlib PIL:pillow onnx:onnx; do
        mod="${pair%%:*}"
        pkg="${pair##*:}"
        # 설치돼 있지 않은 것은 건너뛴다
        "$PY" -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('$mod') else 1)" 2>/dev/null || continue
        check_lib "$mod" "$pkg" "import $mod"
    done
    probe 'from insightface.app import FaceAnalysis' \
        || fail "insightface를 불러오지 못했습니다 ($BOARD). $COMPAT_LOG 내용을 알려 주세요."
    ok "insightface (얼굴 인식) 정상"
fi

# mediapipe(눈 깜빡임 검사) — 사진·화면으로 대리 출석하는 것을 막는 데 쓰므로, 안 되면 이전 버전을 먼저 찾아 본다.
# 버전이 몇 개 없어서(파이썬 3.13 + ARM은 1.0.0~1.1.x) 계열로 묶지 않고 하나씩 다 시험한다.
# 검사기를 '만드는 것'만이 아니라 실제 분석까지 돌려 본다 — 분석하는 순간에만 죽는 경우도 있다.
MP_PROBE="$(cat <<'EOF'
import sys
import numpy as np
import attendance_app as A
detector = A.create_blink_detector()
if detector is None:
    sys.exit(2)  # 모델 파일을 못 읽는 등 CPU와 무관한 문제
A.detect_blink_score(detector, np.zeros((480, 640, 3), np.uint8), 1)
print("blink ok")
EOF
)"
# 'import mediapipe'로 확인하면 불러오는 순간 죽는 경우를 '설치 안 됨'으로 착각하므로, 설치 여부는 pip로 본다
if "$PY" -m pip show mediapipe >/dev/null 2>&1; then
    echo "--- mediapipe" >> "$COMPAT_LOG"
    if probe "$MP_PROBE"; then
        echo "  ✔ mediapipe (눈 깜빡임 검사)"
    else
        MP_VER="$("$PY" -m pip show mediapipe 2>/dev/null | sed -n 's/^Version: //p')"
        warn "mediapipe $MP_VER 이(가) 이 라즈베리파이에서 실행되지 않습니다. 이전 버전으로 바꿔 봅니다."
        if ! repair_by_downgrade mediapipe "$MP_PROBE" all; then
            "$PY" -m pip uninstall -y mediapipe >> "$COMPAT_LOG" 2>&1 || true
            warn "이 라즈베리파이에서 되는 mediapipe 버전이 없어 뺐습니다."
            echo "    → 눈 깜빡임 검사는 mediapipe 대신 얼굴 인식 모델(106점 얼굴 윤곽)로 계속합니다."
        fi
    fi
fi
check_packages || fail "버전을 바꾼 뒤에도 패키지를 불러오지 못했습니다 ($BOARD). $COMPAT_LOG 내용을 알려 주세요."
ok "호환성 시험 완료 (자세한 기록: $COMPAT_LOG)"

step "얼굴 인식 모델 받기 (약 300MB, 처음 한 번만 — 몇 분 걸릴 수 있습니다)"
mkdir -p "$APP_DIR/logs"
PREPARE_LOG="$APP_DIR/logs/prepare.log"
set +e
(cd "$APP_DIR" && "$PY" attendance_app.py --prepare) 2>&1 | tee "$PREPARE_LOG"
PREPARE_CODE=${PIPESTATUS[0]}
set -e
if [ "$PREPARE_CODE" -ne 0 ]; then
    # 같은 '실패'라도 원인이 전혀 달라서, 종료 코드와 접속 시험으로 구분해서 알려준다
    echo
    case "$PREPARE_CODE" in
        137) fail "메모리가 부족해서 강제로 종료됐습니다 (코드 137). 다른 프로그램을 닫고 다시 실행하거나, 메모리 4GB 이상 라즈베리파이를 써 주세요." ;;
        132) fail "이 라즈베리파이 CPU에서 얼굴 인식 엔진이 실행되지 않습니다 (Illegal instruction, 코드 132). $PREPARE_LOG 내용을 알려 주세요." ;;
        139|134) fail "얼굴 인식 엔진을 불러오다 프로그램이 멈췄습니다 (코드 $PREPARE_CODE). $PREPARE_LOG 내용을 알려 주세요." ;;
    esac
    if command -v curl >/dev/null 2>&1 \
        && ! curl -sSfIL --max-time 20 -o /dev/null "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"; then
        fail "인터넷은 되지만 모델이 있는 GitHub 다운로드 서버에 접속하지 못했습니다 (학교·회사 와이파이에서 막는 경우가 많습니다).
   해결: 윈도우 PC에서 raspberrypi\\make_usb_bundle.bat 으로 USB를 다시 만들면 모델이 같이 담깁니다. 그 USB로 다시 설치하세요."
    fi
    fail "모델을 준비하지 못했습니다 (코드 $PREPARE_CODE). 위에 나온 오류나 $PREPARE_LOG 내용을 알려 주세요."
fi
ok "모델 준비 완료"

step "카메라 확인"
if ls /dev/video* >/dev/null 2>&1; then
    v4l2-ctl --list-devices 2>/dev/null | sed 's/^/  /' || true
    ok "카메라 장치가 보입니다"
else
    warn "카메라 장치(/dev/video*)가 보이지 않습니다. USB 웹캠을 꽂거나 카메라 모듈 연결을 확인해 주세요."
fi

# 라즈베리파이에서 USB 카메라가 갑자기 끊기는 가장 흔한 원인은 전원 부족이고, 그다음은 과열이다.
# vcgencmd get_throttled 값의 비트: 0x1/0x10000 전원 부족, 0x4/0x40000 성능 제한, 0x8/0x80000 고온
if command -v vcgencmd >/dev/null 2>&1; then
    THROTTLED="$(vcgencmd get_throttled 2>/dev/null | sed -n 's/^throttled=//p')"
    # 예상한 16진수 값일 때만 판단한다 — 다른 글자가 오면 산술식이 그것을 변수 이름으로 읽어
    # 틀린 결론을 내거나(set -u 때문에) 설치가 멈출 수 있다
    if [[ "$THROTTLED" =~ ^0x[0-9a-fA-F]+$ ]]; then
        UNDER_VOLTAGE=$(( THROTTLED & 0x10001 ))
        # 전원이 부족해도 성능 제한(0x4)이 걸리므로, 과열은 고온 기록(0x8)이 있거나
        # 전원 부족 없이 성능 제한만 걸린 경우(85℃ 이상)로 판단한다
        OVERHEAT=$(( (THROTTLED & 0x80008) || ((THROTTLED & 0x40004) && !UNDER_VOLTAGE) ))
        if (( UNDER_VOLTAGE )); then
            warn "전원 부족이 기록되어 있습니다 (get_throttled=$THROTTLED). USB 카메라가 끊기는 원인이 됩니다."
            echo "    → 라즈베리파이 공식 어댑터(라즈베리파이 4: 5V 3A, 5: 5V 5A)를 쓰고, 카메라 외 USB 기기는 빼 주세요."
        fi
        if (( OVERHEAT )); then
            warn "과열 기록이 있습니다 (get_throttled=$THROTTLED). 방열판이나 팬을 달아 주세요."
        fi
        if (( !UNDER_VOLTAGE && !OVERHEAT )); then
            ok "전원·온도 정상 (부팅 후 전원 부족·과열 기록 없음)"
        fi
    fi
fi

# ---------- 4. 자동 실행 · 바로가기 ----------
step "자동 실행 · 바로가기 등록"
chmod +x "$START" "$APP_DIR/raspberrypi/uninstall.sh"
mkdir -p "$APP_DIR/logs" "$HOME/.config/autostart" "$HOME/.local/share/applications"

write_desktop_entry() {  # $1=만들 파일, $2=start.sh에 넘길 인자
    cat > "$1" <<EOF
[Desktop Entry]
Type=Application
Name=$APP_NAME
Comment=The Eclipse 얼굴 인식 출석체크
Exec="$START" $2
Path=$APP_DIR
Icon=camera-web
Terminal=false
Categories=Utility;
EOF
    chmod +x "$1"
}

# 로그인하면 실행: 매일 06:00에 켜고, 프로그램은 자정에 스스로 종료한다 (start.sh --daemon 참고)
write_desktop_entry "$HOME/.config/autostart/$APP_ID.desktop" "--daemon"
# 메뉴와 바탕화면 바로가기: 지금 바로 한 번 실행
write_desktop_entry "$HOME/.local/share/applications/$APP_ID.desktop" ""
DESKTOP_DIR="$(xdg-user-dir DESKTOP 2>/dev/null || echo "$HOME/Desktop")"
if [ -d "$DESKTOP_DIR" ]; then
    cp "$HOME/.local/share/applications/$APP_ID.desktop" "$DESKTOP_DIR/$APP_ID.desktop"
    gio set "$DESKTOP_DIR/$APP_ID.desktop" metadata::trusted true >/dev/null 2>&1 || true
fi
ok "로그인 자동 실행 · 메뉴(보조 프로그램) · 바탕화면 바로가기 등록 완료"

# ---------- 끝 ----------
echo
printf '\033[1;32m=== 설치 완료 ===\033[0m\n'
echo "  · 설치 위치:        $APP_DIR  (USB는 이제 뽑아도 됩니다)"
echo "  · 다음 로그인부터 자동으로 켜집니다 (06:00~24:00 운영, 자정에 자동 종료 후 다음 날 06:00에 다시 켜짐)."
echo "  · 지금 바로 실행:   bash \"$APP_DIR/raspberrypi/start.sh\"  (또는 메뉴 → 보조 프로그램 → $APP_NAME)"
echo "  · 실행 기록:        $APP_DIR/logs/app.log"
echo "  · 자동 실행 해제:   bash \"$APP_DIR/raspberrypi/uninstall.sh\""
echo "  · 카메라 모듈(CSI)인데 화면이 안 나오면:  THE_ECLIPSE_CAMERA=csi bash \"$APP_DIR/raspberrypi/start.sh\""

if [ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]; then
    read -r -p "지금 바로 실행할까요? [Y/n] " answer || answer="n"
    case "$answer" in
        n|N) ;;
        *) nohup "$START" --daemon >/dev/null 2>&1 & ok "실행했습니다. 잠시 뒤 창이 뜹니다." ;;
    esac
else
    echo "  · 화면이 없는 접속(SSH)이라 지금은 실행하지 않았습니다. 재부팅하면 자동으로 켜집니다."
fi
