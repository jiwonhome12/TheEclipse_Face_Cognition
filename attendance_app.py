import cv2
import numpy as np
import threading
import time
import sqlite3
import json
import os
import shutil
import sys
import multiprocessing
import calendar
import math
from datetime import date, datetime, timedelta
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkinter import font as tkfont
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageTk, ImageSequence

# 프로그램 파일이 있는 폴더 — 어느 폴더에서 실행해도 같은 DB/모델을 쓰도록 기준으로 삼는다
# exe(PyInstaller)로 실행하면 데이터(DB·사진·로그)는 exe 옆에, 같이 묶은 파일은 _internal 폴더에 있다.
IS_FROZEN = getattr(sys, "frozen", False)
if IS_FROZEN:
    APP_DIR = os.path.dirname(os.path.abspath(sys.executable))
    RESOURCE_DIR = getattr(sys, "_MEIPASS", APP_DIR)
else:
    APP_DIR = os.path.dirname(os.path.abspath(__file__))
    RESOURCE_DIR = APP_DIR

# 콘솔 없이(pythonw, 작업 스케줄러) 실행하면 sys.stdout/stderr가 None이라 모델 로딩 중
# 진행 표시 등이 출력하다 죽을 수 있다. 이때는 logs/app.log 파일로 남긴다.
# (얼굴 인식 프로세스도 이 파일을 다시 불러오므로 거기에도 똑같이 적용된다)
if sys.stdout is None or sys.stderr is None:
    os.makedirs(os.path.join(APP_DIR, "logs"), exist_ok=True)
    _log_file = open(os.path.join(APP_DIR, "logs", "app.log"), "a", encoding="utf-8", buffering=1)
    sys.stdout = sys.stdout or _log_file
    sys.stderr = sys.stderr or _log_file

DB_PATH = os.path.join(APP_DIR, "faces.db")
THRESHOLD = 0.50
DEPARTMENT_NAME = "소프트웨어전공"  # 학과는 한 곳뿐이라 입력받지 않고 이 값으로 고정한다

# 사진/화면 영상으로 대리 출석하는 것을 막기 위한 눈 깜빡임 검사 (MediaPipe FaceLandmarker)
# 모델 파일이 없으면 검사는 자동으로 꺼지고, 출퇴근은 예전처럼 그대로 동작한다.
LIVENESS_MODEL_PATH = os.path.join(RESOURCE_DIR, "face_landmarker.task")
LIVENESS_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
BLINK_CLOSED_SCORE = 0.5    # 이 값을 넘으면 눈을 감은 것으로 본다
BLINK_OPEN_SCORE = 0.3      # 다시 이 아래로 내려오면 한 번 깜빡인 것으로 센다
BLINK_VALID_SECONDS = 10    # 출퇴근을 누르기 전 이 시간 안에 깜빡임이 있어야 한다

# 출근/퇴근 확인 중 "눈을 깜빡여 주세요" 안내가 뜰 때 같이 보여주는 힌트 이미지.
# 파일이 없어도 이미지만 안 뜰 뿐 출퇴근은 그대로 동작한다.
BLINK_HINT_IMAGE_PATH = os.path.join(RESOURCE_DIR, "assets", "blink_hint.gif")

# InsightFace 공식 위조판별(liveness) 애드온 기준 점수. 이 값 이상이면 실제 사람으로 본다.
# 모델은 처음 실행할 때 ~/.insightface/addons/liveness.onnx 로 자동 다운로드된다.
LIVENESS_THRESHOLD = 0.8
LIVENESS_VALID_SECONDS = 3  # 출퇴근을 누르기 전 이 시간 안에 '실제 사람' 판정이 있어야 한다

# 출근/퇴근 버튼을 누른 뒤의 흐름
SESSION_TIMEOUT = 30           # 출근/퇴근을 누른 뒤 이 시간(초) 안에 얼굴이 확인되지 않으면 메인 화면으로 돌아간다
TEST_TIMEOUT = 60              # 카메라 테스트 화면도 켜둔 채 잊지 않도록 이 시간이 지나면 메인 화면으로 돌아간다
RECOGNIZE_HOLD_SECONDS = 0.6   # 같은 사람으로 이 시간 동안 계속 인식돼야 기록한다 (지나가는 사람 오인식 방지)
RESULT_POPUP_SECONDS = 3       # "OOO 학생 출근 되었습니다" 안내창이 떠 있는 시간
STATS_POPUP_SECONDS = 20       # [출석 조회] 창이 자동으로 닫히는 시간 (뒤 사람이 보지 않도록)

# 출퇴근할 때 찍힌 얼굴 사진 — 관리자 화면에서만 볼 수 있고, 보관 기간이 지나면 자동으로 지운다
PHOTO_DIR = os.path.join(APP_DIR, "attendance_photos")
PHOTO_WIDTH = 320              # 640x480 원본을 320x240으로 줄여서 저장 (장당 약 10~15KB)
PHOTO_JPEG_QUALITY = 70
PHOTO_RETENTION_DAYS = 90      # 3개월
RIGHT_PANEL_WIDTH = 620  # 우측 UI 패널 폭 — 1600x900 기준 값이고, 실제로는 px()로 화면 배율을 곱해서 쓴다

# 기준 창 크기(창공시스템과 동일). 모니터가 이보다 작으면 UI_SCALE만큼 전체를 줄여서 띄운다.
BASE_WIN_W, BASE_WIN_H = 1600, 900
UI_SCALE = 1.0

# 주간 출석 기준 — 월요일~일요일 동안 채워야 하는 시간/일수
WEEKLY_REQUIRED_SECONDS = 20 * 3600
WEEKLY_REQUIRED_DAYS = 3
# 00:00~06:00 사이에 찍은 출근/퇴근은 무효. 06:00~24:00 사이 기록만 인정한다.
VALID_ATTENDANCE_START = "06:00:00"

# ==========================================
# 윈도우 / macOS 공통 실행을 위한 플랫폼별 설정
# ==========================================
IS_MAC = sys.platform == "darwin"
IS_WINDOWS = sys.platform.startswith("win")

# 화면 글꼴 — 맑은 고딕은 윈도우에만 있어서 macOS에서는 기본 한글 글꼴(애플 SD 산돌고딕 Neo)을 쓴다
UI_FONT = "Apple SD Gothic Neo" if IS_MAC else "맑은 고딕"

# macOS의 Tk는 사진을 화면에 올리는 속도(PhotoImage.paste)가 윈도우보다 3~4배 느리다
# (944x708 기준 윈도우 9ms / 맥 30ms 이상). 카메라 원본이 640x480이라 크게 늘려도 화질은 그대로이므로,
# 맥에서는 표시 배율에 상한을 둬서 UI 스레드가 그리기에 묶이지 않게 한다. (736x552 ≈ 14ms)
MAX_DISPLAY_SCALE = 1.15 if IS_MAC else None


def inference_providers():
    """
    onnxruntime 실행 장치. macOS에서는 CoreML을 먼저 쓴다.
    (i9-9880H 측정: 얼굴 1명 기준 CPU 85ms → CoreML 36ms, 임베딩 코사인 유사도 0.99999 로 결과 동일)
    """
    import onnxruntime
    available = onnxruntime.get_available_providers()
    providers = []
    if IS_MAC and "CoreMLExecutionProvider" in available:
        providers.append("CoreMLExecutionProvider")
    providers.append("CPUExecutionProvider")
    return providers


# 얼굴 인식에 쓸 CPU 스레드 수. 12스레드(6코어)에서 측정: 6개면 화면이 27fps로 끊기고 4개면 29fps.
INFERENCE_THREADS = max(2, (os.cpu_count() or 4) // 3)


# ==========================================
# 0-0. 얼굴 인식 전용 프로세스
# ==========================================
# 같은 프로세스의 스레드에서 추론을 돌리면 파이썬 GIL과 CPU를 화면 그리기와 나눠 써서
# 카메라 화면이 뚝뚝 끊겼다(측정: 인식 끄면 29.6fps, 켜면 25~27fps + 50ms 이상 멈춤 다수).
# 별도 프로세스로 분리하면 UI 프로세스는 그리기만 한다.
class DetectedFace:
    __slots__ = ("bbox", "embedding")

    def __init__(self, bbox, embedding):
        self.bbox = bbox
        self.embedding = embedding


def download_liveness_model():
    """
    깜빡임 검사용 모델(face_landmarker.task, 약 3.7MB)을 내려받는다.
    인터넷이 없거나 실패하면 False를 돌려주고, 이때는 깜빡임 검사 없이 프로그램이 그대로 동작한다.
    """
    import urllib.request

    temp_path = LIVENESS_MODEL_PATH + ".part"
    try:
        print("[출석] 깜빡임 검사 모델을 내려받는 중입니다... (최초 1회)")
        urllib.request.urlretrieve(LIVENESS_MODEL_URL, temp_path)
        os.replace(temp_path, LIVENESS_MODEL_PATH)
        print("[출석] 깜빡임 검사 모델 준비 완료")
        return True
    except Exception as e:
        print(f"[출석] 깜빡임 검사 모델을 받지 못했습니다 ({e}). 깜빡임 검사 없이 실행합니다.")
        try:
            os.remove(temp_path)
        except OSError:
            pass
        return False


def create_blink_detector():
    """
    MediaPipe FaceLandmarker를 준비한다. 눈 깜빡임 정도(blendshape)를 읽어
    사진/화면으로 얼굴을 들이대는 것을 걸러내는 데 쓴다.
    모델 파일이 없거나 MediaPipe를 못 불러오면 None을 돌려주고, 이 경우 깜빡임 검사는 꺼진다.
    """
    if not os.path.exists(LIVENESS_MODEL_PATH):
        # 다른 PC에서 처음 실행할 때 모델 파일이 없으면 한 번 내려받는다 (실패하면 검사만 꺼진다)
        if not download_liveness_model():
            return None
    try:
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision
        # 경로 대신 파일 내용을 넘긴다 — MediaPipe는 한글이 들어간 경로(예: '새 폴더')를 열지 못해서
        # 경로로 넘기면 깜빡임 검사가 소리 없이 꺼진다
        with open(LIVENESS_MODEL_PATH, "rb") as f:
            model_bytes = f.read()
        return vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=model_bytes),
            running_mode=vision.RunningMode.VIDEO,
            output_face_blendshapes=True,   # eyeBlinkLeft / eyeBlinkRight 값을 쓰기 위해
            num_faces=1,
        ))
    except Exception as e:
        print(f"[출석] 깜빡임 검사를 켜지 못했습니다 ({type(e).__name__}: {e})")
        return None


def crop_face_region(frame, bbox, margin=0.35):
    """얼굴 bbox 주변을 여유 있게 잘라낸다. 화면 밖으로 나가지 않게 잘라서 돌려준다."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    x1 = max(0, int(x1 - mx))
    y1 = max(0, int(y1 - my))
    x2 = min(w, int(x2 + mx))
    y2 = min(h, int(y2 + my))
    if x2 - x1 < 20 or y2 - y1 < 20:
        return None
    return frame[y1:y2, x1:x2]


def detect_blink_score(landmarker, frame, timestamp_ms):
    """
    프레임에서 눈이 감긴 정도(0~1)를 구한다. 양쪽 눈 중 더 크게 감긴 쪽 값을 쓴다.
    얼굴을 못 찾았거나 검사가 꺼져 있으면 None.
    주의: 인식 대상 얼굴만 잘라서 넘겨야 한다. 화면 전체를 넘기면 사진을 들고 있는
    사람의 깜빡임이 대신 잡혀서 사진이 통과해 버린다.
    """
    if landmarker is None or frame is None or frame.size == 0:
        return None
    try:
        import mediapipe as mp
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = landmarker.detect_for_video(image, timestamp_ms)
    except Exception:
        return None

    if not result.face_blendshapes:
        return None
    scores = {b.category_name: b.score for b in result.face_blendshapes[0]}
    return max(scores.get("eyeBlinkLeft", 0.0), scores.get("eyeBlinkRight", 0.0))


_instance_mutex = None


def acquire_single_instance():
    """윈도우 명명 뮤텍스로 프로그램이 하나만 켜지게 한다. 이미 켜져 있으면 False. (윈도우 외에는 항상 True)"""
    global _instance_mutex
    if not IS_WINDOWS:
        return True
    import ctypes
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateMutexW.restype = ctypes.c_void_p
    _instance_mutex = kernel32.CreateMutexW(None, False, "Local\\TheEclipseAttendance")
    ERROR_ALREADY_EXISTS = 183
    return ctypes.get_last_error() != ERROR_ALREADY_EXISTS


def create_face_app():
    """InsightFace 얼굴 인식기를 만든다 (처음이면 모델을 ~/.insightface 로 내려받는다)."""
    from insightface.app import FaceAnalysis

    # addons=["liveness"] — InsightFace 공식 위조판별(사진/화면 판별) 애드온.
    # observe 모드로 두면 가짜로 보여도 얼굴 인식은 그대로 하고, 판정 결과만 받아서
    # "사진으로 보입니다" 같은 안내를 우리 쪽에서 띄울 수 있다.
    try:
        app = FaceAnalysis(
            name="buffalo_l",
            allowed_modules=['detection', 'recognition'],
            addons=["liveness"],
            liveness_mode="observe",
            liveness_threshold=LIVENESS_THRESHOLD,
            providers=["CPUExecutionProvider"]
        )
    except Exception as e:
        # 구버전 insightface이거나 애드온을 못 받은 경우 — 위조판별 없이 그대로 동작한다
        print(f"[출석] 위조판별 애드온을 쓸 수 없습니다 ({e}). 깜빡임 검사만 사용합니다.")
        app = FaceAnalysis(
            name="buffalo_l",
            allowed_modules=['detection', 'recognition'],
            providers=["CPUExecutionProvider"]
        )
    app.prepare(ctx_id=0, det_size=(256, 256))
    return app


def face_inference_process(conn, threads):
    import onnxruntime

    app = create_face_app()

    # onnxruntime은 기본으로 CPU 코어를 전부 써서 카메라/화면 쪽이 밀린다.
    # insightface는 세션 옵션을 넘겨받지 않으므로 같은 모델 파일로 세션만 다시 만든다.
    opts = onnxruntime.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    for model in app.models.values():
        model.session = onnxruntime.InferenceSession(
            model.session.model_path, sess_options=opts, providers=inference_providers()
        )

    # 사진/영상 판별용 깜빡임 검출기 (모델 파일이 없으면 None)
    blink_landmarker = create_blink_detector()

    conn.send("ready")
    while True:
        frame = conn.recv()
        faces = app.get(frame)

        # 깜빡임은 "인식된 얼굴"에서만 본다. 화면 전체로 보면 사진을 들고 있는 사람의
        # 깜빡임이 대신 잡혀서 사진이 통과한다. 가장 큰 얼굴(= 인식 대상)만 잘라서 검사한다.
        blink_score = None
        liveness = None   # (is_live, status) — 위조판별 애드온 결과
        if faces:
            main_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            face_crop = crop_face_region(frame, main_face.bbox)
            # MediaPipe VIDEO 모드는 타임스탬프가 계속 커져야 한다
            blink_score = detect_blink_score(blink_landmarker, face_crop, int(time.perf_counter() * 1000))

            result = getattr(main_face, "liveness", None)
            if result is not None:
                liveness = (result.is_live, result.status)

        conn.send(([(f.bbox, f.embedding) for f in faces], blink_score, liveness))


class FlatButton(tk.Label):
    """
    macOS용 버튼. 맥의 tk.Button은 시스템 버튼 모양으로 그려져서 bg(배경색)가 무시되고,
    흰 글씨(fg="white") 버튼은 흰 바탕에 글자가 안 보인다. Label로 같은 모양을 흉내 낸다.
    tk.Button과 같은 옵션(command, bg, activebackground, state ...)을 그대로 받는다.
    """

    def __init__(self, master=None, command=None, activebackground=None, activeforeground=None,
                 overrelief=None, **kw):
        super().__init__(master, **kw)
        self._command = command
        self._active_bg = activebackground
        self._active_fg = activeforeground
        self._normal_bg = self.cget("bg")
        self._normal_fg = self.cget("fg")
        self._hover = False
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<ButtonRelease-1>", self._on_click)

    def configure(self, cnf=None, **kw):
        if cnf:
            kw.update(cnf)
        if "command" in kw:
            self._command = kw.pop("command")
        if "activebackground" in kw:
            self._active_bg = kw.pop("activebackground")
        if "activeforeground" in kw:
            self._active_fg = kw.pop("activeforeground")
        kw.pop("overrelief", None)
        for key, attr in (("bg", "_normal_bg"), ("background", "_normal_bg"),
                          ("fg", "_normal_fg"), ("foreground", "_normal_fg")):
            if key in kw:
                setattr(self, attr, kw[key])
        if not kw:
            return super().configure()
        result = super().configure(**kw)
        if self._hover and ("bg" in kw or "background" in kw):
            self._on_enter()
        return result

    config = configure

    def _enabled(self):
        return str(self.cget("state")) != tk.DISABLED

    def _on_enter(self, _event=None):
        self._hover = True
        if self._enabled():
            if self._active_bg:
                super().configure(bg=self._active_bg)
            if self._active_fg:
                super().configure(fg=self._active_fg)

    def _on_leave(self, _event=None):
        self._hover = False
        super().configure(bg=self._normal_bg, fg=self._normal_fg)

    def _on_click(self, event):
        # 누른 채로 버튼 밖으로 나가서 뗐으면 취소 (일반 버튼과 동일)
        inside = 0 <= event.x < self.winfo_width() and 0 <= event.y < self.winfo_height()
        if inside:
            self.invoke()

    def invoke(self):
        if self._enabled() and self._command:
            return self._command()


# 맥에서는 배경색이 적용되는 FlatButton, 윈도우에서는 원래 tk.Button을 그대로 쓴다
Button = FlatButton if IS_MAC else tk.Button


def px(value):
    """1600x900 기준으로 잡은 픽셀 값을 현재 화면 배율에 맞춰 바꾼다."""
    return max(1, int(round(value * UI_SCALE)))


def _get_work_area(window):
    """작업 표시줄을 뺀 실제 사용 가능한 화면 크기(가로, 세로)를 구한다."""
    try:
        import ctypes
        from ctypes import wintypes
        rect = wintypes.RECT()
        SPI_GETWORKAREA = 0x0030
        if ctypes.windll.user32.SystemParametersInfoW(SPI_GETWORKAREA, 0, ctypes.byref(rect), 0):
            return rect.right - rect.left, rect.bottom - rect.top
    except Exception:
        pass
    if IS_MAC:
        # 메뉴 막대와 Dock을 뺀 영역 (pyobjc가 있을 때만)
        try:
            from AppKit import NSScreen
            frame = NSScreen.mainScreen().visibleFrame()
            return int(frame.size.width), int(frame.size.height)
        except Exception:
            pass
    # 윈도우가 아니거나 조회에 실패하면 작업 표시줄 높이를 대략 빼서 쓴다
    return window.winfo_screenwidth(), window.winfo_screenheight() - 48

def ensure_camera_permission(timeout=60):
    """
    macOS에서 카메라 권한을 미리 요청하고 사용자가 허용/거부할 때까지 기다린다.
    OpenCV도 권한을 요청하지만 응답을 기다리지 않고 바로 실패 처리해서, 처음 실행하면
    [허용]을 눌러도 그 실행에서는 카메라가 안 켜진다. 권한 상태: 0=미결정 1=제한 2=거부 3=허용.
    pyobjc(pyobjc-framework-AVFoundation)가 없으면 아무것도 하지 않는다.
    """
    if not IS_MAC:
        return True
    try:
        import AVFoundation
        from Foundation import NSRunLoop, NSDate
    except ImportError:
        return True
    media = AVFoundation.AVMediaTypeVideo
    status = AVFoundation.AVCaptureDevice.authorizationStatusForMediaType_(media)
    if status != 0:
        return status == 3

    result = {}

    def on_done(granted):
        result["granted"] = bool(granted)

    AVFoundation.AVCaptureDevice.requestAccessForMediaType_completionHandler_(media, on_done)
    deadline = time.time() + timeout
    while "granted" not in result and time.time() < deadline:
        NSRunLoop.currentRunLoop().runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.1))
    return result.get("granted", False)


def open_camera(index=0):
    """
    OS에 맞는 백엔드로 웹캠을 연다.
    - Windows: DSHOW가 MSMF와 FPS는 같고 여는 속도가 훨씬 빠르다
    - macOS: DSHOW가 없으므로 AVFoundation을 쓴다
    - 그 외(리눅스 등)는 V4L2
    원하는 백엔드로 못 열면 OpenCV 기본값(CAP_ANY)으로 한 번 더 시도한다.
    """
    if IS_WINDOWS:
        backend = cv2.CAP_DSHOW
    elif IS_MAC:
        backend = cv2.CAP_AVFOUNDATION
    else:
        backend = cv2.CAP_V4L2
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(index)
    return cap

# ==========================================
# 0-1. 카메라 화면에 한글 이름을 그리기 위한 폰트
# ==========================================
# cv2.putText는 Hershey 계열 폰트만 지원해서 한글을 못 그린다(글자 수만큼 '?'로 깨져 보인다).
# PIL(Pillow)로 트루타입 폰트를 써서 그려야 한글이 제대로 나온다.
_KOREAN_FONT_CACHE = {}
_KOREAN_FONT_CANDIDATES = [
    "malgun.ttf",                              # 맑은 고딕 (윈도우 기본, UI 폰트와 통일)
    r"C:\Windows\Fonts\malgun.ttf",
    r"C:\Windows\Fonts\malgunbd.ttf",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",  # macOS 기본 한글 폰트
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
    "/Library/Fonts/AppleGothic.ttf",
]


def get_korean_font(size=18):
    """지정한 크기의 한글 지원 폰트를 반환한다(캐시해서 매 프레임 다시 읽지 않는다)."""
    if size in _KOREAN_FONT_CACHE:
        return _KOREAN_FONT_CACHE[size]

    font = None
    for candidate in _KOREAN_FONT_CANDIDATES:
        try:
            font = ImageFont.truetype(candidate, size)
            break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()  # 최후의 수단 — 이 경우 한글은 못 그리지만 프로그램은 안 죽는다

    _KOREAN_FONT_CACHE[size] = font
    return font

# ==========================================
# 0. 창공시스템(SeatManagerApp, C#) 연동
# ==========================================
# 두 프로그램이 서로 다른 컴퓨터/설치 경로에서 실행돼도 항상 같은 곳을 가리키도록,
# 소스에 특정 컴퓨터 경로를 적어두지 않고 사용자 홈 폴더를 공유 폴더로 쓴다.
# 창공시스템(C#)도 %USERPROFILE%\SeatManagerApp\face-integration 로 정확히 같은 경로를 계산한다.
# %APPDATA%가 아니라 %USERPROFILE%을 쓰는 이유: Microsoft Store/Python Install Manager로 설치한
# 패키지형 파이썬은 %APPDATA%·%LOCALAPPDATA%를 앱별로 가상화해서, 창공시스템이 그 경로에 쓴 파일이
# 이 프로그램에서는 안 보일 수 있다(실제로 테스트 중 발견됨). 홈 폴더 바로 아래는 가상화 대상이 아니다.
def _face_integration_dir():
    home = os.path.expanduser("~")
    path = os.path.join(home, "SeatManagerApp", "face-integration")
    os.makedirs(path, exist_ok=True)
    return path


ROSTER_PATH = os.path.join(_face_integration_dir(), "roster.json")
ATTENDANCE_EXPORT_PATH = os.path.join(_face_integration_dir(), "attendance.json")


def load_roster():
    """창공시스템이 내보낸 학생 명단(roster.json)을 읽는다. 없거나 깨졌으면 빈 목록."""
    try:
        return load_roster_strict()
    except (FileNotFoundError, ValueError):
        return []


def load_roster_strict():
    """
    관리자 화면의 [새로고침]처럼, 실패했을 때 사용자에게 직접 알려줘야 하는 경우 쓴다.
    파일이 없거나(FileNotFoundError) 형식이 잘못되면(ValueError, json.JSONDecodeError는 ValueError의 하위 클래스)
    예외를 그대로 올려서 호출한 쪽에서 메시지를 보여주게 한다.
    """
    with open(ROSTER_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("roster.json의 최상위 형식이 배열([...])이 아닙니다.")
    return data


def pair_attendance_sessions(rows):
    """
    (user_id, name, log_date, log_type, log_time) 기록을 받아 학번+날짜별로 출근→퇴근 한 쌍씩 묶는다.
    하루에 여러 번 출퇴근할 수 있어서 한 사람이 하루에 여러 건이 나올 수 있다.
    - 퇴근 없이 다시 출근하면 앞의 출근은 퇴근 기록 없음(check_out=None)으로 남는다.
    - 출근 없이 퇴근만 있으면 출근 기록 없음(check_in=None)으로 남긴다.
    반환: {"user_id", "name", "date", "check_in", "check_out"} 목록 (날짜/시간 순)
    """
    sessions = []
    open_session = {}  # (user_id, log_date) -> 아직 퇴근 안 한 세션
    ordered = sorted(
        (r for r in rows if r[0] and r[2]),  # 학번/날짜 없는 이상 데이터는 제외
        key=lambda r: (r[2], r[4] or "", r[0]),
    )
    for user_id, name, log_date, log_type, log_time in ordered:
        key = (user_id, log_date)
        if log_type == "CHECK_IN":
            session = {"user_id": user_id, "name": name, "date": log_date, "check_in": log_time, "check_out": None}
            sessions.append(session)
            open_session[key] = session
        elif log_type == "CHECK_OUT":
            session = open_session.pop(key, None)
            if session is not None:
                session["check_out"] = log_time
            else:
                sessions.append({"user_id": user_id, "name": name, "date": log_date, "check_in": None, "check_out": log_time})
    return sessions


def is_valid_attendance_time(log_time):
    """06:00~24:00 사이에 찍은 기록인지. 00:00~06:00 사이 출근/퇴근은 무효로 본다."""
    return bool(log_time) and log_time >= VALID_ATTENDANCE_START


def format_duration(seconds):
    """초를 'N시간 N분 N초' 형태로 바꾼다."""
    seconds = int(max(0, seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}시간 {minutes:02d}분 {secs:02d}초"


def session_seconds(session):
    """
    출근~퇴근 사이 인정 시간(초).
    출근/퇴근 중 하나라도 없거나, 00:00~06:00 사이에 찍은 기록이거나, 계산이 안 되면 0.
    """
    if not (session["check_in"] and session["check_out"]):
        return 0
    if not (is_valid_attendance_time(session["check_in"]) and is_valid_attendance_time(session["check_out"])):
        return 0
    try:
        t_in = datetime.strptime(f"{session['date']} {session['check_in']}", "%Y-%m-%d %H:%M:%S")
        t_out = datetime.strptime(f"{session['date']} {session['check_out']}", "%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError):
        return 0
    return max(0, (t_out - t_in).total_seconds())


def export_attendance_json():
    """출결 기록 전체를 출근→퇴근 한 쌍씩 묶어서 창공시스템이 읽을 수 있는 attendance.json으로 내보낸다."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT user_id, name, log_date, log_type, log_time
            FROM attendance_logs
            ORDER BY log_date ASC, log_time ASC
        """)
        rows = cursor.fetchall()
        conn.close()

        # 하루에 여러 번 출퇴근하면 같은 학번/날짜로 여러 건이 들어간다 (시간순)
        records = [
            {
                "StudentId": s["user_id"],
                "Name": s["name"],
                "Date": s["date"],
                "CheckIn": s["check_in"],
                "CheckOut": s["check_out"],
            }
            for s in pair_attendance_sessions(rows)
        ]

        with open(ATTENDANCE_EXPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
    except Exception:
        # 내보내기 실패는 치명적이지 않다 — 다음 출결 처리 때 다시 시도한다
        pass


# ==========================================
# 0-2. 출퇴근 사진 저장 / 자동 삭제
# ==========================================
# 사진은 PHOTO_DIR/날짜/시각_학번_IN|OUT.jpg 로 저장하고, DB에는 PHOTO_DIR 기준 상대 경로만 남긴다.
# (폴더를 통째로 옮기거나 구글 드라이브 동기화 폴더로 바꿔도 경로가 깨지지 않게)
def photo_abs_path(rel_path):
    return os.path.join(PHOTO_DIR, *rel_path.split("/"))


def save_attendance_photo(frame, user_id, log_type, when):
    """출퇴근 순간의 카메라 화면을 작게 줄여 JPEG로 저장한다. 저장한 상대 경로, 실패하면 None."""
    if frame is None:
        return None
    try:
        h, w = frame.shape[:2]
        if w > PHOTO_WIDTH:
            frame = cv2.resize(frame, (PHOTO_WIDTH, int(h * PHOTO_WIDTH / w)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, PHOTO_JPEG_QUALITY])
        if not ok:
            return None
        safe_id = "".join(c if c.isalnum() else "_" for c in str(user_id))
        kind = "IN" if log_type == "CHECK_IN" else "OUT"
        rel_path = f"{when:%Y-%m-%d}/{when:%H%M%S}_{safe_id}_{kind}.jpg"
        abs_path = photo_abs_path(rel_path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        # cv2.imwrite는 윈도우에서 한글이 들어간 경로(예: '새 폴더')에 저장하지 못해서 바이트로 직접 쓴다
        with open(abs_path, "wb") as f:
            f.write(buf.tobytes())
        return rel_path
    except Exception as e:
        # 사진 저장 실패로 출퇴근 기록까지 막지는 않는다
        print(f"[출석] 출퇴근 사진을 저장하지 못했습니다 ({e})")
        return None


def delete_photo_files(rel_paths):
    for rel_path in rel_paths:
        if not rel_path:
            continue
        try:
            os.remove(photo_abs_path(rel_path))
        except OSError:
            pass


def cleanup_old_photos(today=None):
    """
    PHOTO_RETENTION_DAYS(3개월)가 지난 날짜 폴더를 통째로 지우고, DB의 사진 경로도 비운다.
    출퇴근 기록(시간) 자체는 그대로 남는다. 지운 폴더 수를 돌려준다.
    """
    cutoff = (today or date.today()) - timedelta(days=PHOTO_RETENTION_DAYS)
    removed = 0
    if os.path.isdir(PHOTO_DIR):
        for name in os.listdir(PHOTO_DIR):
            try:
                folder_date = datetime.strptime(name, "%Y-%m-%d").date()
            except ValueError:
                continue  # 날짜 폴더가 아닌 것은 건드리지 않는다
            if folder_date < cutoff:
                shutil.rmtree(os.path.join(PHOTO_DIR, name), ignore_errors=True)
                removed += 1
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "UPDATE attendance_logs SET photo_path = NULL WHERE photo_path IS NOT NULL AND log_date < ?",
            (cutoff.isoformat(),),
        )
        conn.commit()
        conn.close()
    except sqlite3.Error:
        pass
    return removed

# ==========================================
# 1. DB 초기화 및 관련 함수
# ==========================================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 사용자 테이블 (비밀번호, 권한, 전공, 패널티 필드 추가)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT UNIQUE,
            password TEXT NOT NULL DEFAULT '1234',
            name TEXT NOT NULL,
            major TEXT NOT NULL DEFAULT '소프트웨어전공',
            role TEXT NOT NULL DEFAULT 'student', -- 'admin' 또는 'student'
            embedding BLOB NOT NULL,
            penalty INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # 컬럼 존재 확인 및 마이그레이션
    try:
        cursor.execute("ALTER TABLE users ADD COLUMN password TEXT NOT NULL DEFAULT '1234'")
    except sqlite3.OperationalError:
        pass
    try:
        cursor.execute("ALTER TABLE users ADD COLUMN major TEXT NOT NULL DEFAULT '소프트웨어전공'")
    except sqlite3.OperationalError:
        pass
    try:
        cursor.execute("ALTER TABLE users ADD COLUMN role TEXT NOT NULL DEFAULT 'student'")
    except sqlite3.OperationalError:
        pass
    try:
        cursor.execute("ALTER TABLE users ADD COLUMN penalty INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass

    # 출석 기록 테이블 (구분, 날짜, 시간 필드 보강)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            name TEXT,
            log_type TEXT DEFAULT 'CHECK_IN', -- 'CHECK_IN' (출근) 또는 'CHECK_OUT' (퇴근)
            log_date TEXT,
            log_time TEXT,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        )
    """)
    # 출퇴근 순간 사진 (PHOTO_DIR 기준 상대 경로, 보관 기간이 지나면 NULL)
    try:
        cursor.execute("ALTER TABLE attendance_logs ADD COLUMN photo_path TEXT")
    except sqlite3.OperationalError:
        pass

    # 예전 기본 관리자 계정(dongseo@mail.com / admin1234)이 남아 있으면 새 계정으로 바꾼다
    cursor.execute("SELECT id FROM users WHERE user_id = '1234'")
    if not cursor.fetchone():
        cursor.execute("""
            UPDATE users SET user_id = '1234', password = '1234'
            WHERE user_id = 'dongseo@mail.com' AND role = 'admin'
        """)

    # 기본 관리자 계정 생성 (1234 / 1234)
    cursor.execute("SELECT id FROM users WHERE user_id = '1234'")
    if not cursor.fetchone():
        dummy_embedding = np.zeros(512, dtype=np.float32).tobytes()
        cursor.execute("""
            INSERT INTO users (user_id, password, name, major, role, embedding, penalty)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, ('1234', '1234', '관리자', '시스템관리', 'admin', dummy_embedding, 0))

    conn.commit()
    conn.close()

def load_users():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT user_id, name, major, penalty, embedding FROM users WHERE role = 'student'")
    rows = cursor.fetchall()
    conn.close()

    users = []
    for u_id, name, major, penalty, blob in rows:
        emb = np.frombuffer(blob, dtype=np.float32)
        users.append({
            "user_id": u_id, 
            "name": name, 
            "major": major, 
            "penalty": penalty, 
            "embedding": emb
        })
    return users

def summarize_attendance(logs):
    """
    (user_id, name, log_date, log_type, log_time) 기록으로 (출근 일수, 인정 시간 초)를 구한다.
    학생 화면의 주간 현황과 관리자 출결로그 필터가 같은 규칙을 쓰도록 여기 한 곳에서 계산한다.
    """
    # 00:00~06:00 사이 출근/퇴근은 무효라서 아예 빼고 계산한다
    valid_logs = [log for log in logs if is_valid_attendance_time(log[4])]

    # 출근 일수: 인정되는 출근 기록이 있는 날 (퇴근을 안 찍었어도 일수는 인정)
    attended_days = len({log[2] for log in valid_logs if log[3] == "CHECK_IN"})

    # 출근 시간: 출근→퇴근 한 쌍이 모두 있어야 인정 (하루 여러 번이면 모두 더한다)
    total_seconds = int(sum(session_seconds(s) for s in pair_attendance_sessions(valid_logs)))
    return attended_days, total_seconds


def get_attendance_summary_by_student(start_date, end_date):
    """기간(YYYY-MM-DD, 양 끝 포함) 안의 학생별 출근 일수와 인정 시간. 기록이 없는 등록 학생도 0으로 넣는다."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT user_id, name FROM users WHERE role = 'student'")
    names = {str(user_id): name for user_id, name in cursor.fetchall()}
    cursor.execute("""
        SELECT user_id, name, log_date, log_type, log_time
        FROM attendance_logs
        WHERE log_date BETWEEN ? AND ?
    """, (start_date, end_date))
    logs = cursor.fetchall()
    conn.close()

    logs_by_user = {}
    for log in logs:
        if not log[0]:
            continue
        user_id = str(log[0])
        logs_by_user.setdefault(user_id, []).append(log)
        names.setdefault(user_id, log[1])  # 삭제된 학생의 기록도 이름과 함께 보여준다

    summary = []
    for user_id, name in names.items():
        days, seconds = summarize_attendance(logs_by_user.get(user_id, []))
        summary.append({"user_id": user_id, "name": name or "", "days": days, "seconds": seconds})
    return summary


def get_weekly_attendance_stats(user_id):
    # 이번주 월요일 00:00:00부터 일요일 23:59:59까지의 출석 계산
    today = datetime.now()
    start_of_week = today - timedelta(days=today.weekday())
    start_date_str = start_of_week.strftime("%Y-%m-%d")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT user_id, name, log_date, log_type, log_time
        FROM attendance_logs
        WHERE user_id = ? AND log_date >= ?
        ORDER BY log_date ASC, log_time ASC
    """, (user_id, start_date_str))
    logs = cursor.fetchall()
    conn.close()

    attended_days, total_seconds = summarize_attendance(logs)

    return {
        "sessions": pair_attendance_sessions(logs),  # 이번주 출근→퇴근 목록 ([출석 조회] 창에 표시)
        "days": attended_days,
        "seconds": total_seconds,
        "missing_days": max(0, WEEKLY_REQUIRED_DAYS - attended_days),
        "missing_seconds": max(0, WEEKLY_REQUIRED_SECONDS - total_seconds),
    }

def get_today_attendance_status(user_id):
    """
    오늘 출석 상태를 구한다.
    - finished_seconds : 오늘 퇴근까지 찍어서 인정된 시간(초)
    - working_since    : 지금 출근 중이면 그 출근 시각("HH:MM:SS"), 아니면 None
    - working_seconds  : 지금 출근 중인 시간(초). 출근 중이 아니면 0
    00:00~06:00 사이 기록은 인정되지 않으므로 계산에서 뺀다.
    """
    now = datetime.now()
    today_str = now.strftime("%Y-%m-%d")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT user_id, name, log_date, log_type, log_time
        FROM attendance_logs
        WHERE user_id = ? AND log_date = ?
        ORDER BY log_time ASC
    """, (user_id, today_str))
    logs = [log for log in cursor.fetchall() if is_valid_attendance_time(log[4])]
    conn.close()

    sessions = pair_attendance_sessions(logs)
    finished_seconds = int(sum(session_seconds(s) for s in sessions))

    # 지금 출근 중인지는 "오늘의 마지막 기록이 출근인지"로 본다.
    # 짝이 안 맞는 예전 기록(중복 출근 등)이 남아 있어도 퇴근을 찍으면 바로 멈추게 하기 위함이다.
    working_since = None
    if logs and logs[-1][3] == "CHECK_IN":
        working_since = logs[-1][4]

    working_seconds = 0
    if working_since:
        try:
            started = datetime.strptime(f"{today_str} {working_since}", "%Y-%m-%d %H:%M:%S")
            working_seconds = int(max(0, (now - started).total_seconds()))
        except ValueError:
            working_since = None

    return {
        "finished_seconds": finished_seconds,
        "working_since": working_since,
        "working_seconds": working_seconds,
    }

def log_attendance(user_id, name, log_type, frame=None):
    """
    출근/퇴근을 기록한다. frame을 넘기면 그 순간 카메라 화면을 작게 줄여 사진으로 남긴다.
    반환: (성공 여부, 안내 문구) — 성공이면 기록 시각(+ 미인정 안내), 실패면 그 이유.
    """
    now = datetime.now()
    today_date = now.strftime("%Y-%m-%d")
    now_time = now.strftime("%H:%M:%S")

    # 하루에 여러 번 출퇴근할 수 있지만, 퇴근을 안 한 상태에서 또 출근을 누르는 것은 막는다
    if log_type == 'CHECK_IN' and is_valid_attendance_time(now_time):
        working_since = get_today_attendance_status(user_id)["working_since"]
        if working_since:
            return False, (f"이미 출근 완료 되었습니다. ({working_since} 출근)\n"
                           "퇴근을 먼저 찍은 뒤에 다시 출근할 수 있습니다.")

    photo_path = save_attendance_photo(frame, user_id, log_type, now)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 하루에 출근/퇴근을 여러 번 할 수 있어서 누를 때마다 새 기록으로 남긴다
    cursor.execute("""
        INSERT INTO attendance_logs (user_id, name, log_type, log_date, log_time, photo_path)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (user_id, name, log_type, today_date, now_time, photo_path))
    conn.commit()
    conn.close()
    export_attendance_json()  # 창공시스템이 읽는 출결 파일도 즉시 갱신

    if not is_valid_attendance_time(now_time):
        return True, (f"기록 시각 {now_time}\n"
                      "단, 00:00~06:00 사이 출퇴근은 출석 일수와 시간에 인정되지 않습니다.")
    return True, f"기록 시각 {now_time}"


# ==========================================
# 2. UI 상태 프레임 정의
# ==========================================

WEEKDAY_KO = ["월", "화", "수", "목", "금", "토", "일"]


def format_date_ko(d):
    return f"{d:%Y-%m-%d} ({WEEKDAY_KO[d.weekday()]})"


class DatePicker(Button):
    """누르면 달력이 펼쳐지고, 날짜를 고르면 값이 바뀌는 버튼. (tkcalendar 같은 추가 설치 없이 동작)"""

    def __init__(self, master, initial, on_change=None):
        super().__init__(
            master, bg="white", fg="#0F172A", activebackground="#F1F5F9", relief=tk.SOLID, bd=1,
            font=(UI_FONT, 10), padx=px(8), pady=px(2), cursor="hand2", command=self.toggle_calendar
        )
        self.on_change = on_change
        self.popup = None
        self.set_date(initial)

    def get_date(self):
        return self._date

    def set_date(self, value):
        self._date = value
        self.config(text=f"📅 {format_date_ko(value)}")

    def toggle_calendar(self):
        if self.popup is not None:
            self.close_calendar()
            return

        self.popup = tk.Toplevel(self)
        self.popup.overrideredirect(True)
        self.popup.configure(bg="#CBD5E1")  # 바깥 1px 테두리
        self.body = tk.Frame(self.popup, bg="white")
        self.body.pack(padx=1, pady=1)

        self._view_year, self._view_month = self._date.year, self._date.month
        self._render_month()
        self._place_popup()

        # 달력 밖을 누르면 닫는다 — grab 중에는 창 밖 클릭도 이 팝업으로 전달된다
        self.popup.grab_set()
        self.popup.bind("<ButtonPress-1>", self._on_click)
        self.popup.bind("<Escape>", lambda e: self.close_calendar())
        self.popup.focus_set()

    def close_calendar(self):
        if self.popup is not None:
            self.popup.grab_release()
            self.popup.destroy()
            self.popup = None

    def _place_popup(self):
        self.popup.update_idletasks()
        x = self.winfo_rootx()
        y = self.winfo_rooty() + self.winfo_height() + 2
        # 화면 아래/오른쪽으로 넘치면 안쪽으로 당긴다
        if y + self.popup.winfo_reqheight() > self.winfo_screenheight():
            y = self.winfo_rooty() - self.popup.winfo_reqheight() - 2
        x = min(x, self.winfo_screenwidth() - self.popup.winfo_reqwidth())
        self.popup.geometry(f"+{max(0, x)}+{max(0, y)}")

    def _on_click(self, event):
        p = self.popup
        inside = (p.winfo_rootx() <= event.x_root < p.winfo_rootx() + p.winfo_width()
                  and p.winfo_rooty() <= event.y_root < p.winfo_rooty() + p.winfo_height())
        if not inside:
            self.close_calendar()

    def _shift_month(self, delta):
        index = self._view_month - 1 + delta
        self._view_year += index // 12
        self._view_month = index % 12 + 1
        self._render_month()

    def _pick(self, value):
        self.close_calendar()
        self.set_date(value)
        if self.on_change:
            self.on_change()

    def _render_month(self):
        for widget in self.body.winfo_children():
            widget.destroy()

        header = tk.Frame(self.body, bg="white")
        header.grid(row=0, column=0, columnspan=7, sticky=tk.EW, pady=(px(6), px(4)))
        nav_style = dict(bg="white", fg="#334155", activebackground="#E2E8F0", bd=0,
                         font=(UI_FONT, 11, "bold"), cursor="hand2", padx=px(8))
        Button(header, text="◀", command=lambda: self._shift_month(-1), **nav_style).pack(side=tk.LEFT, padx=px(4))
        Button(header, text="▶", command=lambda: self._shift_month(1), **nav_style).pack(side=tk.RIGHT, padx=px(4))
        tk.Label(header, text=f"{self._view_year}년 {self._view_month}월", bg="white", fg="#0F172A",
                 font=(UI_FONT, 11, "bold")).pack(side=tk.LEFT, expand=True)

        for col, name in enumerate(WEEKDAY_KO):
            color = "#EF4444" if col == 6 else "#3B82F6" if col == 5 else "#64748B"
            tk.Label(self.body, text=name, bg="white", fg=color, width=4,
                     font=(UI_FONT, 9, "bold")).grid(row=1, column=col, pady=(0, px(2)))

        today = date.today()
        weeks = calendar.Calendar(firstweekday=0).monthdatescalendar(self._view_year, self._view_month)
        for row, week in enumerate(weeks, start=2):
            for col, day in enumerate(week):
                selected = day == self._date
                is_today = day == today
                if selected:
                    bg, fg = "#3B82F6", "white"
                elif day.month != self._view_month:
                    bg, fg = "white", "#CBD5E1"
                else:
                    bg = "#DBEAFE" if is_today else "white"
                    fg = "#EF4444" if col == 6 else "#3B82F6" if col == 5 else "#0F172A"
                Button(
                    self.body, text=str(day.day), width=4, bg=bg, fg=fg, bd=0, relief=tk.FLAT,
                    activebackground="#BFDBFE", cursor="hand2",
                    font=(UI_FONT, 10, "bold" if (selected or is_today) else "normal"),
                    command=lambda d=day: self._pick(d)
                ).grid(row=row, column=col, padx=1, pady=1)

        Button(
            self.body, text=f"오늘 ({format_date_ko(today)})", bg="#F1F5F9", fg="#334155", bd=0,
            activebackground="#E2E8F0", font=(UI_FONT, 9, "bold"), cursor="hand2",
            command=lambda: self._pick(today)
        ).grid(row=len(weeks) + 2, column=0, columnspan=7, sticky=tk.EW, padx=px(6), pady=px(6))


# 전체 출결로그 조회 유형 — (키, 화면에 보이는 이름)
LOG_FILTER_MODES = [
    ("today", "오늘 내역"),
    ("recent7", "최근 7일 내역"),
    ("recent7_sum", "최근 7일 합계 (학생별)"),
    ("month", "월 단위 내역"),
    ("week_days", f"주 {WEEKLY_REQUIRED_DAYS}회 이상 출석한 학생 (월~일)"),
    ("week_hours", f"주 {WEEKLY_REQUIRED_SECONDS // 3600}시간 이상 출석한 학생 (월~일)"),
    ("custom", "기간 직접 선택 (내역)"),
]
LOG_SUMMARY_MODES = {"recent7_sum", "week_days", "week_hours"}

ACTION_LABELS = {"CHECK_IN": "출근", "CHECK_OUT": "퇴근"}
ACTION_COLORS = {"CHECK_IN": ("#10B981", "#059669"), "CHECK_OUT": ("#3B82F6", "#2563EB")}


def center_on(window, popup, rel_y=0.4):
    """popup을 window 가운데(세로는 rel_y 위치)에 띄운다."""
    popup.update_idletasks()
    x = window.winfo_rootx() + (window.winfo_width() - popup.winfo_reqwidth()) // 2
    y = window.winfo_rooty() + int((window.winfo_height() - popup.winfo_reqheight()) * rel_y)
    popup.geometry(f"+{max(0, x)}+{max(0, y)}")


class AdminLoginDialog(tk.Toplevel):
    """상단 [관리자] 버튼을 누르면 뜨는 로그인 창."""

    def __init__(self, app):
        super().__init__(app.window)
        self.app = app
        self.title("관리자 로그인")
        self.configure(bg=PANEL)
        self.resizable(False, False)
        self.transient(app.window)

        body = tk.Frame(self, bg=PANEL)
        body.pack(padx=px(40), pady=px(32))

        tk.Label(body, text="관리자 로그인", font=(UI_FONT, 18, "bold"), bg=PANEL, fg=TEXT) \
            .pack(anchor=tk.W, pady=(0, px(20)))

        tk.Label(body, text="아이디", font=(UI_FONT, 10, "bold"), bg=PANEL, fg=MUTED).pack(anchor=tk.W)
        self.ent_id = ttk.Entry(body, font=(UI_FONT, 12), width=28)
        self.ent_id.pack(fill=tk.X, ipady=4, pady=(px(4), px(14)))

        tk.Label(body, text="비밀번호", font=(UI_FONT, 10, "bold"), bg=PANEL, fg=MUTED).pack(anchor=tk.W)
        self.ent_pwd = ttk.Entry(body, font=(UI_FONT, 12), width=28, show="*")
        self.ent_pwd.pack(fill=tk.X, ipady=4, pady=(px(4), px(22)))

        btn_row = tk.Frame(body, bg=PANEL)
        btn_row.pack(fill=tk.X)
        Button(
            btn_row, text="취소", bg=BLUE_SOFT, fg=BLUE, bd=0,
            activebackground=BLUE_SOFT_HOVER, activeforeground=BLUE,
            font=(UI_FONT, 12, "bold"), height=2, cursor="hand2", command=self.destroy
        ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, px(6)))
        Button(
            btn_row, text="로그인", bg=BLUE, fg="white", bd=0,
            activebackground=BLUE_DARK, activeforeground="white",
            font=(UI_FONT, 12, "bold"), height=2, cursor="hand2", command=self.try_login
        ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(px(6), 0))

        self.bind("<Return>", lambda e: self.try_login())
        self.bind("<Escape>", lambda e: self.destroy())

        center_on(app.window, self, rel_y=0.33)
        self.ent_id.focus_set()
        # 창이 화면에 뜬 다음에 잡아야 한다 (뜨기 전에 grab_set하면 'window not viewable' 오류가 날 수 있다)
        self.after(50, self._grab)

    def _grab(self):
        try:
            if self.winfo_exists():
                self.grab_set()
        except tk.TclError:
            pass

    def try_login(self):
        user_id = self.ent_id.get().strip()
        pwd = self.ent_pwd.get().strip()

        if not (user_id and pwd):
            messagebox.showwarning("주의", "아이디와 비밀번호를 입력해주세요.", parent=self)
            return

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT role FROM users WHERE user_id = ? AND password = ?", (user_id, pwd))
        row = cursor.fetchone()
        conn.close()

        if not row:
            messagebox.showerror("오류", "로그인 정보가 틀렸습니다.", parent=self)
            return
        if row[0] != "admin":
            messagebox.showerror("오류", "관리자 권한이 없습니다.", parent=self)
            return

        # 창을 먼저 닫고 나서 화면을 바꾼다 (닫힌 입력칸을 건드리지 않도록)
        self.destroy()
        self.app.set_admin_logged_in(True, user_id)


# ==========================================
# 2-1. 디자인 — 탁한 흰색 + 파란색 두 가지 색만 쓴다
# ==========================================
BG = "#EFEEEA"              # 앱 바탕 (탁한 흰색)
PANEL = "#FAF9F6"           # 카드
PANEL_ALT = "#F2F1ED"       # 카드 안의 보조 영역
LINE = "#DEDCD5"            # 테두리
TEXT = "#1A2233"
TEXT_SUB = "#4A5262"
MUTED = "#8A8F99"
BLUE = "#1F4FBF"
BLUE_DARK = "#173C96"
BLUE_SOFT = "#E3E9F6"
BLUE_SOFT_HOVER = "#D3DDF2"
BLUE_LINE = "#C3D0EC"
CAM_BG = "#111B30"          # 카메라 영상 뒤 배경
# 상태 표시에만 쓰는 색 (카메라 상자 색과 맞춘다)
OK = "#16A34A"
WARN = "#B26A00"
ERR = "#C2362B"

VERIFY_CARD_SIZE = (1160, 640)   # 출근/퇴근/출석 조회 확인 카드 (1600x900 기준)
RESULT_CARD_SIZE = (760, 500)
STATS_CARD_SIZE = (1160, 740)
TEST_PANEL_WIDTH = 510           # 카메라 테스트 오른쪽 안내 패널 — 카메라가 화면의 약 2/3를 쓴다


def _hex_rgb(color):
    color = color.lstrip("#")
    return tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))


_render_cache = {}


def _cached(key, make):
    """같은 모양을 다시 그리지 않도록 PIL 이미지를 기억해 둔다 (창 크기 변경 때만 새로 그린다)."""
    img = _render_cache.get(key)
    if img is None:
        if len(_render_cache) > 256:
            _render_cache.clear()
        img = _render_cache[key] = make()
    return img


def render_round_rect(w, h, radius, fill, bg=None, outline=None, outline_width=1, shadow=0):
    """
    둥근 사각형 그림. Tk 캔버스는 곡선을 부드럽게(안티에일리어싱) 그리지 못해서
    PIL로 크게 그린 뒤 줄여서 가장자리를 매끄럽게 만든다.
    bg=None이면 모서리 바깥이 투명하다. shadow>0이면 그만큼 바깥 여백에 은은한 그림자를 깐다.
    """
    w, h = max(2, int(w)), max(2, int(h))
    radius = max(0, min(int(radius), (w - 2 * shadow) // 2, (h - 2 * shadow) // 2))
    key = ("rr", w, h, radius, fill, bg, outline, outline_width, shadow)

    def make():
        base = Image.new("RGBA", (w, h), (_hex_rgb(bg) + (255,)) if bg else (0, 0, 0, 0))
        if shadow:
            mask = Image.new("L", (w, h), 0)
            ImageDraw.Draw(mask).rounded_rectangle(
                (shadow, shadow + shadow // 3, w - shadow, h - shadow + shadow // 3), radius, fill=34)
            mask = mask.filter(ImageFilter.GaussianBlur(shadow / 2))
            tint = Image.new("RGBA", (w, h), (23, 37, 84, 255))
            tint.putalpha(mask)
            base = Image.alpha_composite(base, tint)
        s = 3 if w * h < 400_000 else 2
        layer = Image.new("RGBA", (w * s, h * s), (0, 0, 0, 0))
        box = (shadow * s, shadow * s, (w - shadow) * s - 1, (h - shadow) * s - 1)
        ImageDraw.Draw(layer).rounded_rectangle(
            box, radius * s, fill=_hex_rgb(fill) + (255,),
            outline=(_hex_rgb(outline) + (255,)) if outline else None,
            width=outline_width * s if outline else 0)
        layer = layer.resize((w, h), Image.LANCZOS)
        return Image.alpha_composite(base, layer)

    return _cached(key, make)


def render_circle(d, fill):
    def make():
        s = 4
        img = Image.new("RGBA", (d * s, d * s), (0, 0, 0, 0))
        ImageDraw.Draw(img).ellipse((0, 0, d * s - 1, d * s - 1), fill=_hex_rgb(fill) + (255,))
        return img.resize((d, d), Image.LANCZOS)
    return _cached(("circle", d, fill), make)


def render_icon(kind, size, color):
    """선 아이콘 (출근·퇴근·얼굴·안내·확인·시계·카메라 꺼짐·주의). 글꼴에 없는 기호를 쓰지 않으려고 직접 그린다."""
    def make():
        s = 4
        S = size * s
        img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        rgb = _hex_rgb(color) + (255,)
        lw = max(2, int(S * 0.09))

        def P(x, y):
            return (x * S, y * S)

        def dot(x, y, r):
            d.ellipse((x * S - r * S, y * S - r * S, x * S + r * S, y * S + r * S), fill=rgb)

        def line(*pts):
            pts = [P(*p) for p in pts]
            d.line(pts, fill=rgb, width=lw, joint="curve")
            r = lw / 2
            for x, y in (pts[0], pts[-1]):
                d.ellipse((x - r, y - r, x + r, y + r), fill=rgb)

        def ring():
            m = lw / 2 + S * 0.06
            d.ellipse((m, m, S - m, S - m), outline=rgb, width=lw)

        if kind == "in":
            line((0.55, 0.14), (0.84, 0.14), (0.84, 0.86), (0.55, 0.86))
            line((0.12, 0.5), (0.62, 0.5))
            line((0.44, 0.32), (0.62, 0.5), (0.44, 0.68))
        elif kind == "out":
            line((0.45, 0.14), (0.16, 0.14), (0.16, 0.86), (0.45, 0.86))
            line((0.38, 0.5), (0.88, 0.5))
            line((0.70, 0.32), (0.88, 0.5), (0.70, 0.68))
        elif kind == "face":
            line((0.08, 0.30), (0.08, 0.08), (0.30, 0.08))
            line((0.70, 0.08), (0.92, 0.08), (0.92, 0.30))
            line((0.92, 0.70), (0.92, 0.92), (0.70, 0.92))
            line((0.30, 0.92), (0.08, 0.92), (0.08, 0.70))
            dot(0.36, 0.42, 0.055)
            dot(0.64, 0.42, 0.055)
            d.arc((*P(0.32, 0.40), *P(0.68, 0.72)), 25, 155, fill=rgb, width=lw)
        elif kind == "info":
            ring()
            dot(0.5, 0.31, 0.06)
            line((0.5, 0.48), (0.5, 0.72))
        elif kind == "alert":
            ring()
            line((0.5, 0.28), (0.5, 0.55))
            dot(0.5, 0.71, 0.06)
        elif kind == "check":
            line((0.18, 0.52), (0.42, 0.75), (0.84, 0.28))
        elif kind == "clock":
            ring()
            line((0.5, 0.28), (0.5, 0.5), (0.68, 0.62))
        elif kind == "camera_off":
            line((0.10, 0.30), (0.66, 0.30), (0.66, 0.76), (0.10, 0.76), (0.10, 0.30))
            line((0.66, 0.46), (0.92, 0.32), (0.92, 0.74), (0.66, 0.60))
            line((0.10, 0.10), (0.90, 0.92))
        return img.resize((size, size), Image.LANCZOS)

    return _cached(("icon", kind, size, color), make)


def render_ring(size, width, track, color=None, start=0, extent=360, bg=PANEL):
    """상태 표시용 원형 링. extent<360이면 끝이 둥근 호(돌아가는 로딩 표시)를 그린다."""
    def make():
        s = 3
        S = size * s
        img = Image.new("RGB", (S, S), _hex_rgb(bg))
        d = ImageDraw.Draw(img)
        m = s * 2
        box = (m, m, S - m, S - m)
        d.ellipse(box, outline=_hex_rgb(track), width=width * s)
        if color:
            rgb = _hex_rgb(color)
            if extent >= 360:
                d.ellipse(box, outline=rgb, width=width * s)
            else:
                d.arc(box, start, start + extent, fill=rgb, width=width * s)
                c = S / 2
                r_mid = (S - 2 * m) / 2 - width * s / 2
                cap = width * s / 2
                for a in (start, start + extent):
                    x = c + r_mid * math.cos(math.radians(a))
                    y = c + r_mid * math.sin(math.radians(a))
                    d.ellipse((x - cap, y - cap, x + cap, y + cap), fill=rgb)
        return img.resize((size, size), Image.LANCZOS)

    return _cached(("ring", size, width, track, color, start, extent, bg), make)


def compose_center(base, icon):
    """base 그림 가운데에 icon을 얹는다 (로고 타일 등)."""
    out = base.copy()
    out.alpha_composite(icon, ((base.width - icon.width) // 2, (base.height - icon.height) // 2))
    return out


class RoundedPanel(tk.Canvas):
    """모서리가 둥근 카드 (배치된 크기를 그대로 채운다). 안쪽 내용은 .body(Frame)에 넣는다."""

    def __init__(self, master, fill=PANEL, bg=BG, radius=None, outline=LINE, shadow=0, pad=None,
                 width=None, height=None):
        super().__init__(master, bg=bg, highlightthickness=0, bd=0, width=width or 1, height=height or 1)
        self.fill, self._bg, self.outline, self.shadow = fill, bg, outline, shadow
        self.radius = px(16) if radius is None else radius
        self.inset = shadow + (px(20) if pad is None else pad)
        self._photo = None
        self._img_item = self.create_image(0, 0, anchor=tk.NW)
        self.body = tk.Frame(self, bg=fill)
        self._win = self.create_window(self.inset, self.inset, window=self.body, anchor=tk.NW)
        self.bind("<Configure>", self._on_configure)

    def _on_configure(self, event):
        img = render_round_rect(event.width, event.height, self.radius, self.fill, self._bg,
                                self.outline, shadow=self.shadow)
        self._photo = ImageTk.PhotoImage(img)
        self.itemconfig(self._img_item, image=self._photo)
        self.itemconfig(self._win, width=max(1, event.width - 2 * self.inset),
                        height=max(1, event.height - 2 * self.inset))


class RoundedBox(tk.Frame):
    """
    높이가 내용에 맞춰지는 둥근 상자. 일반 pack 배치로 크기가 정해지고,
    둥근 배경 그림은 뒤에 깔린 라벨이 크기에 맞춰 다시 그린다.
    """

    def __init__(self, master, fill=PANEL_ALT, bg=PANEL, radius=None, pad=None, outline=None):
        super().__init__(master, bg=bg)
        self.fill, self._bg, self.outline = fill, bg, outline
        self.radius = px(12) if radius is None else radius
        self._photo = None
        self._bg_label = tk.Label(self, bg=bg, bd=0)
        self._bg_label.place(x=0, y=0, relwidth=1, relheight=1)
        self.body = tk.Frame(self, bg=fill)
        pad = px(16) if pad is None else pad
        self.body.pack(fill=tk.BOTH, expand=True, padx=pad, pady=pad)
        self.bind("<Configure>", self._redraw)

    def _redraw(self, event):
        self._photo = ImageTk.PhotoImage(
            render_round_rect(event.width, event.height, self.radius, self.fill, self._bg, self.outline))
        self._bg_label.config(image=self._photo)


class RoundButton(tk.Canvas):
    """둥근 버튼. command가 없으면 글자 표시용 알약(pill)으로도 쓴다."""

    def __init__(self, master, text="", command=None, width=None, height=None, font=None, radius=None,
                 fill=BLUE, hover=BLUE_DARK, fg="white", hover_fg=None, outline=None, bg=PANEL,
                 padx=None, icon=None):
        self._font = font or (UI_FONT, 12, "bold")
        self._padx = px(22) if padx is None else padx
        self._icon = ImageTk.PhotoImage(icon) if icon is not None else None
        self._fixed_width = width
        height = height or px(48)
        self._bw, self._bh = (width or self._measure(text)), height
        super().__init__(master, width=self._bw, height=height, bg=bg, highlightthickness=0, bd=0,
                         cursor="hand2" if command else "")
        self._bg = bg
        self.radius = px(10) if radius is None else radius
        self.command = command
        self._hovering = False
        self._style = None
        self._img_item = self.create_image(0, 0, anchor=tk.NW)
        self._icon_item = self.create_image(0, 0, anchor=tk.W)
        self._text_item = self.create_text(0, 0, text=text, font=self._font, anchor=tk.W)
        self._text = text
        self.set_style(fill, hover, fg, hover_fg, outline)
        if command:
            self.bind("<Enter>", self._on_enter)
            self.bind("<Leave>", self._on_leave)
            self.bind("<ButtonRelease-1>", self._on_click)

    def _measure(self, text):
        width = tkfont.Font(font=self._font).measure(text) + 2 * self._padx
        if self._icon is not None:
            width += self._icon.width() + px(10)
        return width

    def _layout(self):
        text_w = tkfont.Font(font=self._font).measure(self._text)
        icon_w = self._icon.width() + px(10) if self._icon is not None else 0
        x = (self._bw - text_w - icon_w) // 2
        if self._icon is not None:
            self.itemconfig(self._icon_item, image=self._icon)
            self.coords(self._icon_item, x, self._bh // 2)
        self.coords(self._text_item, x + icon_w, self._bh // 2)

    def set_style(self, fill, hover=None, fg="white", hover_fg=None, outline=None):
        style = (fill, hover, fg, hover_fg, outline, self._bw)
        if style == self._style:
            return
        self._style = style
        self._fg, self._hover_fg = fg, hover_fg or fg
        self._img_normal = ImageTk.PhotoImage(
            render_round_rect(self._bw, self._bh, self.radius, fill, self._bg, outline))
        self._img_hover = ImageTk.PhotoImage(
            render_round_rect(self._bw, self._bh, self.radius, hover or fill, self._bg, outline))
        self._apply()
        self._layout()

    def _apply(self):
        self.itemconfig(self._img_item, image=self._img_hover if self._hovering else self._img_normal)
        self.itemconfig(self._text_item, fill=self._hover_fg if self._hovering else self._fg)

    def set_text(self, text):
        if text == self._text:
            return
        self._text = text
        self.itemconfig(self._text_item, text=text)
        if self._fixed_width is None:
            new_w = self._measure(text)
            if new_w != self._bw:
                self._bw = new_w
                self.configure(width=new_w)
                fill, hover, fg, hover_fg, outline, _ = self._style
                self._style = None
                self.set_style(fill, hover, fg, hover_fg, outline)
                return
        self._layout()

    def _on_enter(self, _event=None):
        self._hovering = True
        self._apply()

    def _on_leave(self, _event=None):
        self._hovering = False
        self._apply()

    def _on_click(self, event):
        if 0 <= event.x < self._bw and 0 <= event.y < self._bh and self.command:
            self.command()


class ProgressBar(tk.Canvas):
    """얇은 둥근 진행 막대. marker를 주면 그 위치(0~1)에 기준선을 긋는다."""

    def __init__(self, master, width, height, bg=PANEL, track=BLUE_SOFT, color=BLUE):
        super().__init__(master, width=width, height=height, bg=bg, highlightthickness=0, bd=0)
        self._bw, self._bh, self._bg, self._track, self._color = width, height, bg, track, color
        self._key = None
        self._item = self.create_image(0, 0, anchor=tk.NW)

    def set(self, fraction, color=None, marker=None):
        fraction = min(1.0, max(0.0, fraction))
        color = color or self._color
        key = (round(fraction * self._bw), color, marker)
        if key == self._key:
            return
        self._key = key
        w, h, s = self._bw, self._bh, 3
        img = Image.new("RGB", (w * s, h * s), _hex_rgb(self._bg))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle((0, 0, w * s - 1, h * s - 1), h * s // 2, fill=_hex_rgb(self._track))
        if fraction > 0:
            d.rounded_rectangle((0, 0, max(h * s, int(w * s * fraction)) - 1, h * s - 1), h * s // 2,
                                fill=_hex_rgb(color))
        if marker is not None:
            mx = int(w * s * marker)
            d.rectangle((mx - s, 0, mx + s, h * s), fill=_hex_rgb(TEXT))
        self._photo = ImageTk.PhotoImage(img.resize((w, h), Image.LANCZOS))
        self.itemconfig(self._item, image=self._photo)


def make_pill(master, text, bg=PANEL, fill=BLUE_SOFT, fg=BLUE, font_size=10):
    """상태를 알려주는 작은 알약 모양 표시."""
    h = px(34)
    return RoundButton(master, text=text, height=h, radius=h // 2, fill=fill, fg=fg, bg=bg,
                       font=(UI_FONT, font_size, "bold"), padx=px(16))


def icon_label(master, kind, size, color, bg):
    photo = ImageTk.PhotoImage(render_icon(kind, size, color))
    lbl = tk.Label(master, image=photo, bg=bg, bd=0)
    lbl.image = photo  # 참조를 들고 있어야 그림이 사라지지 않는다
    return lbl


_blink_hint_frames_cache = {}


def _get_blink_hint_frames(size):
    """'눈을 깜빡여 주세요' 안내에 같이 보여줄 힌트 이미지를 size(px) 정사각형으로 불러온다.
    움직이는 GIF/WEBP면 프레임 전체를 (PhotoImage, 이 프레임을 보여줄 시간ms) 목록으로 돌려준다.
    정지 이미지면 프레임이 하나뿐인 목록. 파일이 없거나 못 읽으면 빈 목록(이미지 없이 문구만 보인다)."""
    frames = _blink_hint_frames_cache.get(size)
    if frames is not None:
        return frames
    frames = []
    if os.path.exists(BLINK_HINT_IMAGE_PATH):
        try:
            # 네모난 이미지의 모서리가 바깥 링 위로 튀어나오지 않게 동그랗게 잘라낸다 (가장자리는 부드럽게)
            s = 4
            mask = Image.new("L", (size * s, size * s), 0)
            ImageDraw.Draw(mask).ellipse((0, 0, size * s - 1, size * s - 1), fill=255)
            mask = mask.resize((size, size), Image.LANCZOS)
            src = Image.open(BLINK_HINT_IMAGE_PATH)
            for frame in ImageSequence.Iterator(src):
                duration = frame.info.get("duration", 100) or 100
                resized = frame.convert("RGBA").resize((size, size), Image.LANCZOS)
                alpha = Image.composite(resized.getchannel("A"), Image.new("L", (size, size), 0), mask)
                resized.putalpha(alpha)
                frames.append((ImageTk.PhotoImage(resized), duration))
        except Exception as e:
            print(f"[출석] 눈 깜빡임 안내 이미지를 불러오지 못했습니다 ({e})")
            frames = []
    _blink_hint_frames_cache[size] = frames
    return frames


class StatusVisual(tk.Canvas):
    """
    얼굴 확인 상태를 보여주는 원형 표시.
      search : 링이 돌아간다 (얼굴을 찾는 중)
      face   : 학생을 찾았을 때 — 눈을 깜빡이는 얼굴 GIF
      done / warn / error : 완료 체크 · 주의 표시
    """
    SPIN_FRAMES = 30

    def __init__(self, master, size, bg=PANEL):
        super().__init__(master, width=size, height=size, bg=bg, highlightthickness=0, bd=0)
        self.size = size
        ring_w = max(4, size // 30)
        c = size // 2
        self._spin = [ImageTk.PhotoImage(render_ring(size, ring_w, BLUE_SOFT, BLUE,
                                                     start=i * 360 // self.SPIN_FRAMES - 90, extent=100, bg=bg))
                      for i in range(self.SPIN_FRAMES)]
        self._rings = {
            "track": ImageTk.PhotoImage(render_ring(size, ring_w, BLUE_SOFT, bg=bg)),
            "blue": ImageTk.PhotoImage(render_ring(size, ring_w, BLUE_SOFT, BLUE, bg=bg)),
            "warn": ImageTk.PhotoImage(render_ring(size, ring_w, BLUE_SOFT, WARN, bg=bg)),
            "error": ImageTk.PhotoImage(render_ring(size, ring_w, BLUE_SOFT, ERR, bg=bg)),
        }
        icon_size = int(size * 0.4)
        self._icons = {
            "search": ImageTk.PhotoImage(render_icon("face", icon_size, BLUE_LINE)),
            "done": ImageTk.PhotoImage(render_icon("check", icon_size, BLUE)),
            "warn": ImageTk.PhotoImage(render_icon("alert", icon_size, WARN)),
            "error": ImageTk.PhotoImage(render_icon("alert", icon_size, ERR)),
        }
        self._gif = _get_blink_hint_frames(int(size * 0.68))
        self._ring_item = self.create_image(c, c)
        self._center_item = self.create_image(c, c)
        self._mode = None
        self._job = None
        self._idx = 0
        self.bind("<Destroy>", lambda e: self._cancel() if e.widget is self else None)

    def _cancel(self):
        if self._job is not None:
            self.after_cancel(self._job)
            self._job = None

    def set_mode(self, mode):
        if mode == self._mode:
            return
        self._mode = mode
        self._cancel()
        self._idx = 0
        if mode == "search":
            self.itemconfig(self._center_item, image=self._icons["search"])
            self._spin_step()
        elif mode == "face":
            self.itemconfig(self._ring_item, image=self._rings["blue"])
            if self._gif:
                self._gif_step()
            else:
                self.itemconfig(self._center_item, image=self._icons["search"])
        else:
            ring = {"done": "blue", "warn": "warn", "error": "error"}.get(mode, "track")
            self.itemconfig(self._ring_item, image=self._rings[ring])
            self.itemconfig(self._center_item, image=self._icons.get(mode, ""))

    def _spin_step(self):
        self.itemconfig(self._ring_item, image=self._spin[self._idx])
        self._idx = (self._idx + 1) % self.SPIN_FRAMES
        self._job = self.after(33, self._spin_step)

    def _gif_step(self):
        photo, duration = self._gif[self._idx]
        self.itemconfig(self._center_item, image=photo)
        self._idx = (self._idx + 1) % len(self._gif)
        if len(self._gif) > 1:
            self._job = self.after(duration, self._gif_step)


class StepBox(tk.Canvas):
    """안내 단계 상자 (번호 · 안내 멘트). 상태: idle / active(지금 할 일) / done(완료)."""

    def __init__(self, master, number, title, width, height, bg=PANEL):
        super().__init__(master, width=width, height=height, bg=bg, highlightthickness=0, bd=0)
        self._state = None
        r = px(14)
        self._boxes = {
            "idle": ImageTk.PhotoImage(render_round_rect(width, height, r, PANEL_ALT, bg)),
            "active": ImageTk.PhotoImage(render_round_rect(width, height, r, BLUE_SOFT, bg, BLUE_LINE)),
            "done": ImageTk.PhotoImage(render_round_rect(width, height, r, PANEL_ALT, bg)),
        }
        d = px(40)
        self._circles = {
            "idle": ImageTk.PhotoImage(render_circle(d, BLUE_SOFT)),
            "active": ImageTk.PhotoImage(render_circle(d, BLUE)),
            "done": ImageTk.PhotoImage(compose_center(render_circle(d, BLUE),
                                                      render_icon("check", int(d * 0.55), "#FFFFFF"))),
        }
        left = px(18)
        cx, cy = left + d // 2, height // 2
        text_x = left + d + px(14)
        self._box_item = self.create_image(0, 0, anchor=tk.NW)
        self._circle_item = self.create_image(cx, cy)
        self._num_item = self.create_text(cx, cy, text=str(number), font=(UI_FONT, 12, "bold"))
        self._title_item = self.create_text(text_x, cy, text=title, anchor=tk.W,
                                            font=(UI_FONT, 12, "bold"), width=width - text_x - px(12))
        self.set_state("idle")

    def set_state(self, state):
        if state == self._state:
            return
        self._state = state
        self.itemconfig(self._box_item, image=self._boxes[state])
        self.itemconfig(self._circle_item, image=self._circles[state])
        self.itemconfig(self._num_item, state=tk.HIDDEN if state == "done" else tk.NORMAL,
                        fill=BLUE if state == "idle" else "white")
        self.itemconfig(self._title_item, fill={"idle": TEXT_SUB, "active": BLUE_DARK, "done": TEXT}[state])


class ActionCard(tk.Canvas):
    """메인 화면의 큰 출근/퇴근 버튼 카드."""

    def __init__(self, master, log_type, command, width, height, bg=BG):
        super().__init__(master, width=width, height=height, bg=bg, highlightthickness=0, bd=0, cursor="hand2")
        self.command = command
        primary = log_type == "CHECK_IN"
        sh = px(12)
        r = px(22)
        if primary:
            fill, hover, outline, hover_outline = BLUE, BLUE_DARK, None, None
            title_fg, accent, tile_fill, icon_color, sep = "white", "#BFD0F5", "#3A64C9", "#FFFFFF", "#3D66CC"
        else:
            fill, hover, outline, hover_outline = PANEL, "#F3F5FA", BLUE_LINE, BLUE
            title_fg, accent, tile_fill, icon_color, sep = TEXT, BLUE, BLUE_SOFT, BLUE, LINE
        self._normal = ImageTk.PhotoImage(render_round_rect(width, height, r, fill, bg, outline, shadow=sh))
        self._hover = ImageTk.PhotoImage(render_round_rect(width, height, r, hover, bg, hover_outline,
                                                           outline_width=2, shadow=sh))
        tile = px(64)
        self._tile = ImageTk.PhotoImage(compose_center(
            render_round_rect(tile, tile, px(16), tile_fill),
            render_icon("in" if primary else "out", px(32), icon_color)))

        x0 = sh + px(32)
        y0 = sh + px(30)
        action = ACTION_LABELS[log_type]
        self._bg_item = self.create_image(0, 0, image=self._normal, anchor=tk.NW)
        self.create_image(x0, y0, image=self._tile, anchor=tk.NW)
        self.create_text(width - x0, y0 + px(4), text="CHECK-IN" if primary else "CHECK-OUT",
                         font=(UI_FONT, 10, "bold"), fill=accent, anchor=tk.NE)
        self.create_text(x0, y0 + px(96), text=action, font=(UI_FONT, 40, "bold"), fill=title_fg, anchor=tk.NW)
        line_y = height - sh - px(62)
        self.create_line(x0, line_y, width - x0, line_y, fill=sep)
        self.create_text(x0, height - sh - px(31), text=f"{action} 시작",
                         font=(UI_FONT, 12, "bold"), fill=accent if not primary else "white", anchor=tk.W)
        self.create_text(width - x0, height - sh - px(33), text="→", font=(UI_FONT, 18, "bold"),
                         fill=accent if not primary else "white", anchor=tk.E)

        self.bind("<Enter>", lambda e: self.itemconfig(self._bg_item, image=self._hover))
        self.bind("<Leave>", lambda e: self.itemconfig(self._bg_item, image=self._normal))
        self.bind("<ButtonRelease-1>", self._on_click)

    def _on_click(self, event):
        if 0 <= event.x < int(self.cget("width")) and 0 <= event.y < int(self.cget("height")):
            self.command()


class VerifyCard(RoundedPanel):
    """
    출근 / 퇴근 / 출석 조회 버튼을 누른 뒤 얼굴이 확인될 때까지 보여주는 화면.
    카메라 영상은 띄우지 않는다 — 카메라는 뒤에서 얼굴 확인에만 쓴다.
    """
    STEPS = {
        "loading": ("active", "idle", "idle"),
        "search": ("active", "idle", "idle"),
        "warn": ("active", "idle", "idle"),
        "error": ("active", "idle", "idle"),
        "blink": ("done", "active", "idle"),
        "done": ("done", "done", "done"),
    }
    VISUAL = {"loading": "search", "search": "search", "blink": "face",
              "warn": "warn", "error": "error", "done": "done"}
    TITLE_COLOR = {"loading": TEXT_SUB, "search": TEXT, "blink": BLUE,
                   "warn": WARN, "error": ERR, "done": BLUE}

    def __init__(self, app, kind, master, width, height):
        super().__init__(master, fill=PANEL, bg=BG, radius=px(24), outline=LINE, shadow=px(16), pad=px(40))
        self._limit = SESSION_TIMEOUT
        self._last = None
        b = self.body
        inner_w = width - 2 * self.inset

        if kind == "LOOKUP":
            title = "출석 조회"
            step3 = "확인 되면 출석 현황이 표시 됩니다."
        else:
            action = ACTION_LABELS[kind]
            title = action
            step3 = f"확인 되면 자동으로 {action} 처리 됩니다."

        # 1) 제목
        tk.Label(b, text=title, font=(UI_FONT, 28, "bold"), bg=PANEL, fg=TEXT).pack(anchor=tk.W, pady=(0, px(20)))

        # 2) 안내 멘트 — 지금 해야 할 단계를 파랗게 강조한다
        steps_row = tk.Frame(b, bg=PANEL)
        steps_row.pack(fill=tk.X)
        gap = px(14)
        box_w = (inner_w - 2 * gap) // 3
        self.steps = []
        for i, main in enumerate(("카메라를 정면으로 바라봐 주세요", "눈을 한 번 깜빡여 주세요", step3)):
            box = StepBox(steps_row, i + 1, main, box_w, px(76))
            box.pack(side=tk.LEFT, padx=(0 if i == 0 else gap, 0))
            self.steps.append(box)

        # 4) 하단 — 남은 시간 · 취소 (내용보다 먼저 배치해야 공간이 모자라도 잘리지 않는다)
        bottom = tk.Frame(b, bg=PANEL)
        bottom.pack(side=tk.BOTTOM, fill=tk.X)
        RoundButton(bottom, text="취소", command=app.show_idle, width=px(160), height=px(50),
                    fill=BLUE_SOFT, hover=BLUE_SOFT_HOVER, fg=BLUE, bg=PANEL).pack(side=tk.RIGHT)
        timer = tk.Frame(bottom, bg=PANEL)
        timer.pack(side=tk.LEFT, anchor=tk.W)
        self.lbl_remaining = tk.Label(timer, text="", font=(UI_FONT, 10), bg=PANEL, fg=MUTED)
        self.lbl_remaining.pack(anchor=tk.W, pady=(0, px(6)))
        self.timer_bar = ProgressBar(timer, px(380), px(6))
        self.timer_bar.pack(anchor=tk.W)

        # 3) 얼굴 확인 상태 — "얼굴을 찾는 중.." / "OOO 학생 얼굴 확인 중" + 눈 깜빡이는 얼굴
        status = tk.Frame(b, bg=PANEL)
        status.pack(expand=True)
        self.visual = StatusVisual(status, px(196))
        self.visual.pack()
        self.lbl_title = tk.Label(status, text="", font=(UI_FONT, 20, "bold"), bg=PANEL, fg=TEXT)
        self.lbl_title.pack(pady=(px(14), px(2)))
        self.lbl_detail = tk.Label(status, text="", font=(UI_FONT, 12), bg=PANEL, fg=TEXT_SUB,
                                   justify=tk.CENTER)
        self.lbl_detail.pack()

        self.set_state("search", "얼굴을 찾는 중..")

    def set_state(self, state, title, detail="", sim=None):
        # 화면이 매 프레임 부르므로 내용이 바뀔 때만 다시 그린다
        if (state, title, detail) == self._last:
            return
        self._last = (state, title, detail)
        self.visual.set_mode(self.VISUAL[state])
        self.lbl_title.config(text=title, fg=self.TITLE_COLOR[state])
        self.lbl_detail.config(text=detail)
        for box, step_state in zip(self.steps, self.STEPS[state]):
            box.set_state(step_state)

    def set_remaining(self, seconds):
        text = f"{seconds}초"
        if self.lbl_remaining.cget("text") != text:
            self.lbl_remaining.config(text=text)
        self.timer_bar.set(seconds / self._limit)


class ResultCard(RoundedPanel):
    """출근/퇴근 처리 결과 — RESULT_POPUP_SECONDS 뒤 저절로 메인 화면으로 돌아간다."""

    def __init__(self, app, student, log_type, success, detail, master):
        super().__init__(master, fill=PANEL, bg=BG, radius=px(24), outline=LINE, shadow=px(16), pad=px(40))
        self.app = app
        self.student = student
        b = self.body
        action = ACTION_LABELS[log_type]
        name = student["name"]

        visual = StatusVisual(b, px(150))
        visual.pack(pady=(px(10), px(22)))
        visual.set_mode("done" if success else "warn")
        headline = f"{name} 학생 {action} 되었습니다." if success else f"{name} 학생 {action} 처리가 되지 않았습니다."
        tk.Label(b, text=headline, font=(UI_FONT, 22, "bold"), bg=PANEL, fg=TEXT).pack()
        tk.Label(b, text=detail, font=(UI_FONT, 12), bg=PANEL, fg=TEXT_SUB, justify=tk.CENTER) \
            .pack(pady=(px(8), 0))

        btn_row = tk.Frame(b, bg=PANEL)
        btn_row.pack(side=tk.BOTTOM)
        RoundButton(btn_row, text="출석 조회", command=self.open_stats, width=px(200), height=px(52),
                    fill=BLUE_SOFT, hover=BLUE_SOFT_HOVER, fg=BLUE, bg=PANEL).pack(side=tk.LEFT, padx=px(6))
        self.btn_close = RoundButton(btn_row, text="", command=app.show_idle, width=px(200), height=px(52),
                                     bg=PANEL)
        self.btn_close.pack(side=tk.LEFT, padx=px(6))

        self._remaining = RESULT_POPUP_SECONDS
        self._tick()

    def _tick(self):
        if not self.winfo_exists():
            return
        if self._remaining <= 0:
            self.app.show_idle()
            return
        self.btn_close.set_text(f"메인으로 ({self._remaining})")
        self._remaining -= 1
        self.after(1000, self._tick)

    def open_stats(self):
        self.app.show_stats(self.student)


class StatsCard(RoundedPanel):
    """본인 확인 후 보여주는 이번 주 출석 현황. STATS_POPUP_SECONDS 뒤 저절로 메인 화면으로 돌아간다."""
    MAX_ROWS = 6

    def __init__(self, app, student_info, master, width):
        super().__init__(master, fill=PANEL, bg=BG, radius=px(24), outline=LINE, shadow=px(16), pad=px(36))
        self.app = app
        self.student_info = student_info
        b = self.body
        inner_w = width - 2 * self.inset
        weekly = get_weekly_attendance_stats(student_info["user_id"])

        # 상단 — 이름 · 학번 · 자동 닫힘
        top = tk.Frame(b, bg=PANEL)
        top.pack(fill=tk.X)
        name_box = tk.Frame(top, bg=PANEL)
        name_box.pack(side=tk.LEFT)
        tk.Label(name_box, text=f"{student_info['name']} 학생의 이번 주 출석 현황", font=(UI_FONT, 22, "bold"),
                 bg=PANEL, fg=TEXT).pack(anchor=tk.W)
        tk.Label(name_box, text=f"학번 {student_info['user_id']}",
                 font=(UI_FONT, 10), bg=PANEL, fg=MUTED).pack(anchor=tk.W, pady=(px(4), 0))
        self.btn_close = RoundButton(top, text="", command=app.show_idle, width=px(230), height=px(50), bg=PANEL)
        self.btn_close.pack(side=tk.RIGHT, anchor=tk.N)

        # 지표 4개
        tiles = tk.Frame(b, bg=PANEL)
        tiles.pack(fill=tk.X, pady=(px(22), px(18)))
        gap = px(14)
        tile_w = (inner_w - 3 * gap) // 4

        def tile(index, label, value, note, note_color):
            panel = RoundedPanel(tiles, fill=PANEL_ALT, bg=PANEL, radius=px(16), outline=None, pad=px(18),
                                 width=tile_w, height=px(122))
            panel.pack(side=tk.LEFT, padx=(0 if index == 0 else gap, 0))
            tk.Label(panel.body, text=label, font=(UI_FONT, 10, "bold"), bg=PANEL_ALT, fg=MUTED).pack(anchor=tk.W)
            lbl_value = tk.Label(panel.body, text=value, font=(UI_FONT, 17, "bold"), bg=PANEL_ALT, fg=TEXT)
            lbl_value.pack(anchor=tk.W, pady=(px(6), px(2)))
            lbl_note = tk.Label(panel.body, text=note, font=(UI_FONT, 10, "bold"), bg=PANEL_ALT, fg=note_color)
            lbl_note.pack(anchor=tk.W)
            return lbl_value, lbl_note

        def missing(text):
            return (f"부족 {text}", ERR) if text else ("기준 충족", BLUE)

        tile(0, f"이번 주 출근 일수 (기준 {WEEKLY_REQUIRED_DAYS}일)", f"{weekly['days']}일",
             *missing(f"{weekly['missing_days']}일" if weekly["missing_days"] else ""))
        tile(1, f"이번 주 출근 시간 (기준 {WEEKLY_REQUIRED_SECONDS // 3600}시간)", format_duration(weekly["seconds"]),
             *missing(format_duration(weekly["missing_seconds"]) if weekly["missing_seconds"] else ""))
        self.lbl_today_value, self.lbl_today_note = tile(2, "오늘 출근", "", "", MUTED)
        penalty = student_info.get("penalty") or 0
        tile(3, "패널티", f"{penalty}개",
             "양호" if penalty == 0 else "주의" if penalty == 1 else "경고 (3진 아웃제)",
             BLUE if penalty == 0 else WARN if penalty == 1 else ERR)

        # 이번 주 출퇴근 기록 표
        table = RoundedPanel(b, fill=PANEL, bg=PANEL, radius=px(16), outline=LINE, pad=px(22))
        table.pack(fill=tk.BOTH, expand=True)
        t = table.body
        tk.Label(t, text="이번 주 출퇴근 기록", font=(UI_FONT, 12, "bold"), bg=PANEL, fg=TEXT) \
            .grid(row=0, column=0, columnspan=4, sticky=tk.W, pady=(0, px(10)))
        for col in range(4):
            t.grid_columnconfigure(col, weight=1, uniform="cols")
        for col, heading in enumerate(("날짜", "출근", "퇴근", "인정 시간")):
            tk.Label(t, text=heading, font=(UI_FONT, 9, "bold"), bg=PANEL_ALT, fg=MUTED, anchor=tk.W,
                     padx=px(12), pady=px(7)).grid(row=1, column=col, sticky=tk.EW)
        sessions = weekly["sessions"]
        if not sessions:
            tk.Label(t, text="이번 주 기록이 없습니다", font=(UI_FONT, 11), bg=PANEL, fg=MUTED) \
                .grid(row=2, column=0, columnspan=4, sticky=tk.W, padx=px(12), pady=px(14))
        shown = sessions[-self.MAX_ROWS:]
        for row, s in enumerate(shown):
            try:
                day_text = format_date_ko(datetime.strptime(s["date"], "%Y-%m-%d").date())[5:]
            except (TypeError, ValueError):
                day_text = s["date"]
            seconds = session_seconds(s)
            cells = (day_text, s["check_in"] or "-", s["check_out"] or "-",
                     format_duration(seconds) if seconds else "-")
            for col, value in enumerate(cells):
                tk.Label(t, text=value, font=(UI_FONT, 11, "bold" if col == 3 else "normal"), bg=PANEL,
                         fg=BLUE if (col == 3 and seconds) else TEXT, anchor=tk.W, padx=px(12), pady=px(6)) \
                    .grid(row=2 + row * 2, column=col, sticky=tk.EW)
            tk.Frame(t, bg=PANEL_ALT, height=1).grid(row=3 + row * 2, column=0, columnspan=4, sticky=tk.EW)
        if len(sessions) > len(shown):
            tk.Label(t, text=f"이전 기록 {len(sessions) - len(shown)}건 더 있음", font=(UI_FONT, 9), bg=PANEL,
                     fg=MUTED).grid(row=3 + len(shown) * 2, column=0, columnspan=4, sticky=tk.W,
                                    padx=px(12), pady=(px(6), 0))

        self._refresh_today_status()
        self._remaining = STATS_POPUP_SECONDS
        self._tick()

    def _tick(self):
        if not self.winfo_exists():
            return
        if self._remaining <= 0:
            self.app.show_idle()
            return
        self.btn_close.set_text(f"닫기 ({self._remaining}초)")
        self._remaining -= 1
        self.after(1000, self._tick)

    def _refresh_today_status(self):
        """오늘 출근 시간을 다시 계산해서 표시한다. 출근 중이면 1초 뒤에 다시 갱신한다."""
        if not self.winfo_exists():
            return
        status = get_today_attendance_status(self.student_info["user_id"])
        total_seconds = status["finished_seconds"] + status["working_seconds"]
        self.lbl_today_value.config(text=format_duration(total_seconds))
        if status["working_since"]:
            self.lbl_today_note.config(text=f"출근 중 · {status['working_since']} 출근", fg=BLUE)
            self.after(1000, self._refresh_today_status)
        elif total_seconds > 0:
            self.lbl_today_note.config(text="퇴근 상태", fg=MUTED)
        else:
            self.lbl_today_note.config(text="오늘 출근 기록 없음", fg=MUTED)


class CameraTestCard(RoundedPanel):
    """카메라 테스트 오른쪽 안내 패널 — 출퇴근 기록 없이 카메라와 얼굴 인식만 확인한다."""
    TITLE_COLOR = {"loading": TEXT_SUB, "search": TEXT, "ok": OK, "unknown": ERR, "error": ERR}

    def __init__(self, app, master, width):
        super().__init__(master, fill=PANEL, bg=BG, radius=px(20), outline=LINE, shadow=px(10), pad=px(28))
        self._last = None
        b = self.body
        inner_w = width - 2 * self.inset

        tk.Label(b, text="카메라 테스트", font=(UI_FONT, 22, "bold"), bg=PANEL, fg=TEXT) \
            .pack(anchor=tk.W, pady=(0, px(18)))

        notice = RoundedBox(b, fill=BLUE_SOFT, bg=PANEL, pad=px(14))
        notice.pack(fill=tk.X)
        icon_label(notice.body, "info", px(20), BLUE, BLUE_SOFT).pack(side=tk.LEFT, anchor=tk.N, pady=px(1))
        tk.Label(notice.body, text="카메라 테스트에서는 출퇴근이 기록되지 않습니다.", font=(UI_FONT, 11, "bold"),
                 bg=BLUE_SOFT, fg=BLUE, justify=tk.LEFT, wraplength=inner_w - px(70)) \
            .pack(side=tk.LEFT, padx=(px(10), 0))

        # 하단 — 남은 시간 · 닫기 (먼저 배치해야 공간이 모자라도 잘리지 않는다)
        RoundButton(b, text="닫기", command=app.show_idle, width=inner_w, height=px(52), bg=PANEL) \
            .pack(side=tk.BOTTOM)
        timer = RoundedBox(b, fill=PANEL_ALT, bg=PANEL, pad=px(16))
        timer.pack(side=tk.BOTTOM, fill=tk.X, pady=(0, px(14)))
        row = tk.Frame(timer.body, bg=PANEL_ALT)
        row.pack(fill=tk.X)
        icon_label(row, "clock", px(18), TEXT_SUB, PANEL_ALT).pack(side=tk.LEFT, anchor=tk.N, pady=px(1))
        tk.Label(row, text=f"{TEST_TIMEOUT}초 동안 켜지며, 자동으로 꺼집니다.",
                 font=(UI_FONT, 10), bg=PANEL_ALT, fg=TEXT_SUB, justify=tk.LEFT,
                 wraplength=inner_w - px(70)).pack(side=tk.LEFT, padx=(px(8), 0))
        bar_row = tk.Frame(timer.body, bg=PANEL_ALT)
        bar_row.pack(fill=tk.X, pady=(px(10), 0))
        self.lbl_remaining = tk.Label(bar_row, text="", font=(UI_FONT, 10, "bold"), bg=PANEL_ALT, fg=BLUE)
        self.lbl_remaining.pack(side=tk.RIGHT)
        self.timer_bar = ProgressBar(bar_row, inner_w - 2 * px(16) - px(100), px(6), bg=PANEL_ALT,
                                     track=BLUE_SOFT_HOVER)
        self.timer_bar.pack(side=tk.LEFT)

        # 인식 결과
        tk.Label(b, text="인식 결과", font=(UI_FONT, 10, "bold"), bg=PANEL, fg=MUTED) \
            .pack(anchor=tk.W, pady=(px(20), px(8)))
        result = RoundedBox(b, fill=PANEL_ALT, bg=PANEL, radius=px(14), pad=px(18))
        result.pack(fill=tk.X)
        self.lbl_title = tk.Label(result.body, text="", font=(UI_FONT, 18, "bold"), bg=PANEL_ALT, fg=TEXT,
                                  anchor=tk.W, justify=tk.LEFT, wraplength=inner_w - px(40))
        self.lbl_title.pack(fill=tk.X)
        self.lbl_detail = tk.Label(result.body, text="", font=(UI_FONT, 10), bg=PANEL_ALT, fg=TEXT_SUB,
                                   anchor=tk.W, justify=tk.LEFT, wraplength=inner_w - px(40))
        self.lbl_detail.pack(fill=tk.X, pady=(px(4), px(12)))
        self.sim_bar = ProgressBar(result.body, inner_w - 2 * px(18), px(8), bg=PANEL_ALT, track=BLUE_SOFT_HOVER)
        self.sim_bar.pack(anchor=tk.W)

        self.set_state("loading", "카메라를 켜는 중..")

    def set_state(self, state, title, detail="", sim=None):
        key = (state, title, detail)
        if key != self._last:
            self._last = key
            self.lbl_title.config(text=title, fg=self.TITLE_COLOR.get(state, TEXT))
            self.lbl_detail.config(text=detail)
        if sim is None:
            self.sim_bar.set(0)
        else:
            self.sim_bar.set(sim, OK if state == "ok" else ERR, marker=THRESHOLD)

    def set_remaining(self, seconds):
        text = f"남은 시간 {seconds}초"
        if self.lbl_remaining.cget("text") != text:
            self.lbl_remaining.config(text=text)
        self.timer_bar.set(seconds / TEST_TIMEOUT)


class PhotoViewer(tk.Toplevel):
    """관리자 출결로그에서 출퇴근 사진을 크게 보여주는 창."""

    def __init__(self, master, image, caption):
        super().__init__(master)
        self.title("출퇴근 사진")
        self.configure(bg="#0F172A")
        self.resizable(False, False)

        # 저장본은 320x240으로 작아서 보기 편하게 두 배로 키운다
        shown = image.resize((image.width * 2, image.height * 2), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(shown)  # 참조를 들고 있어야 그림이 사라지지 않는다
        tk.Label(self, image=self._photo, bg="#0F172A").pack(padx=px(12), pady=(px(12), px(6)))
        tk.Label(self, text=caption, font=(UI_FONT, 11, "bold"), bg="#0F172A", fg="white") \
            .pack(pady=(0, px(12)))
        self.bind("<Escape>", lambda e: self.destroy())


class AdminDashboardFrame(tk.Frame):
    """ 세번째 이미지: 관리자 로그인시 화면 """
    def __init__(self, parent, admin_email):
        super().__init__(parent.right_container, bg="white")
        self.parent = parent
        self.admin_email = admin_email

        # 1행: 관리자 정보 + 로그아웃
        top_bar = tk.Frame(self, bg="white")
        top_bar.pack(fill=tk.X, pady=(px(10), px(4)), padx=px(12))

        lbl_admin = tk.Label(top_bar, text=f"🔑 관리자: {self.admin_email}", font=(UI_FONT, 11, "bold"), bg="white", fg="#334155")
        lbl_admin.pack(side=tk.LEFT, pady=5)

        btn_logout = Button(
            top_bar, text="로그아웃", bg="#EF4444", fg="white", font=(UI_FONT, 10, "bold"),
            activebackground="#DC2626", activeforeground="white", bd=0, padx=16, pady=6, cursor="hand2",
            command=self.logout
        )
        btn_logout.pack(side=tk.RIGHT, pady=5)

        # 관리자 작업 중에 카메라 화면을 끄고 켤 수 있는 버튼
        self.btn_camera = Button(
            top_bar, text="📷 화면 끄기", fg="white", font=(UI_FONT, 10, "bold"),
            activeforeground="white", bd=0, padx=16, pady=6, cursor="hand2",
            command=self.toggle_camera
        )
        self.btn_camera.pack(side=tk.RIGHT, pady=5, padx=(0, px(8)))
        # 관리자 화면에 들어오면서 카메라가 켜지므로, 그 다음에 버튼 상태를 맞춘다
        self.after(0, self.sync_camera_button)

        # 2행: 기능 버튼 — 한 줄에 몰아넣으면 글자가 잘려서 아래 줄에 반반 나눠 배치
        action_bar = tk.Frame(self, bg="white")
        action_bar.pack(fill=tk.X, padx=px(12), pady=(0, px(8)))

        btn_refresh = Button(
            action_bar, text="🔄 창공시스템 명단 새로고침", bg="#3B82F6", fg="white", font=(UI_FONT, 10, "bold"),
            activebackground="#2563EB", activeforeground="white", bd=0, pady=8, cursor="hand2",
            command=self.refresh_roster_data
        )
        btn_refresh.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5))

        btn_export = Button(
            action_bar, text="📊 출석 데이터 추출", bg="#10B981", fg="white", font=(UI_FONT, 10, "bold"),
            activebackground="#059669", activeforeground="white", bd=0, pady=8, cursor="hand2",
            command=self.export_attendance_excel
        )
        btn_export.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(5, 0))

        # 탭 뷰 스타일 커스텀
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.tab_students = ttk.Frame(self.notebook)
        self.tab_add_student = ttk.Frame(self.notebook)
        self.tab_logs = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_students, text=" 학생 현황/수정 ")
        self.notebook.add(self.tab_add_student, text=" 학생 등록 ")
        self.notebook.add(self.tab_logs, text=" 전체 출결로그 ")

        self.build_students_tab()
        self.build_add_student_tab()
        self.build_logs_tab()

    def toggle_camera(self):
        """카메라 화면을 켜고 끈다. 꺼두면 얼굴 인식도 멈춰서 학생 등록은 다시 켜야 한다."""
        self.parent.set_camera(not self.parent.camera_on)
        self.sync_camera_button()

    def sync_camera_button(self):
        if not self.btn_camera.winfo_exists():
            return
        if self.parent.camera_on:
            self.btn_camera.config(text="📷 화면 끄기", bg="#334155", activebackground="#475569")
        else:
            self.btn_camera.config(text="📷 화면 켜기", bg="#10B981", activebackground="#059669")

    def logout(self):
        self.parent.set_admin_logged_in(False)

    def refresh_roster_data(self):
        """
        창공시스템이 내보낸 roster.json을 다시 읽어 메모리에 반영한다.
        - 얼굴인식 화면에 표시되는 학생 이름이 즉시 최신 정보로 바뀐다.
        - [학생 등록] 탭의 명단 콤보박스도 같이 갱신한다.
        - 파일이 없거나 형식이 잘못됐으면 프로그램이 죽지 않고 여기서 오류를 보여준다.
        """
        try:
            roster = load_roster_strict()
        except FileNotFoundError:
            messagebox.showerror(
                "새로고침 실패",
                "roster.json 파일을 찾을 수 없습니다.\n"
                "창공시스템을 먼저 실행해서 학생을 등록해주세요.\n\n"
                f"찾아본 위치:\n{ROSTER_PATH}",
            )
            return
        except json.JSONDecodeError as e:
            messagebox.showerror(
                "새로고침 실패",
                f"roster.json 파일의 형식이 올바르지 않습니다 (JSON 오류: {e}).\n"
                "창공시스템 쪽에서 파일이 다 써지는 중에 읽었을 수도 있으니 잠시 후 다시 시도해주세요.",
            )
            return
        except Exception as e:
            messagebox.showerror("새로고침 실패", f"명단을 불러오는 중 오류가 발생했습니다.\n\n{e}")
            return

        self.parent.apply_roster(roster)
        self.reload_roster_combo()

        messagebox.showinfo(
            "새로고침 완료",
            f"창공시스템 명단 {len(roster)}건을 다시 불러왔습니다.\n"
            "얼굴인식 결과에 표시되는 이름에도 바로 반영됩니다.",
        )

    def export_attendance_excel(self):
        """
        현재까지 기록된 출석(attendance_logs) 전체를 학번+날짜별로 묶어서 .xlsx로 저장한다.
        DB 데이터는 읽기만 하고 건드리지 않는다.
        """
        try:
            import openpyxl
            from openpyxl.styles import Font, Alignment, PatternFill
        except ImportError:
            messagebox.showerror(
                "추출 실패",
                "엑셀 내보내기에 필요한 openpyxl 패키지가 설치되어 있지 않습니다.\n\n"
                "터미널에서 아래 명령으로 설치한 뒤 다시 시도해주세요:\n  pip install openpyxl",
            )
            return

        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT user_id, name, log_date, log_type, log_time
                FROM attendance_logs
                ORDER BY log_date ASC, name ASC, log_time ASC
            """)
            rows = cursor.fetchall()
            conn.close()
        except Exception as e:
            messagebox.showerror("추출 실패", f"출석 데이터를 불러오는 중 오류가 발생했습니다.\n\n{e}")
            return

        # 출근→퇴근 한 쌍을 한 행으로 묶는다 (attendance.json 내보내기와 같은 방식, 하루 여러 번이면 여러 행)
        records = sorted(
            pair_attendance_sessions(rows),
            key=lambda r: (r["date"], r["name"] or "", r["check_in"] or r["check_out"] or ""),
        )

        if not records:
            messagebox.showinfo("추출할 데이터 없음", "저장된 출석 기록이 없습니다.")
            return

        default_name = f"출석기록_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        path = filedialog.asksaveasfilename(
            title="출석 데이터 추출",
            defaultextension=".xlsx",
            initialfile=default_name,
            filetypes=[("Excel 파일", "*.xlsx")],
        )
        if not path:
            return  # 사용자가 취소함

        try:
            wb = openpyxl.Workbook()
            ws = wb.active
            ws.title = "출석기록"

            headers = ["학번", "이름", "날짜", "출근시간", "퇴근시간", "근무시간(시간)", "비고"]
            ws.append(headers)
            header_font = Font(bold=True, color="FFFFFF")
            header_fill = PatternFill(start_color="1E293B", end_color="1E293B", fill_type="solid")
            for col_idx in range(1, len(headers) + 1):
                cell = ws.cell(row=1, column=col_idx)
                cell.font = header_font
                cell.fill = header_fill
                cell.alignment = Alignment(horizontal="center")

            for rec in records:
                check_in = rec["check_in"]
                check_out = rec["check_out"]

                work_hours = ""
                seconds = session_seconds(rec)
                if seconds > 0:
                    work_hours = round(seconds / 3600.0, 2)

                if check_in and check_out:
                    note = "정상 출퇴근"
                elif check_in and not check_out:
                    note = "퇴근 기록 없음"
                elif check_out and not check_in:
                    note = "출근 기록 없음"
                else:
                    note = "결석"

                ws.append([
                    rec["user_id"], rec["name"], rec["date"],
                    check_in or "-", check_out or "-",
                    work_hours, note,
                ])

            for col_idx, width in enumerate([12, 10, 12, 10, 10, 14, 14], start=1):
                ws.column_dimensions[openpyxl.utils.get_column_letter(col_idx)].width = width

            wb.save(path)
        except Exception as e:
            messagebox.showerror("추출 실패", f"엑셀 파일을 저장하는 중 오류가 발생했습니다.\n\n{e}")
            return

        messagebox.showinfo("추출 완료", f"출석 데이터 {len(records)}건을 엑셀 파일로 저장했습니다.\n\n{path}")

    def build_students_tab(self):
        # 상단 리스트
        list_frame = ttk.LabelFrame(self.tab_students, text=" 학생 리스트 ")
        list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        scroll = ttk.Scrollbar(list_frame)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.tree_students = ttk.Treeview(
            list_frame, columns=("id", "name", "major", "penalty"), show="headings", 
            yscrollcommand=scroll.set
        )
        self.tree_students.heading("id", text="학번")
        self.tree_students.heading("name", text="이름")
        self.tree_students.heading("major", text="전공")
        self.tree_students.heading("penalty", text="패널티")
        
        self.tree_students.column("id", width=px(95), anchor=tk.CENTER)
        self.tree_students.column("name", width=px(80), anchor=tk.CENTER)
        self.tree_students.column("major", width=px(130), anchor=tk.W)
        self.tree_students.column("penalty", width=px(65), anchor=tk.CENTER)
        self.tree_students.pack(fill=tk.BOTH, expand=True)
        scroll.config(command=self.tree_students.yview)

        self.tree_students.bind("<<TreeviewSelect>>", self.on_student_select)

        # 정보 수정 및 삭제 프레임
        control_frame = ttk.Frame(self.tab_students)
        control_frame.pack(fill=tk.X, padx=5, pady=5)

        # 삭제 제어 영역 (크기 강화) — 편집 폼보다 먼저 배치해야 화면이 작을 때도 버튼이 찌그러지지 않는다
        btn_action_frame = ttk.Frame(control_frame)
        btn_action_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=2)

        # 선택정보 편집 폼
        form_frame = ttk.LabelFrame(control_frame, text=" 학생 데이터 편집 ")
        form_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=2)

        ttk.Label(form_frame, text="이름:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.edit_name = ttk.Entry(form_frame, width=9)
        self.edit_name.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)

        # 이미 등록된 학생의 전공은 그대로 두고 여기서 고칠 수 있게 남겨둔다 (신규 등록만 고정)
        ttk.Label(form_frame, text="전공:").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        self.edit_major = ttk.Entry(form_frame, width=12)
        self.edit_major.grid(row=0, column=3, padx=5, pady=5, sticky=tk.EW)

        ttk.Label(form_frame, text="패널티:").grid(row=0, column=4, padx=5, pady=5, sticky=tk.W)
        self.edit_penalty = ttk.Entry(form_frame, width=4)
        self.edit_penalty.grid(row=0, column=5, padx=5, pady=5)

        btn_update = Button(form_frame, text="수정 완료", command=self.update_student)
        # 입력칸 옆에 두면 화면이 작을 때 잘려서, 입력칸 아래 줄에 폭 전체로 둔다
        btn_update.grid(row=1, column=0, columnspan=6, padx=5, pady=(0, 5), sticky=tk.EW)
        form_frame.columnconfigure(1, weight=2)
        form_frame.columnconfigure(3, weight=3)


        btn_del_sel = Button(btn_action_frame, text="선택 삭제", command=self.delete_selected)
        btn_del_sel.pack(fill=tk.X, ipady=4, pady=2)

        btn_del_all = Button(btn_action_frame, text="일괄 삭제", command=self.delete_all)
        btn_del_all.pack(fill=tk.X, ipady=4, pady=2)

        self.load_students()

    def build_add_student_tab(self):
        frame = ttk.LabelFrame(self.tab_add_student, text=" 학생 얼굴 및 상세 정보 등록 ")
        frame.pack(fill=tk.BOTH, expand=True, padx=px(10), pady=px(10))

        grid_container = tk.Frame(frame, bg="#F8FAFC")
        grid_container.pack(padx=px(20), pady=px(20), fill=tk.BOTH, expand=True)

        # 창공시스템(SeatManagerApp) 명단에서 골라서 자동 입력 — 수기로 다시 안 쳐도 된다
        ttk.Label(grid_container, text="창공시스템 명단:", font=(UI_FONT, 10, "bold"), background="#F8FAFC").grid(row=0, column=0, padx=px(10), pady=px(8), sticky=tk.W)
        self.roster_combo = ttk.Combobox(grid_container, font=(UI_FONT, 10), width=26, state="readonly")
        self.roster_combo.grid(row=0, column=1, padx=px(10), pady=px(8), sticky=tk.W)
        self.roster_combo.bind("<<ComboboxSelected>>", self.on_roster_selected)

        btn_reload_roster = Button(grid_container, text="🔄 명단 새로고침", command=lambda: self.reload_roster_combo(show_message_if_empty=True))
        btn_reload_roster.grid(row=0, column=2, padx=px(10), pady=px(8), sticky=tk.W)

        ttk.Label(grid_container, text="학번(ID):", font=(UI_FONT, 10, "bold"), background="#F8FAFC").grid(row=1, column=0, padx=px(10), pady=px(8), sticky=tk.W)
        self.add_id = ttk.Entry(grid_container, font=(UI_FONT, 11), width=26)
        self.add_id.grid(row=1, column=1, padx=px(10), pady=px(8), sticky=tk.W)

        ttk.Label(grid_container, text="이름:", font=(UI_FONT, 10, "bold"), background="#F8FAFC").grid(row=2, column=0, padx=px(10), pady=px(8), sticky=tk.W)
        self.add_name = ttk.Entry(grid_container, font=(UI_FONT, 11), width=26)
        self.add_name.grid(row=2, column=1, padx=px(10), pady=px(8), sticky=tk.W)

        # 학과는 소프트웨어전공으로 고정이라 입력받지 않고 보여주기만 한다
        ttk.Label(grid_container, text="전공 학과:", font=(UI_FONT, 10, "bold"), background="#F8FAFC").grid(row=3, column=0, padx=px(10), pady=px(8), sticky=tk.W)
        ttk.Label(grid_container, text=DEPARTMENT_NAME, font=(UI_FONT, 11), background="#F8FAFC",
                  foreground="#334155").grid(row=3, column=1, padx=px(10), pady=px(8), sticky=tk.W)

        # 등록 버튼 크기 강화
        btn_register = Button(
            grid_container, text="💾 카메라 인식 얼굴로 등록", bg="#10B981", fg="white", bd=0,
            activebackground="#059669", activeforeground="white", font=(UI_FONT, 12, "bold"),
            height=2, cursor="hand2", command=self.register_student
        )
        btn_register.grid(row=4, column=0, columnspan=3, pady=px(25), sticky=tk.EW)

        self._roster_map = {}
        self.reload_roster_combo()

    def reload_roster_combo(self, show_message_if_empty=False):
        """창공시스템이 내보낸 roster.json을 다시 읽어, 아직 등록 안 된 학생만 콤보박스에 채운다."""
        roster = load_roster()

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT user_id FROM users WHERE role = 'student'")
        already_registered = {str(row[0]) for row in cursor.fetchall()}
        conn.close()

        self._roster_map = {}
        values = []
        for student in roster:
            sid = str(student.get("StudentId", "")).strip()
            if not sid or sid in already_registered:
                continue
            name = student.get("Name", "")
            dept = student.get("Department", "")
            label = f"{sid} - {name} ({dept})" if dept else f"{sid} - {name}"
            self._roster_map[label] = student
            values.append(label)

        self.roster_combo["values"] = values
        self.roster_combo.set("")

        if not roster and show_message_if_empty:
            messagebox.showinfo(
                "명단 없음",
                "창공시스템 명단(roster.json)을 찾지 못했습니다.\n"
                "창공시스템을 한 번 실행해서 학생을 등록해두면 여기서 바로 골라 쓸 수 있습니다.\n\n"
                f"찾아본 위치:\n{ROSTER_PATH}",
            )

    def on_roster_selected(self, event):
        label = self.roster_combo.get()
        student = self._roster_map.get(label)
        if not student:
            return
        self.add_id.delete(0, tk.END)
        self.add_id.insert(0, student.get("StudentId", ""))
        self.add_name.delete(0, tk.END)
        self.add_name.insert(0, student.get("Name", ""))
        # 창공시스템 명단에 학과가 들어 있어도 쓰지 않는다 — 학과는 소프트웨어전공으로 고정

    def build_logs_tab(self):
        # 상단 조회 조건
        filter_frame = ttk.LabelFrame(self.tab_logs, text=" 조회 조건 ")
        filter_frame.pack(fill=tk.X, padx=5, pady=5)
        filter_frame.columnconfigure(1, weight=1)

        # 1행: 조회 유형
        ttk.Label(filter_frame, text="조회 유형:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.cmb_log_mode = ttk.Combobox(
            filter_frame, state="readonly", font=(UI_FONT, 10),
            values=[label for _, label in LOG_FILTER_MODES]
        )
        self.cmb_log_mode.current(1)  # 기본: 최근 7일 내역
        self.cmb_log_mode.grid(row=0, column=1, columnspan=2, padx=5, pady=5, sticky=tk.EW)
        self.cmb_log_mode.bind("<<ComboboxSelected>>", lambda e: self.on_log_mode_changed())

        # 2행: 날짜 (조회 유형에 따라 달력 버튼 개수/의미가 바뀐다)
        self.lbl_log_date = ttk.Label(filter_frame, text="기간:")
        self.lbl_log_date.grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)

        date_row = ttk.Frame(filter_frame)
        date_row.grid(row=1, column=1, columnspan=2, padx=5, pady=5, sticky=tk.W)
        today = date.today()
        self.date_start = DatePicker(date_row, today - timedelta(days=6), on_change=self.load_logs_filtered)
        self.date_start.grid(row=0, column=0)
        self.lbl_date_tilde = ttk.Label(date_row, text="~")
        self.lbl_date_tilde.grid(row=0, column=1, padx=px(6))
        self.date_end = DatePicker(date_row, today, on_change=self.load_logs_filtered)
        self.date_end.grid(row=0, column=2)
        self.lbl_log_range = ttk.Label(date_row, text="", foreground="#475569", font=(UI_FONT, 10, "bold"))
        self.lbl_log_range.grid(row=0, column=3, padx=(px(8), 0))

        # 3행: 이름/학번 검색 + 조회
        ttk.Label(filter_frame, text="이름/학번:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        self.ent_search_name = ttk.Entry(filter_frame, width=16)
        self.ent_search_name.grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)
        self.ent_search_name.bind("<Return>", lambda e: self.load_logs_filtered())

        btn_search = Button(
            filter_frame, text="🔍 조회하기", bg="#3B82F6", fg="white", font=(UI_FONT, 11, "bold"),
            bd=0, activebackground="#2563EB", activeforeground="white", cursor="hand2", padx=px(12), height=1,
            command=self.load_logs_filtered
        )
        btn_search.grid(row=2, column=2, padx=px(10), pady=5, sticky=tk.E)

        # 로그 트리뷰 목록 (조회 유형에 따라 내역/학생별 합계 컬럼으로 바뀐다)
        list_container = tk.Frame(self.tab_logs)
        list_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=(5, 0))

        # 가로/세로 스크롤바 — 컬럼이 화면보다 넓어지면 아래 드래그바로 좌우로 움직인다
        scroll = ttk.Scrollbar(list_container, orient=tk.VERTICAL)
        scroll_x = ttk.Scrollbar(list_container, orient=tk.HORIZONTAL)

        self.tree_logs = ttk.Treeview(list_container, show="headings",
                                      yscrollcommand=scroll.set, xscrollcommand=scroll_x.set)
        self.tree_logs.tag_configure("invalid", foreground="#94A3B8")
        scroll.config(command=self.tree_logs.yview)
        scroll_x.config(command=self.tree_logs.xview)

        self.tree_logs.grid(row=0, column=0, sticky=tk.NSEW)
        scroll.grid(row=0, column=1, sticky=tk.NS)
        scroll_x.grid(row=1, column=0, sticky=tk.EW)
        list_container.rowconfigure(0, weight=1)
        list_container.columnconfigure(0, weight=1)
        self.tree_logs.bind("<Double-1>", lambda e: self.show_log_photo())
        self._log_photos = {}  # 트리뷰 행 id → (사진 상대 경로, 설명)

        result_bar = tk.Frame(self.tab_logs, bg="#F8FAFC")
        result_bar.pack(fill=tk.X, padx=5, pady=(0, 5))
        self.btn_log_photo = Button(
            result_bar, text="📷 사진 보기", bg="#1E293B", fg="white", bd=0,
            activebackground="#334155", activeforeground="white",
            font=(UI_FONT, 9, "bold"), padx=px(10), pady=px(3), cursor="hand2", command=self.show_log_photo
        )
        self.btn_log_photo.pack(side=tk.RIGHT, padx=px(8))
        self.lbl_log_result = tk.Label(result_bar, text="", anchor=tk.W, bg="#F8FAFC", fg="#334155",
                                       font=(UI_FONT, 10, "bold"), padx=px(10), pady=px(6))
        self.lbl_log_result.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Label(self.tab_logs, text=f"📷 행을 더블클릭하면 출퇴근 사진을 봅니다 · 사진은 {PHOTO_RETENTION_DAYS}일 뒤 자동 삭제",
                 anchor=tk.W, bg="white", fg="#94A3B8", font=(UI_FONT, 9)) \
            .pack(fill=tk.X, padx=px(10), pady=(0, 5))

        self.on_log_mode_changed()

    def current_log_mode(self):
        return LOG_FILTER_MODES[self.cmb_log_mode.current()][0]

    def on_log_mode_changed(self):
        mode = self.current_log_mode()

        # 오늘/최근 7일은 날짜를 고를 필요가 없고, 월/주 단위는 기준일 1개, 직접 선택은 시작~종료 2개
        if mode in ("today", "recent7", "recent7_sum"):
            self.lbl_log_date.config(text="기간:")
            self.date_start.grid_remove()
            self.lbl_date_tilde.grid_remove()
            self.date_end.grid_remove()
        elif mode == "custom":
            self.lbl_log_date.config(text="기간:")
            self.date_start.grid()
            self.lbl_date_tilde.grid()
            self.date_end.grid()
        else:
            self.lbl_log_date.config(text="기준일:")
            self.date_start.grid()
            self.lbl_date_tilde.grid_remove()
            self.date_end.grid_remove()

        self.load_logs_filtered()

    def get_log_period(self):
        """현재 조회 유형의 실제 조회 기간 (시작일, 종료일)."""
        mode = self.current_log_mode()
        today = date.today()
        if mode == "today":
            return today, today
        if mode in ("recent7", "recent7_sum"):
            return today - timedelta(days=6), today

        anchor = self.date_start.get_date()
        if mode == "month":
            last_day = calendar.monthrange(anchor.year, anchor.month)[1]
            return anchor.replace(day=1), anchor.replace(day=last_day)
        if mode in ("week_days", "week_hours"):
            monday = anchor - timedelta(days=anchor.weekday())
            return monday, monday + timedelta(days=6)

        start, end = self.date_start.get_date(), self.date_end.get_date()
        if start > end:  # 거꾸로 골랐으면 순서를 바로잡아 보여준다
            self.date_start.set_date(end)
            self.date_end.set_date(start)
            start, end = end, start
        return start, end

    # DB 및 로드 제어 기능들
    def load_students(self):
        for item in self.tree_students.get_children():
            self.tree_students.delete(item)
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT user_id, name, major, penalty FROM users WHERE role = 'student'")
        for r in cursor.fetchall():
            self.tree_students.insert("", tk.END, values=r)
        conn.close()

    def load_logs(self):
        # 기본값 로드 (필터 없이 전체 출력)
        self.load_logs_filtered()

    def load_logs_filtered(self):
        mode = self.current_log_mode()
        start, end = self.get_log_period()
        keyword = self.ent_search_name.get().strip()

        if mode == "today":
            range_text = format_date_ko(start)
        elif mode == "month":
            range_text = f"→ {start.year}년 {start.month}월 전체"
        elif mode == "custom":
            range_text = ""
        else:
            range_text = f"{'→ ' if mode in ('week_days', 'week_hours') else ''}{format_date_ko(start)} ~ {format_date_ko(end)}"
        self.lbl_log_range.config(text=range_text)

        if mode in LOG_SUMMARY_MODES:
            self._show_log_summary(mode, start, end, keyword)
        else:
            self._show_log_rows(start, end, keyword)

    def _set_log_columns(self, columns):
        """columns: (키, 제목, 폭, 정렬) 목록으로 트리뷰 컬럼을 바꾸고 기존 행을 비운다."""
        self.tree_logs.delete(*self.tree_logs.get_children())
        self._log_photos = {}
        self.tree_logs["columns"] = [key for key, _, _, _ in columns]
        self.tree_logs["displaycolumns"] = "#all"
        for key, heading, width, anchor in columns:
            self.tree_logs.heading(key, text=heading)
            # stretch=False — 창 폭에 맞춰 늘어나지 않게 해야 넘칠 때 가로 스크롤바로 움직일 수 있다
            self.tree_logs.column(key, width=px(width), minwidth=px(width), anchor=anchor, stretch=False)

    def _show_log_rows(self, start, end, keyword):
        self._set_log_columns([
            ("user_id", "학번", 100, tk.CENTER),
            ("name", "이름", 80, tk.CENTER),
            ("log_type", "구분", 95, tk.CENTER),
            ("log_date", "날짜", 125, tk.CENTER),
            ("log_time", "시간", 85, tk.CENTER),
            ("photo", "사진", 45, tk.CENTER),
        ])

        query = """
            SELECT user_id, name, log_type, log_date, log_time, photo_path
            FROM attendance_logs
            WHERE log_date BETWEEN ? AND ?
        """
        params = [start.isoformat(), end.isoformat()]
        if keyword:
            query += " AND (name LIKE ? OR user_id LIKE ?)"
            params += [f"%{keyword}%", f"%{keyword}%"]
        query += " ORDER BY log_date DESC, log_time DESC"

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        for user_id, name, log_type, log_date, log_time, photo_path in rows:
            valid = is_valid_attendance_time(log_time)
            type_text = "출근" if log_type == "CHECK_IN" else "퇴근"
            if not valid:
                type_text += " (미인정)"  # 00:00~06:00 기록은 일수/시간 계산에서 빠진다
            try:
                date_text = format_date_ko(datetime.strptime(log_date, "%Y-%m-%d").date())
            except (TypeError, ValueError):
                date_text = log_date
            item = self.tree_logs.insert(
                "", tk.END, values=(user_id, name, type_text, date_text, log_time, "📷" if photo_path else ""),
                tags=() if valid else ("invalid",)
            )
            if photo_path:
                self._log_photos[item] = (photo_path, f"{name} ({user_id}) · {date_text} {log_time} {type_text}")

        check_in = sum(1 for r in rows if r[2] == "CHECK_IN")
        self.lbl_log_result.config(
            text=f"총 {len(rows)}건 (출근 {check_in} · 퇴근 {len(rows) - check_in}) · 학생 {len({r[0] for r in rows})}명"
        )

    def show_log_photo(self):
        """출결로그에서 고른 행의 출퇴근 사진을 연다. (관리자 화면에서만 볼 수 있다)"""
        selected = self.tree_logs.selection()
        if not selected:
            messagebox.showinfo("사진 보기", "사진을 볼 출결 기록을 목록에서 선택하세요.")
            return
        entry = self._log_photos.get(selected[0])
        if not entry:
            messagebox.showinfo(
                "사진 없음",
                "이 기록에는 사진이 없습니다.\n"
                f"(사진 기능 도입 전 기록이거나, {PHOTO_RETENTION_DAYS}일 보관 기간이 지나 삭제되었습니다)",
            )
            return
        rel_path, caption = entry
        try:
            with Image.open(photo_abs_path(rel_path)) as img:
                image = img.convert("RGB")
        except (OSError, ValueError):
            messagebox.showinfo("사진 없음", "사진 파일을 찾을 수 없습니다. 이미 삭제되었을 수 있습니다.")
            return
        PhotoViewer(self, image, caption)

    def _show_log_summary(self, mode, start, end, keyword):
        self._set_log_columns([
            ("user_id", "학번", 110, tk.CENTER),
            ("name", "이름", 90, tk.CENTER),
            ("days", "출근 일수", 90, tk.CENTER),
            ("seconds", "출석 시간 합계", 170, tk.CENTER),
        ])

        summary = get_attendance_summary_by_student(start.isoformat(), end.isoformat())
        if keyword:
            summary = [s for s in summary if keyword in s["name"] or keyword in s["user_id"]]

        # 기준은 학생 화면의 주간 현황과 같다: 인정 출근 일수 ≥ 3일, 인정 시간 ≥ 20시간 (월~일)
        if mode == "week_days":
            summary = [s for s in summary if s["days"] >= WEEKLY_REQUIRED_DAYS]
            summary.sort(key=lambda s: (-s["days"], -s["seconds"], s["name"]))
        elif mode == "week_hours":
            summary = [s for s in summary if s["seconds"] >= WEEKLY_REQUIRED_SECONDS]
            summary.sort(key=lambda s: (-s["seconds"], -s["days"], s["name"]))
        else:
            summary.sort(key=lambda s: (-s["seconds"], -s["days"], s["name"]))

        for s in summary:
            self.tree_logs.insert("", tk.END, values=(s["user_id"], s["name"], f"{s['days']}일",
                                                      format_duration(s["seconds"])))

        if mode == "recent7_sum":
            total = sum(s["seconds"] for s in summary)
            attended = sum(1 for s in summary if s["days"] > 0)
            text = f"학생 {len(summary)}명 중 출석 {attended}명 · 전체 출석 시간 {format_duration(total)}"
        else:
            text = f"조건을 충족한 학생 {len(summary)}명"
        self.lbl_log_result.config(text=text)

    def on_student_select(self, event):
        selected = self.tree_students.selection()
        if not selected:
            return
        values = self.tree_students.item(selected[0], "values")
        
        self.edit_name.delete(0, tk.END)
        self.edit_name.insert(0, values[1])

        self.edit_major.delete(0, tk.END)
        self.edit_major.insert(0, values[2])


        self.edit_penalty.delete(0, tk.END)
        self.edit_penalty.insert(0, values[3])

    def update_student(self):
        selected = self.tree_students.selection()
        if not selected:
            messagebox.showwarning("선택 없음", "수정할 대상을 리스트에서 선택하세요.")
            return
        
        student_id = self.tree_students.item(selected[0], "values")[0]
        name = self.edit_name.get().strip()
        major = self.edit_major.get().strip()
        penalty_str = self.edit_penalty.get().strip()

        if not (name and major and penalty_str):
            messagebox.showwarning("경고", "수정 데이터를 모두 기입하세요.")
            return

        try:
            penalty = int(penalty_str)
        except ValueError:
            messagebox.showerror("오류", "패널티는 정수형만 입력 가능합니다.")
            return

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE users SET name = ?, major = ?, penalty = ?
            WHERE user_id = ?
        """, (name, major, penalty, student_id))
        conn.commit()
        conn.close()

        messagebox.showinfo("수정 성공", "학생 정보 변경이 완료되었습니다.")
        self.load_students()
        self.parent.reload_users()

    def delete_selected(self):
        selected = self.tree_students.selection()
        if not selected:
            messagebox.showwarning("선택 없음", "삭제할 대상을 리스트에서 선택하세요.")
            return
        
        student_id = self.tree_students.item(selected[0], "values")[0]
        name = self.tree_students.item(selected[0], "values")[1]

        if messagebox.askyesno("삭제 확인", f"[{name}] 학생 정보를 완전히 삭제하시겠습니까?"):
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # SQLite 타입 유연성을 고려하여 문자열 및 정수형 모두 삭제 쿼리 반영
            alt_id = int(student_id) if str(student_id).isdigit() else student_id
            cursor.execute("SELECT photo_path FROM attendance_logs WHERE user_id = ? OR user_id = ?",
                           (str(student_id), alt_id))
            photo_paths = [row[0] for row in cursor.fetchall()]
            cursor.execute("DELETE FROM users WHERE user_id = ? OR user_id = ?", (str(student_id), alt_id))
            cursor.execute("DELETE FROM attendance_logs WHERE user_id = ? OR user_id = ?", (str(student_id), alt_id))

            conn.commit()
            conn.close()
            delete_photo_files(photo_paths)  # 출결 기록과 함께 그 학생의 출퇴근 사진도 지운다

            messagebox.showinfo("성공", "선택 정보가 삭제되었습니다.")
            self.load_students()
            self.load_logs()
            self.parent.reload_users()

    def delete_all(self):
        if messagebox.askyesno("전체 삭제 경고", "모든 학생 정보 및 모든 출결 이력이 영구 삭제됩니다. 진행하시겠습니까?"):
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM users WHERE role = 'student'")
            cursor.execute("DELETE FROM attendance_logs")
            conn.commit()
            conn.close()
            shutil.rmtree(PHOTO_DIR, ignore_errors=True)  # 출퇴근 사진도 전부 삭제

            messagebox.showinfo("성공", "전체 데이터가 삭제되었습니다.")
            self.load_students()
            self.load_logs()
            self.parent.reload_users()

    def _find_duplicate_face(self, embedding):
        """새로 등록하려는 얼굴 임베딩이 기존 등록자와 동일 인물로 보이는지 확인한다.
        인식 화면에서 쓰는 것과 같은 코사인 유사도 THRESHOLD를 기준으로 삼는다."""
        enrolled_users = self.parent.enrolled_users
        embed_matrix = self.parent.embed_matrix
        if not enrolled_users or embed_matrix is None or len(embed_matrix) == 0:
            return None

        norm = np.linalg.norm(embedding)
        if norm == 0:
            return None
        normed = embedding / norm

        sims = embed_matrix @ normed
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        if best_sim >= THRESHOLD:
            return enrolled_users[best_idx]
        return None

    def register_student(self):
        s_id = self.add_id.get().strip()
        name = self.add_name.get().strip()
        major = DEPARTMENT_NAME

        if not (s_id and name):
            messagebox.showwarning("입력 미달", "모든 정보를 입력하세요.")
            return

        if not self.parent.cached_faces:
            messagebox.showerror("얼굴 인식 실패", "카메라 영역에 등록할 학생의 얼굴이 감지되지 않았습니다.")
            return

        # 사진으로 학생이 등록되면 이후 출석도 사진으로 뚫리므로 등록할 때도 사람인지 확인한다
        if not self.parent.passes_antispoof():
            messagebox.showerror("얼굴 인식 실패", self.parent.liveness_reject_message())
            return

        if not self.parent.has_recent_blink():
            messagebox.showerror(
                "얼굴 인식 실패",
                "사람 얼굴이 아닌 것으로 보입니다.\n"
                "등록할 학생이 카메라를 바라보고 눈을 한 번 깜빡인 뒤 다시 눌러주세요.",
            )
            return

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE user_id = ?", (s_id,))
        if cursor.fetchone():
            messagebox.showerror("오류", "이미 가입된 학번 ID입니다.")
            conn.close()
            return

        new_embedding = self.parent.cached_faces[0].embedding.astype(np.float32)
        dup_user = self._find_duplicate_face(new_embedding)
        if dup_user:
            messagebox.showerror(
                "중복 얼굴 감지",
                f"이미 등록된 얼굴과 유사도가 높습니다.\n"
                f"기존 등록자: {dup_user['name']} ({dup_user['user_id']})\n"
                "동일 인물로 판단되어 등록을 진행하지 않습니다."
            )
            conn.close()
            return

        emb = new_embedding.tobytes()
        cursor.execute("""
            INSERT INTO users (user_id, name, major, role, embedding, penalty)
            VALUES (?, ?, ?, ?, ?, 0)
        """, (s_id, name, major, 'student', emb))
        conn.commit()
        conn.close()

        messagebox.showinfo("성공", f"[{name}] 학생이 성공적으로 등록되었습니다.")
        self.add_id.delete(0, tk.END)
        self.add_name.delete(0, tk.END)

        self.load_students()
        self.parent.reload_users()
        self.reload_roster_combo()  # 방금 등록한 학생은 명단 목록에서 빠지도록


# ==========================================
# 3. 메인 어플리케이션
# ==========================================
class AttendanceApp:
    def __init__(self, window, kiosk=False):
        self.window = window
        # 무인 운영(--kiosk): 00:00~06:00(출퇴근 미인정 시간)이 되면 스스로 종료한다.
        # 06:00 재시작은 윈도우 작업 스케줄러가 맡는다 (windows/install.bat 참고).
        self.kiosk = kiosk
        self.window.title("창의공간 얼굴인식 출석체크 시스템")
        self.window.configure(bg=BG)  # 탁한 흰색 배경

        # 창공시스템(SeatManagerApp, MainWindow.xaml)과 동일한 1600x900 레이아웃을 기준으로 한다.
        # 리사이즈를 허용하면 내부 UI 폭/높이가 바뀌면서 학생 등록 탭 같은 화면이 잘려 보일 수 있어서
        # 크기는 고정하되, 모니터가 1600x900보다 작으면 창/글자/여백을 같은 비율로 줄여서 전체가 보이게 한다.
        global UI_SCALE
        work_w, work_h = _get_work_area(self.window)
        # 창 테두리와 제목 표시줄이 차지하는 만큼 여유를 둔다
        UI_SCALE = min(1.0, (work_w - 16) / BASE_WIN_W, (work_h - 40) / BASE_WIN_H)
        UI_SCALE = max(UI_SCALE, 0.5)  # 너무 작아져서 글자를 못 읽는 것만 방지

        # 포인트(pt) 단위로 지정한 모든 글꼴이 같은 비율로 줄어들게 한다
        base_tk_scaling = float(self.window.tk.call("tk", "scaling"))
        if IS_MAC:
            # 맥 Tk는 72dpi(1.0) 기준이라 같은 pt 글자가 윈도우(96dpi, 1.333)보다 25% 작게 나온다.
            # 레이아웃이 윈도우 기준 픽셀로 짜여 있으므로 윈도우와 같은 배율로 맞춘다.
            base_tk_scaling = 96 / 72
        self.window.tk.call("tk", "scaling", base_tk_scaling * UI_SCALE)
        # 글꼴을 따로 지정하지 않은 라벨/입력칸이 쓰는 기본 글꼴은 픽셀 단위라 위 설정으로 안 줄어서 직접 줄인다
        for font_name in tkfont.names(self.window):
            named_font = tkfont.nametofont(font_name, root=self.window)
            size = int(named_font.cget("size"))
            if size < 0:
                named_font.configure(size=-max(1, round(-size * UI_SCALE)))

        win_w, win_h = px(BASE_WIN_W), px(BASE_WIN_H)
        pos_x = max(0, (work_w - win_w) // 2)
        pos_y = max(0, (work_h - 40 - win_h) // 2)

        self.window.geometry(f"{win_w}x{win_h}+{pos_x}+{pos_y}")
        # 최대화(전체 화면) 버튼을 쓸 수 있게 크기 조절은 허용하되, 기본 크기보다 작게 줄이면
        # 오른쪽 패널 내용이 잘리므로 최소 크기를 기본 크기로 막는다. 늘어난 공간은 카메라 화면이 쓴다.
        self.window.resizable(True, True)
        self.window.minsize(win_w, win_h)
        self.window_w, self.window_h = win_w, win_h

        # UI 스타일 테마 통합 구성
        style = ttk.Style()
        style.theme_use('clam')
        
        # Notebook (Tabs) 스타일링
        style.configure('TNotebook', background='#F1F5F9', borderwidth=0)
        style.configure('TNotebook.Tab', background='#E2E8F0', foreground='#475569', padding=[px(15), px(6)], font=(UI_FONT, 10, 'bold'))
        style.map('TNotebook.Tab', background=[('selected', '#ffffff')], foreground=[('selected', '#1E293B')])
        
        # Treeview 스타일링
        style.configure('Treeview', background='#ffffff', fieldbackground='#ffffff', rowheight=px(28), font=(UI_FONT, 9))
        style.configure('Treeview.Heading', background='#E2E8F0', foreground='#1E293B', font=(UI_FONT, 10, 'bold'))
        style.map('Treeview', background=[('selected', '#3B82F6')], foreground=[('selected', '#ffffff')])
        
        # LabelFrame 스타일링
        style.configure('TLabelframe', background='#ffffff', bordercolor='#CBD5E1', borderwidth=1)
        style.configure('TLabelframe.Label', background='#ffffff', foreground='#1E293B', font=(UI_FONT, 10, 'bold'))

        # Ttk Button 스타일링 크기 및 패딩 조절로 해상도 대폭 업그레이드
        style.configure('TButton', font=(UI_FONT, 11, 'bold'), padding=px(8))

        # DB 초기화
        init_db()

        # 창공시스템이 항상 최신 출결 데이터를 읽어갈 수 있도록 시작할 때도 한 번 내보낸다
        export_attendance_json()

        # InsightFace는 별도 프로세스에서 로드/실행한다 (face_inference_process 참고)
        self.face_conn, child_conn = multiprocessing.Pipe()
        self.face_proc = multiprocessing.Process(
            target=face_inference_process, args=(child_conn, INFERENCE_THREADS), daemon=True
        )
        self.face_proc.start()
        self.enrolled_users = load_users()
        self.embed_matrix = None
        self._rebuild_embedding_matrix()

        # 창공시스템 명단(roster.json) — 얼굴 인식 결과에 표시할 이름을 여기서 우선 찾는다
        self.roster_lookup = {}
        self.apply_roster(load_roster())

        # 화면 상태
        #   idle       : 메인 화면. 카메라/얼굴 인식 꺼짐, 출근·퇴근 버튼만 보인다
        #   attendance : 출근/퇴근을 누른 뒤. 카메라가 켜지고 얼굴이 확인되면 자동으로 기록한다
        #   lookup     : 상단 [출석 조회]. 얼굴로 본인 확인만 하고 기록은 하지 않는다
        #   result     : 출근/퇴근 처리 결과 (카메라 꺼짐, 잠시 뒤 메인 화면으로)
        #   stats      : 본인 출석 현황 (카메라 꺼짐, 잠시 뒤 메인 화면으로)
        #   test       : 상단 [카메라 테스트]. 카메라와 인식 결과만 보여주고 기록은 하지 않는다
        #   admin      : 관리자 로그인 후. 학생 등록에 쓰도록 카메라가 켜져 있다
        self.mode = "idle"
        self.admin_logged_in = False
        self.admin_email = ""
        self.pending_action = None      # attendance 모드에서 누른 버튼 ("CHECK_IN" / "CHECK_OUT")
        self.session_started = 0.0      # attendance/test 모드에 들어간 시각 (시간 초과 계산용)
        self._session_card = None       # 오른쪽 안내 카드 (AttendanceWaitingFrame / CameraTestFrame)
        self._candidate_id = None       # 지금 인식되고 있는 학생 — RECOGNIZE_HOLD_SECONDS 동안 유지돼야 기록
        self._candidate_since = 0.0
        self.ai_ready = False           # 얼굴 인식 모델 로딩이 끝났는지
        self.ai_failed = False          # 모델을 받지 못했거나 인식 프로세스가 죽었는지
        self.camera_on = False          # 캡처 스레드가 이 값에 맞춰 카메라 장치를 열고 닫는다
        self.camera_error = False
        self.camera_generation = 0      # 카메라를 켜고 끌 때마다 1씩 증가 — 이전 세션의 늦은 인식 결과를 버리는 데 쓴다

        # 사진/영상 판별용 깜빡임 상태 (인식 프로세스가 깜빡임 값을 보내주면 켜진다)
        self.liveness_enabled = False
        self.last_blink_time = 0.0
        self._eye_closed = False

        # InsightFace 위조판별 애드온 상태 (애드온이 결과를 보내주면 켜진다)
        self.antispoof_enabled = False
        self.last_live_time = 0.0
        self.last_fake_time = 0.0
        self.last_liveness_status = None

        # UI 레이아웃 구성
        self.create_widgets()

        # macOS 카메라 권한은 앱을 켤 때 미리 받아둔다 (처음 한 번만 허용 창이 뜬다).
        # 카메라 장치 자체는 출근/퇴근 버튼을 눌렀을 때 캡처 스레드가 연다.
        ensure_camera_permission()

        # 3개월 지난 출퇴근 사진 정리 — 시작할 때 한 번, 이후 날짜가 바뀔 때마다 (_update_clock)
        self._photo_cleanup_date = date.today()
        threading.Thread(target=cleanup_old_photos, daemon=True).start()

        self.latest_frame = None
        self.frame_seq = 0          # 캡처 스레드가 새 프레임을 받을 때마다 1씩 증가
        self.cached_faces = []
        self.is_running = True

        # 화면 그리기 상태 (PhotoImage를 매 프레임 새로 만들지 않고 재사용한다)
        self._photo = None
        self._rendered_seq = -1
        self._fps_count = 0
        self._fps_since = time.perf_counter()

        # 카메라 읽기(cap.read는 다음 프레임이 올 때까지 최대 33ms 블로킹)와 인식 결과 주고받기는
        # 각자 백그라운드 스레드에서 돌리고, UI 스레드는 그리기만 한다
        self.capture_thread = threading.Thread(target=self.capture_worker, daemon=True)
        self.capture_thread.start()
        self.ai_thread = threading.Thread(target=self.ai_worker, daemon=True)
        self.ai_thread.start()

        self.show_idle()
        self.update_video()
        self._update_clock()
        if self.kiosk:
            # 창이 다 뜬 뒤부터 확인한다 (만드는 도중에 닫으면 오류가 난다)
            self.window.after(1000, self._check_operating_hours)

    def create_widgets(self):
        # 상단 헤더 — 좌측: 로고 · 제목 / 우측: 날짜·시간, [카메라 테스트] [출석 조회] [관리자]
        header = tk.Frame(self.window, bg=PANEL, height=px(84))
        header.pack(fill=tk.X, side=tk.TOP)
        header.pack_propagate(False)
        tk.Frame(self.window, bg=LINE, height=1).pack(fill=tk.X, side=tk.TOP)

        header_inner = tk.Frame(header, bg=PANEL)
        header_inner.pack(fill=tk.BOTH, expand=True, padx=px(28))

        logo = ImageTk.PhotoImage(compose_center(
            render_round_rect(px(50), px(50), px(14), BLUE_SOFT, PANEL), render_icon("face", px(28), BLUE)))
        lbl_logo = tk.Label(header_inner, image=logo, bg=PANEL, bd=0)
        lbl_logo.image = logo
        lbl_logo.pack(side=tk.LEFT)

        title_box = tk.Frame(header_inner, bg=PANEL)
        title_box.pack(side=tk.LEFT, padx=(px(14), 0))
        tk.Label(title_box, text="창의공간 얼굴 인식 출석체크 시스템", bg=PANEL, fg=TEXT,
                 font=(UI_FONT, 17, "bold")).pack(anchor=tk.W)

        nav = dict(height=px(44), bg=PANEL, font=(UI_FONT, 11, "bold"), padx=px(18))
        # 로그인 후에는 [관리자 로그아웃]으로 바뀐다
        self.btn_admin_header = RoundButton(header_inner, text="관리자", command=self.on_admin_header_button, **nav)
        self.btn_admin_header.pack(side=tk.RIGHT, padx=(px(6), 0))
        # 관리자 화면에서는 숨기는 버튼 묶음
        self.header_tools = tk.Frame(header_inner, bg=PANEL)
        self.header_tools.pack(side=tk.RIGHT, padx=(px(16), 0))
        # 얼굴 인식이 안 될 때 카메라가 제대로 나오는지 확인하는 용도 (기록은 하지 않는다)
        self.btn_camera_test = RoundButton(self.header_tools, text="카메라 테스트", command=self.start_camera_test,
                                           **nav)
        self.btn_camera_test.pack(side=tk.LEFT)
        # 얼굴로 본인 확인 후 이번주 출근 시간을 본다 (출퇴근 기록 없이)
        self.btn_lookup = RoundButton(self.header_tools, text="출석 조회", command=self.start_lookup, **nav)
        self.btn_lookup.pack(side=tk.LEFT, padx=(px(6), 0))
        self._update_nav()

        clock_font = (UI_FONT, 13, "bold")
        self.lbl_clock = RoundButton(
            header_inner, text="0000-00-00  00:00:00", height=px(44), font=clock_font, bg=PANEL,
            fill=PANEL_ALT, fg=TEXT, padx=px(16), icon=render_icon("clock", px(18), TEXT_SUB),
            width=tkfont.Font(font=clock_font).measure("0000-00-00  00:00:00") + px(18 + 10 + 34))
        self.lbl_clock.pack(side=tk.RIGHT)

        # 메인 콘텐츠 컨테이너 — 메인 / 확인 중(카드) / 카메라 화면을 번갈아 보여준다
        content_frame = tk.Frame(self.window, bg=BG)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=px(24), pady=px(22))
        self._build_idle_view(content_frame)

        # 출근/퇴근·출석 조회 확인 중, 결과, 출석 현황 — 카메라 영상 없이 카드만 가운데에 보여준다
        self.checking_view = tk.Frame(content_frame, bg=BG)

        # 카메라 화면 — 카메라 테스트 / 관리자 화면에서만 영상이 보인다
        self.session_view = tk.Frame(content_frame, bg=BG)

        # 좌측: 카메라 패널 (상단 상태바 + 영상 영역)
        self.camera_panel = RoundedPanel(self.session_view, fill=CAM_BG, bg=BG, radius=px(20), outline=None,
                                         pad=px(14))
        self.camera_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        cam = self.camera_panel.body

        cam_bar = tk.Frame(cam, bg=CAM_BG, height=px(40))
        cam_bar.pack(fill=tk.X, side=tk.TOP)
        cam_bar.pack_propagate(False)

        self.lbl_cam_status = tk.Label(cam_bar, text="● OFF", bg=CAM_BG, fg="#F87171",
                                       font=(UI_FONT, 11, "bold"))
        self.lbl_cam_status.pack(side=tk.LEFT, padx=(px(8), px(14)))

        # 깜빡임(사람 확인) 상태 — 사진을 들이대면 여기가 회색으로 남는다
        self.lbl_liveness = tk.Label(cam_bar, text="", bg=CAM_BG, fg="#94A3B8", font=(UI_FONT, 10, "bold"))
        self.lbl_liveness.pack(side=tk.LEFT)

        # 영상 영역 — 이미지 크기 계산은 상태바를 뺀 이 영역 기준으로 한다
        self.video_area = tk.Frame(cam, bg=CAM_BG)
        self.video_area.pack(fill=tk.BOTH, expand=True)

        self.video_label = tk.Label(self.video_area, bg=CAM_BG)
        self.video_label.pack(padx=px(10), pady=px(10), fill=tk.BOTH, expand=True)

        # 우측: 가변 상태 패널 컨테이너 (카메라 테스트 안내 / 관리자 화면)
        self.right_container = tk.Frame(self.session_view, width=px(RIGHT_PANEL_WIDTH), bg=BG)
        self.right_container.pack(side=tk.RIGHT, fill=tk.Y, expand=False, padx=(px(20), 0))
        self.right_container.pack_propagate(False)

    def _build_idle_view(self, parent):
        """메인 화면 — 카메라는 꺼져 있고 출근/퇴근 버튼만 가운데에 보여준다."""
        self.idle_view = tk.Frame(parent, bg=BG)

        center = tk.Frame(self.idle_view, bg=BG)
        center.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        card_w, card_h, gap = px(470), px(340), px(24)
        btn_row = tk.Frame(center, bg=BG)
        btn_row.pack()
        for i, log_type in enumerate(("CHECK_IN", "CHECK_OUT")):
            ActionCard(btn_row, log_type, lambda t=log_type: self.start_attendance(t), card_w, card_h) \
                .pack(side=tk.LEFT, padx=(0 if i == 0 else gap, 0))

        # 얼굴 인식 준비 상태 — 처음 실행하면 모델(약 300MB)을 받느라 몇 분 걸린다 (_update_clock에서 갱신)
        self.lbl_ai_status = tk.Label(center, text="", font=(UI_FONT, 11, "bold"), bg=BG, fg=WARN,
                                      justify=tk.CENTER)
        self.lbl_ai_status.pack(pady=(px(14), 0))

    def capture_worker(self):
        """
        camera_on에 맞춰 카메라를 열고 닫으며, 켜져 있는 동안 최신 프레임 1장만 보관한다.
        카메라는 이 스레드에서만 다룬다 (읽는 도중에 다른 스레드가 release하면 드라이버가 멈출 수 있다).
        꺼져 있을 때는 장치를 완전히 닫아서 카메라 불도 꺼진다.
        """
        cap = None
        while self.is_running:
            if not self.camera_on:
                if cap is not None:
                    cap.release()
                    cap = None
                time.sleep(0.05)
                continue

            if cap is None:
                cap = open_camera(0)  # OS별로 맞는 백엔드를 고른다 (open_camera 참고)
                if not cap.isOpened():
                    cap.release()
                    cap = None
                    self.camera_error = True
                    time.sleep(1.0)  # 다른 프로그램이 놓아줄 수도 있으니 켜져 있는 동안 1초마다 다시 시도
                    continue
                self.camera_error = False
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 밀린 옛 프레임 대신 항상 최신 프레임

            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            if self.camera_on:  # 읽는 사이에 꺼졌다면 버린다
                self.latest_frame = cv2.flip(frame, 1)
                self.frame_seq += 1

        if cap is not None:
            cap.release()

    def ai_worker(self):
        """최신 프레임을 인식 프로세스로 보내고 결과를 받아온다 (한 번에 1장씩만 보내서 밀리지 않게)."""
        try:
            self.face_conn.recv()  # 인식 프로세스의 모델 로딩 완료 신호
            self.ai_ready = True
            last_seq = -1
            while self.is_running:
                frame = self.latest_frame
                seq = self.frame_seq
                generation = self.camera_generation
                # 같은 프레임을 두 번 분석하지 않는다 (관리자 탭에서도 학생 등록용 스캔은 계속 유지)
                if self.camera_on and frame is not None and seq != last_seq:
                    last_seq = seq
                    self.face_conn.send(frame)
                    results, blink_score, liveness = self.face_conn.recv()
                    # 분석하는 사이에 카메라를 껐다 켰다면(다음 사람) 예전 얼굴 결과는 버린다
                    if self.camera_on and generation == self.camera_generation:
                        self.cached_faces = [DetectedFace(bbox, emb) for bbox, emb in results]
                        self._update_blink_state(blink_score)
                        self._update_liveness_state(liveness)
                else:
                    time.sleep(0.005)
        except (EOFError, OSError):
            # 프로그램 종료로 닫힌 경우가 아니면, 모델을 받지 못했거나 인식 프로세스가 죽은 것이다
            if self.is_running:
                self.ai_failed = True

    def _update_blink_state(self, blink_score):
        """
        눈을 감았다가(BLINK_CLOSED_SCORE 이상) 다시 떴을 때(BLINK_OPEN_SCORE 이하)
        한 번 깜빡인 것으로 보고 시각을 기록한다.
        사진은 눈을 감지 못하므로 이 시각이 갱신되지 않는다.
        """
        if blink_score is None:
            return  # 얼굴을 못 찾았거나 깜빡임 검사가 꺼져 있음

        self.liveness_enabled = True
        if blink_score >= BLINK_CLOSED_SCORE:
            self._eye_closed = True
        elif self._eye_closed and blink_score <= BLINK_OPEN_SCORE:
            self._eye_closed = False
            self.last_blink_time = time.time()

    def has_recent_blink(self):
        """최근 BLINK_VALID_SECONDS 안에 눈을 깜빡였는지. 검사가 꺼져 있으면 항상 통과."""
        if not self.liveness_enabled:
            return True
        return (time.time() - self.last_blink_time) <= BLINK_VALID_SECONDS

    def _update_liveness_state(self, liveness):
        """
        InsightFace 위조판별 애드온 결과를 기록한다.
        liveness = (is_live, status) 또는 None(애드온 없음/얼굴 없음)
        """
        if liveness is None:
            return

        is_live, status = liveness
        self.antispoof_enabled = True
        self.last_liveness_status = status
        if status == "ok":
            if is_live:
                self.last_live_time = time.time()
            else:
                self.last_fake_time = time.time()

    def passes_antispoof(self):
        """
        최근 LIVENESS_VALID_SECONDS 안에 '실제 사람' 판정이 있었는지.
        애드온이 없으면 항상 통과(이때는 깜빡임 검사가 대신 막는다).
        """
        if not self.antispoof_enabled:
            return True
        return (time.time() - self.last_live_time) <= LIVENESS_VALID_SECONDS

    def liveness_reject_message(self):
        """위조판별에서 막혔을 때 사용자에게 보여줄 안내 문구."""
        if (time.time() - self.last_fake_time) <= LIVENESS_VALID_SECONDS:
            return ("사진이나 화면으로 판단됩니다.\n"
                    "본인이 직접 카메라 앞에 서서 다시 시도해주세요.")
        if self.last_liveness_status == "input_rejected":
            return ("얼굴이 너무 크게 잡혀 확인할 수 없습니다.\n"
                    "카메라에서 조금 물러난 뒤 다시 시도해주세요.")
        return ("본인 확인을 하지 못했습니다.\n"
                "카메라를 정면으로 바라본 뒤 다시 시도해주세요.")

    def update_video(self):
        frame = self.latest_frame
        seq = self.frame_seq

        # 새 프레임이 들어왔을 때만 그린다 — 같은 프레임을 다시 그리는 CPU 낭비 방지
        if self.camera_on and frame is not None and seq != self._rendered_seq:
            self._rendered_seq = seq
            self._render_frame(frame)
        elif self.camera_on and self.camera_error and self._photo is None:
            msg = "📷\n\n카메라를 열 수 없습니다\n다른 프로그램이 카메라를 쓰고 있는지 확인하세요"
            if IS_MAC:
                msg += "\n\n시스템 설정 > 개인정보 보호 및 보안 > 카메라에서\n실행한 앱(터미널 등)을 허용한 뒤 다시 실행하세요"
            if self.video_label.cget("text") != msg:
                self.video_label.config(image="", text=msg, fg="#F87171", font=(UI_FONT, 13, "bold"))
                self.lbl_cam_status.config(text="● 카메라 오류", fg="#F87171")
            if self._session_card is not None:
                self._session_card.set_state("error", "카메라를 열 수 없습니다",
                                             "관리자에게 알려주세요")

        # 출근/퇴근·카메라 테스트 중 시간이 지나면 메인 화면으로 (카메라도 꺼진다)
        if self.mode in ("attendance", "lookup", "test") and self._session_card is not None:
            limit = TEST_TIMEOUT if self.mode == "test" else SESSION_TIMEOUT
            remaining = limit - (time.time() - self.session_started)
            if remaining <= 0:
                self.show_idle()
            else:
                self._session_card.set_remaining(int(remaining) + 1)

        if self.is_running:
            self.window.after(5, self.update_video)

    def _render_frame(self, frame):
        # 표시 크기로 먼저 줄이고 그 위에 박스/글자를 그린다 (PIL LANCZOS 11ms → cv2 1ms)
        panel_w = self.video_area.winfo_width() - 20
        panel_h = self.video_area.winfo_height() - 20
        src_h, src_w = frame.shape[:2]
        scale = min(panel_w / src_w, panel_h / src_h) if panel_w > 0 and panel_h > 0 else 1.0
        if MAX_DISPLAY_SCALE is not None:
            scale = min(scale, MAX_DISPLAY_SCALE)
        scale = max(scale, 0.1)
        out_w, out_h = max(1, int(src_w * scale)), max(1, int(src_h * scale))
        display = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

        best_user = None
        best_sim = -1.0
        # 사각형은 cv2로 그대로 그리고, 글자(한글)는 아래에서 PIL로 한 번에 그린다.
        # cv2.putText는 Hershey 폰트만 지원해서 한글을 못 그려 '?'로 깨져 보이는 문제가 있었다.
        labels_to_draw = []  # (x, y, text, RGB색상)

        # 얼굴 바운딩 박스 렌더링 (등록된 전원과의 유사도를 행렬곱으로 한번에 계산)
        faces = self.cached_faces
        if faces:
            face_embs = np.asarray([f.embedding for f in faces], dtype=np.float32)
            face_norms = np.linalg.norm(face_embs, axis=1, keepdims=True)
            face_norms[face_norms == 0] = 1.0
            face_embs_normed = face_embs / face_norms

            has_users = self.embed_matrix is not None and len(self.embed_matrix) > 0
            if has_users:
                sims = face_embs_normed @ self.embed_matrix.T  # (num_faces, num_users)
                match_idx = np.argmax(sims, axis=1)
                match_sims = sims[np.arange(len(faces)), match_idx]

            for i, face in enumerate(faces):
                x1, y1, x2, y2 = (face.bbox * scale).astype(int)  # 원본 좌표 → 표시 크기 좌표
                sim = float(match_sims[i]) if has_users else -1.0
                user = self.enrolled_users[match_idx[i]] if has_users else None

                if sim >= THRESHOLD and user:
                    # 이름은 창공시스템 명단(roster.json)을 우선 쓴다 — 새로고침하면 바로 반영된다.
                    display_name = self._resolve_display_name(user)
                    label = f"{display_name}  유사도 {sim:.2f}"
                    color_bgr = (74, 163, 22)  # 초록 (#16A34A) — cv2는 BGR 순서
                else:
                    label = f"미등록  유사도 {sim:.2f}" if sim >= 0 else "미등록"
                    color_bgr = (43, 54, 194)  # 빨강 (#C2362B) — cv2는 BGR 순서

                cv2.rectangle(display, (x1, y1), (x2, y2), color_bgr, 3)
                color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
                labels_to_draw.append((x1, y1, label, color_rgb))

                if sim > best_sim:
                    best_sim = sim
                    best_user = user

        matched_user = best_user if (best_user and best_sim >= THRESHOLD) else None
        if self.mode in ("attendance", "lookup") and self._session_card is not None:
            self._process_attendance(frame, faces, matched_user)
        elif self.mode == "test" and self._session_card is not None:
            self._process_camera_test(faces, matched_user, best_sim)

        if self.mode in ("attendance", "lookup"):
            return  # 출근/퇴근·출석 조회 중에는 카메라 영상을 화면에 그리지 않는다

        img = Image.fromarray(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))

        # 얼굴마다 이름 · 유사도를 상자 색 띠 위에 흰 글씨로, 한글이 제대로 보이게 PIL로 그린다
        if labels_to_draw:
            draw = ImageDraw.Draw(img)
            font = get_korean_font(px(18))
            pad_x, pad_y = px(8), px(5)
            for box_x, box_y, text, color_rgb in labels_to_draw:
                left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
                tag_w, tag_h = right - left + 2 * pad_x, bottom - top + 2 * pad_y
                tag_x = max(0, min(box_x - 1, img.width - tag_w))
                tag_y = box_y - tag_h if box_y - tag_h >= 0 else box_y  # 위가 잘리면 상자 안쪽에
                draw.rectangle((tag_x, tag_y, tag_x + tag_w, tag_y + tag_h), fill=color_rgb)
                draw.text((tag_x + pad_x - left, tag_y + pad_y - top), text, font=font, fill=(255, 255, 255))

        if self._photo is None:  # 카메라를 켠 뒤 첫 화면 — fps가 계산되기 전에도 LIVE로 바꿔 둔다
            self.lbl_cam_status.config(text="● LIVE", fg="#8FB0FF")

        # PhotoImage는 크기가 같으면 새로 만들지 않고 픽셀만 덮어쓴다 (20ms → 9ms)
        if self._photo is None or (self._photo.width(), self._photo.height()) != img.size:
            self._photo = ImageTk.PhotoImage(image=img)
            self.video_label.configure(image=self._photo)
        else:
            self._photo.paste(img)

        self._update_fps_label()

    def _update_fps_label(self):
        self._fps_count += 1
        now = time.perf_counter()
        elapsed = now - self._fps_since
        if elapsed >= 1.0:
            self.lbl_cam_status.config(text=f"● LIVE  {self._fps_count / elapsed:.0f} fps", fg="#8FB0FF")
            self._fps_count = 0
            self._fps_since = now
            self._update_liveness_label()

    def _update_liveness_label(self):
        """실제 사람으로 확인된 상태인지 카메라 상태바에 표시한다."""
        if self.antispoof_enabled and (time.time() - self.last_fake_time) <= LIVENESS_VALID_SECONDS:
            self.lbl_liveness.config(text="·  사진/화면으로 판단됨", fg="#F87171")
        elif not (self.antispoof_enabled or self.liveness_enabled):
            self.lbl_liveness.config(text="")
        elif self.passes_antispoof() and self.has_recent_blink():
            self.lbl_liveness.config(text="·  사람 확인됨 (눈 깜빡임 확인)", fg="#86EFAC")
        elif not self.has_recent_blink():
            self.lbl_liveness.config(text="·  눈을 깜빡여 주세요", fg="#94A3B8")
        else:
            self.lbl_liveness.config(text="·  카메라를 바라봐 주세요", fg="#94A3B8")

    def _process_attendance(self, frame, faces, matched_user):
        """
        출근/퇴근을 누른 뒤 매 프레임 불린다. 같은 학생이 RECOGNIZE_HOLD_SECONDS 동안 계속 인식되고
        사람 확인(위조판별 + 눈 깜빡임)까지 통과하면 그 순간 화면을 사진으로 남기고 자동으로 기록한다.
        """
        card = self._session_card
        if not self.ai_ready:
            card.set_state("loading", "얼굴 인식 준비 중..")
            return
        if matched_user is None:
            self._candidate_id = None
            if faces:
                card.set_state("warn", "등록되지 않은 얼굴입니다")
            else:
                card.set_state("search", "얼굴을 찾는 중..")
            return

        # 학생을 찾으면 "OOO 학생 얼굴 확인 중" + 눈 깜빡이는 얼굴을 보여준다.
        # 깜빡임은 최근 BLINK_VALID_SECONDS 안의 것을 인정하므로 확인 중에 미리 깜빡여도 된다.
        name = self._resolve_display_name(matched_user)
        checking = f"{name} 학생 얼굴 확인 중"
        now = time.time()
        if self._candidate_id != matched_user["user_id"]:
            self._candidate_id = matched_user["user_id"]
            self._candidate_since = now
        if now - self._candidate_since < RECOGNIZE_HOLD_SECONDS:
            card.set_state("blink", checking, "눈을 한 번 깜빡여 주세요")
            return
        if not self.passes_antispoof():
            if (now - self.last_fake_time) <= LIVENESS_VALID_SECONDS:
                card.set_state("error", "사진이나 화면으로 판단됩니다", "본인이 직접 카메라 앞에 서 주세요")
            elif self.last_liveness_status == "input_rejected":
                card.set_state("warn", "얼굴이 너무 가깝습니다", "카메라에서 조금 물러나 주세요")
            else:
                card.set_state("warn", checking, "카메라를 정면으로 바라봐 주세요")
            return
        if not self.has_recent_blink():
            card.set_state("blink", checking, "눈을 한 번 깜빡여 주세요")
            return

        # 확인 완료 — 출근/퇴근이면 기록한다 (출석 조회는 조회만). 사진은 남기지 않는다.
        # 카드를 먼저 비워서 두 번 처리되지 않게 한다.
        self._session_card = None
        student = dict(matched_user)
        student["name"] = name  # 최신 창공시스템 명단 이름 우선
        self.window.after_idle(lambda: self.complete_attendance(student))

    def complete_attendance(self, student):
        if self.mode == "lookup":
            self.show_stats(student)  # 기록·사진 없이 본인 출석 현황만
            return
        log_type = self.pending_action
        success, detail = log_attendance(student["user_id"], student["name"], log_type)
        self.show_result(student, log_type, success, detail)  # 카메라를 끄고 결과를 3초 동안 보여준다

    def _process_camera_test(self, faces, matched_user, best_sim):
        card = self._session_card
        if not self.ai_ready:
            card.set_state("loading", "얼굴 인식 준비 중..")
        elif matched_user is not None:
            card.set_state("ok", f"{self._resolve_display_name(matched_user)} 학생 인식",
                           f"유사도 {best_sim:.2f}",
                           round(best_sim, 2))
        elif faces:
            card.set_state("unknown", "등록되지 않은 얼굴입니다",
                           f"유사도 {max(best_sim, 0):.2f}", round(max(best_sim, 0), 2))
        else:
            card.set_state("search", "얼굴을 찾는 중..")

    def set_camera(self, on):
        """카메라와 얼굴 인식을 켜고 끈다. 실제 장치 열기/닫기는 캡처 스레드가 한다."""
        if on == self.camera_on:
            return
        self.camera_generation += 1
        self.camera_on = on
        self.latest_frame = None
        self.cached_faces = []  # 꺼지거나 새로 켜면 예전 얼굴로 출결/등록이 되지 않게 비운다
        self._photo = None
        self._fps_count = 0
        self._fps_since = time.perf_counter()
        self.lbl_liveness.config(text="")
        if on:
            self.camera_error = False
            self.lbl_cam_status.config(text="● 카메라 켜는 중", fg="#FBBF24")
            self.video_label.config(image="", text="카메라를 켜는 중입니다...",
                                    fg="#8A94A8", font=(UI_FONT, 14, "bold"))
        else:
            self.lbl_cam_status.config(text="● OFF", fg="#F87171")
            self.video_label.config(image="", text="")

    def _reset_liveness(self):
        """다음 사람이 앞 사람의 깜빡임/실제 사람 판정을 이어받지 않도록 비운다."""
        self.last_blink_time = 0.0
        self._eye_closed = False
        self.last_live_time = 0.0
        self.last_fake_time = 0.0
        self.last_liveness_status = None

    def _update_clock(self):
        if not self.is_running:
            return
        now = datetime.now()
        self.lbl_clock.set_text(now.strftime("%Y-%m-%d  %H:%M:%S"))
        if self.ai_failed:
            ai_text, ai_color = ("얼굴 인식을 시작하지 못했습니다.\n"
                                 "인터넷 연결을 확인하고 프로그램을 다시 실행하세요. (logs\\app.log 참고)"), ERR
        elif not self.ai_ready:
            ai_text, ai_color = ("얼굴 인식을 준비하고 있습니다...\n"
                                 "처음 실행하면 모델(약 300MB)을 받느라 몇 분 걸립니다."), WARN
        else:
            ai_text, ai_color = "", WARN
        if self.lbl_ai_status.cget("text") != ai_text:
            self.lbl_ai_status.config(text=ai_text, fg=ai_color)
        # 프로그램을 며칠씩 켜두어도 3개월 지난 사진이 지워지도록 날짜가 바뀌면 한 번 정리한다
        if now.date() != self._photo_cleanup_date:
            self._photo_cleanup_date = now.date()
            threading.Thread(target=cleanup_old_photos, daemon=True).start()
        self.window.after(1000, self._update_clock)

    def _check_operating_hours(self):
        """
        무인 운영(--kiosk)이면 운영 시간(06:00~24:00)이 끝나는 자정에 종료한다.
        관리자가 작업 중이면 로그아웃할 때까지 기다리고, 새벽에 직접 켠 경우(점검용)는 끄지 않는다.
        """
        if not self.is_running:
            return
        now = datetime.now()
        in_hours = now.strftime("%H:%M:%S") >= VALID_ATTENDANCE_START
        if in_hours:
            self._seen_operating_hours = True
        elif getattr(self, "_seen_operating_hours", False) and self.mode != "admin":
            print(f"[출석] {now:%Y-%m-%d %H:%M:%S} 운영 시간이 끝나 종료합니다.")
            self.on_close()
            return
        self.window.after(1000, self._check_operating_hours)

    def _rebuild_embedding_matrix(self):
        # 정규화된 임베딩을 미리 쌓아두면 프레임마다 norm을 다시 계산하지 않고
        # 행렬곱 한 번으로 전체 등록자와의 유사도를 구할 수 있다.
        if not self.enrolled_users:
            self.embed_matrix = np.empty((0, 512), dtype=np.float32)
            return
        mat = np.stack([u["embedding"] for u in self.enrolled_users]).astype(np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.embed_matrix = mat / norms

    def reload_users(self):
        self.enrolled_users = load_users()
        self._rebuild_embedding_matrix()

    def apply_roster(self, roster):
        """창공시스템 명단(roster.json)을 얼굴 인식 이름 표시용 조회표로 반영한다."""
        self.roster_lookup = {
            str(s.get("StudentId", "")).strip(): s
            for s in roster
            if str(s.get("StudentId", "")).strip()
        }

    def _resolve_display_name(self, user):
        """
        얼굴 인식된 사용자의 표시 이름.
        창공시스템 명단(roster.json, 최신)을 우선 쓰고, 거기 없으면 등록 당시 이름,
        그마저 없으면 '?' 로 표시해서 최소한 빈 문자열로 비어 보이지 않게 한다.
        """
        roster_entry = self.roster_lookup.get(str(user.get("user_id", "")).strip())
        if roster_entry and roster_entry.get("Name"):
            return roster_entry["Name"]
        if user.get("name"):
            return user["name"]
        return "?"

    # ---------- 화면 전환 ----------
    def _show_view(self, view):
        """메인 / 카드(확인 중·결과·출석 현황) / 카메라 화면 중 하나만 보이게 한다."""
        for other in (self.idle_view, self.checking_view, self.session_view):
            if other is not view:
                other.pack_forget()
        view.pack(fill=tk.BOTH, expand=True)

    def _show_session_layout(self):
        self._show_view(self.session_view)

    def _clear_session_cards(self):
        self.clear_right_container()
        for widget in self.checking_view.winfo_children():
            widget.destroy()

    def _show_card(self, card, size):
        """카드 하나를 콘텐츠 영역 가운데에 띄운다 (카메라 영상 없이)."""
        width, height = (px(v) for v in size)
        card.place(relx=0.5, rely=0.5, anchor=tk.CENTER, width=width, height=height)
        self._show_view(self.checking_view)

    def _update_nav(self):
        """상단 버튼 중 지금 화면에 해당하는 버튼을 파랗게 칠한다."""
        active = {"test": self.btn_camera_test, "lookup": self.btn_lookup, "stats": self.btn_lookup,
                  "admin": self.btn_admin_header}.get(self.mode)
        for btn in (self.btn_camera_test, self.btn_lookup, self.btn_admin_header):
            if btn is active:
                btn.set_style(BLUE, BLUE_DARK, "white")
            else:
                btn.set_style(PANEL, BLUE_SOFT, TEXT_SUB, hover_fg=BLUE)

    def _start_session(self, mode, card_factory):
        self.mode = mode
        self.session_started = time.time()
        self._candidate_id = None
        self._reset_liveness()
        self._clear_session_cards()
        self._session_card = card_factory()
        if mode in ("attendance", "lookup"):
            # 카메라는 뒤에서만 돌고, 화면에는 안내 카드만 보인다
            self._show_card(self._session_card, VERIFY_CARD_SIZE)
        else:
            # 카메라 테스트 — 카메라가 왼쪽 약 2/3, 안내 패널이 오른쪽 1/3
            self.right_container.config(width=px(TEST_PANEL_WIDTH))
            self._session_card.pack(fill=tk.BOTH, expand=True)
            self._show_session_layout()
        self._update_nav()
        self.set_camera(True)

    def start_attendance(self, log_type):
        """메인 화면에서 출근/퇴근을 눌렀을 때 — 카메라를 (화면에 띄우지 않고) 켜고 얼굴을 기다린다."""
        self.pending_action = log_type
        self._start_session("attendance", lambda: VerifyCard(self, log_type, self.checking_view,
                                                             *(px(v) for v in VERIFY_CARD_SIZE)))

    def start_lookup(self):
        """상단 [출석 조회] — 얼굴로 본인 확인 후 그 학생의 출석 시간만 보여준다 (기록하지 않는다)."""
        self.pending_action = None
        self._start_session("lookup", lambda: VerifyCard(self, "LOOKUP", self.checking_view,
                                                         *(px(v) for v in VERIFY_CARD_SIZE)))

    def start_camera_test(self):
        """상단 [카메라 테스트] — 기록 없이 카메라 화면과 인식 결과만 보여준다."""
        self.pending_action = None
        self._start_session("test", lambda: CameraTestCard(self, self.right_container, px(TEST_PANEL_WIDTH)))

    def _end_session(self, mode):
        """확인이 끝나면 카메라를 끄고 다음 화면(결과 / 출석 현황)으로 넘어갈 준비를 한다."""
        self.mode = mode
        self.pending_action = None
        self._candidate_id = None
        self._session_card = None
        self.set_camera(False)
        self._clear_session_cards()
        self._update_nav()

    def show_result(self, student, log_type, success, detail):
        """출근/퇴근 처리 결과 — RESULT_POPUP_SECONDS 뒤 저절로 메인 화면으로 돌아간다."""
        self._end_session("result")
        self._show_card(ResultCard(self, student, log_type, success, detail, self.checking_view),
                        RESULT_CARD_SIZE)

    def show_stats(self, student):
        """본인 확인된 학생의 이번주 출석 현황 — STATS_POPUP_SECONDS 뒤 저절로 메인 화면으로 돌아간다."""
        self._end_session("stats")
        self._show_card(StatsCard(self, student, self.checking_view, px(STATS_CARD_SIZE[0])), STATS_CARD_SIZE)

    def show_idle(self):
        """메인 화면 — 카메라와 얼굴 인식을 끄고 출근/퇴근 버튼만 보여준다."""
        self.mode = "idle"
        self.pending_action = None
        self._candidate_id = None
        self._session_card = None
        self._clear_session_cards()
        self.set_camera(False)
        self._show_view(self.idle_view)
        self.btn_admin_header.set_text("관리자")
        if not self.header_tools.winfo_ismapped():
            self.header_tools.pack(side=tk.RIGHT, padx=(px(16), 0), before=self.lbl_clock)
        self._update_nav()

    def on_admin_header_button(self):
        if self.admin_logged_in:
            self.set_admin_logged_in(False)
            return
        AdminLoginDialog(self)

    def set_admin_logged_in(self, logged_in, admin_email=""):
        self.admin_logged_in = logged_in
        self.admin_email = admin_email
        if not logged_in:
            self.show_idle()
            return

        self.mode = "admin"
        self.pending_action = None
        self._session_card = None
        self._clear_session_cards()
        self.right_container.config(width=px(RIGHT_PANEL_WIDTH))
        frame = AdminDashboardFrame(self, admin_email)
        frame.pack(fill=tk.BOTH, expand=True)
        self._show_session_layout()
        self.set_camera(True)  # 학생 등록 탭에서 얼굴을 찍어야 하므로 켜둔다
        self.btn_admin_header.set_text("관리자 로그아웃")
        self.header_tools.pack_forget()  # 관리자 화면은 이미 카메라가 켜져 있다
        self._update_nav()

    def clear_right_container(self):
        for widget in self.right_container.winfo_children():
            widget.destroy()

    def on_close(self):
        self.is_running = False
        self.capture_thread.join(timeout=1.0)  # 카메라는 캡처 스레드가 닫고 끝난다
        self.face_proc.terminate()
        self.window.destroy()


# ==========================================
# 4. 앱 실행
# ==========================================
def run_selftest(app, seconds=10):
    """
    --selftest : 설치 후 자가진단. 카메라 테스트 화면을 seconds초 동안 켜서
    카메라·얼굴 인식·사진 차단 검사가 도는지 logs/selftest.txt 에 적고 종료한다.
    """
    stats = {"frames_start": None, "faces": 0, "blink": False, "liveness": False}
    original = app._update_blink_state

    def spy(score):  # 인식 결과가 올 때마다 불린다
        if app.cached_faces:
            stats["faces"] += 1
        original(score)
    app._update_blink_state = spy

    def start():
        if not app.ai_ready and not app.ai_failed and time.time() - started < 600:
            app.window.after(500, start)  # 처음이면 모델을 받는 중이다 (최대 10분 대기)
            return
        app.start_camera_test()
        stats["frames_start"] = app.frame_seq
        app.window.after(seconds * 1000, finish)

    def finish():
        frames = app.frame_seq - (stats["frames_start"] or 0)
        lines = [
            f"자가진단 {datetime.now():%Y-%m-%d %H:%M:%S}",
            f"얼굴 인식 모델 : {'정상' if app.ai_ready else '실패'}",
            f"카메라         : {'정상' if frames > 0 and not app.camera_error else '실패'} ({frames / seconds:.0f} fps)",
            f"얼굴 감지      : {stats['faces']}회 (카메라 앞에 사람이 있어야 잡힙니다)",
            f"눈 깜빡임 검사 : {'사용' if app.liveness_enabled else '확인 안 됨 (얼굴이 없었거나 모델 없음)'}",
            f"위조판별       : {'사용' if app.antispoof_enabled else '확인 안 됨 (얼굴이 없었거나 애드온 없음)'}",
        ]
        ok = app.ai_ready and frames > 0 and not app.camera_error
        lines.append("결과           : " + ("정상" if ok else "문제 있음"))
        os.makedirs(os.path.join(APP_DIR, "logs"), exist_ok=True)
        with open(os.path.join(APP_DIR, "logs", "selftest.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        app.on_close()

    started = time.time()
    app.window.after(500, start)


if __name__ == "__main__":
    multiprocessing.freeze_support()  # exe로 패키징해도 인식 프로세스가 앱을 다시 띄우지 않게

    # --prepare : 얼굴 인식·위조판별·깜빡임 모델을 미리 내려받고 끝낸다 (설치할 때 한 번)
    if "--prepare" in sys.argv:
        create_face_app()
        ready = create_blink_detector() is not None
        print(f"[출석] 모델 준비 완료 (깜빡임 검사: {'사용' if ready else '사용 안 함'})")
        sys.exit(0)

    # --scheduled : 작업 스케줄러/시작프로그램이 띄운 경우. 00:00~06:00이면 켜지 않고, 자정에 스스로 종료한다.
    scheduled = "--scheduled" in sys.argv
    if scheduled and datetime.now().strftime("%H:%M:%S") < VALID_ATTENDANCE_START:
        sys.exit(0)

    # 카메라는 한 프로그램만 쓸 수 있으므로 이미 켜져 있으면 하나 더 띄우지 않는다
    if not acquire_single_instance():
        if not scheduled:
            _root = tk.Tk()
            _root.withdraw()
            messagebox.showinfo("창의공간 출석체크", "출석체크 프로그램이 이미 실행 중입니다.")
        sys.exit(0)

    root = tk.Tk()
    app = AttendanceApp(root, kiosk=scheduled or "--kiosk" in sys.argv)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    if "--selftest" in sys.argv:
        run_selftest(app)
    root.mainloop()