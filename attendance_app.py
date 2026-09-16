import cv2
import numpy as np
import threading
import time
import sqlite3
import json
import os
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, filedialog
from tkinter import font as tkfont
from PIL import Image, ImageDraw, ImageFont, ImageTk
from insightface.app import FaceAnalysis

DB_PATH = "faces.db"
THRESHOLD = 0.50
RIGHT_PANEL_WIDTH = 620  # 우측 UI 패널 폭 — 1600x900 기준 값이고, 실제로는 px()로 화면 배율을 곱해서 쓴다

# 기준 창 크기(창공시스템과 동일). 모니터가 이보다 작으면 UI_SCALE만큼 전체를 줄여서 띄운다.
BASE_WIN_W, BASE_WIN_H = 1600, 900
UI_SCALE = 1.0

# 주간 출석 기준 — 월요일~일요일 동안 채워야 하는 시간/일수
WEEKLY_REQUIRED_SECONDS = 20 * 3600
WEEKLY_REQUIRED_DAYS = 3
# 00:00~06:00 사이에 찍은 출근/퇴근은 무효. 06:00~24:00 사이 기록만 인정한다.
VALID_ATTENDANCE_START = "06:00:00"


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
    # 윈도우가 아니거나 조회에 실패하면 작업 표시줄 높이를 대략 빼서 쓴다
    return window.winfo_screenwidth(), window.winfo_screenheight() - 48

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
            major TEXT NOT NULL DEFAULT '컴퓨터 공학 전공',
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
        cursor.execute("ALTER TABLE users ADD COLUMN major TEXT NOT NULL DEFAULT '컴퓨터 공학 전공'")
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

    # 00:00~06:00 사이 출근/퇴근은 무효라서 아예 빼고 계산한다
    valid_logs = [log for log in logs if is_valid_attendance_time(log[4])]

    # 출근 일수: 인정되는 출근 기록이 있는 날 (퇴근을 안 찍었어도 일수는 인정)
    attended_days = len({log[2] for log in valid_logs if log[3] == "CHECK_IN"})

    # 출근 시간: 출근→퇴근 한 쌍이 모두 있어야 인정 (하루 여러 번이면 모두 더한다)
    total_seconds = int(sum(session_seconds(s) for s in pair_attendance_sessions(valid_logs)))

    return {
        "days": attended_days,
        "seconds": total_seconds,
        "missing_days": max(0, WEEKLY_REQUIRED_DAYS - attended_days),
        "missing_seconds": max(0, WEEKLY_REQUIRED_SECONDS - total_seconds),
    }

def log_attendance(user_id, name, log_type):
    today_date = datetime.now().strftime("%Y-%m-%d")
    now_time = datetime.now().strftime("%H:%M:%S")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 하루에 출근/퇴근을 여러 번 할 수 있어서 누를 때마다 새 기록으로 남긴다
    cursor.execute("""
        INSERT INTO attendance_logs (user_id, name, log_type, log_date, log_time)
        VALUES (?, ?, ?, ?, ?)
    """, (user_id, name, log_type, today_date, now_time))
    conn.commit()
    conn.close()
    export_attendance_json()  # 창공시스템이 읽는 출결 파일도 즉시 갱신

    type_str = "출근" if log_type == 'CHECK_IN' else "퇴근"
    if not is_valid_attendance_time(now_time):
        return True, (f"{now_time} {type_str}이 기록되었습니다.\n"
                      "단, 00:00~06:00 사이 출퇴근은 출석 일수와 시간에 인정되지 않습니다.")
    return True, f"{now_time} {type_str}이 확인 되었습니다."


# ==========================================
# 2. UI 상태 프레임 정의
# ==========================================

class AdminLoginFrame(tk.Frame):
    """ 두번째 이미지: 기본 화면 (관리자 로그인 기능) """
    def __init__(self, parent):
        super().__init__(parent.right_container, bg="white", bd=1, relief=tk.SOLID)
        self.parent = parent

        # 외부 마진 시뮬레이션용 프레임
        card_inner = tk.Frame(self, bg="white")
        card_inner.pack(padx=px(60), pady=px(60), fill=tk.BOTH, expand=True)

        title_row = tk.Frame(card_inner, bg="white")
        title_row.pack(fill=tk.X, pady=(0, px(30)))

        lbl_title = tk.Label(title_row, text="관리자 로그인", font=("맑은 고딕", 20, "bold"), bg="white", fg="#1E293B")
        lbl_title.pack(side=tk.LEFT)

        # 학생 화면에서 [관리자]를 눌러 여기로 온 경우 되돌아갈 수 있게 (기본 화면에선 눌러도 그대로라 무해함)
        btn_cancel = tk.Button(
            title_row, text="✕ 취소", bg="white", fg="#94A3B8", bd=0,
            activebackground="white", activeforeground="#475569",
            font=("맑은 고딕", 10, "bold"), cursor="hand2", command=self.parent.reset_to_default_view
        )
        btn_cancel.pack(side=tk.RIGHT)

        # 이메일 입력 그룹 (기입된 글자 없도록 비워둠)
        lbl_email = tk.Label(card_inner, text="이메일 주소", font=("맑은 고딕", 10, "bold"), bg="white", fg="#64748B")
        lbl_email.pack(anchor=tk.W, pady=(0, 4))
        self.ent_email = ttk.Entry(card_inner, font=("맑은 고딕", 12))
        self.ent_email.pack(fill=tk.X, ipady=4, pady=(0, 15))

        # 비밀번호 입력 그룹 (기입된 글자 없도록 비워둠)
        lbl_pwd = tk.Label(card_inner, text="비밀번호", font=("맑은 고딕", 10, "bold"), bg="white", fg="#64748B")
        lbl_pwd.pack(anchor=tk.W, pady=(0, 4))
        self.ent_pwd = ttk.Entry(card_inner, font=("맑은 고딕", 12), show="*")
        self.ent_pwd.pack(fill=tk.X, ipady=4, pady=(0, 25))

        # 로그인 버튼 해상도(크기) 강화
        btn_login = tk.Button(
            card_inner, text="로그인 (Log In)", bg="#3B82F6", fg="white", bd=0,
            activebackground="#2563EB", activeforeground="white",
            font=("맑은 고딕", 13, "bold"), height=2, cursor="hand2", command=self.try_login
        )
        btn_login.pack(fill=tk.X, pady=5)

    def try_login(self):
        email = self.ent_email.get().strip()
        pwd = self.ent_pwd.get().strip()

        if not (email and pwd):
            messagebox.showwarning("주의", "이메일과 비밀번호를 입력해주세요.")
            return

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT name, role FROM users WHERE user_id = ? AND password = ?", (email, pwd))
        row = cursor.fetchone()
        conn.close()

        if row:
            name, role = row
            if role == "admin":
                self.parent.set_admin_logged_in(True, email)
                self.ent_email.delete(0, tk.END)
                self.ent_pwd.delete(0, tk.END)
            else:
                messagebox.showerror("오류", "관리자 권한이 없습니다.")
        else:
            messagebox.showerror("오류", "로그인 정보가 틀렸습니다.")


class StudentInfoFrame(tk.Frame):
    """ 첫번째 이미지: 학생 얼굴 인식 시 뜨는 화면 (비밀번호 추가 검증 단계 포함) """
    def __init__(self, parent, student_info):
        super().__init__(parent.right_container, bg="white")
        self.parent = parent
        self.student_info = student_info

        # 학생 얼굴이 인식된 상태에서도 관리자가 바로 로그인 화면으로 넘어갈 수 있는 버튼
        top_bar = tk.Frame(self, bg="white")
        top_bar.pack(fill=tk.X, padx=px(15), pady=(px(10), 0))

        btn_admin = tk.Button(
            top_bar, text="🔑 관리자", bg="#1E293B", fg="white", bd=0,
            activebackground="#334155", activeforeground="white",
            font=("맑은 고딕", 9, "bold"), padx=10, pady=4, cursor="hand2",
            command=self.parent.show_admin_login
        )
        btn_admin.pack(side=tk.RIGHT)

        # 이번주 출석 현황 박스 (Clean Card Style)
        stats_frame = tk.Frame(self, bg="#F8FAFC", bd=1, relief=tk.SOLID)
        stats_frame.pack(fill=tk.X, padx=px(15), pady=px(10))

        lbl_stats_title = tk.Label(stats_frame, text="이번주 출석 현황", font=("맑은 고딕", 13, "bold"), bg="#F8FAFC", fg="#1E293B")
        lbl_stats_title.pack(anchor=tk.W, padx=px(15), pady=(px(12), px(8)))

        weekly = get_weekly_attendance_stats(student_info["user_id"])

        # 항목 / 이번주 기록 / 부족분을 표처럼 맞춰서 보여준다. 기준에 못 미친 부족분은 빨간색으로 강조.
        stats_grid = tk.Frame(stats_frame, bg="#F8FAFC")
        stats_grid.pack(anchor=tk.W, fill=tk.X, padx=px(25), pady=px(2))

        def add_stat_row(row, title, value_text, missing_text):
            tk.Label(stats_grid, text=f"• {title} :", font=("맑은 고딕", 11), bg="#F8FAFC", fg="#334155") \
                .grid(row=row, column=0, sticky=tk.W, pady=px(2))
            tk.Label(stats_grid, text=value_text, font=("맑은 고딕", 11, "bold"), bg="#F8FAFC", fg="#0F172A") \
                .grid(row=row, column=1, sticky=tk.W, padx=(px(8), 0), pady=px(2))
            if missing_text:
                tk.Label(stats_grid, text=f"부족 {missing_text}", font=("맑은 고딕", 11, "bold"), bg="#F8FAFC", fg="#EF4444") \
                    .grid(row=row + 1, column=1, sticky=tk.W, padx=(px(8), 0), pady=(0, px(4)))
            else:
                tk.Label(stats_grid, text="기준 충족", font=("맑은 고딕", 10, "bold"), bg="#F8FAFC", fg="#10B981") \
                    .grid(row=row + 1, column=1, sticky=tk.W, padx=(px(8), 0), pady=(0, px(4)))

        add_stat_row(
            0, "출근 일수",
            f"{weekly['days']}일 / {WEEKLY_REQUIRED_DAYS}일",
            f"{weekly['missing_days']}일" if weekly["missing_days"] else "",
        )
        add_stat_row(
            2, "출근 시간",
            f"{format_duration(weekly['seconds'])} / {WEEKLY_REQUIRED_SECONDS // 3600}시간",
            format_duration(weekly["missing_seconds"]) if weekly["missing_seconds"] else "",
        )

        # 3진 아웃제 기준 패널티 색상 경고 표기
        penalty_val = student_info['penalty']
        penalty_color = "#EF4444" if penalty_val >= 2 else "#F59E0B" if penalty_val == 1 else "#10B981"
        
        penalty_container = tk.Frame(stats_frame, bg="#F8FAFC")
        penalty_container.pack(anchor=tk.W, padx=px(25), pady=(px(2), px(4)))
        
        lbl_penalty_bullet = tk.Label(penalty_container, text="• 패널티 현황 : ", font=("맑은 고딕", 11), bg="#F8FAFC", fg="#334155")
        lbl_penalty_bullet.pack(side=tk.LEFT)
        
        lbl_penalty_value = tk.Label(penalty_container, text=f"{penalty_val}개", font=("맑은 고딕", 11, "bold"), bg="#F8FAFC", fg=penalty_color)
        lbl_penalty_value.pack(side=tk.LEFT)

        lbl_rule = tk.Label(
            stats_frame, text="06:00~24:00 출퇴근만 인정 · 퇴근을 찍어야 시간 인정",
            font=("맑은 고딕", 9), bg="#F8FAFC", fg="#94A3B8"
        )
        lbl_rule.pack(anchor=tk.W, padx=px(25), pady=(0, px(12)))

        # 학생 프로필 정보 박스
        profile_frame = tk.Frame(self, bg="white", bd=1, relief=tk.SOLID)
        profile_frame.pack(fill=tk.BOTH, expand=True, padx=px(15), pady=px(10))

        info_inner = tk.Frame(profile_frame, bg="white")
        info_inner.pack(padx=px(20), pady=px(20), fill=tk.BOTH, expand=True)

        lbl_name_tag = tk.Label(info_inner, text="이름", font=("맑은 고딕", 9, "bold"), bg="white", fg="#94A3B8")
        lbl_name_tag.pack(anchor=tk.W, pady=(0, 1))
        lbl_name = tk.Label(info_inner, text=student_info["name"], font=("맑은 고딕", 14, "bold"), bg="white", fg="#0F172A")
        lbl_name.pack(anchor=tk.W, pady=(0, 12))

        lbl_id_tag = tk.Label(info_inner, text="학번", font=("맑은 고딕", 9, "bold"), bg="white", fg="#94A3B8")
        lbl_id_tag.pack(anchor=tk.W, pady=(0, 1))
        lbl_id = tk.Label(info_inner, text=student_info['user_id'], font=("맑은 고딕", 14, "bold"), bg="white", fg="#0F172A")
        lbl_id.pack(anchor=tk.W, pady=(0, 12))

        lbl_major_tag = tk.Label(info_inner, text="전공 학과", font=("맑은 고딕", 9, "bold"), bg="white", fg="#94A3B8")
        lbl_major_tag.pack(anchor=tk.W, pady=(0, 1))
        lbl_major = tk.Label(info_inner, text=student_info['major'], font=("맑은 고딕", 13, "bold"), bg="white", fg="#0F172A")
        lbl_major.pack(anchor=tk.W, pady=(0, 15))

        # 출근 / 퇴근 버튼 구성 (크기 강화)
        btn_frame = tk.Frame(info_inner, bg="white")
        btn_frame.pack(fill=tk.X, pady=5)

        btn_in = tk.Button(
            btn_frame, text="출 근 (IN)", bg="#10B981", fg="white", font=("맑은 고딕", 13, "bold"), 
            bd=0, activebackground="#059669", activeforeground="white", height=2, cursor="hand2",
            command=lambda: self.handle_action("CHECK_IN")
        )
        btn_in.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)

        btn_out = tk.Button(
            btn_frame, text="퇴 근 (OUT)", bg="#3B82F6", fg="white", font=("맑은 고딕", 13, "bold"), 
            bd=0, activebackground="#2563EB", activeforeground="white", height=2, cursor="hand2",
            command=lambda: self.handle_action("CHECK_OUT")
        )
        btn_out.pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=5)

    def handle_action(self, log_type):
        # 도용 방지 비밀번호 확인 다이얼로그 띄우기
        pwd_input = simpledialog.askstring("도용 방지", "본인 확인을 위해 비밀번호를 입력해주세요:", show="*", parent=self)
        if pwd_input is None:
            return # 취소 시 동작 안 함
            
        # DB에서 저장된 본인 비밀번호와 매칭 검사
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT password FROM users WHERE user_id = ?", (self.student_info["user_id"],))
        row = cursor.fetchone()
        conn.close()

        if row and row[0] == pwd_input:
            success, msg = log_attendance(self.student_info["user_id"], self.student_info["name"], log_type)
            if success:
                messagebox.showinfo("확인 완료", msg)
            else:
                messagebox.showwarning("주의", msg)
            self.parent.reset_to_default_view()
        else:
            messagebox.showerror("오류", "비밀번호가 일치하지 않습니다. 도용 방지를 위해 요청을 중단합니다.")


class AdminDashboardFrame(tk.Frame):
    """ 세번째 이미지: 관리자 로그인시 화면 """
    def __init__(self, parent, admin_email):
        super().__init__(parent.right_container, bg="white")
        self.parent = parent
        self.admin_email = admin_email

        # 1행: 관리자 정보 + 로그아웃
        top_bar = tk.Frame(self, bg="white")
        top_bar.pack(fill=tk.X, pady=(px(10), px(4)), padx=px(12))

        lbl_admin = tk.Label(top_bar, text=f"🔑 관리자: {self.admin_email}", font=("맑은 고딕", 11, "bold"), bg="white", fg="#334155")
        lbl_admin.pack(side=tk.LEFT, pady=5)

        btn_logout = tk.Button(
            top_bar, text="로그아웃", bg="#EF4444", fg="white", font=("맑은 고딕", 10, "bold"),
            activebackground="#DC2626", activeforeground="white", bd=0, padx=16, pady=6, cursor="hand2",
            command=self.logout
        )
        btn_logout.pack(side=tk.RIGHT, pady=5)

        # 2행: 기능 버튼 — 한 줄에 몰아넣으면 글자가 잘려서 아래 줄에 반반 나눠 배치
        action_bar = tk.Frame(self, bg="white")
        action_bar.pack(fill=tk.X, padx=px(12), pady=(0, px(8)))

        btn_refresh = tk.Button(
            action_bar, text="🔄 창공시스템 명단 새로고침", bg="#3B82F6", fg="white", font=("맑은 고딕", 10, "bold"),
            activebackground="#2563EB", activeforeground="white", bd=0, pady=8, cursor="hand2",
            command=self.refresh_roster_data
        )
        btn_refresh.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5))

        btn_export = tk.Button(
            action_bar, text="📊 출석 데이터 추출", bg="#10B981", fg="white", font=("맑은 고딕", 10, "bold"),
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

        ttk.Label(form_frame, text="전공:").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        self.edit_major = ttk.Entry(form_frame, width=12)
        self.edit_major.grid(row=0, column=3, padx=5, pady=5, sticky=tk.EW)

        ttk.Label(form_frame, text="패널티:").grid(row=0, column=4, padx=5, pady=5, sticky=tk.W)
        self.edit_penalty = ttk.Entry(form_frame, width=4)
        self.edit_penalty.grid(row=0, column=5, padx=5, pady=5)

        btn_update = ttk.Button(form_frame, text="수정 완료", command=self.update_student)
        # 입력칸 옆에 두면 화면이 작을 때 잘려서, 입력칸 아래 줄에 폭 전체로 둔다
        btn_update.grid(row=1, column=0, columnspan=6, padx=5, pady=(0, 5), sticky=tk.EW)
        form_frame.columnconfigure(1, weight=2)
        form_frame.columnconfigure(3, weight=3)


        btn_del_sel = ttk.Button(btn_action_frame, text="선택 삭제", command=self.delete_selected)
        btn_del_sel.pack(fill=tk.X, ipady=4, pady=2)

        btn_del_all = ttk.Button(btn_action_frame, text="일괄 삭제", command=self.delete_all)
        btn_del_all.pack(fill=tk.X, ipady=4, pady=2)

        self.load_students()

    def build_add_student_tab(self):
        frame = ttk.LabelFrame(self.tab_add_student, text=" 학생 얼굴 및 상세 정보 등록 ")
        frame.pack(fill=tk.BOTH, expand=True, padx=px(10), pady=px(10))

        grid_container = tk.Frame(frame, bg="#F8FAFC")
        grid_container.pack(padx=px(20), pady=px(20), fill=tk.BOTH, expand=True)

        # 창공시스템(SeatManagerApp) 명단에서 골라서 자동 입력 — 수기로 다시 안 쳐도 된다
        ttk.Label(grid_container, text="창공시스템 명단:", font=("맑은 고딕", 10, "bold"), background="#F8FAFC").grid(row=0, column=0, padx=px(10), pady=px(8), sticky=tk.W)
        self.roster_combo = ttk.Combobox(grid_container, font=("맑은 고딕", 10), width=26, state="readonly")
        self.roster_combo.grid(row=0, column=1, padx=px(10), pady=px(8), sticky=tk.W)
        self.roster_combo.bind("<<ComboboxSelected>>", self.on_roster_selected)

        btn_reload_roster = ttk.Button(grid_container, text="🔄 명단 새로고침", command=lambda: self.reload_roster_combo(show_message_if_empty=True))
        btn_reload_roster.grid(row=0, column=2, padx=px(10), pady=px(8), sticky=tk.W)

        ttk.Label(grid_container, text="학번(ID):", font=("맑은 고딕", 10, "bold"), background="#F8FAFC").grid(row=1, column=0, padx=px(10), pady=px(8), sticky=tk.W)
        self.add_id = ttk.Entry(grid_container, font=("맑은 고딕", 11), width=26)
        self.add_id.grid(row=1, column=1, padx=px(10), pady=px(8), sticky=tk.W)

        ttk.Label(grid_container, text="비밀번호:", font=("맑은 고딕", 10, "bold"), background="#F8FAFC").grid(row=2, column=0, padx=px(10), pady=px(8), sticky=tk.W)
        self.add_pwd = ttk.Entry(grid_container, show="*", font=("맑은 고딕", 11), width=26)
        self.add_pwd.grid(row=2, column=1, padx=px(10), pady=px(8), sticky=tk.W)

        ttk.Label(grid_container, text="이름:", font=("맑은 고딕", 10, "bold"), background="#F8FAFC").grid(row=3, column=0, padx=px(10), pady=px(8), sticky=tk.W)
        self.add_name = ttk.Entry(grid_container, font=("맑은 고딕", 11), width=26)
        self.add_name.grid(row=3, column=1, padx=px(10), pady=px(8), sticky=tk.W)

        ttk.Label(grid_container, text="전공 학과:", font=("맑은 고딕", 10, "bold"), background="#F8FAFC").grid(row=4, column=0, padx=px(10), pady=px(8), sticky=tk.W)
        self.add_major = ttk.Entry(grid_container, font=("맑은 고딕", 11), width=26)
        self.add_major.grid(row=4, column=1, padx=px(10), pady=px(8), sticky=tk.W)
        self.add_major.insert(0, "컴퓨터 공학 전공")

        # 등록 버튼 크기 강화
        btn_register = tk.Button(
            grid_container, text="💾 카메라 인식 얼굴로 등록", bg="#10B981", fg="white", bd=0,
            activebackground="#059669", activeforeground="white", font=("맑은 고딕", 12, "bold"),
            height=2, cursor="hand2", command=self.register_student
        )
        btn_register.grid(row=5, column=0, columnspan=3, pady=px(25), sticky=tk.EW)

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
        if student.get("Department"):
            self.add_major.delete(0, tk.END)
            self.add_major.insert(0, student.get("Department", ""))

    def build_logs_tab(self):
        # 상단 필터/검색 영역 바
        filter_frame = ttk.LabelFrame(self.tab_logs, text=" 기간 및 학생 검색 조건 ")
        filter_frame.pack(fill=tk.X, padx=5, pady=5)

        # 기본 기간을 오늘 기준으로 세팅
        today_str = datetime.now().strftime("%Y-%m-%d")
        week_ago_str = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

        ttk.Label(filter_frame, text="시작일:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.ent_start_date = ttk.Entry(filter_frame, width=10)
        self.ent_start_date.insert(0, week_ago_str)
        self.ent_start_date.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(filter_frame, text="종료일:").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        self.ent_end_date = ttk.Entry(filter_frame, width=10)
        self.ent_end_date.insert(0, today_str)
        self.ent_end_date.grid(row=0, column=3, padx=5, pady=5)

        ttk.Label(filter_frame, text="이름 검색:").grid(row=0, column=4, padx=5, pady=5, sticky=tk.W)
        self.ent_search_name = ttk.Entry(filter_frame, width=9)
        self.ent_search_name.grid(row=0, column=5, padx=5, pady=5)

        # 조회 버튼 크기 강화
        btn_search = tk.Button(
            filter_frame, text="🔍 조회하기", bg="#3B82F6", fg="white", font=("맑은 고딕", 11, "bold"),
            bd=0, activebackground="#2563EB", cursor="hand2", padx=px(12), height=1
        )
        btn_search.grid(row=0, column=6, padx=px(10), pady=5)
        btn_search.config(command=self.load_logs_filtered)

        # 로그 트리뷰 목록
        list_container = tk.Frame(self.tab_logs)
        list_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        scroll = ttk.Scrollbar(list_container)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.tree_logs = ttk.Treeview(
            list_container, columns=("user_id", "name", "log_type", "log_date", "log_time"), show="headings",
            yscrollcommand=scroll.set
        )
        self.tree_logs.heading("user_id", text="학번")
        self.tree_logs.heading("name", text="이름")
        self.tree_logs.heading("log_type", text="구분")
        self.tree_logs.heading("log_date", text="날짜")
        self.tree_logs.heading("log_time", text="시간")
        
        self.tree_logs.column("user_id", width=px(100), anchor=tk.CENTER)
        self.tree_logs.column("name", width=px(80), anchor=tk.CENTER)
        self.tree_logs.column("log_type", width=px(70), anchor=tk.CENTER)
        self.tree_logs.column("log_date", width=px(110), anchor=tk.CENTER)
        self.tree_logs.column("log_time", width=px(110), anchor=tk.CENTER)
        self.tree_logs.pack(fill=tk.BOTH, expand=True)
        scroll.config(command=self.tree_logs.yview)

        self.load_logs()

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
        for item in self.tree_logs.get_children():
            self.tree_logs.delete(item)

        start_date = self.ent_start_date.get().strip()
        end_date = self.ent_end_date.get().strip()
        search_name = self.ent_search_name.get().strip()

        # 값이 누락되었을 시 기본값 보강
        if not start_date: start_date = "1970-01-01"
        if not end_date: end_date = "2999-12-31"

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        if search_name:
            query = """
                SELECT user_id, name, log_type, log_date, log_time 
                FROM attendance_logs 
                WHERE name LIKE ? AND log_date BETWEEN ? AND ? 
                ORDER BY id DESC
            """
            cursor.execute(query, (f"%{search_name}%", start_date, end_date))
        else:
            query = """
                SELECT user_id, name, log_type, log_date, log_time 
                FROM attendance_logs 
                WHERE log_date BETWEEN ? AND ? 
                ORDER BY id DESC
            """
            cursor.execute(query, (start_date, end_date))

        for r in cursor.fetchall():
            display_type = "출근" if r[2] == 'CHECK_IN' else "퇴근"
            self.tree_logs.insert("", tk.END, values=(r[0], r[1], display_type, r[3], r[4]))
        conn.close()

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
            cursor.execute("DELETE FROM users WHERE user_id = ? OR user_id = ?", (str(student_id), alt_id))
            cursor.execute("DELETE FROM attendance_logs WHERE user_id = ? OR user_id = ?", (str(student_id), alt_id))
            
            conn.commit()
            conn.close()

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

            messagebox.showinfo("성공", "전체 데이터가 삭제되었습니다.")
            self.load_students()
            self.load_logs()
            self.parent.reload_users()

    def register_student(self):
        s_id = self.add_id.get().strip()
        pwd = self.add_pwd.get().strip()
        name = self.add_name.get().strip()
        major = self.add_major.get().strip()

        if not (s_id and pwd and name and major):
            messagebox.showwarning("입력 미달", "모든 정보를 입력하세요.")
            return

        if not self.parent.cached_faces:
            messagebox.showerror("얼굴 인식 실패", "카메라 영역에 등록할 학생의 얼굴이 감지되지 않았습니다.")
            return

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE user_id = ?", (s_id,))
        if cursor.fetchone():
            messagebox.showerror("오류", "이미 가입된 학번 ID입니다.")
            conn.close()
            return

        emb = self.parent.cached_faces[0].embedding.tobytes()
        cursor.execute("""
            INSERT INTO users (user_id, password, name, major, role, embedding, penalty)
            VALUES (?, ?, ?, ?, ?, ?, 0)
        """, (s_id, pwd, name, major, 'student', emb))
        conn.commit()
        conn.close()

        messagebox.showinfo("성공", f"[{name}] 학생이 성공적으로 등록되었습니다.")
        self.add_id.delete(0, tk.END)
        self.add_pwd.delete(0, tk.END)
        self.add_name.delete(0, tk.END)

        self.load_students()
        self.parent.reload_users()
        self.reload_roster_combo()  # 방금 등록한 학생은 명단 목록에서 빠지도록


# ==========================================
# 3. 메인 어플리케이션
# ==========================================
class AttendanceApp:
    def __init__(self, window):
        self.window = window
        self.window.title("창의공간 얼굴인식 출석체크 시스템")
        self.window.configure(bg="#F1F5F9") # 깔끔한 slate 연회색 배경

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
        style.configure('TNotebook.Tab', background='#E2E8F0', foreground='#475569', padding=[px(15), px(6)], font=('맑은 고딕', 10, 'bold'))
        style.map('TNotebook.Tab', background=[('selected', '#ffffff')], foreground=[('selected', '#1E293B')])
        
        # Treeview 스타일링
        style.configure('Treeview', background='#ffffff', fieldbackground='#ffffff', rowheight=px(28), font=('맑은 고딕', 9))
        style.configure('Treeview.Heading', background='#E2E8F0', foreground='#1E293B', font=('맑은 고딕', 10, 'bold'))
        style.map('Treeview', background=[('selected', '#3B82F6')], foreground=[('selected', '#ffffff')])
        
        # LabelFrame 스타일링
        style.configure('TLabelframe', background='#ffffff', bordercolor='#CBD5E1', borderwidth=1)
        style.configure('TLabelframe.Label', background='#ffffff', foreground='#1E293B', font=('맑은 고딕', 10, 'bold'))

        # Ttk Button 스타일링 크기 및 패딩 조절로 해상도 대폭 업그레이드
        style.configure('TButton', font=('맑은 고딕', 11, 'bold'), padding=px(8))

        # DB 초기화
        init_db()

        # 창공시스템이 항상 최신 출결 데이터를 읽어갈 수 있도록 시작할 때도 한 번 내보낸다
        export_attendance_json()

        # InsightFace 초기화
        self.app = FaceAnalysis(
            name="buffalo_l",
            allowed_modules=['detection', 'recognition'],
            providers=["CPUExecutionProvider"]
        )
        self.app.prepare(ctx_id=0, det_size=(256, 256))
        self.enrolled_users = load_users()
        self.embed_matrix = None
        self._rebuild_embedding_matrix()

        # 창공시스템 명단(roster.json) — 얼굴 인식 결과에 표시할 이름을 여기서 우선 찾는다
        self.roster_lookup = {}
        self.apply_roster(load_roster())

        # 상태 제어 필드
        self.admin_logged_in = False
        self.admin_email = ""
        self.current_view_state = None  # "login", "student", "admin"
        self.current_student_id = None
        self.last_face_time = 0.0
        self.admin_login_requested = False  # 학생 화면에서 [관리자]를 눌러 로그인 화면으로 넘어온 상태
        self.camera_on = True

        # UI 레이아웃 구성
        self.create_widgets()

        # 기본 화면으로 전환
        self.reset_to_default_view()

        # 웹캠 기동
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.latest_frame = None
        self.cached_faces = []
        self.is_running = True

        # 비디오 리프레시 및 추론 백그라운드
        self.ai_thread = threading.Thread(target=self.ai_worker, daemon=True)
        self.ai_thread.start()

        self.update_video()
        self._update_clock()

    def create_widgets(self):
        # 상단 헤더 배너 (Modern Dark Slate) — 좌측 타이틀 / 우측 실시간 시계
        title_frame = tk.Frame(self.window, bg="#0F172A", height=px(84))
        title_frame.pack(fill=tk.X, side=tk.TOP)
        title_frame.pack_propagate(False)

        header_inner = tk.Frame(title_frame, bg="#0F172A")
        header_inner.pack(fill=tk.BOTH, expand=True, padx=px(28))

        title_box = tk.Frame(header_inner, bg="#0F172A")
        title_box.pack(side=tk.LEFT, fill=tk.Y)

        lbl_title = tk.Label(title_box, text="창의공간 얼굴인식 출석체크 시스템",
                             bg="#0F172A", fg="white", font=("맑은 고딕", 21, "bold"))
        lbl_title.pack(anchor=tk.W, pady=(px(17), 0))

        lbl_subtitle = tk.Label(title_box, text="동서대학교 창의공간 · Face Recognition Attendance",
                                bg="#0F172A", fg="#64748B", font=("맑은 고딕", 9))
        lbl_subtitle.pack(anchor=tk.W)

        self.lbl_clock = tk.Label(header_inner, text="", bg="#0F172A", fg="#CBD5E1",
                                  font=("맑은 고딕", 15, "bold"))
        self.lbl_clock.pack(side=tk.RIGHT)

        # 메인 콘텐츠 컨테이너
        content_frame = tk.Frame(self.window, bg="#F1F5F9")
        content_frame.pack(fill=tk.BOTH, expand=True, padx=px(20), pady=px(20))

        # 좌측: 카메라 패널 (상단 상태바 + 영상 영역)
        self.camera_panel = tk.Frame(content_frame, bg="#0F172A",
                                     highlightthickness=1, highlightbackground="#CBD5E1")
        self.camera_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        cam_bar = tk.Frame(self.camera_panel, bg="#1E293B", height=px(48))
        cam_bar.pack(fill=tk.X, side=tk.TOP)
        cam_bar.pack_propagate(False)

        self.lbl_cam_status = tk.Label(cam_bar, text="● LIVE", bg="#1E293B", fg="#10B981",
                                       font=("맑은 고딕", 11, "bold"))
        self.lbl_cam_status.pack(side=tk.LEFT, padx=px(16))

        self.btn_camera_toggle = tk.Button(
            cam_bar, text="📷 화면 끄기", bg="#334155", fg="white", bd=0,
            activebackground="#475569", activeforeground="white",
            font=("맑은 고딕", 10, "bold"), padx=px(16), pady=px(5), cursor="hand2",
            command=self.toggle_camera
        )
        self.btn_camera_toggle.pack(side=tk.RIGHT, padx=px(14))

        # 영상 영역 — 이미지 크기 계산은 상태바를 뺀 이 영역 기준으로 한다
        self.video_area = tk.Frame(self.camera_panel, bg="#0F172A")
        self.video_area.pack(fill=tk.BOTH, expand=True)

        self.video_label = tk.Label(self.video_area, bg="#0F172A")
        self.video_label.pack(padx=px(10), pady=px(10), fill=tk.BOTH, expand=True)

        # 우측: 가변 상태 패널 컨테이너
        self.right_container = tk.Frame(content_frame, width=px(RIGHT_PANEL_WIDTH), bg="#F1F5F9")
        self.right_container.pack(side=tk.RIGHT, fill=tk.Y, expand=False, padx=(px(20), 0))
        self.right_container.pack_propagate(False)

    def ai_worker(self):
        while self.is_running:
            if self.camera_on and self.latest_frame is not None:
                # 관리자 탭에서도 학생 신규 등록을 위해 인물 스캔 연산은 항상 유지
                frame_to_process = self.latest_frame.copy()
                faces = self.app.get(frame_to_process)
                self.cached_faces = faces
            time.sleep(0.04)

    def update_video(self):
        # 화면을 꺼둔 동안은 프레임을 읽지도, 얼굴을 인식하지도 않는다
        ret = False
        if self.camera_on:
            ret, frame = self.cap.read()

        if ret:
            frame = cv2.flip(frame, 1)
            self.latest_frame = frame
            display = frame.copy()

            best_user = None
            best_sim = -1.0
            # 사각형은 cv2로 그대로 그리고, 글자(한글)는 아래에서 PIL로 한 번에 그린다.
            # cv2.putText는 Hershey 폰트만 지원해서 한글을 못 그려 '?'로 깨져 보이는 문제가 있었다.
            labels_to_draw = []  # (x, y, text, RGB색상)

            # 얼굴 바운딩 박스 렌더링 (등록된 전원과의 유사도를 행렬곱으로 한번에 계산)
            if self.cached_faces:
                face_embs = np.asarray([f.embedding for f in self.cached_faces], dtype=np.float32)
                face_norms = np.linalg.norm(face_embs, axis=1, keepdims=True)
                face_norms[face_norms == 0] = 1.0
                face_embs_normed = face_embs / face_norms

                has_users = self.embed_matrix is not None and len(self.embed_matrix) > 0
                if has_users:
                    sims = face_embs_normed @ self.embed_matrix.T  # (num_faces, num_users)
                    match_idx = np.argmax(sims, axis=1)
                    match_sims = sims[np.arange(len(self.cached_faces)), match_idx]

                for i, face in enumerate(self.cached_faces):
                    bbox = face.bbox.astype(int)
                    sim = float(match_sims[i]) if has_users else -1.0
                    user = self.enrolled_users[match_idx[i]] if has_users else None

                    # 테두리 및 텍스트 렌더링을 깔끔하게 개선
                    if sim >= THRESHOLD and user:
                        # 이름은 창공시스템 명단(roster.json)을 우선 쓴다 — 새로고침하면 바로 반영된다.
                        display_name = self._resolve_display_name(user)
                        label = f"{display_name} ({sim:.2f})"
                        color_bgr = (16, 185, 129)  # Neon Green (#10B981)
                    else:
                        label = "UNKNOWN"
                        color_bgr = (239, 68, 68)  # Rose Red (#EF4444)

                    cv2.rectangle(display, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color_bgr, 2)
                    label_x, label_y = int(bbox[0]), max(0, int(bbox[1]) - 26)
                    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
                    labels_to_draw.append((label_x, label_y, label, color_rgb))

                    if sim > best_sim:
                        best_sim = sim
                        best_user = user

            # 관리자가 로그인하지 않았고, 학생 화면에서 [관리자]를 눌러 로그인 중도 아닐 때만
            # 얼굴 인식 결과로 화면을 자동으로 바꾼다 (안 그러면 로그인 화면이 자꾸 밀려난다)
            if not self.admin_logged_in and not self.admin_login_requested:
                if best_sim >= THRESHOLD and best_user:
                    self.last_face_time = time.time()
                    # 새로운 사용자를 보았을 때 화면 카드 교환
                    if self.current_student_id != best_user["user_id"]:
                        self.current_student_id = best_user["user_id"]
                        # 표시용 이름은 최신 창공시스템 명단을 우선한다
                        shown_user = dict(best_user)
                        shown_user["name"] = self._resolve_display_name(best_user)
                        self.show_student_view(shown_user)
                else:
                    # 감지된 얼굴이 없으면 4초 카운트 다운 후 기본 관리자 로그인 카드로 복귀
                    if self.current_view_state == "student":
                        if time.time() - self.last_face_time > 4.0:
                            self.reset_to_default_view()

            cv2image = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)

            # 얼굴마다 이름을 한 번씩만, 한글이 제대로 보이게 PIL로 그린다
            if labels_to_draw:
                draw = ImageDraw.Draw(img)
                font = get_korean_font(18)
                for label_x, label_y, text, color_rgb in labels_to_draw:
                    draw.text((label_x + 1, label_y + 1), text, font=font, fill=(0, 0, 0))  # 가독성용 그림자
                    draw.text((label_x, label_y), text, font=font, fill=color_rgb)

            # 영상 영역에 비율을 유지하며 카메라 화면을 축소/확대한다 (화면 밖으로 안 나가게)
            panel_w = self.video_area.winfo_width()
            panel_h = self.video_area.winfo_height()
            if panel_w > 20 and panel_h > 20:
                img = self._fit_image_to_panel(img, panel_w - 20, panel_h - 20)

            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        if self.is_running:
            self.window.after(20, self.update_video)

    def toggle_camera(self):
        """카메라 화면 표시를 켜고 끈다. 꺼져 있는 동안은 얼굴 인식도 같이 멈춘다."""
        self.camera_on = not self.camera_on

        if self.camera_on:
            self.lbl_cam_status.config(text="● LIVE", fg="#10B981")
            self.btn_camera_toggle.config(text="📷 화면 끄기")
            self.video_label.config(text="")
            return

        self.latest_frame = None
        self.cached_faces = []  # 꺼진 동안 예전 얼굴로 학생이 등록되지 않게 비운다
        self.current_student_id = None

        self.lbl_cam_status.config(text="● OFF", fg="#EF4444")
        self.btn_camera_toggle.config(text="📷 화면 켜기")
        self.video_label.imgtk = None
        self.video_label.config(
            image="", text="📷\n\n카메라 화면이 꺼져 있습니다\n[화면 켜기]를 누르면 다시 표시됩니다",
            fg="#64748B", font=("맑은 고딕", 14, "bold")
        )

        # 학생 카드가 떠 있었다면 기본 화면으로 (관리자 작업 중이면 그대로 둔다)
        if not self.admin_logged_in and self.current_view_state == "student":
            self.reset_to_default_view()

    def _update_clock(self):
        if not self.is_running:
            return
        self.lbl_clock.config(text=datetime.now().strftime("%Y-%m-%d  %H:%M:%S"))
        self.window.after(1000, self._update_clock)

    @staticmethod
    def _fit_image_to_panel(img, max_w, max_h):
        """가로세로 비율을 유지하면서 주어진 영역 안에 들어오도록 이미지를 맞춘다."""
        if max_w <= 0 or max_h <= 0:
            return img
        w, h = img.size
        if w == 0 or h == 0:
            return img
        scale = min(max_w / w, max_h / h)
        scale = max(scale, 0.1)  # 극단적으로 작아지는 것만 방지
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        return img.resize(new_size, Image.LANCZOS)

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

    def show_student_view(self, student_info):
        self.clear_right_container()
        self.current_view_state = "student"

        frame = StudentInfoFrame(self, student_info)
        frame.pack(fill=tk.BOTH, expand=True)

    def show_admin_login(self):
        """학생 얼굴 인식 화면에서 [관리자]를 눌렀을 때 — 로그인 화면으로 바로 전환한다."""
        if self.admin_logged_in:
            self.set_admin_logged_in(True, self.admin_email)
            return

        self.admin_login_requested = True
        self.clear_right_container()
        self.current_view_state = "login"

        frame = AdminLoginFrame(self)
        frame.pack(fill=tk.BOTH, expand=True)

    def reset_to_default_view(self):
        self.clear_right_container()
        self.current_student_id = None
        self.current_view_state = "login"
        self.admin_login_requested = False

        frame = AdminLoginFrame(self)
        frame.pack(fill=tk.BOTH, expand=True)

    def set_admin_logged_in(self, logged_in, admin_email=""):
        self.admin_logged_in = logged_in
        self.admin_email = admin_email
        self.admin_login_requested = False
        self.clear_right_container()

        if logged_in:
            self.current_view_state = "admin"
            frame = AdminDashboardFrame(self, admin_email)
            frame.pack(fill=tk.BOTH, expand=True)
        else:
            self.reset_to_default_view()

    def clear_right_container(self):
        for widget in self.right_container.winfo_children():
            widget.destroy()

    def on_close(self):
        self.is_running = False
        self.cap.release()
        self.window.destroy()


# ==========================================
# 4. 앱 실행
# ==========================================
if __name__ == "__main__":
    root = tk.Tk()
    app = AttendanceApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()