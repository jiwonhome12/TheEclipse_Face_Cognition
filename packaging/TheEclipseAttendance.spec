# -*- mode: python ; coding: utf-8 -*-
# 창의공간 얼굴인식 출석체크 — 윈도우 exe 빌드 설정 (PyInstaller)
#   python -m PyInstaller packaging/TheEclipseAttendance.spec
# 결과: dist/TheEclipseAttendance/TheEclipseAttendance.exe (+ _internal 폴더)
#
# InsightFace 얼굴 모델(buffalo_l)은 비상업 연구용 라이선스라 exe에 넣지 않는다.
# 처음 실행할 때 ~/.insightface 로 자동으로 내려받는다.
import os
from PyInstaller.utils.hooks import collect_all

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))

datas = [(os.path.join(ROOT, "face_landmarker.task"), ".")]  # 눈 깜빡임 검사 모델 (Apache 2.0)
binaries = []
hiddenimports = []
for package in ("mediapipe", "insightface"):
    pkg_datas, pkg_binaries, pkg_hidden = collect_all(package)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hidden

# insightface는 exe로 실행되면 _internal/objects 에서 기본 도형 파일을 찾는다 (insightface/data/pickle_object.py)
import insightface
datas.append((os.path.join(os.path.dirname(insightface.__file__), "data", "objects"), "objects"))

a = Analysis(
    [os.path.join(ROOT, "attendance_app.py")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["IPython", "jupyter", "notebook", "pytest", "PyQt5", "PySide6", "torch", "tensorflow"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="TheEclipseAttendance",
    console=False,  # 콘솔 창 없이 실행 (출력은 logs/app.log)
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="TheEclipseAttendance", upx=False)
