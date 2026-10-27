; 창의공간 얼굴인식 출석체크 — 윈도우 설치 프로그램 (Inno Setup 6)
;   packaging\build_release.ps1 이 exe를 빌드한 뒤 이 스크립트로 설치 파일을 만든다.
;   직접: ISCC.exe /DMyAppVersion=2026.10.27 packaging\installer\TheEclipseAttendance.iss
; 결과: dist\TheEclipseAttendance-Setup.exe
;
; 학생 DB(faces.db)·출퇴근 사진·로그·설정 파일은 설치 파일에 들어 있지 않다.
; 그래서 새 버전을 위에 설치(업데이트)해도 지워지지 않고, 제거(uninstall)해도 남는다.

#define MyAppName "창의공간 얼굴인식 출석체크"
#define MyAppExe "TheEclipseAttendance.exe"
#ifndef MyAppVersion
  #define MyAppVersion "1.0.0"
#endif
#ifndef SourceDir
  #define SourceDir "..\..\dist\TheEclipseAttendance"
#endif

[Setup]
AppId={{6B0F2D7A-3C54-4E1B-9A6D-7E2B5C8F1A40}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher=The Eclipse
; 프로그램이 faces.db 같은 데이터를 exe 옆에 쓰므로 Program Files가 아니라 C:\ 아래에 설치한다
DefaultDirName=C:\TheEclipseAttendance
DisableProgramGroupPage=yes
PrivilegesRequired=admin
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\..\dist
OutputBaseFilename=TheEclipseAttendance-Setup
Compression=lzma2/normal
SolidCompression=yes
WizardStyle=modern
UninstallDisplayIcon={app}\{#MyAppExe}
; 켜져 있는 프로그램은 아래 [Code]에서 직접 끈다 (카메라를 쥐고 있어서 파일 교체가 안 되므로)
CloseApplications=no

[Languages]
Name: "korean"; MessagesFile: "compiler:Languages\Korean.isl"

[Tasks]
Name: "desktopicon"; Description: "바탕화면에 바로가기 만들기"; GroupDescription: "추가 작업:"
Name: "autostart"; Description: "매일 06:00에 자동 실행 + PC를 켜고 로그인하면 자동 실행 (00:00에 자동 종료)"; GroupDescription: "추가 작업:"

[Dirs]
; 프로그램이 이 폴더 안에 DB·로그·설정을 쓰므로 일반 사용자 계정도 쓸 수 있게 한다
Name: "{app}"; Permissions: users-modify

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs; Excludes: "install.ps1,install.bat,uninstall.ps1,uninstall.bat,README.txt"
Source: "register_task.ps1"; DestDir: "{app}\tools"; Flags: ignoreversion
Source: "unregister_task.ps1"; DestDir: "{app}\tools"; Flags: ignoreversion

[Icons]
; 운영 시간에 켜면 자정에 스스로 종료한다 (--kiosk). 시작프로그램은 00:00~06:00이면 켜지지 않는다 (--scheduled).
Name: "{commondesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExe}"; Parameters: "--kiosk"; WorkingDir: "{app}"; Tasks: desktopicon
Name: "{commonstartup}\{#MyAppName}"; Filename: "{app}\{#MyAppExe}"; Parameters: "--scheduled"; WorkingDir: "{app}"; Tasks: autostart
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExe}"; Parameters: "--kiosk"; WorkingDir: "{app}"
Name: "{autoprograms}\{#MyAppName} 제거"; Filename: "{uninstallexe}"

[Run]
; 얼굴 인식 모델을 미리 받는다 (처음 한 번, 약 300MB). 못 받아도 프로그램을 처음 켤 때 다시 받는다.
; 모델은 사용자 폴더(~/.insightface)에 저장되므로 설치 계정이 아니라 실제 로그인한 사용자로 실행한다.
Filename: "{app}\{#MyAppExe}"; Parameters: "--prepare"; WorkingDir: "{app}"; StatusMsg: "얼굴 인식 모델을 내려받는 중입니다 (처음 한 번, 약 300MB — 몇 분 걸릴 수 있습니다)..."; Flags: waituntilterminated runasoriginaluser
Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\tools\register_task.ps1"" -Exe ""{app}\{#MyAppExe}"" -WorkDir ""{app}"""; StatusMsg: "자동 실행 작업을 등록하는 중..."; Flags: runhidden waituntilterminated runasoriginaluser; Tasks: autostart
Filename: "{app}\{#MyAppExe}"; Parameters: "--kiosk"; WorkingDir: "{app}"; Description: "지금 바로 실행"; Flags: nowait postinstall skipifsilent runasoriginaluser

[UninstallRun]
Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\tools\unregister_task.ps1"""; RunOnceId: "RemoveTask"; Flags: runhidden waituntilterminated

[Code]
procedure StopRunningApp();
var
  ResultCode: Integer;
begin
  // 얼굴 인식 프로세스도 같은 이름이라 한 번에 같이 꺼진다
  Exec(ExpandConstant('{sys}\taskkill.exe'), '/F /IM {#MyAppExe}', '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
  Sleep(500);
end;

function PrepareToInstall(var NeedsRestart: Boolean): String;
begin
  StopRunningApp();
  Result := '';
end;

function InitializeUninstall(): Boolean;
begin
  StopRunningApp();
  Result := True;
end;

function NextButtonClick(CurPageID: Integer): Boolean;
begin
  Result := True;
  if (CurPageID = wpSelectDir) and (Pos('onedrive', LowerCase(WizardDirValue)) > 0) then
    Result := MsgBox('OneDrive 폴더 안에 설치하면 학생 DB가 동기화되면서 충돌할 수 있습니다.' + #13#10 +
                     'C:\TheEclipseAttendance 같은 곳을 권장합니다. 그래도 이 위치에 설치할까요?',
                     mbConfirmation, MB_YESNO) = IDYES;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usPostUninstall then
    MsgBox('프로그램을 제거했습니다.' + #13#10 +
           '학생 DB(faces.db)·출퇴근 사진·로그는 안전하게 남겨 두었습니다: ' + ExpandConstant('{app}'),
           mbInformation, MB_OK);
end;
