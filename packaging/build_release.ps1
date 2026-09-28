# 윈도우 배포판 zip 만들기 — exe 빌드 후 설치 파일을 넣어 압축한다.
#   powershell -ExecutionPolicy Bypass -File packaging\build_release.ps1
# 결과: dist\TheEclipseAttendance-windows.zip
# (GitHub에 v* 태그를 올리면 .github/workflows/release-windows.yml 이 이 스크립트로 자동 빌드한다)
param([string]$OutDir = "")
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
# 경로가 길면(260자 제한) 빌드가 실패해서 작업 폴더는 짧은 임시 경로를 쓴다
$Work = Join-Path $env:TEMP "teb-work"
$Dist = if ($OutDir) { $OutDir } else { Join-Path $Root "dist" }
$AppOut = Join-Path $Dist "TheEclipseAttendance"
$Zip = Join-Path $Dist "TheEclipseAttendance-windows.zip"

python -m PyInstaller (Join-Path $PSScriptRoot "TheEclipseAttendance.spec") --noconfirm --distpath $Dist --workpath $Work
if ($LASTEXITCODE -ne 0) { throw "PyInstaller 빌드 실패" }

# 설치 파일 복사 (.ps1 은 한글이 깨지지 않도록 UTF-8 BOM 그대로, .bat 은 ASCII)
Copy-Item (Join-Path $PSScriptRoot "release\*") $AppOut -Force
foreach ($name in @("install", "uninstall")) {
    $bat = "@echo off`r`npowershell -NoProfile -ExecutionPolicy Bypass -File `"%~dp0$name.ps1`"`r`npause`r`n"
    [IO.File]::WriteAllText((Join-Path $AppOut "$name.bat"), $bat, [Text.Encoding]::ASCII)
}

if (Test-Path $Zip) { Remove-Item $Zip }
# 압축 풀면 TheEclipseAttendance 폴더 하나가 나오도록
Compress-Archive -Path $AppOut -DestinationPath $Zip -CompressionLevel Optimal
Write-Host ("완료: {0} ({1:N0} MB)" -f $Zip, ((Get-Item $Zip).Length / 1MB))
