@echo off
rem Double-click to run uninstall.ps1 (PowerShell script in the same folder)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0uninstall.ps1"
pause
