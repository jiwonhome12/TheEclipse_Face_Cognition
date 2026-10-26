@echo off
rem Double-click to copy the Raspberry Pi install files to a USB drive (runs make_usb_bundle.ps1)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0make_usb_bundle.ps1"
pause
