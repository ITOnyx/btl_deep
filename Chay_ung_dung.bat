@echo off
rem Chạy app Windows (cửa sổ riêng). Dùng khi không có shortcut hoặc cần xem lỗi trong console.
chcp 65001 >nul
cd /d "%~dp0"
python app\desktop.py
if errorlevel 1 pause
