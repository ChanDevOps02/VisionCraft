@echo off
setlocal

cd /d "%~dp0"
set "MPLCONFIGDIR=%~dp0.mplconfig"
if not exist "%MPLCONFIGDIR%" mkdir "%MPLCONFIGDIR%"

if not exist ".venv\Scripts\python.exe" (
  echo Virtual environment Python not found: .venv\Scripts\python.exe
  echo Run setup first.
  pause
  exit /b 1
)

".venv\Scripts\python.exe" -u app.py
pause
