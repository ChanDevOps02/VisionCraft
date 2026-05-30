@echo off
setlocal

cd /d "%~dp0"

echo [VisionCraft] Windows setup
echo.

call :find_python
if errorlevel 1 (
  echo Python 3.11 was not found.
  echo.
  echo Install Python 3.11, then open a new terminal and run this file again.
  echo Recommended command:
  echo winget install --id Python.Python.3.11 -e --scope user --accept-package-agreements --accept-source-agreements
  echo.
  pause
  exit /b 1
)

echo Using Python command: %PYTHON_CMD%

if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  %PYTHON_CMD% -m venv .venv
  if errorlevel 1 (
    echo Failed to create virtual environment.
    pause
    exit /b 1
  )
) else (
  echo Existing .venv found.
)

echo Upgrading pip...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 (
  echo Failed to upgrade pip.
  pause
  exit /b 1
)

echo Installing requirements...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
  echo Failed to install requirements.
  pause
  exit /b 1
)

if not exist ".mplconfig" mkdir ".mplconfig"

echo.
echo Setup complete.
echo Run run_visioncraft.bat to start the application.
pause
exit /b 0

:find_python
if exist "%LocalAppData%\Programs\Python\Python311\python.exe" (
  "%LocalAppData%\Programs\Python\Python311\python.exe" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1
  if not errorlevel 1 (
    set "PYTHON_CMD="%LocalAppData%\Programs\Python\Python311\python.exe""
    exit /b 0
  )
)

py -3.11 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1
if not errorlevel 1 (
  set "PYTHON_CMD=py -3.11"
  exit /b 0
)

py -3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1
if not errorlevel 1 (
  set "PYTHON_CMD=py -3"
  exit /b 0
)

python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1
if not errorlevel 1 (
  set "PYTHON_CMD=python"
  exit /b 0
)

python3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1
if not errorlevel 1 (
  set "PYTHON_CMD=python3"
  exit /b 0
)

exit /b 1
