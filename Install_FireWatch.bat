@echo off
setlocal
pushd "%~dp0"

if not exist "requirements.txt" (
    echo ERROR: requirements.txt was not found in the project folder.
    goto :failed
)

where py >nul 2>nul
if errorlevel 1 (
    echo ERROR: Python Launcher was not found. Install Python 3.11 and enable the Python Launcher, then run this file again.
    goto :failed
)

py -3.11 --version >nul 2>nul
if errorlevel 1 (
    echo ERROR: Python 3.11 was not found. Install Python 3.11, then run this file again.
    goto :failed
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating the project virtual environment...
    py -3.11 -m venv .venv
    if errorlevel 1 goto :failed
)

call ".venv\Scripts\activate.bat"
if errorlevel 1 goto :failed

echo Upgrading pip in the project virtual environment...
python -m pip install --upgrade pip
if errorlevel 1 goto :failed

echo Installing all project dependencies from requirements.txt...
python -m pip install -r "requirements.txt"
if errorlevel 1 goto :failed

echo.
echo FireWatch dependencies installed successfully in:
echo %~dp0.venv
echo Double-click Start_FireWatch.bat to launch the interface.
popd
pause
exit /b 0

:failed
echo.
echo Installation did not complete. Review the message above, fix the issue, then run Install_FireWatch.bat again.
popd
pause
exit /b 1
