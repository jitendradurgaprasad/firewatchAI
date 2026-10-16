@echo off
setlocal
cd /d "%~dp0"

set "FW_PYTHON="
if exist "%~dp0.venv\Scripts\python.exe" set "FW_PYTHON=%~dp0.venv\Scripts\python.exe"
if not defined FW_PYTHON if defined LOCALAPPDATA if exist "%LOCALAPPDATA%\FireWatchAI\venv\Scripts\python.exe" set "FW_PYTHON=%LOCALAPPDATA%\FireWatchAI\venv\Scripts\python.exe"

if defined FW_PYTHON (
    "%FW_PYTHON%" -m streamlit run "%~dp0app.py" --server.headless false --browser.gatherUsageStats false
) else (
    where py >nul 2>nul
    if not errorlevel 1 (
        py -3.11 -m streamlit run "%~dp0app.py" --server.headless false --browser.gatherUsageStats false
    ) else (
        python -m streamlit run "%~dp0app.py" --server.headless false --browser.gatherUsageStats false
    )
)

if errorlevel 1 (
    echo.
    echo FireWatch could not start. Confirm Python, project dependencies, and the local .env configuration.
    pause
)

endlocal
