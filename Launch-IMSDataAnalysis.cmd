@echo off
setlocal

set "REPO_DIR=%~dp0"
pushd "%REPO_DIR%" >nul

set "VENV_PY=.venv\Scripts\python.exe"
if not exist "%VENV_PY%" (
    echo [IMSDataAnalysis2026] Local environment not found. Running setup_env.bat...
    call "%REPO_DIR%setup_env.bat"
    if errorlevel 1 (
        echo [ERROR] Setup failed. Application did not launch.
        popd >nul
        pause
        exit /b 1
    )
)

echo [IMSDataAnalysis2026] Launching application...
"%VENV_PY%" main.py
if errorlevel 1 (
    echo [ERROR] Application exited with an error.
    popd >nul
    pause
    exit /b 1
)

popd >nul
exit /b 0
