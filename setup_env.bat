@echo off
setlocal

set "REPO_DIR=%~dp0"
pushd "%REPO_DIR%" >nul

echo [IMSDataAnalysis2026] Setting up local Python environment...

set "PY_CMD="
where python >nul 2>&1
if %ERRORLEVEL%==0 (
    set "PY_CMD=python"
) else (
    where py >nul 2>&1
    if %ERRORLEVEL%==0 (
        set "PY_CMD=py -3"
    )
)

if "%PY_CMD%"=="" (
    echo [ERROR] Python was not found on PATH.
    echo Install Python and ensure python.exe or py.exe is available from Command Prompt.
    popd >nul
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo [IMSDataAnalysis2026] Creating .venv...
    call %PY_CMD% -m venv ".venv"
    if errorlevel 1 (
        echo [ERROR] Failed to create .venv.
        popd >nul
        pause
        exit /b 1
    )
) else (
    echo [IMSDataAnalysis2026] Using existing .venv.
)

set "VENV_PY=.venv\Scripts\python.exe"
if not exist "%VENV_PY%" (
    echo [ERROR] Expected interpreter not found at %VENV_PY%.
    popd >nul
    pause
    exit /b 1
)

echo [IMSDataAnalysis2026] Upgrading pip...
"%VENV_PY%" -m pip install --upgrade pip
if errorlevel 1 (
    echo [ERROR] pip upgrade failed.
    popd >nul
    pause
    exit /b 1
)

echo [IMSDataAnalysis2026] Installing required packages...
"%VENV_PY%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Package installation failed.
    popd >nul
    pause
    exit /b 1
)

echo [IMSDataAnalysis2026] Environment ready.
popd >nul
exit /b 0
