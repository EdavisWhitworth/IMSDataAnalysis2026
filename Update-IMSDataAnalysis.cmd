@echo off
setlocal

set "REPO_DIR=%~dp0"
pushd "%REPO_DIR%" >nul

echo [IMSDataAnalysis2026] Updating repository from Git...
where git >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Git was not found on PATH.
    popd >nul
    pause
    exit /b 1
)

if not exist ".git" (
    echo [ERROR] This folder is not a Git repository.
    popd >nul
    pause
    exit /b 1
)

git pull
if errorlevel 1 (
    echo [ERROR] git pull failed. Resolve Git issues and try again.
    popd >nul
    pause
    exit /b 1
)

echo [IMSDataAnalysis2026] Refreshing environment after update...
call "%REPO_DIR%setup_env.bat"
if errorlevel 1 (
    echo [ERROR] Environment refresh failed after update.
    popd >nul
    pause
    exit /b 1
)

echo [IMSDataAnalysis2026] Update complete.
popd >nul
exit /b 0
