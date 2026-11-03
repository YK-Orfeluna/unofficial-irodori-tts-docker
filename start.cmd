@echo off
setlocal
chcp 65001 >nul
pushd "%~dp0" || exit /b 1
if not exist ".venv\Scripts\python.exe" (
    echo Run setup.cmd first, then run start.cmd again.
    popd
    pause
    exit /b 1
)
".venv\Scripts\python.exe" main.py --open-browser %*
set "result=%errorlevel%"
popd
if not "%result%"=="0" pause
exit /b %result%
