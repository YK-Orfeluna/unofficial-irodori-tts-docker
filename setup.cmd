@echo off
setlocal
chcp 65001 >nul
pushd "%~dp0" || exit /b 1
"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -ExecutionPolicy Bypass -File "scripts\bootstrap_uv.ps1"
set "result=%errorlevel%"
popd
if not "%result%"=="0" echo Setup failed. Check the message above and run setup.cmd again.
pause
exit /b %result%
