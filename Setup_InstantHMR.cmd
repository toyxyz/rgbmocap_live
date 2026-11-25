@echo off
setlocal DisableDelayedExpansion
cd /d "%~dp0"
if not exist "runtime\python.exe" (
  echo Run install.cmd first.
  pause
  exit /b 1
)
"runtime\python.exe" -I -X utf8 -B tools\setup_instanthmr.py %*
set "rgb_instant_result=%errorlevel%"
if not defined RGB_MOCAP_NO_PAUSE pause
exit /b %rgb_instant_result%
