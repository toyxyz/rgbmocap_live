@echo off
setlocal DisableDelayedExpansion
cd /d "%~dp0"
if not exist "runtime\python.exe" (
  echo RGB MoCap Live is not installed. Run install.cmd first.
  exit /b 1
)
"%~dp0runtime\python.exe" -I -X utf8 -u -B "%~dp0launch_live.py" %*
exit /b %errorlevel%
