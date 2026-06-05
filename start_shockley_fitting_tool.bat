@echo off
cd /d "%~dp0"
python shockley_gui.py
if errorlevel 1 (
  echo.
  echo Shockley fitting tool failed to start.
  pause
)
