@echo off
rem Robotonic Teach Pendant, twin only (no robot needed). Double-click to start.
title Robotonic Teach Pendant (twin only)
cd /d C:\dev\cobot-teaching\product\sim
call C:\dev\venvs\teach\Scripts\activate.bat
python -m so101_sim view %*
echo.
echo Teach Pendant closed. Press any key to close this window.
pause >nul
