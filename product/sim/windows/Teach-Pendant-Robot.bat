@echo off
rem Robotonic Teach Pendant with the real SO-101 on COM4. Double-click to start.
rem Put the arm in its folded rest pose first: closing the window brings it back there slowly.
title Robotonic Teach Pendant (real arm, COM4)
cd /d C:\dev\cobot-teaching\product\sim
call C:\dev\venvs\teach\Scripts\activate.bat
python -m so101_sim view --port COM4 %*
echo.
echo Teach Pendant closed. Press any key to close this window.
pause >nul
