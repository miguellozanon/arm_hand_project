@echo off
title Arm Hand Robot - stop
REM Stops the running job (only what this launcher started) and the launcher itself.
curl.exe -s --max-time 30 -X POST -H "X-AHR: 1" http://127.0.0.1:8790/api/shutdown >nul 2>&1
if errorlevel 1 (echo The launcher was not running.) else (echo Launcher and its job stopped.)
timeout /t 3 >nul
