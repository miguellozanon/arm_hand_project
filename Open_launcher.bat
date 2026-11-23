@echo off
title Arm Hand Robot - opening the launcher
REM Launcher page: http://127.0.0.1:8790 (Go2 8765, B2 8770, Performance Lab 8090, Airbox 8780).
REM Runs hidden with Isaac Sim's Python (standard library only). Log: C:\arm_hand_robot\logs\launcher\launcher.log
set "PY=C:\isaacsim\kit\python\python.exe"
if not exist "%PY%" set "PY=C:\Python314\python.exe"
if not exist "%PY%" set "PY=python"
if not exist "%~dp0logs\launcher" mkdir "%~dp0logs\launcher"
curl.exe -s --max-time 2 http://127.0.0.1:8790/api/report >nul 2>&1
if not errorlevel 1 goto open
echo Starting the Arm Hand Robot launcher...
powershell -NoProfile -Command "Start-Process -WindowStyle Hidden -FilePath '%PY%' -ArgumentList '-u','%~dp0launcher\ahr_launcher.py' -RedirectStandardOutput '%~dp0logs\launcher\launcher.log' -RedirectStandardError '%~dp0logs\launcher\launcher_err.log'"
for /l %%i in (1,1,30) do (
  timeout /t 1 /nobreak >nul
  curl.exe -s --max-time 1 http://127.0.0.1:8790/api/report >nul 2>&1 && goto open
)
echo.
echo The launcher did not start. See %~dp0logs\launcher\launcher_err.log
pause
exit /b 1
:open
if /i "%~1"=="--no-browser" exit /b 0
start "" http://127.0.0.1:8790
