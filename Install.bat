@echo off
title Arm Hand Robot - install
echo ============================================================
echo   Arm Hand Robot - install
echo ============================================================
echo.
echo 1/3  Task package into Isaac Sim's Python and the local LLM (Ollama)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\windows\setup_windows.ps1"
if errorlevel 1 (echo. & echo ERROR in the setup. Copy the text above and give it to Claude. & pause & exit /b 1)
echo.
echo 2/3  Shortcut "Arm Hand Robot" in %~dp0
powershell -NoProfile -ExecutionPolicy Bypass -Command "$s=(New-Object -ComObject WScript.Shell).CreateShortcut('%~dp0Arm Hand Robot.lnk'); $s.TargetPath='%~dp0Open_launcher.bat'; $s.WorkingDirectory='%~dp0'; $s.WindowStyle=7; $s.IconLocation='%SystemRoot%\System32\shell32.dll,22'; $s.Description='Arm Hand Robot launcher (http://127.0.0.1:8790)'; $s.Save()"
echo.
echo 3/3  Start the launcher automatically when you log in?
echo      (lets Claude open http://127.0.0.1:8790 in the browser without you starting it first)
choice /C YN /M "Start at login"
if errorlevel 2 goto done
powershell -NoProfile -ExecutionPolicy Bypass -Command "$p=[Environment]::GetFolderPath('Startup'); $s=(New-Object -ComObject WScript.Shell).CreateShortcut((Join-Path $p 'Arm Hand Robot launcher.lnk')); $s.TargetPath='%~dp0Open_launcher.bat'; $s.Arguments='--no-browser'; $s.WorkingDirectory='%~dp0'; $s.WindowStyle=7; $s.Save()"
echo Added to Startup (remove it with Win+R, shell:startup).
:done
echo.
echo Done. Open the launcher with "Arm Hand Robot" in %~dp0 or Open_launcher.bat.
pause
