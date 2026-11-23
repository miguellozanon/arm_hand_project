@echo off
rem Runs list_tasks.ps1 without changing the PowerShell execution policy.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0list_tasks.ps1" %*
