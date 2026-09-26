@echo off
setlocal
title Gemini Native v1.0 Preview - Install License
cd /d "%~dp0"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "tools\Install-GeminiLicense.ps1"
set "RESULT=%ERRORLEVEL%"
echo.
if "%RESULT%"=="0" (
  echo License file installed. Reopen Gemini Native.
) else (
  echo License installation failed. Exit code: %RESULT%
)
pause
exit /b %RESULT%
