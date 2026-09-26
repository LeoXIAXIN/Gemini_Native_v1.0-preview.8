@echo off
setlocal
title Gemini Native v1.0 Preview
cd /d "%~dp0"
"runtime-gmr\python.exe" -u "src\controller\native_app.py" --open-browser
set "RESULT=%ERRORLEVEL%"
echo.
if not "%RESULT%"=="0" echo Gemini Native exited with code %RESULT%.
pause
exit /b %RESULT%
