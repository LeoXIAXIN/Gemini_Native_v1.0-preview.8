@echo off
setlocal
cd /d "%~dp0"
echo Active Windows IPv4 adapters:
powershell.exe -NoProfile -Command "Get-NetIPConfiguration | Where-Object { $_.NetAdapter.Status -eq 'Up' } | ForEach-Object { $c=$_; $c.IPv4Address | ForEach-Object { [pscustomobject]@{Adapter=$c.InterfaceAlias; IPv4=$_.IPAddress; Description=$c.NetAdapter.InterfaceDescription} } } | Format-Table -AutoSize"
echo.
echo Use the Windows Ethernet IPv4, not the mocap server IPv4.
set /p "UNITREE_PC_IP=Enter the Windows Ethernet IPv4 connected to G1 (this PC: 192.168.123.120): "
if not defined UNITREE_PC_IP set "UNITREE_PC_IP=192.168.123.120"
echo.
echo Before the receive-only check, put G1 on a loaded support rig,
echo enter damping/zero-torque mode with the physical remote, then press
echo L2+R2 to enter Develop mode. This script never creates a LowCmd writer.
set /p "UNITREE_READY=Type READY after the robot is safely in Develop mode: "
if /i not "%UNITREE_READY%"=="READY" (
  echo Cancelled. No DDS probe was started.
  exit /b 2
)
"runtime-hgpt\python.exe" "src\adapters\unitree_lowstate_probe_windows.py" --interface-address "%UNITREE_PC_IP%" --duration 60 --report "storage\reports\unitree_lowstate_windows_report.json" --trace "storage\logs\cyclonedds_windows.log"
set "RESULT=%ERRORLEVEL%"
echo.
if "%RESULT%"=="0" (
  echo LowState acceptance passed. The web UI can now authorize a non-debug real task.
) else (
  echo LowState acceptance failed. Keep the web UI in read-only debug mode.
)
pause
exit /b %RESULT%
