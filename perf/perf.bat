@echo off
rem ============================================================================
rem  Lions Racing - Performance Lab (LR26 lap time simulation)
rem
rem  Double-click this file (or "Performance Lab.bat" in the project folder, or
rem  the desktop shortcut). It opens the Performance Lab window, from which the
rem  Lab is started in the web browser. No WSL, ROS or Isaac Sim needed.
rem
rem  First time on a PC it sets everything up, asking before each step:
rem    1. Python 3.12 for your user (no admin rights): winget, else python.org
rem    2. the Python packages numpy, scipy, pyyaml (pip, for your user), and casadi (optional)
rem  Needs internet only for these two steps.
rem
rem  perf.bat --console   runs the Lab in this window instead (no launcher window)
rem
rem  The car comes from lapsim\LR26_LapSim (lr26_params.m and competition_*.m);
rem  setups are saved in lapsim\configs, uploaded logs in perf\uploads.
rem ============================================================================
setlocal EnableExtensions
title Lions Racing - Performance Lab
cd /d "%~dp0"
set "MODE=window"
if /i "%~1"=="--console" set "MODE=console"
set "PY_VERSION=3.12.10"

call :find_python
if not defined PY call :install_python
if not defined PY goto no_python
echo Python: %PY%

"%PY%" -c "import numpy, scipy, yaml" >nul 2>&1
if errorlevel 1 call :install_packages
"%PY%" -c "import numpy, scipy, yaml" >nul 2>&1
if errorlevel 1 goto no_packages

if "%MODE%"=="console" goto console
"%PY%" -c "import tkinter" >nul 2>&1
if errorlevel 1 (
    echo This Python has no Tk ^(window toolkit^): starting the Lab in this window instead.
    goto console
)
set "PYW=%PY:python.exe=pythonw.exe%"
if not exist "%PYW%" set "PYW=%PY%"
start "" "%PYW%" "%~dp0launcher\perf_launcher.pyw"
exit /b 0

:console
"%PY%" "%~dp0app.py"
if errorlevel 1 pause
exit /b 0

rem ---------------------------------------------------------------------------
:find_python
rem A Python 3.9+ as a full path to python.exe (the Microsoft Store stub does not count).
set "PY="
for /f "delims=" %%i in ('py -3 -c "import sys; print(sys.executable) if sys.version_info >= (3, 9) else None" 2^>nul') do set "PY=%%i"
if defined PY if exist "%PY%" exit /b 0
set "PY="
for /f "delims=" %%i in ('python -c "import sys; print(sys.executable) if sys.version_info >= (3, 9) else None" 2^>nul') do set "PY=%%i"
if defined PY if exist "%PY%" exit /b 0
set "PY="
for %%v in (314 313 312 311 310 39) do (
    if not defined PY if exist "%LOCALAPPDATA%\Programs\Python\Python%%v\python.exe" set "PY=%LOCALAPPDATA%\Programs\Python\Python%%v\python.exe"
)
exit /b 0

:install_python
echo.
echo  The Performance Lab needs Python ^(free, from python.org^). It is not installed on this PC.
echo  It can be installed now for your Windows user only ^(no administrator rights, about 2 minutes^).
echo.
choice /c YN /m "Install Python %PY_VERSION% now"
if errorlevel 2 exit /b 0
where winget >nul 2>&1
if errorlevel 1 goto install_python_download
echo Installing with winget ...
winget install -e --id Python.Python.3.12 --scope user --silent --accept-package-agreements --accept-source-agreements
call :find_python
if defined PY exit /b 0
:install_python_download
echo Downloading the installer from python.org ...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference = 'Stop'; [Net.ServicePointManager]::SecurityProtocol = 'Tls12'; $f = Join-Path $env:TEMP 'python-%PY_VERSION%-amd64.exe'; Invoke-WebRequest -UseBasicParsing -Uri 'https://www.python.org/ftp/python/%PY_VERSION%/python-%PY_VERSION%-amd64.exe' -OutFile $f; Start-Process -Wait -FilePath $f -ArgumentList '/quiet','InstallAllUsers=0','PrependPath=1','Include_launcher=1','Include_tcltk=1','Include_test=0'"
call :find_python
exit /b 0

:install_packages
echo.
echo  The Performance Lab needs the Python packages numpy, scipy and pyyaml.
choice /c YN /m "Install them now (for your user, a few minutes)"
if errorlevel 2 exit /b 0
"%PY%" -m pip install --user --disable-pip-version-check numpy scipy pyyaml
rem optional: CasADi for the optimal-control lap time models (Methods page); the Lab works without it
"%PY%" -m pip install --user --disable-pip-version-check casadi
exit /b 0

:no_python
echo.
echo  Python could not be installed automatically.
echo  Install it by hand: https://www.python.org/downloads/  (tick "Add python.exe to PATH"),
echo  then double-click this file again.
echo.
pause
exit /b 1

:no_packages
echo.
echo  numpy, scipy and pyyaml are still missing. Check the internet connection and run this file again,
echo  or install them by hand:  "%PY%" -m pip install --user numpy scipy pyyaml
echo.
pause
exit /b 1
