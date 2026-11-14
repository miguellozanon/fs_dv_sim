@echo off
rem ============================================================================
rem  Lions Racing - FSG DV simulator launcher
rem
rem  Double-click: starts the launcher in WSL and opens it in the browser
rem  (http://localhost:8070). From there: Isaac Sim / lite / offline demo, the
rem  dashboard, RL training and the Performance tools - no terminals needed.
rem
rem  Which copy of the project it runs:
rem    1. the folder this file is in, if it is the project (fsg-dv-sim\run.bat)
rem    2. a "fsg-dv-sim" folder next to this file (e.g. C:\formula_student\FSG DV Launcher.bat)
rem    3. otherwise ~/fsg-dv-sim inside WSL (installed from Downloads\fsg-dv-sim.zip if missing)
rem  A project on the Windows drive runs straight from there (WSL reads it as /mnt/c/...);
rem  its build files go to ~/.fsg-dv-sim in WSL.
rem
rem  Close this window (or press "Quit launcher" in the page) to stop everything.
rem ============================================================================
setlocal
set "DISTRO=Ubuntu-24.04"
set "PORT=8070"
set "REPO="

title Lions Racing - DV launcher

rem ---- which project copy -----------------------------------------------------
set "HERE=%~dp0"
if "%HERE:~-1%"=="\" set "HERE=%HERE:~0,-1%"
set "REPO_WIN="
if exist "%HERE%\tools\launcher\launcher.py" set "REPO_WIN=%HERE%"
if not defined REPO_WIN if exist "%HERE%\fsg-dv-sim\tools\launcher\launcher.py" set "REPO_WIN=%HERE%\fsg-dv-sim"
cd /d "%USERPROFILE%"

rem Already running? Just open the page.
curl.exe -s -o nul -m 2 http://localhost:%PORT%/api/ping >nul 2>&1
if not errorlevel 1 (
    echo The launcher is already running: opening http://localhost:%PORT%
    start "" http://localhost:%PORT%
    timeout /t 3 >nul
    exit /b 0
)

where wsl.exe >nul 2>&1
if errorlevel 1 (
    echo WSL is not installed. See docs\WINDOWS_WSL.md.
    pause
    exit /b 1
)

if defined REPO_WIN (
    for /f "usebackq delims=" %%P in (`wsl.exe -d %DISTRO% -- wslpath -a "%REPO_WIN%"`) do set "REPO=%%P"
) else (
    set "REPO=~/fsg-dv-sim"
)
if not defined REPO (
    echo Could not reach WSL '%DISTRO%'. Check the name with:  wsl -l   and change DISTRO in this file.
    pause
    exit /b 1
)

if not defined REPO_WIN (
    rem First run inside WSL, or an older copy without the launcher: install from the Downloads zip.
    wsl.exe -d %DISTRO% -- bash -c "test -f ~/fsg-dv-sim/tools/launcher/launcher.py"
    if errorlevel 1 call :install || exit /b 1
)

echo Starting the FSG DV launcher in WSL (%DISTRO%) for %REPO% ...
echo The page opens in your browser: http://localhost:%PORT%
echo Keep this window open while you use the simulator; closing it stops everything.
echo.
wsl.exe -d %DISTRO% --cd ~ -- bash -c "exec python3 '%REPO%/tools/launcher/launcher.py' --port %PORT%"
if errorlevel 1 (
    echo.
    echo The launcher stopped with an error. Check that WSL '%DISTRO%' exists ^(wsl -l^) and that
    echo python3 works in it. Change DISTRO at the top of this file if needed.
    pause
)
endlocal
exit /b 0

:install
set "ZIP=%USERPROFILE%\Downloads\fsg-dv-sim.zip"
if not exist "%ZIP%" (
    echo The project is not installed in WSL yet ^(~/fsg-dv-sim^) and there is no
    echo %ZIP% to install it from. Put this file next to the fsg-dv-sim folder instead.
    pause
    exit /b 1
)
echo Installing %ZIP% into WSL ^(%DISTRO%: ~/fsg-dv-sim^) - a few minutes ...
for /f "usebackq delims=" %%P in (`wsl.exe -d %DISTRO% -- wslpath -a "%ZIP%"`) do set "ZIPW=%%P"
wsl.exe -d %DISTRO% --cd ~ -- unzip -o -q "%ZIPW%" -d .
if errorlevel 1 (
    echo Unzipping failed. Is unzip installed in WSL ^(sudo apt install unzip^)?
    pause
    exit /b 1
)
echo.
exit /b 0
