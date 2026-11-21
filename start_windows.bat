@echo off
title AdversarialLab - Demarrage
setlocal EnableDelayedExpansion
set WSL_UTF8=1

set DISTRO=Ubuntu
set PROJECT_WSL=/mnt/r/ESP/M2SSI/Robustess des IA/Examen/adversarial-lab
set PORT=8000
set URL=http://localhost:%PORT%

echo ==========================================================
echo   AdversarialLab - demarrage complet (WSL Ubuntu + GPU)
echo ==========================================================
echo.

REM -- 0) WSL disponible ? --
wsl.exe --status >nul 2>&1
if errorlevel 1 (
    echo [ERREUR] WSL indisponible sur cette machine.
    echo          Installation : wsl --install -d Ubuntu
    pause
    exit /b 1
)

REM -- 1) libere le port 8000 (serveur WSL precedent OU processus Windows) --
echo [1/4] Liberation du port %PORT% si deja occupe...
wsl.exe -d %DISTRO% -e bash -lc "fuser -k %PORT%/tcp 2>/dev/null; pkill -f '[u]vicorn app.main' 2>/dev/null; exit 0"
for /f "tokens=5" %%p in ('netstat -ano ^| findstr :%PORT% ^| findstr LISTENING') do taskkill /F /PID %%p >nul 2>&1
ping -n 3 127.0.0.1 >nul

REM -- 2) demarre le serveur dans une fenetre dediee (logs visibles) --
echo [2/4] Demarrage du serveur (venv WSL + PyTorch CUDA)...
echo       (premier lancement sur une nouvelle machine : installation des
echo        dependances possible, cela peut prendre plusieurs minutes)
start "AdversarialLab serveur" wsl.exe -d %DISTRO% -e bash -lc "cd '%PROJECT_WSL%' && bash run.sh"

REM -- 3) attend que le serveur reponde --
echo [3/4] Attente du serveur sur %URL% ...
set /a TRIES=0
:waitloop
ping -n 4 127.0.0.1 >nul
set CODE=
for /f %%i in ('curl -s -o NUL -w "%%{http_code}" %URL% 2^>nul') do set CODE=%%i
if "!CODE!"=="200" goto ready
set /a TRIES+=1
if !TRIES! geq 100 (
    echo.
    echo [ERREUR] Le serveur ne repond pas apres 5 minutes.
    echo          Regardez la fenetre "AdversarialLab serveur" pour l'erreur.
    pause
    exit /b 1
)
goto waitloop

:ready
echo [4/4] Serveur pret (HTTP 200) - ouverture du navigateur...
start "" %URL%
echo.
echo ==========================================================
echo   AdversarialLab tourne sur %URL%
echo   - laissez la fenetre "AdversarialLab serveur" ouverte
echo   - pour tout eteindre : stop_windows.bat
echo ==========================================================
ping -n 9 127.0.0.1 >nul
exit /b 0
