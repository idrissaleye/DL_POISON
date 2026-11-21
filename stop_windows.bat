@echo off
title AdversarialLab - Arret complet
echo ==========================================================
echo   AdversarialLab - arret du serveur puis de la VM WSL
echo ==========================================================
echo.

echo [1/3] Arret du serveur AdversarialLab (port 8000)...
wsl.exe -d Ubuntu -e bash -lc "fuser -k 8000/tcp 2>/dev/null; pkill -f '[u]vicorn app.main' 2>/dev/null; exit 0"

REM tue aussi un eventuel serveur lance cote Windows (PyCharm par exemple)
for /f "tokens=5" %%p in ('netstat -ano ^| findstr :8000 ^| findstr LISTENING') do (
    echo       processus Windows PID %%p arrete
    taskkill /F /PID %%p >nul 2>&1
)
ping -n 3 127.0.0.1 >nul

echo [2/3] Extinction de WSL (toutes distributions, libere la memoire vmmem)...
wsl.exe --shutdown
ping -n 4 127.0.0.1 >nul

echo [3/3] Verification finale...
netstat -ano | findstr :8000 | findstr LISTENING >nul 2>&1
if errorlevel 1 (
    echo       port 8000 : libre
) else (
    echo       ATTENTION : le port 8000 est encore occupe cote Windows
)
echo.
echo Etat de WSL :
wsl.exe --list --running
echo.
echo ==========================================================
echo   Tout est eteint (serveur + VM WSL).
echo ==========================================================
ping -n 7 127.0.0.1 >nul
exit /b 0
