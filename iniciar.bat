@echo off
setlocal EnableExtensions
chcp 65001 >nul 2>&1
cd /d "%~dp0"

set "PY=%~dp0.venv\Scripts\pythonw.exe"
set "PYC=%~dp0.venv\Scripts\python.exe"

if not exist "%PYC%" (
    echo ========================================================================
    echo  [FALHA] Ambiente nao configurado.
    echo  Execute configurar.bat uma vez ^(com internet^) antes de iniciar.
    echo ========================================================================
    pause
    exit /b 1
)

rem Confere modelos e dependencias antes de abrir a janela (RF01).
"%PYC%" "%~dp0app.py" --verificar --sem-hash
if errorlevel 1 (
    echo.
    echo ========================================================================
    echo  [FALHA] Ambiente incompleto - execute configurar.bat.
    echo ========================================================================
    pause
    exit /b 1
)

echo.
echo Abrindo EmotionScan...

rem pythonw.exe abre a interface sem manter uma janela de console.
if exist "%PY%" (
    start "" "%PY%" "%~dp0app.py"
) else (
    "%PYC%" "%~dp0app.py"
)
exit /b 0
