@echo off
setlocal EnableExtensions
chcp 65001 >nul 2>&1
cd /d "%~dp0"

set "PY=%~dp0.venv\Scripts\python.exe"

if not exist "%PY%" (
    echo [FALHA] Ambiente nao configurado. Execute configurar.bat primeiro.
    pause
    exit /b 1
)

"%PY%" "%~dp0scripts\verificar_ambiente.py" %*
set "CODIGO=%ERRORLEVEL%"

echo.
if "%CODIGO%"=="0" echo Situacao: PRONTO PARA APRESENTAR.
if "%CODIGO%"=="2" echo Situacao: PRONTO COM RESSALVAS - leia os avisos acima.
if "%CODIGO%"=="1" echo Situacao: NAO PRONTO - corrija os itens criticos acima.

pause
exit /b %CODIGO%
