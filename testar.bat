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

echo ========================================================================
echo  EMOTIONSCAN - Testes automatizados
echo ========================================================================
echo.

if /i "%~1"=="--lentos" (
    echo Suite COMPLETA, incluindo execucao prolongada.
    echo Duracao da execucao prolongada: %EXPRESSAO_NEURAL_DURACAO_MIN% minuto^(s^).
    echo Defina EXPRESSAO_NEURAL_DURACAO_MIN=30 para o ensaio completo.
    echo.
    "%PY%" -m pytest -m "lento or not lento" -v
) else (
    echo Suite padrao ^(testes de longa duracao ignorados^).
    echo Use: testar.bat --lentos   para a suite completa.
    echo.
    "%PY%" -m pytest -m "not lento"
)
set "CODIGO=%ERRORLEVEL%"

echo.
if "%CODIGO%"=="0" (
    echo RESULTADO: todos os testes executados passaram.
) else (
    echo RESULTADO: houve falha nos testes - veja o relatorio acima.
)
pause
exit /b %CODIGO%
