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
echo  EXPRESSAO NEURAL - teste de funcionamento OFFLINE
echo ========================================================================
echo.
echo  ANTES DE CONTINUAR, desligue a rede:
echo.
echo    1. Clique no icone de rede na bandeja ^(canto inferior direito^)
echo    2. Desligue o Wi-Fi
echo    3. Desconecte o cabo de rede, se houver
echo.
echo  O script CONFIRMA que a rede esta fora antes de testar. Se ela ainda
echo  estiver acessivel, ele aborta - validar modo offline com internet
echo  ligada nao prova nada.
echo.
echo ========================================================================
echo.
pause

"%PY%" "%~dp0scripts\testar_offline.py" %*
set "CODIGO=%ERRORLEVEL%"

echo.
if "%CODIGO%"=="0" (
    echo RESULTADO: APROVADO - o sistema funciona sem internet.
    echo Evidencia gravada em evidencias\teste_offline.md
)
if "%CODIGO%"=="2" (
    echo RESULTADO: aborted ou com ressalvas - leia as mensagens acima.
)
if "%CODIGO%"=="1" (
    echo RESULTADO: REPROVADO - veja as falhas acima.
)
echo.
echo Nao esqueca de RELIGAR a rede depois do teste.
echo.
pause
exit /b %CODIGO%
