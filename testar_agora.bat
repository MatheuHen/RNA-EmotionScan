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
echo  EXPRESSAO NEURAL - modo de teste de sensibilidade
echo ========================================================================
echo.
echo  O que mudou nesta versao:
echo.
echo    - Limiar de confianca ....... 0,50  ^-^>  0,35
echo    - Inferencia ................ 1 a cada 2 quadros  ^-^>  TODO quadro
echo    - Suavizacao ................ 0,40  ^-^>  0,55 ^(responde mais rapido^)
echo    - Previsao sutil ............ agora aparece, marcada com "?"
echo                                  e "confianca baixa" em ambar
echo.
echo  A camera NAO inicia sozinha. Clique em "Iniciar camera".
echo.
echo  O que testar:
echo    1. Sorriso leve, de conversa      -^> deve marcar Alegria
echo    2. Sobrancelhas levantadas        -^> deve marcar Surpresa
echo    3. Rosto em repouso               -^> deve marcar Neutra
echo    4. Mude de expressao rapidamente  -^> o painel deve acompanhar
echo.
echo  Nos Ajustes ^(painel direito, role para baixo^) voce pode:
echo    - mover "Limiar de confianca" para mais ou menos sensivel
echo    - desligar "Exibir previsao mesmo com confianca baixa"
echo    - desligar "Estabilidade temporal" para ver a saida crua
echo.
echo ========================================================================
echo.
pause

"%PY%" "%~dp0app.py"
exit /b 0
