@echo off
setlocal EnableExtensions
chcp 65001 >nul 2>&1
cd /d "%~dp0"

echo ========================================================================
echo  EMOTIONSCAN - Configuracao do ambiente
echo  Esta e a UNICA etapa que precisa de internet.
echo ========================================================================
echo.

set "VENV=%~dp0.venv"
set "PY=%VENV%\Scripts\python.exe"

rem ---------------------------------------------------------------- Python
set "BASEPY="
for %%V in (3.12 3.11 3.10 3.9) do (
    if not defined BASEPY (
        py -%%V -c "import sys" >nul 2>&1 && set "BASEPY=py -%%V"
    )
)
if not defined BASEPY (
    python -c "import sys; raise SystemExit(0 if sys.version_info[:2]>=(3,9) else 1)" >nul 2>&1 && set "BASEPY=python"
)
if not defined BASEPY (
    echo [FALHA] Nao encontrei um Python 3.9 a 3.12 compativel.
    echo         Instale o Python 3.11 em https://www.python.org/downloads/windows/
    echo         e marque "Add python.exe to PATH" durante a instalacao.
    goto :erro
)
echo [ OK ] Interprete base: %BASEPY%
%BASEPY% --version

rem ------------------------------------------------------- ambiente virtual
if exist "%PY%" (
    echo [ OK ] Ambiente virtual ja existe em .venv
) else (
    echo [ .. ] Criando ambiente virtual exclusivo em .venv
    %BASEPY% -m venv "%VENV%"
    if errorlevel 1 (
        echo [FALHA] Nao foi possivel criar o ambiente virtual.
        goto :erro
    )
    echo [ OK ] Ambiente virtual criado.
)

rem ---------------------------------------------------------- dependencias
rem Primeiro tentamos com os certificados padrao do pip (pacote certifi), que
rem e a cadeia de confianca mais restrita e, portanto, a mais segura.
rem O bundle de CAs do Windows e usado APENAS como alternativa, quando a rede
rem faz inspecao de TLS e a verificacao padrao falha.
echo.
echo [ .. ] Instalando dependencias (verificacao padrao de certificados)
set "PIPCERT="
"%PY%" -m pip install --upgrade pip --disable-pip-version-check -q
"%PY%" -m pip install -r "%~dp0requirements.txt" --disable-pip-version-check

if errorlevel 1 (
    echo.
    echo [AVISO] A instalacao falhou com a verificacao padrao de certificados.
    echo         Isso costuma indicar proxy corporativo ou antivirus com
    echo         inspecao de TLS. Tentando com as autoridades certificadoras
    echo         ja confiadas por este computador.
    echo.
    powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\exportar_certificados.ps1" "%~dp0scripts\windows-ca-bundle.pem"
    if not exist "%~dp0scripts\windows-ca-bundle.pem" (
        echo [FALHA] Nao foi possivel gerar o bundle de certificados.
        goto :erro
    )
    set "PIPCERT=--cert "%~dp0scripts\windows-ca-bundle.pem""
    echo [ .. ] Reinstalando dependencias com o bundle do Windows
    "%PY%" -m pip install --cert "%~dp0scripts\windows-ca-bundle.pem" --upgrade pip --disable-pip-version-check -q
    "%PY%" -m pip install --cert "%~dp0scripts\windows-ca-bundle.pem" -r "%~dp0requirements.txt" --disable-pip-version-check
    if errorlevel 1 (
        echo.
        echo [FALHA] Nao foi possivel instalar as dependencias.
        echo         Verifique a conexao, o proxy e o antivirus.
        goto :erro
    )
)
echo [ OK ] Dependencias instaladas.

rem --------------------------------------------------------------- modelos
echo.
echo [ .. ] Obtendo os modelos neurais ONNX
"%PY%" "%~dp0scripts\baixar_modelos.py"
if errorlevel 1 (
    echo.
    echo [FALHA] Os modelos neurais nao ficaram disponiveis.
    goto :erro
)

rem -------------------------------------------------------------- validacao
echo.
echo [ .. ] Validando o ambiente configurado
"%PY%" "%~dp0app.py" --verificar
if errorlevel 1 (
    echo [FALHA] A verificacao final encontrou problemas.
    goto :erro
)

echo.
echo ========================================================================
echo  CONFIGURACAO CONCLUIDA
echo.
echo  A partir daqui o sistema funciona SEM internet.
echo    iniciar.bat    abre a aplicacao
echo    verificar.bat  confere a prontidao antes de apresentar
echo    testar.bat     executa os testes automatizados
echo ========================================================================
pause
exit /b 0

:erro
echo.
echo ========================================================================
echo  CONFIGURACAO INTERROMPIDA - veja a mensagem acima.
echo ========================================================================
pause
exit /b 1
