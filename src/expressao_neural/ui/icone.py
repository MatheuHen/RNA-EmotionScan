"""Aplica o icone proprio da aplicacao em todas as janelas.

Por que nao basta `iconbitmap()`
--------------------------------
No Windows, a barra de tarefas agrupa janelas pelo *AppUserModelID* do
processo. Um processo iniciado por `python.exe`/`pythonw.exe` herda o
identificador do interpretador, e o Windows usa o icone DELE -- mesmo que a
janela ja tenha outro icone definido. Resultado pratico: o icone muda na barra
de titulo e no Alt+Tab, mas a barra de tarefas continua exibindo o logotipo do
Python.

A correcao e declarar um AppUserModelID proprio ANTES de criar a janela, com
`SetCurrentProcessExplicitAppUserModelID`. Isso e documentado pela Microsoft
para aplicacoes que rodam sob um interpretador hospedeiro.

Tudo aqui e tolerante a falha: sem o arquivo de icone, ou fora do Windows, a
aplicacao abre normalmente com o icone padrao.
"""
from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path

from ..config import RAIZ

ARQUIVO_ICO = RAIZ / "assets" / "icone.ico"
ARQUIVO_PNG = RAIZ / "assets" / "icone_256.png"

# Identificador no formato recomendado: Empresa.Produto.Componente.Versao
APP_USER_MODEL_ID = "UniRV.EmotionScan.Aplicacao.1"

_foto_icone: tk.PhotoImage | None = None  # evita coleta pelo GC


def declarar_identidade_no_windows(identificador: str = APP_USER_MODEL_ID) -> bool:
    """Declara o AppUserModelID. Chame ANTES de criar a janela Tk.

    Devolve True se a chamada foi aceita. Em outros sistemas, ou se a API não
    estiver disponível, devolve False sem levantar exceção.
    """
    if sys.platform != "win32":
        return False
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(identificador)
        return True
    except Exception:  # noqa: BLE001 - recurso cosmético, nunca fatal
        return False


def aplicar(janela: tk.Misc, caminho: Path | None = None) -> bool:
    """Aplica o ícone a uma janela (raiz ou Toplevel).

    Devolve True se algum método funcionou. Ausência do arquivo não é erro:
    a aplicação deve abrir mesmo sem o ícone.
    """
    global _foto_icone

    ico = caminho or ARQUIVO_ICO
    aplicado = False

    # 1. .ico nativo — é o que o Windows usa na barra de título e no Explorer.
    if ico.is_file():
        try:
            janela.iconbitmap(default=str(ico))
            aplicado = True
        except tk.TclError:
            try:
                janela.iconbitmap(str(ico))
                aplicado = True
            except tk.TclError:
                pass

    # 2. iconphoto — funciona em qualquer plataforma e cobre o caso de o
    #    iconbitmap falhar. `default=True` propaga para as Toplevel futuras.
    if ARQUIVO_PNG.is_file():
        try:
            if _foto_icone is None:
                _foto_icone = tk.PhotoImage(master=janela, file=str(ARQUIVO_PNG))
            janela.iconphoto(True, _foto_icone)
            aplicado = True
        except tk.TclError:
            pass

    return aplicado


def preparar() -> None:
    """Atalho para o ponto de entrada: declara a identidade antes da janela."""
    declarar_identidade_no_windows()
