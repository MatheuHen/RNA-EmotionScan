"""Verificacao de ambiente (RF01) usada pela aplicacao e por verificar.bat."""
from __future__ import annotations

import importlib
import platform
import sys
from dataclasses import dataclass

from .. import config
from ..core import modelos

VERSAO_MINIMA_PYTHON = (3, 9)
DEPENDENCIAS = (
    ("cv2", "opencv-python"),
    ("numpy", "numpy"),
    ("PIL", "pillow"),
    ("tkinter", "tkinter (biblioteca padrão do Python)"),
)


@dataclass
class Item:
    nome: str
    ok: bool
    detalhe: str
    critico: bool = True


@dataclass
class Relatorio:
    itens: list[Item]

    @property
    def ok(self) -> bool:
        return all(i.ok for i in self.itens if i.critico)

    @property
    def problemas(self) -> list[Item]:
        return [i for i in self.itens if not i.ok]

    @property
    def problemas_criticos(self) -> list[Item]:
        return [i for i in self.itens if not i.ok and i.critico]

    def texto(self) -> str:
        linhas = []
        for i in self.itens:
            marca = "[ OK ]" if i.ok else ("[FALHA]" if i.critico else "[AVISO]")
            linhas.append(f"{marca} {i.nome}: {i.detalhe}")
        return "\n".join(linhas)


def _versao(mod_nome: str) -> str:
    try:
        mod = importlib.import_module(mod_nome)
    except Exception as exc:  # noqa: BLE001
        raise ImportError(str(exc)) from exc
    for atributo in ("__version__", "VERSION", "TkVersion"):
        valor = getattr(mod, atributo, None)
        if valor:
            return str(valor)
    return "instalado"


def verificar_ambiente(conferir_hash: bool = True) -> Relatorio:
    itens: list[Item] = []

    # ---------------------------------------------------------------- Python
    atual = sys.version_info[:3]
    itens.append(
        Item(
            "Python",
            atual[:2] >= VERSAO_MINIMA_PYTHON,
            f"{platform.python_version()} ({platform.machine()}) — mínimo "
            f"{'.'.join(map(str, VERSAO_MINIMA_PYTHON))}",
        )
    )

    # ----------------------------------------------------------- dependencias
    for modulo, pacote in DEPENDENCIAS:
        try:
            itens.append(Item(pacote, True, f"versão {_versao(modulo)}"))
        except ImportError as exc:
            itens.append(
                Item(
                    pacote,
                    False,
                    f"não disponível ({exc}). Execute configurar.bat para "
                    f"reinstalar as dependências.",
                )
            )

    # ---------------------------------------------------------------- modelos
    for status in modelos.verificar_todos(conferir_hash=conferir_hash):
        itens.append(Item(f"Modelo {status.info.chave}", status.ok, status.mensagem))

    # -------------------------------------------------------- pastas de apoio
    for pasta in (config.DIR_MODELOS, config.DIR_EVIDENCIAS):
        existe = pasta.is_dir()
        itens.append(
            Item(
                f"Pasta {pasta.name}",
                existe,
                str(pasta) if existe else f"ausente: {pasta}",
                critico=(pasta is config.DIR_MODELOS),
            )
        )

    # ------------------------------------------------- backends de video (aviso)
    try:
        import cv2 as cv

        backends = [
            n for n in ("CAP_DSHOW", "CAP_MSMF", "CAP_ANY") if hasattr(cv, n)
        ]
        itens.append(
            Item(
                "Backends de vídeo",
                bool(backends),
                ", ".join(backends) or "nenhum identificado",
                critico=False,
            )
        )
    except ImportError:
        pass

    return Relatorio(itens)
