"""Identidade visual do Expressao Neural.

Paleta escura de alto contraste (tecnologica e academica), tipografia Segoe UI
com degradacao automatica para fontes sempre presentes. Nenhum recurso remoto:
sem CDN, sem fonte baixada, sem icone externo (secao 9 do escopo).
"""
from __future__ import annotations

import tkinter.font as tkfont
from tkinter import ttk

# --------------------------------------------------------------------- cores
FUNDO = "#0b1120"            # fundo da janela
SUPERFICIE = "#111a2e"       # cartoes
SUPERFICIE_ALTA = "#18233c"  # cartoes destacados / hover
BORDA = "#223153"
BORDA_CLARA = "#2f4269"

TEXTO = "#e8eefc"
TEXTO_SECUNDARIO = "#9fb0d0"
TEXTO_FRACO = "#6b7fa3"

PRIMARIA = "#38bdf8"         # ciano tecnologico
PRIMARIA_ESCURA = "#0b7fb0"
ACENTO = "#a78bfa"           # violeta (rede neural)
SUCESSO = "#22c55e"
ALERTA = "#f59e0b"
ERRO = "#ef4444"
NEUTRO = "#64748b"

GRADE = "#1c283f"

# ------------------------------------------------------------------- fontes
_PREFERIDAS = ("Segoe UI", "Calibri", "DejaVu Sans", "Helvetica", "Arial")
_MONO = ("Cascadia Mono", "Consolas", "DejaVu Sans Mono", "Courier New")


class Fontes:
    """Resolve as familias disponiveis uma unica vez por aplicacao."""

    def __init__(self, raiz) -> None:
        disponiveis = {f.lower() for f in tkfont.families(raiz)}
        self.familia = next(
            (f for f in _PREFERIDAS if f.lower() in disponiveis), "TkDefaultFont"
        )
        self.familia_mono = next(
            (f for f in _MONO if f.lower() in disponiveis), "TkFixedFont"
        )
        f, m = self.familia, self.familia_mono
        self.titulo = (f, 20, "bold")
        self.subtitulo = (f, 10)
        self.secao = (f, 11, "bold")
        self.corpo = (f, 10)
        self.corpo_bold = (f, 10, "bold")
        self.pequeno = (f, 9)
        self.micro = (f, 8)
        self.destaque = (f, 30, "bold")
        self.destaque_medio = (f, 15, "bold")
        self.botao = (f, 10, "bold")
        self.mono = (m, 9)
        self.mono_bold = (m, 10, "bold")
        self.marca = (f, 17, "bold")


def ativar_consciencia_dpi() -> None:
    """Evita interface borrada em telas com escala do Windows acima de 100%.

    Deve ser chamada ANTES de criar a janela Tk. Silenciosa fora do Windows.
    """
    try:
        import ctypes

        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)  # Windows 8.1+
        except (AttributeError, OSError):
            ctypes.windll.user32.SetProcessDPIAware()  # Windows Vista+
    except Exception:  # noqa: BLE001 - outras plataformas ou API indisponivel
        pass


def aplicar(raiz) -> Fontes:
    """Configura o tema ttk da aplicacao e devolve as fontes resolvidas."""
    fontes = Fontes(raiz)
    estilo = ttk.Style(raiz)
    try:
        estilo.theme_use("clam")  # unico tema ttk que aceita cores customizadas no Windows
    except Exception:  # pragma: no cover - ambiente sem o tema clam
        pass

    raiz.configure(bg=FUNDO)

    estilo.configure(".", background=FUNDO, foreground=TEXTO, font=fontes.corpo)
    estilo.configure("TFrame", background=FUNDO)
    estilo.configure("Cartao.TFrame", background=SUPERFICIE)
    estilo.configure("CartaoAlto.TFrame", background=SUPERFICIE_ALTA)
    estilo.configure("Borda.TFrame", background=BORDA)

    estilo.configure("TLabel", background=FUNDO, foreground=TEXTO)
    estilo.configure("Cartao.TLabel", background=SUPERFICIE, foreground=TEXTO)
    estilo.configure(
        "CartaoFraco.TLabel", background=SUPERFICIE, foreground=TEXTO_SECUNDARIO,
        font=fontes.pequeno,
    )
    estilo.configure(
        "Titulo.TLabel", background=FUNDO, foreground=TEXTO, font=fontes.titulo
    )
    estilo.configure(
        "Subtitulo.TLabel", background=FUNDO, foreground=TEXTO_SECUNDARIO,
        font=fontes.subtitulo,
    )
    estilo.configure(
        "Secao.TLabel", background=SUPERFICIE, foreground=PRIMARIA, font=fontes.secao
    )
    estilo.configure(
        "Rodape.TLabel", background=SUPERFICIE, foreground=TEXTO_FRACO,
        font=fontes.pequeno,
    )
    estilo.configure(
        "Mono.TLabel", background=SUPERFICIE, foreground=TEXTO, font=fontes.mono
    )

    # ------------------------------------------------------------- botoes
    def _botao(nome: str, cor: str, cor_hover: str, texto: str = "#06121f") -> None:
        estilo.configure(
            nome,
            background=cor,
            foreground=texto,
            font=fontes.botao,
            borderwidth=0,
            focusthickness=1,
            focuscolor=PRIMARIA,
            padding=(14, 9),
            relief="flat",
        )
        estilo.map(
            nome,
            background=[("disabled", BORDA), ("pressed", cor_hover), ("active", cor_hover)],
            foreground=[("disabled", TEXTO_FRACO)],
        )

    _botao("Primario.TButton", PRIMARIA, "#7dd3fc")
    _botao("Sucesso.TButton", SUCESSO, "#4ade80")
    _botao("Alerta.TButton", ALERTA, "#fbbf24")
    _botao("Perigo.TButton", ERRO, "#f87171", texto="#fff5f5")
    _botao("Acento.TButton", ACENTO, "#c4b5fd")

    estilo.configure(
        "Neutro.TButton",
        background=SUPERFICIE_ALTA,
        foreground=TEXTO,
        font=fontes.botao,
        borderwidth=1,
        bordercolor=BORDA_CLARA,
        padding=(14, 9),
        relief="flat",
    )
    estilo.map(
        "Neutro.TButton",
        background=[("disabled", SUPERFICIE), ("active", BORDA_CLARA)],
        foreground=[("disabled", TEXTO_FRACO)],
    )

    # ------------------------------------------------- controles auxiliares
    estilo.configure(
        "TCombobox",
        fieldbackground=SUPERFICIE_ALTA,
        background=SUPERFICIE_ALTA,
        foreground=TEXTO,
        arrowcolor=PRIMARIA,
        bordercolor=BORDA_CLARA,
        selectbackground=SUPERFICIE_ALTA,
        selectforeground=TEXTO,
        padding=6,
    )
    estilo.map("TCombobox", fieldbackground=[("readonly", SUPERFICIE_ALTA)])

    estilo.configure(
        "TCheckbutton",
        background=SUPERFICIE,
        foreground=TEXTO_SECUNDARIO,
        font=fontes.pequeno,
        focuscolor=PRIMARIA,
    )
    estilo.map("TCheckbutton", foreground=[("active", TEXTO)])

    estilo.configure(
        "TScale", background=SUPERFICIE, troughcolor=BORDA, bordercolor=BORDA
    )
    estilo.configure(
        "Separador.TFrame", background=BORDA
    )
    estilo.configure(
        "TNotebook", background=FUNDO, borderwidth=0, tabmargins=(4, 4, 0, 0)
    )
    estilo.configure(
        "TNotebook.Tab",
        background=SUPERFICIE,
        foreground=TEXTO_SECUNDARIO,
        padding=(16, 8),
        font=fontes.corpo_bold,
        borderwidth=0,
    )
    estilo.map(
        "TNotebook.Tab",
        background=[("selected", SUPERFICIE_ALTA)],
        foreground=[("selected", PRIMARIA)],
    )
    return fontes


def cor_confianca(valor: float, limiar: float) -> str:
    """Verde acima do limiar, ambar na faixa intermediaria, vermelho abaixo."""
    if valor >= limiar:
        return SUCESSO
    if valor >= limiar * 0.6:
        return ALERTA
    return ERRO


def hex_para_bgr(cor_hex: str) -> tuple[int, int, int]:
    """Converte '#rrggbb' para a tupla BGR usada pelo OpenCV."""
    c = cor_hex.lstrip("#")
    r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    return b, g, r
