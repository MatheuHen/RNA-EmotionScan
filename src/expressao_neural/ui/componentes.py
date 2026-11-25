"""Componentes visuais reutilizaveis (Canvas puro, sem dependencia externa)."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import numpy as np

from .. import config
from . import tema


class Cartao(tk.Frame):
    """Painel com borda sutil e titulo opcional.

    Os filhos devem ser criados em `cartao.corpo`, que ja respeita o
    espacamento interno.
    """

    def __init__(self, pai, titulo: str | None = None, fontes=None, padding: int = 12) -> None:
        super().__init__(
            pai,
            bg=tema.SUPERFICIE,
            highlightbackground=tema.BORDA,
            highlightcolor=tema.BORDA,
            highlightthickness=1,
            bd=0,
        )
        self.fontes = fontes
        if titulo:
            cab = tk.Frame(self, bg=tema.SUPERFICIE)
            cab.pack(fill="x", padx=padding, pady=(padding, 0))
            tk.Frame(cab, bg=tema.PRIMARIA, width=3, height=15).pack(
                side="left", padx=(0, 8)
            )
            ttk.Label(cab, text=titulo.upper(), style="Secao.TLabel").pack(side="left")
        self.corpo = tk.Frame(self, bg=tema.SUPERFICIE)
        self.corpo.pack(fill="both", expand=True, padx=padding, pady=padding)


class Selo(tk.Label):
    """Etiqueta compacta de status (ex.: PROCESSAMENTO 100% LOCAL)."""

    def __init__(self, pai, texto: str, cor: str, fonte, fundo=tema.FUNDO) -> None:
        super().__init__(
            pai,
            text=texto,
            font=fonte,
            fg=cor,
            bg=fundo,
            padx=9,
            pady=4,
            borderwidth=1,
            relief="solid",
        )
        self.configure(highlightbackground=cor)
        self._cor = cor

    def atualizar(self, texto: str, cor: str | None = None) -> None:
        self.configure(text=texto)
        if cor and cor != self._cor:
            self.configure(fg=cor)
            self._cor = cor


class LinhaIndicador(ttk.Frame):
    """Par rotulo/valor alinhado, usado nos indicadores tecnicos (RF07)."""

    def __init__(self, pai, rotulo: str, fontes, valor: str = "—") -> None:
        super().__init__(pai, style="Cartao.TFrame")
        self.pack(fill="x", pady=2)
        ttk.Label(self, text=rotulo, style="CartaoFraco.TLabel").pack(side="left")
        self._valor = tk.Label(
            self,
            text=valor,
            font=fontes.mono,
            fg=tema.TEXTO,
            bg=tema.SUPERFICIE,
            anchor="e",
        )
        self._valor.pack(side="right")

    def definir(self, texto: str, cor: str = tema.TEXTO) -> None:
        if self._valor.cget("text") != texto:
            self._valor.configure(text=texto)
        if self._valor.cget("fg") != cor:
            self._valor.configure(fg=cor)


class PainelRolavel(tk.Frame):
    """Contêiner com rolagem vertical, para caber em notebooks de tela baixa.

    Os filhos devem ser criados em `painel.conteudo`. A barra de rolagem so
    aparece quando o conteudo realmente excede a altura disponivel.
    """

    def __init__(self, pai, largura: int) -> None:
        super().__init__(pai, bg=tema.FUNDO)
        self._largura = largura
        self._canvas = tk.Canvas(
            self, bg=tema.FUNDO, highlightthickness=0, bd=0, width=largura
        )
        self._barra = ttk.Scrollbar(self, orient="vertical", command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=self._ao_rolar)
        self._canvas.pack(side="left", fill="both", expand=True)

        self.conteudo = tk.Frame(self._canvas, bg=tema.FUNDO)
        self._janela = self._canvas.create_window(
            (0, 0), window=self.conteudo, anchor="nw", width=largura
        )
        self.conteudo.bind("<Configure>", self._ao_mudar_conteudo)
        self._canvas.bind("<Configure>", self._ao_mudar_canvas)
        self.bind_all("<MouseWheel>", self._ao_girar_roda, add="+")
        self._barra_visivel = False

    # ----------------------------------------------------------------- interno
    def _ao_mudar_conteudo(self, _evento=None) -> None:
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _ao_mudar_canvas(self, evento) -> None:
        self._canvas.itemconfigure(self._janela, width=evento.width)

    def _ao_rolar(self, inicio: str, fim: str) -> None:
        precisa = not (float(inicio) <= 0.0 and float(fim) >= 1.0)
        if precisa and not self._barra_visivel:
            self._barra.pack(side="right", fill="y")
            self._barra_visivel = True
        elif not precisa and self._barra_visivel:
            self._barra.pack_forget()
            self._barra_visivel = False
        self._barra.set(inicio, fim)

    def _ao_girar_roda(self, evento) -> None:
        if not self._barra_visivel or not self.winfo_exists():
            return
        # Rola somente quando o ponteiro esta sobre este painel.
        sob = self.winfo_containing(evento.x_root, evento.y_root)
        alvo = sob
        while alvo is not None:
            if alvo is self:
                break
            alvo = getattr(alvo, "master", None)
        if alvo is not self:
            return
        self._canvas.yview_scroll(-1 if evento.delta > 0 else 1, "units")


class IconeExpressao(tk.Canvas):
    """Disco colorido com a inicial da expressao predominante.

    Substitui emojis: o Tk 8.6 no Windows nao renderiza emoji colorido, e um
    glifo quebrado passaria a impressao de componente decorativo sem funcao.
    """

    LADO = 56

    def __init__(self, pai, fontes) -> None:
        super().__init__(
            pai, width=self.LADO, height=self.LADO, bg=tema.SUPERFICIE,
            highlightthickness=0, bd=0,
        )
        m = 3
        self._anel = self.create_oval(
            m, m, self.LADO - m, self.LADO - m, outline=tema.BORDA_CLARA, width=3
        )
        self._disco = self.create_oval(
            m + 7, m + 7, self.LADO - m - 7, self.LADO - m - 7,
            fill=tema.GRADE, outline="",
        )
        self._letra = self.create_text(
            self.LADO / 2, self.LADO / 2, text="—",
            fill=tema.TEXTO_FRACO, font=(fontes.familia, 18, "bold"),
        )

    def definir(self, indice: int | None) -> None:
        if indice is None or not 0 <= indice < len(config.CLASSES_PT):
            self.itemconfigure(self._anel, outline=tema.BORDA_CLARA)
            self.itemconfigure(self._disco, fill=tema.GRADE)
            self.itemconfigure(self._letra, text="?", fill=tema.TEXTO_FRACO)
            return
        cor = config.CORES_CLASSES[indice]
        self.itemconfigure(self._anel, outline=cor)
        self.itemconfigure(self._disco, fill=cor)
        self.itemconfigure(
            self._letra, text=config.CLASSES_PT[indice][0].upper(), fill="#0b1120"
        )


class GraficoProbabilidades(tk.Canvas):
    """Grafico de barras horizontais das 7 categorias (RF06).

    Os itens do Canvas sao criados uma unica vez e apenas tem coordenadas e
    cores atualizadas -- evita recriar objetos a cada quadro.
    """

    ALTURA_LINHA = 30
    MARGEM_X = 12
    LARGURA_ROTULO = 92
    LARGURA_VALOR = 52

    def __init__(self, pai, fontes) -> None:
        self._n = len(config.CLASSES_PT)
        super().__init__(
            pai,
            height=self.ALTURA_LINHA * self._n + 10,
            bg=tema.SUPERFICIE,
            highlightthickness=0,
            bd=0,
        )
        self.fontes = fontes
        self._itens: list[dict] = []
        self._valores = np.zeros(self._n, dtype=np.float32)
        self._destaque = -1
        self._construir()
        self.bind("<Configure>", lambda _e: self._redesenhar())

    # ---------------------------------------------------------------- montagem
    def _construir(self) -> None:
        for i, nome in enumerate(config.CLASSES_PT):
            y = 8 + i * self.ALTURA_LINHA
            # Marcador colorido desenhado no proprio Canvas: nao depende de
            # fonte com emoji colorido, que o Tk 8.6 nao renderiza no Windows.
            self.create_oval(
                self.MARGEM_X,
                y + 5,
                self.MARGEM_X + 8,
                y + 13,
                fill=config.CORES_CLASSES[i],
                outline="",
            )
            rotulo = self.create_text(
                self.MARGEM_X + 15,
                y + 9,
                text=nome,
                anchor="w",
                fill=tema.TEXTO_SECUNDARIO,
                font=self.fontes.pequeno,
            )
            trilha = self.create_rectangle(0, 0, 0, 0, fill=tema.GRADE, outline="")
            barra = self.create_rectangle(
                0, 0, 0, 0, fill=config.CORES_CLASSES[i], outline=""
            )
            valor = self.create_text(
                0,
                y + 9,
                text="0,0%",
                anchor="e",
                fill=tema.TEXTO_FRACO,
                font=self.fontes.mono,
            )
            self._itens.append(
                {"rotulo": rotulo, "trilha": trilha, "barra": barra, "valor": valor, "y": y}
            )

    # -------------------------------------------------------------- atualizacao
    def atualizar(self, probabilidades: np.ndarray | None, destaque: int = -1) -> None:
        if probabilidades is None:
            self._valores = np.zeros(self._n, dtype=np.float32)
            self._destaque = -1
        else:
            v = np.asarray(probabilidades, dtype=np.float32).reshape(-1)
            if v.size != self._n:
                return
            self._valores = v
            self._destaque = int(destaque)
        self._redesenhar()

    def _redesenhar(self) -> None:
        largura = max(self.winfo_width(), 260)
        x0 = self.MARGEM_X + self.LARGURA_ROTULO
        x1 = largura - self.MARGEM_X - self.LARGURA_VALOR
        faixa = max(10, x1 - x0)

        for i, item in enumerate(self._itens):
            y = item["y"]
            valor = float(np.clip(self._valores[i], 0.0, 1.0))
            ativo = i == self._destaque

            self.coords(item["trilha"], x0, y + 2, x1, y + 16)
            self.coords(item["barra"], x0, y + 2, x0 + faixa * valor, y + 16)
            self.itemconfigure(
                item["barra"],
                fill=config.CORES_CLASSES[i] if (ativo or valor > 0.02) else tema.BORDA_CLARA,
            )
            self.coords(item["valor"], largura - self.MARGEM_X, y + 9)
            self.itemconfigure(item["valor"], text=f"{valor * 100:.1f}%".replace(".", ","))
            self.itemconfigure(
                item["valor"],
                fill=config.CORES_CLASSES[i] if ativo else tema.TEXTO_FRACO,
                font=self.fontes.mono_bold if ativo else self.fontes.mono,
            )
            self.itemconfigure(
                item["rotulo"],
                fill=tema.TEXTO if ativo else tema.TEXTO_SECUNDARIO,
                font=self.fontes.corpo_bold if ativo else self.fontes.pequeno,
            )


class MedidorConfianca(tk.Canvas):
    """Barra unica de confianca com marcador do limiar vigente (RF05)."""

    def __init__(self, pai, fontes) -> None:
        super().__init__(pai, height=16, bg=tema.SUPERFICIE, highlightthickness=0, bd=0)
        self.fontes = fontes
        self._valor = 0.0
        self._limiar = 0.5
        self._trilha = self.create_rectangle(0, 0, 0, 0, fill=tema.GRADE, outline="")
        self._barra = self.create_rectangle(0, 0, 0, 0, fill=tema.NEUTRO, outline="")
        self._marca = self.create_line(0, 0, 0, 0, fill=tema.TEXTO_SECUNDARIO, dash=(2, 2))
        self.bind("<Configure>", lambda _e: self._redesenhar())

    def atualizar(self, valor: float, limiar: float) -> None:
        self._valor = float(np.clip(valor, 0.0, 1.0))
        self._limiar = float(np.clip(limiar, 0.0, 1.0))
        self._redesenhar()

    def _redesenhar(self) -> None:
        largura = max(self.winfo_width(), 160)
        self.coords(self._trilha, 0, 4, largura, 14)
        self.coords(self._barra, 0, 4, largura * self._valor, 14)
        self.itemconfigure(self._barra, fill=tema.cor_confianca(self._valor, self._limiar))
        x = largura * self._limiar
        self.coords(self._marca, x, 1, x, 16)
