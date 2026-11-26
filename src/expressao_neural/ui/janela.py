"""Janela principal do Expressao Neural.

Regra de ouro: a thread do Tkinter NUNCA executa inferencia nem leitura de
camera. Ela apenas consome o quadro mais recente publicado pelo
`TrabalhadorCamera` e atualiza widgets.
"""
from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2 as cv
import numpy as np
from PIL import Image, ImageTk

from .. import config
from ..core import imagem_estatica
from ..core.camera import TrabalhadorCamera, listar_cameras
from ..core.detector import ErroModelo
from ..core.pipeline import PipelineExpressao, ResultadoQuadro
from ..utils import ambiente
from . import componentes, desenho, sobre, tema

INTERVALO_TICK_MS = 25
LARGURA_PAINEL = 352


class JanelaPrincipal(tk.Tk):
    # ------------------------------------------------------------------ setup
    def __init__(self) -> None:
        super().__init__()
        self.ajustes = config.Ajustes.carregar()
        self.title(f"{config.APP_NOME} — {config.APP_DESCRICAO}")
        self.geometry("1280x800")
        self.minsize(1024, 640)
        self.protocol("WM_DELETE_WINDOW", self.encerrar)

        self.fontes = tema.aplicar(self)

        self._pipeline: PipelineExpressao | None = None
        self._trabalhador: TrabalhadorCamera | None = None
        # Captura que não encerrou no prazo e ainda pode deter a câmera.
        self._trabalhador_preso: TrabalhadorCamera | None = None
        self._foto: ImageTk.PhotoImage | None = None
        self._quadro_exibido: np.ndarray | None = None
        self._ultimo_resultado: ResultadoQuadro | None = None
        self._modo = "parado"  # parado | ativo | pausado | imagem | erro
        self._tela_cheia = False
        self._tick: str | None = None
        self._fps = 0.0
        self._cameras: list[int] = [self.ajustes.indice_camera]
        self._carregando_modelos = False
        self._mensagem_status = "Pronto para iniciar."
        self._cor_status = tema.TEXTO_SECUNDARIO

        self._var_limiar = tk.DoubleVar(value=self.ajustes.limiar_confianca)
        self._var_suavizar = tk.BooleanVar(value=self.ajustes.suavizacao_ativa)
        self._var_espelhar = tk.BooleanVar(value=self.ajustes.espelhar_imagem)
        self._var_multiplos = tk.BooleanVar(value=self.ajustes.multiplos_rostos)
        self._var_passo = tk.IntVar(value=self.ajustes.inferir_a_cada)
        self._var_camera = tk.StringVar(value=f"Câmera {self.ajustes.indice_camera}")

        self._montar()
        self._atalhos()
        self._verificar_ambiente_inicial()
        self._agendar_tick()

    # ------------------------------------------------------------------ layout
    def _montar(self) -> None:
        self._montar_cabecalho()

        corpo = tk.Frame(self, bg=tema.FUNDO)
        corpo.pack(fill="both", expand=True, padx=16, pady=(0, 10))
        corpo.columnconfigure(0, weight=1)
        corpo.columnconfigure(1, minsize=LARGURA_PAINEL, weight=0)
        corpo.rowconfigure(0, weight=1)

        self._area_video = tk.Frame(corpo, bg=tema.FUNDO)
        self._area_video.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        self._montar_video(self._area_video)

        # Painel rolavel: garante que todos os cartoes continuem acessiveis em
        # notebooks de 1366x768 ou com escala de tela elevada.
        self._painel = componentes.PainelRolavel(corpo, LARGURA_PAINEL)
        self._painel.grid(row=0, column=1, sticky="nsew")
        self._montar_painel(self._painel.conteudo)

        self._montar_rodape()

    def _montar_cabecalho(self) -> None:
        barra = tk.Frame(self, bg=tema.SUPERFICIE)
        barra.pack(fill="x")
        tk.Frame(barra, bg=tema.PRIMARIA, height=3).pack(fill="x", side="top")

        interno = tk.Frame(barra, bg=tema.SUPERFICIE)
        interno.pack(fill="x", padx=18, pady=12)

        marca = tk.Canvas(
            interno, width=44, height=44, bg=tema.SUPERFICIE, highlightthickness=0
        )
        marca.pack(side="left", padx=(0, 12))
        self._desenhar_marca(marca)

        textos = tk.Frame(interno, bg=tema.SUPERFICIE)
        textos.pack(side="left")
        ttk.Label(
            textos, text=config.APP_NOME, style="Cartao.TLabel", font=self.fontes.marca
        ).pack(anchor="w")
        ttk.Label(
            textos,
            text=f"{config.APP_DESCRICAO} · YuNet + MobileFaceNet (ONNX)",
            style="CartaoFraco.TLabel",
        ).pack(anchor="w")

        acoes = tk.Frame(interno, bg=tema.SUPERFICIE)
        acoes.pack(side="right")
        self._selo_local = componentes.Selo(
            acoes,
            "● PROCESSAMENTO 100% LOCAL",
            tema.SUCESSO,
            self.fontes.micro,
            fundo=tema.SUPERFICIE,
        )
        self._selo_local.pack(side="left", padx=(0, 12))
        ttk.Button(
            acoes, text="Sobre o projeto", style="Neutro.TButton", command=self.abrir_sobre
        ).pack(side="left", padx=(0, 8))
        self._btn_tela_cheia = ttk.Button(
            acoes,
            text="Apresentação (F11)",
            style="Acento.TButton",
            command=self.alternar_tela_cheia,
        )
        self._btn_tela_cheia.pack(side="left")

    def _desenhar_marca(self, c: tk.Canvas) -> None:
        """Logotipo: pequeno grafo de neuronios desenhado em Canvas (sem arquivo)."""
        nos = [(8, 10), (8, 22), (8, 34), (22, 16), (22, 28), (36, 22)]
        ligacoes = [(0, 3), (1, 3), (1, 4), (2, 4), (3, 5), (4, 5)]
        for a, b in ligacoes:
            c.create_line(*nos[a], *nos[b], fill=tema.BORDA_CLARA, width=1)
        for i, (x, y) in enumerate(nos):
            cor = tema.PRIMARIA if i >= 3 else tema.ACENTO
            c.create_oval(x - 3, y - 3, x + 3, y + 3, fill=cor, outline="")

    def _montar_video(self, pai: tk.Frame) -> None:
        moldura = tk.Frame(
            pai, bg=tema.SUPERFICIE, highlightbackground=tema.BORDA, highlightthickness=1
        )
        moldura.pack(fill="both", expand=True)

        self._canvas = tk.Canvas(
            moldura, bg=tema.SUPERFICIE, highlightthickness=0, bd=0
        )
        self._canvas.pack(fill="both", expand=True)
        self._img_canvas = self._canvas.create_image(0, 0, anchor="center")

        self._barra_controles = tk.Frame(pai, bg=tema.FUNDO)
        self._barra_controles.pack(fill="x", pady=(10, 0))
        self._montar_controles(self._barra_controles)

    def _montar_controles(self, pai: tk.Frame) -> None:
        self._btn_iniciar = ttk.Button(
            pai, text="▶  Iniciar câmera", style="Sucesso.TButton", command=self.iniciar_camera
        )
        self._btn_iniciar.pack(side="left", padx=(0, 8))

        self._btn_pausar = ttk.Button(
            pai, text="⏸  Pausar", style="Alerta.TButton", command=self.alternar_pausa,
            state="disabled",
        )
        self._btn_pausar.pack(side="left", padx=(0, 8))

        self._btn_reiniciar = ttk.Button(
            pai, text="⟳  Reiniciar", style="Neutro.TButton", command=self.reiniciar_camera,
            state="disabled",
        )
        self._btn_reiniciar.pack(side="left", padx=(0, 8))

        self._btn_encerrar = ttk.Button(
            pai, text="⏹  Encerrar", style="Perigo.TButton",
            command=self.parar_camera, state="disabled",
        )
        self._btn_encerrar.pack(side="left", padx=(0, 16))

        self._btn_imagem = ttk.Button(
            pai, text="Analisar imagem", style="Primario.TButton",
            command=self.analisar_imagem,
        )
        self._btn_imagem.pack(side="left", padx=(0, 16))

        # Seletor de dispositivo ancorado a direita: continua visivel mesmo
        # quando a janela e estreitada ate o tamanho minimo.
        self._btn_procurar = ttk.Button(
            pai, text="Procurar", style="Neutro.TButton", command=self.procurar_cameras
        )
        self._btn_procurar.pack(side="right")

        self._cmb_camera = ttk.Combobox(
            pai,
            textvariable=self._var_camera,
            values=[f"Câmera {i}" for i in self._cameras],
            state="readonly",
            width=11,
        )
        self._cmb_camera.pack(side="right", padx=(0, 8))
        self._cmb_camera.bind("<<ComboboxSelected>>", self._ao_trocar_camera)

        ttk.Label(pai, text="Dispositivo:", style="Subtitulo.TLabel").pack(
            side="right", padx=(12, 6)
        )

    def _montar_painel(self, pai: tk.Frame) -> None:
        # ------------------------------------------- expressao predominante
        cartao = componentes.Cartao(pai, "Expressão predominante", self.fontes)
        cartao.pack(fill="x", padx=(0, 16))
        corpo = cartao.corpo

        linha = tk.Frame(corpo, bg=tema.SUPERFICIE)
        linha.pack(fill="x")
        self._icone = componentes.IconeExpressao(linha, self.fontes)
        self._icone.pack(side="left", padx=(0, 12))
        bloco = tk.Frame(linha, bg=tema.SUPERFICIE)
        bloco.pack(side="left", fill="x", expand=True)
        self._lbl_expressao = tk.Label(
            bloco, text="Aguardando", font=self.fontes.destaque_medio,
            bg=tema.SUPERFICIE, fg=tema.TEXTO_SECUNDARIO, anchor="w",
        )
        self._lbl_expressao.pack(fill="x")
        self._lbl_confianca = tk.Label(
            bloco, text="confiança —", font=self.fontes.pequeno,
            bg=tema.SUPERFICIE, fg=tema.TEXTO_FRACO, anchor="w",
        )
        self._lbl_confianca.pack(fill="x")

        self._medidor = componentes.MedidorConfianca(corpo, self.fontes)
        self._medidor.pack(fill="x", pady=(10, 4))

        self._lbl_aviso_conf = tk.Label(
            corpo,
            text="",
            font=self.fontes.pequeno,
            bg=tema.SUPERFICIE,
            fg=tema.ALERTA,
            wraplength=LARGURA_PAINEL - 90,
            justify="left",
            anchor="w",
        )
        self._lbl_aviso_conf.pack(fill="x")

        # ----------------------------------------------------------- grafico
        cartao_graf = componentes.Cartao(
            pai, "Pontuações das 7 categorias", self.fontes
        )
        cartao_graf.pack(fill="x", padx=(0, 16), pady=(10, 0))
        self._grafico = componentes.GraficoProbabilidades(
            cartao_graf.corpo, self.fontes
        )
        self._grafico.pack(fill="x")
        tk.Label(
            cartao_graf.corpo,
            text="Valores produzidos pela inferência do modelo (softmax dos logits).",
            font=self.fontes.micro,
            bg=tema.SUPERFICIE,
            fg=tema.TEXTO_FRACO,
            wraplength=LARGURA_PAINEL - 90,
            justify="left",
            anchor="w",
        ).pack(fill="x", pady=(8, 0))

        # ------------------------------------------------------- indicadores
        cartao_ind = componentes.Cartao(pai, "Indicadores em tempo real", self.fontes)
        cartao_ind.pack(fill="x", padx=(0, 16), pady=(10, 0))
        ci = cartao_ind.corpo
        self._ind = {
            "camera": componentes.LinhaIndicador(ci, "Câmera", self.fontes),
            "modelos": componentes.LinhaIndicador(ci, "Modelos neurais", self.fontes),
            "rostos": componentes.LinhaIndicador(ci, "Rostos detectados", self.fontes),
            "fps": componentes.LinhaIndicador(ci, "FPS", self.fontes),
            "inferencia": componentes.LinhaIndicador(ci, "Inferência", self.fontes),
            "deteccao": componentes.LinhaIndicador(ci, "Detecção", self.fontes),
            "estabilidade": componentes.LinhaIndicador(ci, "Estabilidade", self.fontes),
            "local": componentes.LinhaIndicador(ci, "Processamento", self.fontes),
        }
        self._ind["local"].definir("100% local", tema.SUCESSO)

        # ---------------------------------------------------------- ajustes
        cartao_aj = componentes.Cartao(pai, "Ajustes", self.fontes)
        cartao_aj.pack(fill="x", padx=(0, 16), pady=(10, 0))
        ca = cartao_aj.corpo

        topo = tk.Frame(ca, bg=tema.SUPERFICIE)
        topo.pack(fill="x")
        ttk.Label(topo, text="Limiar de confiança", style="CartaoFraco.TLabel").pack(
            side="left"
        )
        self._lbl_limiar = tk.Label(
            topo, text="50%", font=self.fontes.mono, bg=tema.SUPERFICIE, fg=tema.PRIMARIA
        )
        self._lbl_limiar.pack(side="right")
        ttk.Scale(
            ca, from_=0.10, to=0.95, variable=self._var_limiar, command=self._ao_mudar_limiar
        ).pack(fill="x", pady=(2, 8))

        ttk.Checkbutton(
            ca, text="Estabilidade temporal (suavização)", variable=self._var_suavizar,
            command=self._ao_mudar_ajuste,
        ).pack(fill="x")
        ttk.Checkbutton(
            ca, text="Espelhar imagem (modo espelho)", variable=self._var_espelhar,
            command=self._ao_mudar_ajuste,
        ).pack(fill="x")
        ttk.Checkbutton(
            ca, text="Classificar múltiplos rostos", variable=self._var_multiplos,
            command=self._ao_mudar_ajuste,
        ).pack(fill="x")

        passo = tk.Frame(ca, bg=tema.SUPERFICIE)
        passo.pack(fill="x", pady=(8, 0))
        ttk.Label(passo, text="Inferir a cada N quadros", style="CartaoFraco.TLabel").pack(
            side="left"
        )
        ttk.Combobox(
            passo,
            textvariable=self._var_passo,
            values=[1, 2, 3, 4],
            state="readonly",
            width=4,
        ).pack(side="right")
        self._var_passo.trace_add("write", lambda *_: self._ao_mudar_ajuste())

        self._ao_mudar_limiar()

    def _montar_rodape(self) -> None:
        rodape = tk.Frame(self, bg=tema.SUPERFICIE)
        rodape.pack(fill="x", side="bottom")
        tk.Frame(rodape, bg=tema.BORDA, height=1).pack(fill="x")

        linha1 = tk.Frame(rodape, bg=tema.SUPERFICIE)
        linha1.pack(fill="x", padx=18, pady=(8, 2))
        self._pt_status = tk.Canvas(
            linha1, width=10, height=10, bg=tema.SUPERFICIE, highlightthickness=0
        )
        self._pt_status.pack(side="left", padx=(0, 8))
        self._ponto = self._pt_status.create_oval(1, 1, 9, 9, fill=tema.NEUTRO, outline="")
        self._lbl_status = tk.Label(
            linha1, text=self._mensagem_status, font=self.fontes.pequeno,
            bg=tema.SUPERFICIE, fg=tema.TEXTO_SECUNDARIO, anchor="w",
        )
        self._lbl_status.pack(side="left", fill="x", expand=True)
        self._lbl_modelos = tk.Label(
            linha1,
            text="YuNet 2023mar · MobileFaceNet 2022july · ONNX / cv2.dnn (CPU)",
            font=self.fontes.micro, bg=tema.SUPERFICIE, fg=tema.TEXTO_FRACO,
        )
        self._lbl_modelos.pack(side="right")

        linha2 = tk.Frame(rodape, bg=tema.SUPERFICIE)
        linha2.pack(fill="x", padx=18, pady=(0, 10))
        tk.Label(
            linha2,
            text=config.AVISO_PRIVACIDADE,
            font=self.fontes.micro,
            bg=tema.SUPERFICIE,
            fg=tema.ALERTA,
            wraplength=820,
            justify="left",
            anchor="w",
        ).pack(side="left", fill="x", expand=True)
        tk.Label(
            linha2,
            text=(
                f"{config.ACADEMICO} · {config.EMAIL}\n"
                f"{config.CURSO} · {config.DISCIPLINA} · Prof. {config.PROFESSOR}"
            ),
            font=self.fontes.micro, bg=tema.SUPERFICIE, fg=tema.TEXTO_FRACO,
            justify="right",
        ).pack(side="right")

    def _atalhos(self) -> None:
        self.bind("<F11>", lambda _e: self.alternar_tela_cheia())
        self.bind("<Escape>", lambda _e: self._sair_tela_cheia())
        self.bind("<space>", lambda _e: self.alternar_pausa())
        self.bind("<F1>", lambda _e: self.abrir_sobre())
        self.bind("<Control-o>", lambda _e: self.analisar_imagem())
        self.bind("<Control-q>", lambda _e: self.encerrar())

    # ------------------------------------------------------------- ambiente
    def _verificar_ambiente_inicial(self) -> None:
        """RF01: diagnostico antes de qualquer uso da camera."""
        rel = ambiente.verificar_ambiente(conferir_hash=False)
        if rel.ok:
            self._ind["modelos"].definir("verificados", tema.SUCESSO)
            self._definir_status(
                "Ambiente verificado. Clique em “Iniciar câmera” para começar.",
                tema.SUCESSO,
            )
            self._mostrar_placa(
                [
                    "EXPRESSAO NEURAL",
                    "Clique em Iniciar camera para comecar",
                    "A camera so e acionada por acao sua",
                ]
            )
            return

        self._modo = "erro"
        self._ind["modelos"].definir("com problema", tema.ERRO)
        problemas = "\n\n".join(f"• {i.nome}: {i.detalhe}" for i in rel.problemas_criticos)
        self._definir_status("Ambiente incompleto — veja as instruções.", tema.ERRO)
        self._mostrar_placa(
            ["AMBIENTE INCOMPLETO", "Execute configurar.bat", "Detalhes na mensagem exibida"]
        )
        self.after(
            250,
            lambda: messagebox.showerror(
                "Ambiente incompleto",
                "O sistema não pode iniciar porque faltam recursos:\n\n"
                f"{problemas}\n\n"
                "Como resolver: feche a aplicação e execute configurar.bat "
                "(precisa de internet apenas nesta etapa). Depois use verificar.bat "
                "para confirmar e iniciar.bat para abrir novamente.",
                parent=self,
            ),
        )

    def _garantir_pipeline(self) -> bool:
        """Carrega os modelos sob demanda (secao 8: carregamento tardio)."""
        if self._pipeline is not None:
            return True
        if self._carregando_modelos:
            return False
        self._carregando_modelos = True
        self._definir_status("Carregando modelos neurais…", tema.PRIMARIA)
        self._ind["modelos"].definir("carregando…", tema.ALERTA)
        self.update_idletasks()
        try:
            self._pipeline = PipelineExpressao(self.ajustes)
        except ErroModelo as exc:
            self._carregando_modelos = False
            self._ind["modelos"].definir("falha", tema.ERRO)
            self._definir_status(str(exc), tema.ERRO)
            messagebox.showerror("Modelos neurais", str(exc), parent=self)
            return False
        except Exception as exc:  # noqa: BLE001
            self._carregando_modelos = False
            self._ind["modelos"].definir("falha", tema.ERRO)
            self._definir_status(f"Falha ao carregar modelos: {exc}", tema.ERRO)
            messagebox.showerror(
                "Modelos neurais",
                f"Falha inesperada ao carregar os modelos:\n\n{exc}",
                parent=self,
            )
            return False
        self._carregando_modelos = False
        self._ind["modelos"].definir(
            f"ativos ({self._pipeline.ms_aquecimento:.0f} ms aquec.)", tema.SUCESSO
        )
        return True

    # ------------------------------------------------------------- camera
    def iniciar_camera(self) -> None:
        if self._modo in ("ativo", "pausado"):
            return
        if not self._garantir_pipeline():
            return
        assert self._pipeline is not None
        self._pipeline.aplicar_ajustes(self.ajustes)
        self._pipeline.reiniciar_estado()

        self._encerrar_trabalhador()
        if self._camera_ainda_ocupada():
            # Reabrir agora falharia com "dispositivo em uso" e confundiria o
            # usuário: a câmera está presa na captura anterior, não ocupada por
            # outro programa.
            self._definir_status(
                "Aguardando a câmera anterior ser liberada. Tente novamente em "
                "alguns segundos.",
                tema.ALERTA,
            )
            self._mostrar_placa(
                ["LIBERANDO A CAMERA", "Aguarde e clique em Iniciar camera novamente"]
            )
            self._modo = "parado"
            self._atualizar_botoes()
            return

        self._definir_status("Abrindo câmera…", tema.PRIMARIA)
        self._mostrar_placa(["ABRINDO CAMERA", "Aguarde alguns instantes"])
        self.update_idletasks()

        self._trabalhador = TrabalhadorCamera(self._pipeline, self.ajustes)
        self._trabalhador.start()
        self._modo = "ativo"
        self._atualizar_botoes()
        self.after(1200, self._conferir_abertura)

    def _conferir_abertura(self) -> None:
        t = self._trabalhador
        if t is None:
            return
        if t.erro:
            self._modo = "erro"
            self._atualizar_botoes()
            self._ind["camera"].definir("indisponível", tema.ERRO)
            self._definir_status(t.erro, tema.ERRO)
            self._mostrar_placa(
                ["CAMERA INDISPONIVEL", "Use Analisar imagem como alternativa"]
            )
            messagebox.showerror("Câmera", t.erro, parent=self)
            return
        if not t.ativo:
            self._modo = "erro"
            self._atualizar_botoes()
            self._definir_status("A captura foi encerrada inesperadamente.", tema.ERRO)
            return
        self._definir_status("Câmera ativa — classificação em tempo real.", tema.SUCESSO)

    def alternar_pausa(self) -> None:
        t = self._trabalhador
        if t is None or not t.ativo or self._modo not in ("ativo", "pausado"):
            return
        if t.pausado:
            t.retomar()
            self._modo = "ativo"
            self._definir_status("Captura retomada.", tema.SUCESSO)
        else:
            t.pausar()
            self._modo = "pausado"
            self._definir_status("Captura pausada — a câmera continua reservada.", tema.ALERTA)
        self._atualizar_botoes()

    def reiniciar_camera(self) -> None:
        if self._trabalhador is None:
            self.iniciar_camera()
            return
        self._definir_status("Reiniciando captura…", tema.PRIMARIA)
        self._encerrar_trabalhador()
        if self._pipeline is not None:
            self._pipeline.reiniciar_estado()
        self._limpar_resultados()
        self.after(300, self.iniciar_camera)

    def parar_camera(self) -> None:
        self._encerrar_trabalhador()
        self._modo = "parado"
        self._limpar_resultados()
        self._ind["camera"].definir("desligada", tema.NEUTRO)
        self._ind["fps"].definir("—")
        self._definir_status("Câmera encerrada e liberada.", tema.TEXTO_SECUNDARIO)
        self._mostrar_placa(["CAMERA ENCERRADA", "Clique em Iniciar camera para retomar"])
        self._atualizar_botoes()

    def _encerrar_trabalhador(self) -> bool:
        """Encerra a captura. Devolve False se a thread não terminou a tempo."""
        if self._trabalhador is None:
            return True
        parou = self._trabalhador.parar()
        if not parou:
            # A thread ficou presa em cap.read(): a câmera ainda está ocupada.
            # Guardamos a referência para não tentar reabrir o dispositivo.
            self._trabalhador_preso = self._trabalhador
        self._trabalhador = None
        return parou

    def _camera_ainda_ocupada(self) -> bool:
        preso = getattr(self, "_trabalhador_preso", None)
        if preso is None:
            return False
        if preso.is_alive():
            return True
        self._trabalhador_preso = None  # a thread finalmente liberou o dispositivo
        return False

    def procurar_cameras(self) -> None:
        """Varredura de dispositivos em thread propria (operacao lenta)."""
        self._btn_procurar.configure(state="disabled")
        self._definir_status("Procurando câmeras disponíveis…", tema.PRIMARIA)
        estava_ativa = self._modo in ("ativo", "pausado")
        if estava_ativa:
            self._encerrar_trabalhador()
            self._modo = "parado"
            self._atualizar_botoes()

        def tarefa() -> None:
            encontrados = listar_cameras(5)
            self.after(0, lambda: self._cameras_encontradas(encontrados))

        threading.Thread(target=tarefa, name="BuscaCameras", daemon=True).start()

    def _cameras_encontradas(self, encontrados: list[int]) -> None:
        self._btn_procurar.configure(state="normal")
        if not encontrados:
            self._cameras = []
            self._cmb_camera.configure(values=[])
            self._definir_status(
                "Nenhuma câmera encontrada. Use “Analisar imagem” como contingência.",
                tema.ERRO,
            )
            messagebox.showwarning(
                "Câmera",
                "Nenhuma câmera foi encontrada.\n\nVerifique a conexão, feche "
                "programas que possam estar usando a câmera e confirme a permissão "
                "de câmera nas configurações de privacidade do Windows.\n\n"
                "A função “Analisar imagem” continua disponível.",
                parent=self,
            )
            return
        self._cameras = encontrados
        self._cmb_camera.configure(values=[f"Câmera {i}" for i in encontrados])
        if self.ajustes.indice_camera not in encontrados:
            self.ajustes.indice_camera = encontrados[0]
        self._var_camera.set(f"Câmera {self.ajustes.indice_camera}")
        self._definir_status(
            f"{len(encontrados)} câmera(s) encontrada(s): "
            f"{', '.join(str(i) for i in encontrados)}.",
            tema.SUCESSO,
        )

    def _ao_trocar_camera(self, _evento=None) -> None:
        try:
            indice = int(self._var_camera.get().split()[-1])
        except (ValueError, IndexError):
            return
        if indice == self.ajustes.indice_camera and self._modo in ("ativo", "pausado"):
            return
        self.ajustes.indice_camera = indice
        if self._trabalhador is not None and self._trabalhador.ativo:
            self._trabalhador.trocar_camera(indice)
            self._definir_status(f"Trocando para a câmera {indice}…", tema.PRIMARIA)
            self.after(1500, self._conferir_abertura)

    # ------------------------------------------------------ imagem estatica
    def analisar_imagem(self) -> None:
        """RF09: contingencia para falha de webcam na apresentacao."""
        if not self._garantir_pipeline():
            return
        tipos = [
            ("Imagens", "*.jpg *.jpeg *.png *.bmp *.webp"),
            ("Todos os arquivos", "*.*"),
        ]
        caminho = filedialog.askopenfilename(
            title="Selecionar imagem para análise", filetypes=tipos, parent=self
        )
        if not caminho:
            return

        try:
            imagem = imagem_estatica.carregar(caminho)
        except imagem_estatica.ImagemInvalida as exc:
            self._definir_status(str(exc), tema.ERRO)
            messagebox.showerror("Imagem inválida", str(exc), parent=self)
            return

        # Encerra a captura SEMPRE que houver thread viva -- inclusive em modo
        # "erro", quando o trabalhador continua tentando reconectar. Sem isso,
        # os modelos seriam usados por duas threads ao mesmo tempo.
        self._encerrar_trabalhador()
        self._modo = "imagem"
        self._atualizar_botoes()

        imagem = imagem_estatica.redimensionar_para_analise(imagem)
        assert self._pipeline is not None
        self._pipeline.reiniciar_estado()
        resultado = self._pipeline.processar(imagem, suavizar=False)

        self._ind["camera"].definir("imagem local", tema.PRIMARIA)
        self._ind["fps"].definir("n/a (imagem)")
        self._quadro_exibido = imagem
        self._ultimo_resultado = resultado
        self._fps = 0.0
        self._renderizar(imagem, resultado)
        self._atualizar_paineis(resultado)

        nome = Path(caminho).name
        if resultado.erro:
            self._definir_status(f"{nome}: {resultado.erro}", tema.ERRO)
        elif resultado.quantidade_rostos == 0:
            self._definir_status(
                f"{nome}: nenhum rosto detectado — nenhuma classificação foi feita.",
                tema.ALERTA,
            )
        else:
            self._definir_status(
                f"{nome}: {resultado.quantidade_rostos} rosto(s) classificado(s) "
                f"em {resultado.tempo_total_ms:.0f} ms.",
                tema.SUCESSO,
            )

    # --------------------------------------------------------------- ajustes
    def _ao_mudar_limiar(self, _valor=None) -> None:
        valor = float(self._var_limiar.get())
        self.ajustes.limiar_confianca = valor
        self._lbl_limiar.configure(text=f"{valor * 100:.0f}%")
        if self._modo == "imagem" and self._ultimo_resultado is not None:
            self._renderizar(self._quadro_exibido, self._ultimo_resultado)
            self._atualizar_paineis(self._ultimo_resultado)

    def _ao_mudar_ajuste(self) -> None:
        self.ajustes.suavizacao_ativa = bool(self._var_suavizar.get())
        self.ajustes.espelhar_imagem = bool(self._var_espelhar.get())
        self.ajustes.multiplos_rostos = bool(self._var_multiplos.get())
        try:
            self.ajustes.inferir_a_cada = max(1, min(4, int(self._var_passo.get())))
        except (tk.TclError, ValueError):
            pass
        if self._pipeline is not None:
            self._pipeline.aplicar_ajustes(self.ajustes)

    # ---------------------------------------------------------- apresentacao
    def alternar_tela_cheia(self) -> None:
        self._tela_cheia = not self._tela_cheia
        self.attributes("-fullscreen", self._tela_cheia)
        self._btn_tela_cheia.configure(
            text="Sair da apresentação (Esc)" if self._tela_cheia
            else "Apresentação (F11)"
        )

    def _sair_tela_cheia(self) -> None:
        if self._tela_cheia:
            self.alternar_tela_cheia()

    def abrir_sobre(self) -> tk.Toplevel:
        return sobre.abrir(self, self.fontes)

    # ------------------------------------------------------------ laco de UI
    def _agendar_tick(self) -> None:
        self._tick = self.after(INTERVALO_TICK_MS, self._passo)

    def _passo(self) -> None:
        try:
            self._atualizar_quadro()
        except Exception as exc:  # a interface nunca deve cair por um quadro ruim
            self._definir_status(f"Falha ao atualizar a tela: {exc}", tema.ERRO)
        finally:
            self._agendar_tick()

    def _atualizar_quadro(self) -> None:
        t = self._trabalhador
        if t is None:
            return

        erro = t.erro
        if erro:
            self._ind["camera"].definir("com falha", tema.ERRO)
            if self._mensagem_status != erro:
                self._definir_status(erro, tema.ERRO)
        elif t.pausado:
            self._ind["camera"].definir("pausada", tema.ALERTA)
        elif t.ativo:
            self._ind["camera"].definir(
                f"ativa (dispositivo {t.indice_camera})", tema.SUCESSO
            )

        pronto = t.obter()
        if pronto is None:
            return

        self._fps = pronto.fps
        self._quadro_exibido = pronto.quadro
        self._ultimo_resultado = pronto.resultado
        self._renderizar(pronto.quadro, pronto.resultado)
        self._atualizar_paineis(pronto.resultado)

    def _renderizar(self, quadro: np.ndarray | None, resultado: ResultadoQuadro) -> None:
        if quadro is None or quadro.size == 0:
            return
        largura = max(self._canvas.winfo_width(), 160)
        altura = max(self._canvas.winfo_height(), 120)
        anotado = desenho.anotar(quadro, resultado, self.ajustes.limiar_confianca)
        ajustado = desenho.ajustar_para_area(anotado, largura, altura)
        rgb = cv.cvtColor(ajustado, cv.COLOR_BGR2RGB)
        self._foto = ImageTk.PhotoImage(Image.fromarray(rgb))
        self._canvas.itemconfigure(self._img_canvas, image=self._foto)
        self._canvas.coords(self._img_canvas, largura // 2, altura // 2)

    def _mostrar_placa(self, linhas: list[str]) -> None:
        largura = max(self._canvas.winfo_width(), 640)
        altura = max(self._canvas.winfo_height(), 360)
        placa = desenho.placa_mensagem(largura, altura, linhas)
        rgb = cv.cvtColor(placa, cv.COLOR_BGR2RGB)
        self._foto = ImageTk.PhotoImage(Image.fromarray(rgb))
        self._canvas.itemconfigure(self._img_canvas, image=self._foto)
        self._canvas.coords(self._img_canvas, largura // 2, altura // 2)

    def _atualizar_paineis(self, resultado: ResultadoQuadro) -> None:
        limiar = self.ajustes.limiar_confianca
        self._ind["rostos"].definir(
            str(resultado.quantidade_rostos),
            tema.SUCESSO if resultado.quantidade_rostos else tema.TEXTO_FRACO,
        )
        if self._modo != "imagem":
            self._ind["fps"].definir(f"{self._fps:.1f}" if self._fps else "—")
        self._ind["inferencia"].definir(
            f"{resultado.tempo_inferencia_ms:.1f} ms"
            if resultado.tempo_inferencia_ms
            else "—"
        )
        self._ind["deteccao"].definir(
            f"{resultado.tempo_deteccao_ms:.1f} ms"
            if resultado.tempo_deteccao_ms
            else "—"
        )

        alvo = resultado.principal
        if alvo is None:
            self._limpar_resultados(manter_contadores=True)
            self._status_da_captura(resultado)
            return

        indice = alvo.indice
        confiavel = alvo.confiavel(limiar)
        cor = config.CORES_CLASSES[indice] if confiavel else tema.NEUTRO

        self._icone.definir(indice if confiavel else None)
        self._lbl_expressao.configure(
            text=alvo.rotulo if confiavel else "Indefinida", fg=cor
        )
        self._lbl_confianca.configure(
            text=f"confiança {alvo.confianca * 100:.1f}%".replace(".", ",")
        )
        self._medidor.atualizar(alvo.confianca, limiar)
        self._lbl_aviso_conf.configure(
            text="" if confiavel else config.MSG_BAIXA_CONFIANCA
        )
        self._grafico.atualizar(alvo.probabilidades, indice if confiavel else -1)
        self._ind["estabilidade"].definir(
            f"{alvo.estabilidade * 100:.0f}%" if self.ajustes.suavizacao_ativa else "desligada",
            tema.SUCESSO if alvo.estabilidade > 0.7 else tema.ALERTA,
        )
        self._status_da_captura(resultado)

    def _status_da_captura(self, resultado: ResultadoQuadro) -> None:
        """Mantém o rodapé coerente com o quadro ATUAL.

        Antes, o status só era escrito quando havia mais de um rosto e nunca
        voltava atrás: uma detecção falsa momentânea de 2 rostos deixava o
        rodapé congelado contradizendo o painel lateral.
        """
        if self._modo not in ("ativo", "pausado"):
            return  # imagem estática e erros têm mensagens próprias
        t = self._trabalhador
        if t is not None and t.erro:
            return  # a mensagem de erro tem prioridade
        if t is not None and t.pausado:
            self._definir_status(
                "Captura pausada — a câmera continua reservada.", tema.ALERTA
            )
            return

        limiar = self.ajustes.limiar_confianca
        quantidade = resultado.quantidade_rostos
        if quantidade == 0:
            self._definir_status(
                "Câmera ativa — nenhum rosto no enquadramento.", tema.TEXTO_SECUNDARIO
            )
            return
        if quantidade == 1:
            self._definir_status(
                "Câmera ativa — classificação em tempo real.", tema.SUCESSO
            )
            return

        alvo = resultado.principal
        extras = ", ".join(r.rotulo_exibicao(limiar) for r in resultado.rostos[1:4])
        if quantidade > 4:
            extras += ", …"
        self._definir_status(
            f"{quantidade} rostos — principal: {alvo.rotulo_exibicao(limiar)}; "
            f"outros: {extras}.",
            tema.SUCESSO,
        )

    def _limpar_resultados(self, manter_contadores: bool = False) -> None:
        """RF10: ao perder o rosto, nenhum resultado antigo permanece na tela."""
        self._icone.definir(None)
        self._lbl_expressao.configure(text="Nenhum rosto", fg=tema.TEXTO_SECUNDARIO)
        self._lbl_confianca.configure(text="confiança —")
        self._lbl_aviso_conf.configure(text="")
        self._medidor.atualizar(0.0, self.ajustes.limiar_confianca)
        self._grafico.atualizar(None, -1)
        self._ind["estabilidade"].definir("—", tema.TEXTO_FRACO)
        if not manter_contadores:
            self._ind["rostos"].definir("0", tema.TEXTO_FRACO)
            self._ind["inferencia"].definir("—")
            self._ind["deteccao"].definir("—")

    def _definir_status(self, mensagem: str, cor: str) -> None:
        if mensagem == self._mensagem_status and cor == self._cor_status:
            return
        self._mensagem_status = mensagem
        self._cor_status = cor
        self._lbl_status.configure(text=mensagem, fg=cor)
        self._pt_status.itemconfigure(self._ponto, fill=cor)

    def _atualizar_botoes(self) -> None:
        ativo = self._modo in ("ativo", "pausado")
        t = self._trabalhador
        self._btn_iniciar.configure(state="disabled" if ativo else "normal")
        self._btn_pausar.configure(state="normal" if ativo else "disabled")
        self._btn_pausar.configure(
            text="▶  Retomar" if (t is not None and t.pausado) else "⏸  Pausar"
        )
        self._btn_reiniciar.configure(state="normal" if ativo else "disabled")
        self._btn_encerrar.configure(state="normal" if ativo else "disabled")

    # ------------------------------------------------------------ encerramento
    def encerrar(self) -> None:
        if self._tick is not None:
            try:
                self.after_cancel(self._tick)
            except tk.TclError:
                pass
            self._tick = None
        self._encerrar_trabalhador()
        if self._pipeline is not None:
            self._pipeline.liberar()
            self._pipeline = None
        self._foto = None
        self.ajustes.salvar()
        try:
            self.destroy()
        except tk.TclError:
            pass


def executar() -> int:
    """Ponto de entrada da interface. Devolve o codigo de saida do processo."""
    tema.ativar_consciencia_dpi()
    janela = JanelaPrincipal()
    try:
        janela.mainloop()
    except KeyboardInterrupt:
        janela.encerrar()
    return 0





