"""Coleta um conjunto de amostras ROTULADAS a partir da webcam.

Cada amostra e o recorte facial 112x112 ja alinhado -- exatamente a entrada que
o classificador recebe. Salvas em dados/amostras/<classe>/.

O rotulo verdadeiro e a expressao SOLICITADA ao voluntario. Isso deve ser
declarado no relatorio: sao expressoes encenadas, nao espontaneas.

Uso:
    .venv\\Scripts\\python.exe scripts\\coletar_amostras.py
    .venv\\Scripts\\python.exe scripts\\coletar_amostras.py --por-classe 40

As imagens contem rosto humano e NAO sao versionadas (ver .gitignore).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
os.environ["EXPRESSAO_NEURAL_SEM_PERSISTENCIA"] = "1"

import cv2 as cv  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageTk  # noqa: E402

from expressao_neural.core.alinhamento import ErroAlinhamento, alinhar_rosto  # noqa: E402
from expressao_neural.core.camera import ErroCamera, _abrir  # noqa: E402
from expressao_neural.core.detector import DetectorFacial, ErroModelo  # noqa: E402
from expressao_neural.ui import tema  # noqa: E402

DESTINO = RAIZ / "dados" / "amostras"

# Orientacao pratica por classe. Expressoes "faceis" primeiro, para o
# voluntario pegar o ritmo antes das dificeis.
ORIENTACAO = {
    "Neutra": "Rosto em repouso, sem sorrir e sem franzir.",
    "Alegria": "Sorriso natural. Varie entre leve e aberto.",
    "Surpresa": "Sobrancelhas levantadas, boca levemente aberta.",
    "Tristeza": "Cantos da boca para baixo, sobrancelhas caídas.",
    "Raiva": "Sobrancelhas franzidas e juntas, lábios tensos.",
    "Nojo": "Nariz enrugado, lábio superior levantado.",
    "Medo": "Olhos arregalados, sobrancelhas erguidas e juntas.",
}
ORDEM = ["Neutra", "Alegria", "Surpresa", "Tristeza", "Raiva", "Nojo", "Medo"]


class Coletor(tk.Tk):
    def __init__(self, por_classe: int, indice_camera: int) -> None:
        super().__init__()
        self.title("Coleta de amostras — Expressão Neural")
        self.configure(bg=tema.FUNDO)
        self.geometry("1100x760")
        self.protocol("WM_DELETE_WINDOW", self.encerrar)
        self.fontes = tema.aplicar(self)

        self.por_classe = por_classe
        try:
            self.detector = DetectorFacial(limiar_confianca=0.80)
            self.cap = _abrir(indice_camera)
        except Exception:
            # A janela Tk ja existe: destrui-la evita deixar uma janela orfa
            # na tela quando a camera ou o modelo nao estao disponiveis.
            self.destroy()
            raise
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

        self.classe_atual = 0
        self.coletando = False
        self.salvas: dict[str, int] = {c: 0 for c in ORDEM}
        self._foto = None
        self._ultimo_salvo = 0.0
        self._encerrado = False

        self._montar()
        self._contar_existentes()
        self._atualizar_textos()
        self.after(30, self._tick)

    # ------------------------------------------------------------------ layout
    def _montar(self) -> None:
        topo = tk.Frame(self, bg=tema.SUPERFICIE)
        topo.pack(fill="x")
        interno = tk.Frame(topo, bg=tema.SUPERFICIE)
        interno.pack(fill="x", padx=20, pady=14)
        tk.Label(
            interno, text="Coleta de amostras rotuladas",
            font=self.fontes.destaque_medio, bg=tema.SUPERFICIE, fg=tema.TEXTO,
        ).pack(anchor="w")
        tk.Label(
            interno,
            text="As imagens ficam apenas no seu computador e não são versionadas.",
            font=self.fontes.pequeno, bg=tema.SUPERFICIE, fg=tema.TEXTO_SECUNDARIO,
        ).pack(anchor="w")

        corpo = tk.Frame(self, bg=tema.FUNDO)
        corpo.pack(fill="both", expand=True, padx=16, pady=12)

        self.canvas = tk.Canvas(corpo, bg=tema.SUPERFICIE, highlightthickness=0,
                                width=640, height=480)
        self.canvas.pack(side="left")
        self.img_id = self.canvas.create_image(0, 0, anchor="center")

        lado = tk.Frame(corpo, bg=tema.FUNDO)
        lado.pack(side="left", fill="both", expand=True, padx=(16, 0))

        self.lbl_classe = tk.Label(
            lado, text="", font=(self.fontes.familia, 26, "bold"),
            bg=tema.FUNDO, fg=tema.PRIMARIA, anchor="w",
        )
        self.lbl_classe.pack(fill="x")
        self.lbl_orientacao = tk.Label(
            lado, text="", font=self.fontes.corpo, bg=tema.FUNDO,
            fg=tema.TEXTO, wraplength=380, justify="left", anchor="w",
        )
        self.lbl_orientacao.pack(fill="x", pady=(6, 14))
        self.lbl_progresso = tk.Label(
            lado, text="", font=self.fontes.mono_bold, bg=tema.FUNDO,
            fg=tema.TEXTO_SECUNDARIO, anchor="w",
        )
        self.lbl_progresso.pack(fill="x")
        self.lbl_estado = tk.Label(
            lado, text="", font=self.fontes.mono, bg=tema.FUNDO,
            fg=tema.TEXTO_FRACO, anchor="w",
        )
        self.lbl_estado.pack(fill="x", pady=(8, 16))

        self.btn = tk.Button(
            lado, text="▶  COLETAR ESTA EXPRESSÃO", command=self.alternar,
            font=(self.fontes.familia, 13, "bold"), bg=tema.SUCESSO, fg="#06121f",
            relief="flat", cursor="hand2", padx=16, pady=12, borderwidth=0,
        )
        self.btn.pack(fill="x")

        linha = tk.Frame(lado, bg=tema.FUNDO)
        linha.pack(fill="x", pady=(10, 0))
        tk.Button(
            linha, text="Pular classe", command=self.proxima,
            font=self.fontes.corpo, bg=tema.SUPERFICIE_ALTA, fg=tema.TEXTO,
            relief="flat", cursor="hand2", padx=12, pady=10, borderwidth=0,
        ).pack(side="left", padx=(0, 8))
        tk.Button(
            linha, text="Encerrar", command=self.encerrar,
            font=self.fontes.corpo, bg=tema.SUPERFICIE_ALTA, fg=tema.ERRO,
            relief="flat", cursor="hand2", padx=12, pady=10, borderwidth=0,
        ).pack(side="left")

        tk.Label(
            lado,
            text=("Dica: varie levemente a pose, a distância e a intensidade "
                  "durante a coleta. Variedade no conjunto de treino melhora a "
                  "generalização."),
            font=self.fontes.pequeno, bg=tema.FUNDO, fg=tema.TEXTO_FRACO,
            wraplength=380, justify="left",
        ).pack(fill="x", pady=(18, 0))

        self.bind("<space>", lambda _e: self.alternar())
        self.bind("<Escape>", lambda _e: self.encerrar())

    # ----------------------------------------------------------------- estado
    def _contar_existentes(self) -> None:
        for classe in ORDEM:
            pasta = DESTINO / classe
            if pasta.is_dir():
                self.salvas[classe] = len(list(pasta.glob("*.png")))

    @property
    def classe(self) -> str:
        return ORDEM[self.classe_atual]

    def _atualizar_textos(self) -> None:
        c = self.classe
        self.lbl_classe.configure(text=c.upper())
        self.lbl_orientacao.configure(text=ORIENTACAO[c])
        total = sum(self.salvas.values())
        self.lbl_progresso.configure(
            text=f"{c}: {self.salvas[c]}/{self.por_classe}     "
                 f"total: {total}/{self.por_classe * len(ORDEM)}"
        )

    def alternar(self) -> None:
        self.coletando = not self.coletando
        self.btn.configure(
            text="⏸  PAUSAR COLETA" if self.coletando else "▶  COLETAR ESTA EXPRESSÃO",
            bg=tema.ALERTA if self.coletando else tema.SUCESSO,
        )

    def proxima(self) -> None:
        self.coletando = False
        self.btn.configure(text="▶  COLETAR ESTA EXPRESSÃO", bg=tema.SUCESSO)
        if self.classe_atual + 1 >= len(ORDEM):
            self.encerrar()
            return
        self.classe_atual += 1
        self._atualizar_textos()

    # ------------------------------------------------------------------- laco
    def _tick(self) -> None:
        if self._encerrado:
            return
        try:
            ok, quadro = self.cap.read()
            if ok and quadro is not None:
                quadro = cv.flip(quadro, 1)
                self._processar(quadro)
        except Exception as exc:  # noqa: BLE001
            self.lbl_estado.configure(text=f"erro: {exc}", fg=tema.ERRO)
        self.after(30, self._tick)

    def _processar(self, quadro: np.ndarray) -> None:
        rostos = self.detector.detectar(quadro)
        exibir = quadro.copy()

        if not rostos:
            self.lbl_estado.configure(text="nenhum rosto detectado", fg=tema.TEXTO_FRACO)
        else:
            r = rostos[0]
            cor = (74, 222, 128) if self.coletando else (200, 200, 200)
            cv.rectangle(exibir, (r.x, r.y), (r.x + r.largura, r.y + r.altura), cor, 2)

            if self.coletando and self.salvas[self.classe] < self.por_classe:
                agora = time.time()
                # Intervalo entre amostras: evita salvar dezenas de quadros
                # praticamente identicos, que nao acrescentam informacao.
                if agora - self._ultimo_salvo >= 0.25:
                    self._salvar(quadro, r)
                    self._ultimo_salvo = agora
            elif self.coletando:
                self.lbl_estado.configure(
                    text=f"{self.classe}: meta atingida", fg=tema.SUCESSO
                )
                self.proxima()
                if self._encerrado:
                    return  # ultima classe concluida: a janela ja foi destruida

        rgb = cv.cvtColor(exibir, cv.COLOR_BGR2RGB)
        self._foto = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.itemconfigure(self.img_id, image=self._foto)
        self.canvas.coords(self.img_id, 320, 240)

    def _salvar(self, quadro: np.ndarray, rosto) -> None:
        try:
            recorte = alinhar_rosto(quadro, rosto.pontos)
        except ErroAlinhamento:
            return
        pasta = DESTINO / self.classe
        pasta.mkdir(parents=True, exist_ok=True)
        carimbo = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        caminho = pasta / f"{self.classe}_{carimbo}.png"
        ok, buf = cv.imencode(".png", recorte)
        if not ok:
            return
        caminho.write_bytes(buf.tobytes())
        self.salvas[self.classe] += 1
        self._atualizar_textos()
        self.lbl_estado.configure(
            text=f"amostra {self.salvas[self.classe]} salva", fg=tema.SUCESSO
        )

    def encerrar(self) -> None:
        self._encerrado = True
        try:
            self.cap.release()
        except Exception:  # noqa: BLE001
            pass
        try:
            self.destroy()
        except tk.TclError:
            pass


def main() -> int:
    ap = argparse.ArgumentParser(description="Coleta amostras rotuladas pela webcam.")
    ap.add_argument("--por-classe", type=int, default=30,
                    help="amostras por expressão (padrão: 30)")
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()

    print("=" * 72)
    print(" Expressão Neural — coleta de amostras rotuladas")
    print("=" * 72)
    print(f"\n Meta: {args.por_classe} amostras por expressão "
          f"({args.por_classe * len(ORDEM)} no total).")
    print(" As imagens ficam em dados/amostras/ e NÃO são versionadas.")
    print("\n ESPAÇO inicia/pausa a coleta · ESC encerra\n")

    tema.ativar_consciencia_dpi()
    try:
        app = Coletor(args.por_classe, args.camera)
    except (ErroCamera, ErroModelo) as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        return 1
    app.mainloop()

    total = 0
    print("\n" + "=" * 72)
    print(" Amostras disponíveis:")
    for classe in ORDEM:
        pasta = DESTINO / classe
        n = len(list(pasta.glob("*.png"))) if pasta.is_dir() else 0
        total += n
        marca = "OK " if n >= 10 else "!! "
        print(f"   {marca}{classe:10s} {n:4d}")
    print(f"\n Total: {total} amostras em {DESTINO}")
    if total >= 70:
        print("\n Próximo passo:")
        print("   .venv\\Scripts\\python.exe scripts\\treinar_calibracao.py")
    else:
        print("\n Colete mais amostras antes de treinar (mínimo recomendado: 10 por classe).")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
