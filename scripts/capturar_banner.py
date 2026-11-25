"""Captura assistida das telas do sistema para o banner academico.

Abre a aplicacao REAL em 1920x1080, acompanha a inferencia ao vivo e salva a
tela quando voce confirmar. Nenhuma montagem: o que e salvo e exatamente o que
esta na tela, com resultado produzido pela rede neural.

Uso:
    .venv\\Scripts\\python.exe scripts\\capturar_banner.py

Roteiro automatico:
    Figura 01  tela principal                 (nao exige rosto)
    Figura 02  expressao de ALEGRIA           (exige voce diante da webcam)
    Figura 03  expressao de SURPRESA          (exige voce diante da webcam)
    Figura 04  expressao NEUTRA               (exige voce diante da webcam)
    Figura 05  painel de classificacao        (recorte do quadro atual)
    Figura 06  janela "Sobre o Projeto"       (nao exige rosto)

As imagens vao para docs/banner/02_capturas/.
"""
from __future__ import annotations

import os
import sys
import time
import tkinter as tk
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
os.environ["EXPRESSAO_NEURAL_SEM_PERSISTENCIA"] = "1"

from PIL import ImageGrab  # noqa: E402

from expressao_neural import config  # noqa: E402
from expressao_neural.ui import sobre, tema  # noqa: E402
from expressao_neural.ui.janela import JanelaPrincipal  # noqa: E402

DESTINO = RAIZ / "docs" / "banner" / "02_capturas"
LARGURA, ALTURA = 1920, 1080

# (arquivo, titulo, instrucao, classe_alvo_ou_None)
ROTEIRO = [
    ("figura_01_tela_principal.png", "Figura 01 — Tela principal",
     "Deixe a tela como está. Se a câmera estiver ligada, ótimo.", None),
    ("figura_02_alegria.png", "Figura 02 — Expressão de ALEGRIA",
     "Sorria abertamente, mostrando os dentes, olhando para a câmera.", "Alegria"),
    ("figura_03_surpresa.png", "Figura 03 — Expressão de SURPRESA",
     "Arregale os olhos e abra a boca, sobrancelhas bem levantadas.", "Surpresa"),
    ("figura_04_neutra.png", "Figura 04 — Expressão NEUTRA",
     "Relaxe o rosto completamente, sem sorrir e sem franzir a testa.", "Neutra"),
    ("figura_05_painel_classificacao.png", "Figura 05 — Painel de classificação",
     "Mantenha o rosto enquadrado; será salvo um recorte do painel lateral.", None),
    ("figura_06_sobre_o_projeto.png", "Figura 06 — Janela Sobre o Projeto",
     "A janela será aberta automaticamente.", None),
]


class Assistente(tk.Toplevel):
    """Painel flutuante com a instrucao da captura atual."""

    def __init__(self, pai) -> None:
        super().__init__(pai)
        self.title("Roteiro de captura")
        self.configure(bg=tema.SUPERFICIE)
        self.attributes("-topmost", True)
        self.geometry("520x250+40+40")
        self.protocol("WM_DELETE_WINDOW", lambda: None)

        self.lbl_titulo = tk.Label(
            self, text="", font=("Segoe UI", 15, "bold"), bg=tema.SUPERFICIE,
            fg=tema.PRIMARIA, wraplength=480, justify="left",
        )
        self.lbl_titulo.pack(anchor="w", padx=18, pady=(16, 4))
        self.lbl_instr = tk.Label(
            self, text="", font=("Segoe UI", 11), bg=tema.SUPERFICIE,
            fg=tema.TEXTO, wraplength=480, justify="left",
        )
        self.lbl_instr.pack(anchor="w", padx=18)
        self.lbl_estado = tk.Label(
            self, text="", font=("Consolas", 11, "bold"), bg=tema.SUPERFICIE,
            fg=tema.TEXTO_SECUNDARIO, wraplength=480, justify="left",
        )
        self.lbl_estado.pack(anchor="w", padx=18, pady=(12, 0))
        self.lbl_ajuda = tk.Label(
            self,
            text="ENTER = capturar agora     P = pular esta figura     ESC = encerrar",
            font=("Segoe UI", 10), bg=tema.SUPERFICIE, fg=tema.TEXTO_FRACO,
        )
        self.lbl_ajuda.pack(side="bottom", pady=12)

    def definir(self, titulo: str, instrucao: str) -> None:
        self.lbl_titulo.configure(text=titulo)
        self.lbl_instr.configure(text=instrucao)

    def estado(self, texto: str, cor: str) -> None:
        self.lbl_estado.configure(text=texto, fg=cor)


def _salvar_tela(janela: tk.Tk, caminho: Path, recorte=None) -> bool:
    janela.update()
    time.sleep(0.25)
    x, y = janela.winfo_rootx(), janela.winfo_rooty()
    caixa = (x, y, x + janela.winfo_width(), y + janela.winfo_height())
    try:
        img = ImageGrab.grab(bbox=caixa)
    except Exception as exc:  # noqa: BLE001
        print(f"   ! falha ao capturar: {exc}")
        return False
    if recorte:
        img = img.crop(recorte)
    img.save(caminho, "PNG", dpi=(300, 300))
    print(f"   + {caminho.relative_to(RAIZ)}  ({img.width}x{img.height} px)")
    return True


def main() -> int:  # noqa: C901
    DESTINO.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print(" Expressão Neural — captura assistida para o banner")
    print("=" * 72)
    print("\nA aplicação vai abrir. Inicie a câmera quando solicitado.")
    print("Em cada figura, ajuste-se e pressione ENTER na janela 'Roteiro'.\n")

    tema.ativar_consciencia_dpi()
    janela = JanelaPrincipal()
    janela.geometry(f"{LARGURA}x{ALTURA}+0+0")
    janela.update()

    assistente = Assistente(janela)
    estado = {"indice": 0, "acao": None, "encerrar": False}

    def capturar(_e=None):
        estado["acao"] = "capturar"

    def pular(_e=None):
        estado["acao"] = "pular"

    def encerrar(_e=None):
        estado["encerrar"] = True

    for alvo in (assistente, janela):
        alvo.bind("<Return>", capturar)
        alvo.bind("<KP_Enter>", capturar)
        alvo.bind("<p>", pular)
        alvo.bind("<P>", pular)
        alvo.bind("<Escape>", encerrar)
    assistente.focus_force()

    salvas: list[str] = []
    puladas: list[str] = []

    try:
        for arquivo, titulo, instrucao, classe_alvo in ROTEIRO:
            if estado["encerrar"]:
                break
            estado["acao"] = None
            assistente.definir(titulo, instrucao)
            print(f"\n>> {titulo}\n   {instrucao}")

            if arquivo.endswith("sobre_o_projeto.png"):
                dialogo = sobre.abrir(janela, janela.fontes)
                dialogo.geometry("1100x760+120+120")
                for _ in range(10):
                    janela.update()
                    time.sleep(0.05)
                janela.update()
                time.sleep(0.4)
                x, y = dialogo.winfo_rootx(), dialogo.winfo_rooty()
                caixa = (x, y, x + dialogo.winfo_width(), y + dialogo.winfo_height())
                try:
                    ImageGrab.grab(bbox=caixa).save(DESTINO / arquivo, "PNG", dpi=(300, 300))
                    print(f"   + {(DESTINO / arquivo).relative_to(RAIZ)}")
                    salvas.append(arquivo)
                except Exception as exc:  # noqa: BLE001
                    print(f"   ! {exc}")
                    puladas.append(arquivo)
                dialogo.destroy()
                continue

            # Espera a confirmacao, mostrando o que a rede esta classificando
            while estado["acao"] is None and not estado["encerrar"]:
                janela.update()
                assistente.update()
                res = janela._ultimo_resultado
                alvo_res = res.principal if res else None
                if alvo_res is None:
                    assistente.estado("nenhum rosto detectado", tema.TEXTO_FRACO)
                else:
                    limiar = janela.ajustes.limiar_confianca
                    rotulo = alvo_res.rotulo_exibicao(limiar)
                    conf = alvo_res.confianca * 100
                    if classe_alvo and rotulo == classe_alvo:
                        cor, marca = tema.SUCESSO, "PRONTO  "
                    elif classe_alvo:
                        cor, marca = tema.ALERTA, "ajuste   "
                    else:
                        cor, marca = tema.PRIMARIA, "         "
                    assistente.estado(
                        f"{marca}{rotulo}  {conf:.0f}%"
                        + (f"   (alvo: {classe_alvo})" if classe_alvo else ""),
                        cor,
                    )
                time.sleep(0.03)

            if estado["encerrar"]:
                break
            if estado["acao"] == "pular":
                print("   (pulada)")
                puladas.append(arquivo)
                continue

            assistente.withdraw()  # o assistente nao pode aparecer na captura
            janela.update()
            time.sleep(0.35)
            recorte = None
            if "painel_classificacao" in arquivo:
                larg = janela.winfo_width()
                recorte = (larg - 400, 60, larg, 760)
            ok = _salvar_tela(janela, DESTINO / arquivo, recorte)
            salvas.append(arquivo) if ok else puladas.append(arquivo)
            assistente.deiconify()
            assistente.focus_force()
    finally:
        try:
            assistente.destroy()
        except tk.TclError:
            pass
        janela.encerrar()

    print("\n" + "=" * 72)
    print(f" Capturas salvas: {len(salvas)}")
    for s in salvas:
        print(f"   OK  {s}")
    if puladas:
        print(f" Pendentes: {len(puladas)}")
        for s in puladas:
            print(f"   --  {s}")
    print(f"\n Pasta: {DESTINO}")
    print("=" * 72)
    print("\nLembrete: use apenas o SEU rosto. Não capture terceiros sem")
    print("autorização por escrito.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
