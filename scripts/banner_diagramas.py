"""Gera os diagramas do banner em SVG e PNG de alta resolucao.

Fundo branco, texto escuro, caixas simples e setas -- adequado a impressao.
O SVG e escrito diretamente (vetorial, escala livre) e o PNG e rasterizado com
Pillow em 4x, sem dependencia externa de conversor.

Todo o conteudo vem da implementacao real: nomes de modelos, arquivos,
dimensoes e etapas conferem com o codigo em src/.
"""
from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

DESTINO = RAIZ / "docs" / "banner" / "03_diagramas"
ESCALA_PNG = 4  # 4x -> PNG com densidade adequada a impressao em banner

# Paleta sobria para impressao
PRETO = "#101418"
CINZA = "#55606e"
CINZA_CLARO = "#c3cad4"
AZUL = "#1f4e79"
AZUL_CLARO = "#eaf0f7"
VERDE = "#1e6b3a"
VERDE_CLARO = "#e8f3ec"
ROXO = "#5b3a8e"
ROXO_CLARO = "#f0eaf8"
LARANJA = "#9c5a10"
LARANJA_CLARO = "#fbf0e2"
BRANCO = "#ffffff"


def _fonte(tamanho: int, negrito: bool = False):
    candidatos = (
        ["segoeuib.ttf", "arialbd.ttf", "calibrib.ttf"]
        if negrito
        else ["segoeui.ttf", "arial.ttf", "calibri.ttf"]
    )
    for nome in candidatos:
        caminho = Path("C:/Windows/Fonts") / nome
        if caminho.is_file():
            try:
                return ImageFont.truetype(str(caminho), tamanho)
            except OSError:
                continue
    return ImageFont.load_default()


# ---------------------------------------------------------------------- SVG
def _svg_cabecalho(largura: int, altura: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{largura}" '
        f'height="{altura}" viewBox="0 0 {largura} {altura}" '
        f'font-family="Segoe UI, Calibri, Arial, sans-serif">',
        '  <defs>',
        '    <marker id="seta" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="7" markerHeight="7" orient="auto-start-reverse">',
        f'      <path d="M 0 0 L 10 5 L 0 10 z" fill="{CINZA}"/>',
        '    </marker>',
        '  </defs>',
        f'  <rect width="{largura}" height="{altura}" fill="{BRANCO}"/>',
    ]


def _svg_caixa(x, y, w, h, titulo, linhas, borda, fundo, r=6) -> list[str]:
    partes = [
        f'  <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" '
        f'fill="{fundo}" stroke="{borda}" stroke-width="2"/>',
        f'  <text x="{x + w / 2}" y="{y + 22}" text-anchor="middle" '
        f'font-size="15" font-weight="600" fill="{borda}">{titulo}</text>',
    ]
    for i, linha in enumerate(linhas):
        partes.append(
            f'  <text x="{x + w / 2}" y="{y + 42 + i * 17}" text-anchor="middle" '
            f'font-size="12.5" fill="{CINZA}">{linha}</text>'
        )
    return partes


def _svg_seta(x1, y1, x2, y2, rotulo="") -> list[str]:
    partes = [
        f'  <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{CINZA}" '
        f'stroke-width="2" marker-end="url(#seta)"/>'
    ]
    if rotulo:
        partes.append(
            f'  <text x="{(x1 + x2) / 2}" y="{y1 - 9}" text-anchor="middle" '
            f'font-size="11" fill="{CINZA}">{rotulo}</text>'
        )
    return partes


# ---------------------------------------------------------------------- PNG
class Lona:
    """Desenha o mesmo conteudo do SVG em PNG, com escala multiplicada."""

    def __init__(self, largura: int, altura: int, escala: int = ESCALA_PNG) -> None:
        self.e = escala
        self.img = Image.new("RGB", (largura * escala, altura * escala), BRANCO)
        self.d = ImageDraw.Draw(self.img)

    def _f(self, tam: int, negrito: bool = False):
        return _fonte(int(tam * self.e), negrito)

    def caixa(self, x, y, w, h, titulo, linhas, borda, fundo, r=6) -> None:
        e = self.e
        self.d.rounded_rectangle(
            [x * e, y * e, (x + w) * e, (y + h) * e],
            radius=r * e, fill=fundo, outline=borda, width=max(1, 2 * e // 2),
        )
        self.d.text(
            ((x + w / 2) * e, (y + 15) * e), titulo, font=self._f(15, True),
            fill=borda, anchor="mm",
        )
        for i, linha in enumerate(linhas):
            self.d.text(
                ((x + w / 2) * e, (y + 37 + i * 17) * e), linha,
                font=self._f(12.5), fill=CINZA, anchor="mm",
            )

    def seta(self, x1, y1, x2, y2, rotulo="") -> None:
        e = self.e
        self.d.line([x1 * e, y1 * e, x2 * e, y2 * e], fill=CINZA, width=max(1, e))
        # ponta
        import math

        ang = math.atan2(y2 - y1, x2 - x1)
        tam = 8 * e
        for d_ang in (2.6, -2.6):
            self.d.line(
                [x2 * e, y2 * e,
                 x2 * e + tam * math.cos(ang + d_ang),
                 y2 * e + tam * math.sin(ang + d_ang)],
                fill=CINZA, width=max(1, e),
            )
        if rotulo:
            self.d.text(
                (((x1 + x2) / 2) * e, (y1 - 13) * e), rotulo,
                font=self._f(11), fill=CINZA, anchor="mm",
            )

    def texto(self, x, y, txt, tam=12, cor=CINZA, negrito=False, ancora="lm") -> None:
        self.d.text(
            (x * self.e, y * self.e), txt, font=self._f(tam, negrito),
            fill=cor, anchor=ancora,
        )

    def linha(self, x1, y1, x2, y2, cor=CINZA_CLARO, largura=1) -> None:
        e = self.e
        self.d.line([x1 * e, y1 * e, x2 * e, y2 * e], fill=cor, width=max(1, largura * e))

    def salvar(self, caminho: Path) -> None:
        self.img.save(caminho, "PNG", dpi=(300, 300))
        print(f"   + {caminho.relative_to(RAIZ)}  ({self.img.width}x{self.img.height} px)")


# ================================================= 1. arquitetura do sistema
def diagrama_arquitetura() -> None:
    L, A = 1180, 322
    svg = _svg_cabecalho(L, A)
    png = Lona(L, A)

    titulo = "Expressão Neural — fluxo de processamento"
    svg.append(
        f'  <text x="24" y="30" font-size="17" font-weight="700" fill="{PRETO}">'
        f'{titulo}</text>'
    )
    png.texto(24, 24, titulo, 17, PRETO, True)
    svg.append(f'  <line x1="24" y1="42" x2="{L - 24}" y2="42" stroke="{CINZA_CLARO}"/>')
    png.linha(24, 42, L - 24, 42)

    y, h = 78, 92
    caixas = [
        (24, 150, "Webcam", ["OpenCV VideoCapture", "640×480 @ 30 FPS", "thread dedicada"], CINZA, "#f4f6f8"),
        (212, 190, "Detecção facial", ["YuNet (ONNX)", "cv2.FaceDetectorYN", "caixa + 5 pontos"], AZUL, AZUL_CLARO),
        (440, 186, "Alinhamento", ["similaridade não reflexiva", "(mínimos quadrados)", "recorte 112×112 RGB"], ROXO, ROXO_CLARO),
        (664, 196, "Classificação", ["MobileFaceNet /", "Progressive Teacher", "(ONNX) → 7 logits"], VERDE, VERDE_CLARO),
        (898, 150, "Softmax", ["7 probabilidades", "que somam 1,0"], LARANJA, LARANJA_CLARO),
    ]
    for x, w, tit, linhas, borda, fundo in caixas:
        svg += _svg_caixa(x, y, w, h, tit, linhas, borda, fundo)
        png.caixa(x, y, w, h, tit, linhas, borda, fundo)

    meio = y + h / 2
    for (x1, w1), (x2, _) in zip(
        [(c[0], c[1]) for c in caixas[:-1]], [(c[0], c[1]) for c in caixas[1:]]
    ):
        svg += _svg_seta(x1 + w1 + 4, meio, x2 - 6, meio)
        png.seta(x1 + w1 + 4, meio, x2 - 6, meio)

    # segunda linha: suavizacao e interface
    y2, h2 = 208, 70
    svg += _svg_caixa(
        330, y2, 250, h2, "Estabilidade temporal",
        ["média móvel exponencial", "por rosto acompanhado"], ROXO, "#faf8fe",
    )
    png.caixa(330, y2, 250, h2, "Estabilidade temporal",
              ["média móvel exponencial", "por rosto acompanhado"], ROXO, "#faf8fe")
    svg += _svg_caixa(
        640, y2, 360, h2, "Interface (Tkinter)",
        ["vídeo anotado · gráfico das 7 classes",
         "confiança · FPS · tempos de inferência"],
        PRETO, "#f4f6f8",
    )
    png.caixa(640, y2, 360, h2, "Interface (Tkinter)",
              ["vídeo anotado · gráfico das 7 classes",
               "confiança · FPS · tempos de inferência"],
              PRETO, "#f4f6f8")

    # Softmax -> (desce, vai a esquerda) -> Estabilidade temporal
    cot = 180  # altura do trecho horizontal de ligacao
    for seg in [(973, y + h + 2, 973, cot), (973, cot, 455, cot)]:
        svg.append(
            f'  <line x1="{seg[0]}" y1="{seg[1]}" x2="{seg[2]}" y2="{seg[3]}" '
            f'stroke="{CINZA}" stroke-width="2"/>'
        )
        png.linha(seg[0], seg[1], seg[2], seg[3], CINZA, 1)
    svg += _svg_seta(455, cot, 455, y2 - 6)
    png.seta(455, cot, 455, y2 - 6)

    # Estabilidade temporal -> Interface
    svg += _svg_seta(580, y2 + h2 / 2, 636, y2 + h2 / 2)
    png.seta(580, y2 + h2 / 2, 636, y2 + h2 / 2)

    rodape = ("Inferência ONNX executada por cv2.dnn somente em CPU — "
              "sem GPU dedicada, sem acesso à rede, 100% local")
    svg.append(
        f'  <text x="24" y="{A - 14}" font-size="12" fill="{CINZA}">{rodape}</text>'
    )
    png.texto(24, A - 18, rodape, 12, CINZA)

    svg.append("</svg>")
    (DESTINO / "arquitetura_sistema.svg").write_text("\n".join(svg), encoding="utf-8")
    print(f"   + {(DESTINO / 'arquitetura_sistema.svg').relative_to(RAIZ)}")
    png.salvar(DESTINO / "arquitetura_sistema.png")


# ===================================================== 2. rede neural (FER)
def diagrama_rede_neural() -> None:
    """Estrutura do classificador. Apenas o que e comprovado pela fonte oficial."""
    L, A = 1180, 330
    svg = _svg_cabecalho(L, A)
    png = Lona(L, A)

    titulo = "Classificador de expressões — MobileFaceNet (método Progressive Teacher)"
    svg.append(
        f'  <text x="24" y="30" font-size="17" font-weight="700" fill="{PRETO}">'
        f'{titulo}</text>'
    )
    png.texto(24, 24, titulo, 17, PRETO, True)
    sub = "Modelo pré-treinado (OpenCV Zoo, Apache 2.0). Não houve treinamento neste trabalho."
    svg.append(f'  <text x="24" y="50" font-size="12.5" fill="{CINZA}">{sub}</text>')
    png.texto(24, 46, sub, 12.5, CINZA)
    svg.append(f'  <line x1="24" y1="64" x2="{L - 24}" y2="64" stroke="{CINZA_CLARO}"/>')
    png.linha(24, 64, L - 24, 64)

    y, h = 96, 104
    blocos = [
        (24, 164, "Entrada", ["tensor 1×3×112×112", "RGB, nó 'data'",
                              "(px/255 − 0,5) / 0,5"], AZUL, AZUL_CLARO),
        (212, 188, "Convolução inicial", ["extração de bordas", "e texturas do rosto",
                                          "já alinhado"], VERDE, VERDE_CLARO),
        (436, 236, "Backbone MobileFaceNet", ["pilha de convoluções",
                                              "separáveis em profundidade",
                                              "(depthwise separable)"], VERDE, VERDE_CLARO),
        (708, 196, "Vetor de características", ["embedding facial",
                                                "aprendido pela rede"], ROXO, ROXO_CLARO),
        (940, 216, "Camada final", ["7 logits brutos,", "nó de saída 'label'"],
         LARANJA, LARANJA_CLARO),
    ]
    for x, w, tit, linhas, borda, fundo in blocos:
        svg += _svg_caixa(x, y, w, h, tit, linhas, borda, fundo)
        png.caixa(x, y, w, h, tit, linhas, borda, fundo)
    meio = y + h / 2
    for (x1, w1), (x2, _) in zip(
        [(b[0], b[1]) for b in blocos[:-1]], [(b[0], b[1]) for b in blocos[1:]]
    ):
        svg += _svg_seta(x1 + w1 + 4, meio, x2 - 6, meio)
        png.seta(x1 + w1 + 4, meio, x2 - 6, meio)

    # saida: 7 classes
    ys = 234
    rot = "Softmax → distribuição de probabilidades sobre as 7 classes"
    svg.append(f'  <text x="24" y="{ys}" font-size="13.5" font-weight="600" '
               f'fill="{PRETO}">{rot}</text>')
    png.texto(24, ys - 5, rot, 13.5, PRETO, True)

    classes = [
        ("0 angry", "Raiva"), ("1 disgust", "Nojo"), ("2 fearful", "Medo"),
        ("3 happy", "Alegria"), ("4 neutral", "Neutra"), ("5 sad", "Tristeza"),
        ("6 surprised", "Surpresa"),
    ]
    larg, gap = 150, 11
    x0, yb, hb = 24, 250, 54
    for i, (idx, pt) in enumerate(classes):
        x = x0 + i * (larg + gap)
        svg += _svg_caixa(x, yb, larg, hb, pt, [idx], AZUL, AZUL_CLARO, r=5)
        png.caixa(x, yb, larg, hb, pt, [idx], AZUL, AZUL_CLARO, r=5)

    nota = ("A ordem dos índices é definida por FacialExpressionRecog.getDesc "
            "(OpenCV Zoo) e não pode ser alterada.")
    svg.append(f'  <text x="24" y="{A - 10}" font-size="11.5" fill="{CINZA}">{nota}</text>')
    png.texto(24, A - 14, nota, 11.5, CINZA)

    svg.append("</svg>")
    (DESTINO / "rede_neural.svg").write_text("\n".join(svg), encoding="utf-8")
    print(f"   + {(DESTINO / 'rede_neural.svg').relative_to(RAIZ)}")
    png.salvar(DESTINO / "rede_neural.png")


def main() -> int:
    DESTINO.mkdir(parents=True, exist_ok=True)
    print("Gerando diagramas do banner (SVG + PNG 4x)")
    diagrama_arquitetura()
    diagrama_rede_neural()
    print("Concluido.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
