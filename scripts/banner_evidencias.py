"""Renderiza as evidencias textuais reais como imagem, para uso no banner.

Nao reescreve nem embeleza a saida: le o arquivo .txt gravado pela execucao
real e o desenha em estilo de terminal. O .txt permanece ao lado da imagem,
de modo que qualquer afirmacao do banner possa ser auditada na origem.
"""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

EVID = RAIZ / "docs" / "banner" / "05_evidencias"
GRAF = RAIZ / "docs" / "banner" / "04_graficos"

FUNDO = "#11161d"
TEXTO = "#dbe3ee"
VERDE = "#4ade80"
VERMELHO = "#f87171"
AMARELO = "#fbbf24"
CIANO = "#67e8f9"
CINZA = "#64748b"


def _mono(tam: int, negrito: bool = False):
    nomes = (["consolab.ttf", "CascadiaMono-Bold.ttf"] if negrito
             else ["consola.ttf", "CascadiaMono.ttf", "cour.ttf"])
    for n in nomes:
        p = Path("C:/Windows/Fonts") / n
        if p.is_file():
            try:
                return ImageFont.truetype(str(p), tam)
            except OSError:
                continue
    return ImageFont.load_default()


def _cor_da_linha(linha: str) -> str:
    if "PASSED" in linha or "passed" in linha:
        return VERDE
    if "FAILED" in linha or "failed" in linha or "ERROR" in linha:
        return VERMELHO
    if "SKIPPED" in linha or "deselected" in linha:
        return AMARELO
    if linha.startswith("===") or linha.startswith("---"):
        return CIANO
    return TEXTO


def renderizar_texto(
    origem: Path, destino: Path, titulo: str,
    max_linhas: int = 60, escala: int = 2,
) -> None:
    linhas_brutas = origem.read_text(encoding="utf-8", errors="ignore").splitlines()
    linhas = [l.rstrip() for l in linhas_brutas if l.strip()]

    # Mantem o cabecalho e o fim (resumo), que e o que interessa no banner.
    if len(linhas) > max_linhas:
        corte = max_linhas // 2
        linhas = (
            linhas[:corte]
            + [f"... ({len(linhas) - max_linhas} linhas omitidas; saída completa "
               f"em {origem.name}) ..."]
            + linhas[-(max_linhas - corte):]
        )

    alt_linha, margem, topo = 17, 22, 56
    largura_car = 8.1
    larg_max = max((len(l) for l in linhas), default=80)
    L = int(margem * 2 + larg_max * largura_car) + 20
    L = max(900, min(L, 1700))
    A = topo + len(linhas) * alt_linha + margem + 16

    img = Image.new("RGB", (L * escala, A * escala), FUNDO)
    d = ImageDraw.Draw(img)
    f = _mono(int(12 * escala))
    fb = _mono(int(13 * escala), True)

    d.text((margem * escala, 20 * escala), titulo, font=fb, fill=CIANO)
    d.line([margem * escala, 42 * escala, (L - margem) * escala, 42 * escala],
           fill="#223049", width=escala)

    y = topo
    for linha in linhas:
        d.text((margem * escala, y * escala), linha[:220],
               font=f, fill=_cor_da_linha(linha))
        y += alt_linha

    d.text((margem * escala, (A - 20) * escala),
           f"Saída real de execução — arquivo: {origem.name}",
           font=_mono(int(10 * escala)), fill=CINZA)

    img.save(destino, "PNG", dpi=(300, 300))
    print(f"   + {destino.relative_to(RAIZ)}  ({img.width}x{img.height} px)")


def _resumo_testes(origem: Path, destino: Path, escala: int = 3) -> None:
    """Imagem compacta com apenas o resultado final, legivel a distancia."""
    txt = origem.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"(\d+) passed(?:, (\d+) deselected)?.*?in ([\d.]+)s", txt)
    aprov = m.group(1) if m else "?"
    desel = m.group(2) if (m and m.group(2)) else "0"
    dur = m.group(3) if m else "?"
    reprov = str(len(re.findall(r"\bFAILED\b", txt)))

    L, A = 760, 230
    img = Image.new("RGB", (L * escala, A * escala), "#ffffff")
    d = ImageDraw.Draw(img)

    def f(t, b=False):
        nomes = (["segoeuib.ttf", "arialbd.ttf"] if b else ["segoeui.ttf", "arial.ttf"])
        for n in nomes:
            p = Path("C:/Windows/Fonts") / n
            if p.is_file():
                return ImageFont.truetype(str(p), int(t * escala))
        return ImageFont.load_default()

    d.text((26 * escala, 24 * escala), "Resultado da suíte automatizada",
           font=f(17, True), fill="#101418")
    d.line([26 * escala, 54 * escala, (L - 26) * escala, 54 * escala],
           fill="#c3cad4", width=escala)

    cards = [
        (aprov, "testes aprovados", "#1e6b3a"),
        (reprov, "reprovados", "#b91c1c" if reprov != "0" else "#55606e"),
        (f"{dur}s".replace(".", ","), "duração total", "#1f4e79"),
    ]
    larg = (L - 52) / 3
    for i, (valor, rotulo, cor) in enumerate(cards):
        cx = 26 + larg * i + larg / 2
        d.text((cx * escala, 115 * escala), valor, font=f(44, True), fill=cor, anchor="mm")
        d.text((cx * escala, 152 * escala), rotulo, font=f(12.5), fill="#55606e", anchor="mm")

    nota = (f"Inferência executada com os modelos ONNX reais. "
            f"{desel} teste de longa duração executado à parte.")
    d.text((26 * escala, 196 * escala), nota, font=f(11), fill="#55606e")
    img.save(destino, "PNG", dpi=(300, 300))
    print(f"   + {destino.relative_to(RAIZ)}  ({img.width}x{img.height} px)")


def main() -> int:
    EVID.mkdir(parents=True, exist_ok=True)
    print("Gerando evidências visuais a partir das saídas reais")

    testes_txt = EVID / "testes_automatizados.txt"
    if testes_txt.is_file():
        renderizar_texto(
            testes_txt, EVID / "figura_07_testes_automatizados.png",
            "pytest — execução real da suíte automatizada",
        )
        _resumo_testes(testes_txt, EVID / "figura_07b_resumo_testes.png")
    else:
        print("   ! testes_automatizados.txt ausente — execute banner_metricas.py")

    log30 = RAIZ / "evidencias" / "execucao_prolongada.log"
    if log30.is_file() and log30.stat().st_size > 0:
        destino_txt = EVID / "execucao_prolongada_30min.txt"
        shutil.copy2(log30, destino_txt)
        renderizar_texto(
            destino_txt, EVID / "execucao_prolongada_30min.png",
            "pytest — ensaio de 30 minutos contínuos",
        )

    # Figura 08: desempenho (reaproveita o gráfico já gerado dos dados reais)
    origem = GRAF / "desempenho_inferencia.png"
    if origem.is_file():
        shutil.copy2(origem, EVID / "figura_08_desempenho.png")
        print(f"   + {(EVID / 'figura_08_desempenho.png').relative_to(RAIZ)}")

    # Verificação de prontidão, se existir
    for nome in ("metricas.md",):
        src = RAIZ / "docs" / "banner" / "06_metricas" / nome
        if src.is_file():
            shutil.copy2(src, EVID / nome)

    print("Concluído.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
