"""Sobreposicao visual no quadro de video (delimitacao e rotulos).

Todo o texto desenhado passa por `cv2.putText`, cuja fonte Hershey nao possui
acentuacao. Os rotulos sobrepostos usam, portanto, forma sem acento; os rotulos
acentuados completos aparecem nos paineis Tkinter.
"""
from __future__ import annotations

import cv2 as cv
import numpy as np

from .. import config
from ..core.pipeline import ResultadoQuadro
from . import tema

_SEM_ACENTO = {
    "Raiva": "Raiva",
    "Nojo": "Nojo",
    "Medo": "Medo",
    "Alegria": "Alegria",
    "Neutra": "Neutra",
    "Tristeza": "Tristeza",
    "Surpresa": "Surpresa",
    "Indefinida": "Indefinida",
}

_FONTE = cv.FONT_HERSHEY_DUPLEX
_CINZA = tema.hex_para_bgr(tema.NEUTRO)


def _texto_seguro(valor: str) -> str:
    return _SEM_ACENTO.get(valor, valor.encode("ascii", "ignore").decode() or "?")


def _cantos(img, x1, y1, x2, y2, cor, espessura, tamanho) -> None:
    """Moldura em 'L' nos quatro cantos -- leitura mais limpa que um retangulo."""
    for px, py, dx, dy in (
        (x1, y1, 1, 1),
        (x2, y1, -1, 1),
        (x1, y2, 1, -1),
        (x2, y2, -1, -1),
    ):
        cv.line(img, (px, py), (px + dx * tamanho, py), cor, espessura, cv.LINE_AA)
        cv.line(img, (px, py), (px, py + dy * tamanho), cor, espessura, cv.LINE_AA)


def anotar(
    quadro: np.ndarray,
    resultado: ResultadoQuadro,
    limiar: float,
    mostrar_pontos: bool = True,
) -> np.ndarray:
    """Devolve uma COPIA do quadro com as marcacoes dos rostos detectados."""
    if quadro is None or quadro.size == 0:
        return quadro
    img = quadro.copy()
    altura, largura = img.shape[:2]
    escala_texto = max(0.45, min(0.75, largura / 1100.0))

    for item in resultado.rostos:
        r = item.rosto
        x1 = max(0, r.x)
        y1 = max(0, r.y)
        x2 = min(largura - 1, r.x + r.largura)
        y2 = min(altura - 1, r.y + r.altura)
        if x2 <= x1 or y2 <= y1:
            continue

        confiavel = item.confiavel(limiar)
        cor = (
            tema.hex_para_bgr(config.CORES_CLASSES[item.indice]) if confiavel else _CINZA
        )

        _cantos(img, x1, y1, x2, y2, cor, 2, max(14, (x2 - x1) // 6))
        cv.rectangle(img, (x1, y1), (x2, y2), cor, 1, cv.LINE_AA)

        if mostrar_pontos:
            for px, py in r.pontos.astype(np.int32):
                if 0 <= px < largura and 0 <= py < altura:
                    cv.circle(img, (int(px), int(py)), 2, cor, -1, cv.LINE_AA)

        rotulo = _texto_seguro(item.rotulo_exibicao(limiar))
        if confiavel:
            rotulo = f"{rotulo} {item.confianca * 100:.0f}%"
        (tw, th), base = cv.getTextSize(rotulo, _FONTE, escala_texto, 1)

        # Faixa do rotulo acima da caixa; abaixo quando nao houver espaco.
        topo = y1 - th - base - 10
        acima = topo >= 0
        fy1 = topo if acima else min(altura - th - base - 10, y2 + 2)
        fy2 = fy1 + th + base + 10
        fx2 = min(largura - 1, x1 + tw + 16)

        fy1 = max(0, fy1)
        fy2 = min(altura - 1, fy2)
        cv.rectangle(img, (x1, fy1), (fx2, fy2), cor, -1)
        cv.putText(
            img,
            rotulo,
            (x1 + 8, fy2 - base - 4),
            _FONTE,
            escala_texto,
            (10, 18, 32),
            1,
            cv.LINE_AA,
        )
    return img


def placa_mensagem(largura: int, altura: int, linhas: list[str]) -> np.ndarray:
    """Imagem de espera/erro exibida quando nao ha video disponivel."""
    img = np.full((altura, largura, 3), tema.hex_para_bgr(tema.SUPERFICIE), dtype=np.uint8)
    # Grade discreta, reforcando a identidade tecnologica.
    grade = tema.hex_para_bgr(tema.GRADE)
    for x in range(0, largura, 40):
        cv.line(img, (x, 0), (x, altura), grade, 1)
    for y in range(0, altura, 40):
        cv.line(img, (0, y), (largura, y), grade, 1)

    y = altura // 2 - (len(linhas) * 30) // 2
    for i, linha in enumerate(linhas):
        texto = _texto_seguro(linha) if not linha.isascii() else linha
        escala = 0.9 if i == 0 else 0.55
        cor = tema.hex_para_bgr(tema.PRIMARIA if i == 0 else tema.TEXTO_SECUNDARIO)
        (tw, _), _ = cv.getTextSize(texto, _FONTE, escala, 1)
        cv.putText(
            img, texto, ((largura - tw) // 2, y), _FONTE, escala, cor, 1, cv.LINE_AA
        )
        y += 36 if i == 0 else 26
    return img


def ajustar_para_area(quadro: np.ndarray, largura: int, altura: int) -> np.ndarray:
    """Redimensiona preservando a proporcao e preenche o restante com o fundo."""
    if largura < 10 or altura < 10 or quadro is None or quadro.size == 0:
        return quadro
    h, w = quadro.shape[:2]
    fator = min(largura / w, altura / h)
    nw, nh = max(1, int(w * fator)), max(1, int(h * fator))
    interp = cv.INTER_AREA if fator < 1 else cv.INTER_LINEAR
    redim = cv.resize(quadro, (nw, nh), interpolation=interp)
    if (nw, nh) == (largura, altura):
        return redim
    tela = np.full((altura, largura, 3), tema.hex_para_bgr(tema.FUNDO), dtype=np.uint8)
    ox, oy = (largura - nw) // 2, (altura - nh) // 2
    tela[oy:oy + nh, ox:ox + nw] = redim
    return tela
