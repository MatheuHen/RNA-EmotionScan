"""Analise de imagem local como contingencia da webcam (RF09).

Validacao em camadas antes de qualquer decodificacao pesada:
  1. caminho existe e e arquivo regular;
  2. extensao permitida;
  3. tamanho dentro do limite;
  4. cabecalho reconhecivel e dimensoes sanas (Pillow, `verify` + `open`);
  5. decodificacao efetiva pelo OpenCV.
"""
from __future__ import annotations

from pathlib import Path

import cv2 as cv
import numpy as np
from PIL import Image, UnidentifiedImageError

from ..config import EXTENSOES_IMAGEM, LADO_MAX_IMAGEM, TAMANHO_MAX_IMAGEM_BYTES


class ImagemInvalida(ValueError):
    """Arquivo recusado pela validacao."""


def _mb(valor: int) -> str:
    return f"{valor / (1024 * 1024):.1f} MB"


def validar_caminho(caminho: str | Path) -> Path:
    p = Path(caminho)
    if not p.exists():
        raise ImagemInvalida(f"Arquivo não encontrado: {p.name}")
    if not p.is_file():
        raise ImagemInvalida(f"O caminho informado não é um arquivo: {p.name}")
    if p.suffix.lower() not in EXTENSOES_IMAGEM:
        permitidas = ", ".join(e.lstrip(".").upper() for e in EXTENSOES_IMAGEM)
        raise ImagemInvalida(
            f"Extensão não suportada ({p.suffix or 'sem extensão'}). "
            f"Use: {permitidas}."
        )
    tamanho = p.stat().st_size
    if tamanho == 0:
        raise ImagemInvalida(f"O arquivo {p.name} está vazio.")
    if tamanho > TAMANHO_MAX_IMAGEM_BYTES:
        raise ImagemInvalida(
            f"Imagem muito grande ({_mb(tamanho)}). "
            f"Limite: {_mb(TAMANHO_MAX_IMAGEM_BYTES)}."
        )
    return p


def carregar(caminho: str | Path) -> np.ndarray:
    """Devolve a imagem BGR validada, pronta para o pipeline."""
    p = validar_caminho(caminho)

    # Camada Pillow: detecta arquivo corrompido ou com extensao mentirosa.
    try:
        with Image.open(p) as img:
            img.verify()
        with Image.open(p) as img:
            largura, altura = img.size
            formato = (img.format or "").upper()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImagemInvalida(
            f"O arquivo {p.name} não é uma imagem válida ou está corrompido."
        ) from exc

    if formato not in {"JPEG", "PNG", "BMP", "WEBP", "MPO"}:
        raise ImagemInvalida(
            f"Formato interno não suportado ({formato or 'desconhecido'}) em {p.name}."
        )
    if largura < 32 or altura < 32:
        raise ImagemInvalida(
            f"Imagem muito pequena ({largura}x{altura}). Mínimo 32x32 pixels."
        )
    if largura > LADO_MAX_IMAGEM or altura > LADO_MAX_IMAGEM:
        raise ImagemInvalida(
            f"Imagem muito grande ({largura}x{altura}). "
            f"Limite de {LADO_MAX_IMAGEM} pixels por lado."
        )

    # Camada OpenCV: o que efetivamente alimenta os modelos.
    dados = np.fromfile(str(p), dtype=np.uint8)  # suporta caminho com acentos
    imagem = cv.imdecode(dados, cv.IMREAD_COLOR)
    if imagem is None or imagem.size == 0:
        raise ImagemInvalida(f"Não foi possível decodificar a imagem {p.name}.")
    if imagem.ndim != 3 or imagem.shape[2] != 3:
        raise ImagemInvalida(f"A imagem {p.name} não possui 3 canais de cor.")
    return imagem


def redimensionar_para_analise(imagem: np.ndarray, lado_max: int = 1280) -> np.ndarray:
    """Reduz imagens muito grandes para manter a inferencia rapida."""
    altura, largura = imagem.shape[:2]
    maior = max(altura, largura)
    if maior <= lado_max:
        return imagem
    fator = lado_max / float(maior)
    return cv.resize(
        imagem,
        (max(1, int(largura * fator)), max(1, int(altura * fator))),
        interpolation=cv.INTER_AREA,
    )
