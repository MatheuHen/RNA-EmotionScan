"""Deteccao facial com a rede neural YuNet (RF03)."""
from __future__ import annotations

from dataclasses import dataclass

import cv2 as cv
import numpy as np

from . import modelos


class ErroModelo(RuntimeError):
    """Falha ao carregar ou executar um modelo neural."""


@dataclass(frozen=True)
class Rosto:
    """Um rosto detectado. Coordenadas em pixels da imagem original."""

    x: int
    y: int
    largura: int
    altura: int
    confianca: float
    pontos: np.ndarray  # (5, 2) float32

    @property
    def caixa(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.largura, self.altura

    @property
    def centro(self) -> tuple[float, float]:
        return self.x + self.largura / 2.0, self.y + self.altura / 2.0

    @property
    def area(self) -> int:
        return max(0, self.largura) * max(0, self.altura)


class DetectorFacial:
    """Encapsula `cv2.FaceDetectorYN`. Instancia unica reutilizada entre quadros."""

    def __init__(
        self,
        limiar_confianca: float = 0.85,
        limiar_nms: float = 0.30,
        max_rostos: int = 5,
    ) -> None:
        info = modelos.DETECTOR
        if not info.caminho.is_file():
            raise ErroModelo(modelos.verificar(info, conferir_hash=False).mensagem)
        try:
            self._net = cv.FaceDetectorYN.create(
                model=str(info.caminho),
                config="",
                input_size=(320, 320),
                score_threshold=float(limiar_confianca),
                nms_threshold=float(limiar_nms),
                top_k=5000,
            )
        except cv.error as exc:
            raise ErroModelo(
                f"Nao foi possivel carregar o detector YuNet ({info.arquivo}): {exc}"
            ) from exc
        self._tamanho_atual: tuple[int, int] = (320, 320)
        self._max_rostos = max(1, int(max_rostos))
        self.info = info

    # ---------------------------------------------------------------- ajustes
    @property
    def max_rostos(self) -> int:
        return self._max_rostos

    @max_rostos.setter
    def max_rostos(self, valor: int) -> None:
        self._max_rostos = max(1, int(valor))

    def definir_limiares(self, confianca: float, nms: float) -> None:
        self._net.setScoreThreshold(float(confianca))
        self._net.setNMSThreshold(float(nms))

    # --------------------------------------------------------------- execucao
    def detectar(self, quadro: np.ndarray) -> list[Rosto]:
        if quadro is None or quadro.ndim != 3:
            return []
        altura, largura = quadro.shape[:2]
        if largura <= 0 or altura <= 0:
            return []
        if self._tamanho_atual != (largura, altura):
            self._net.setInputSize((largura, altura))
            self._tamanho_atual = (largura, altura)

        try:
            _, deteccoes = self._net.detect(quadro)
        except cv.error as exc:
            raise ErroModelo(f"Falha na deteccao facial: {exc}") from exc

        if deteccoes is None or len(deteccoes) == 0:
            return []

        rostos: list[Rosto] = []
        for linha in deteccoes:
            x, y, w, h = (float(v) for v in linha[0:4])
            if w <= 1 or h <= 1:
                continue
            pontos = np.asarray(linha[4:14], dtype=np.float32).reshape(5, 2)
            if not np.isfinite(pontos).all():
                continue
            rostos.append(
                Rosto(
                    x=int(round(x)),
                    y=int(round(y)),
                    largura=int(round(w)),
                    altura=int(round(h)),
                    confianca=float(linha[14]),
                    pontos=pontos,
                )
            )

        # Maiores rostos primeiro: o rosto principal da apresentacao e o mais proximo.
        rostos.sort(key=lambda r: r.area, reverse=True)
        return rostos[: self._max_rostos]
