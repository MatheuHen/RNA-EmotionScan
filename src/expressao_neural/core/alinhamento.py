"""Alinhamento facial por transformacao de similaridade nao reflexiva.

Porte direto da implementacao de referencia do OpenCV Zoo
(`models/facial_expression_recognition/facial_fer_model.py`, classe FaceAlignment),
para garantir que o pre-processamento seja identico ao esperado pelo modelo.

Resolucao por minimos quadrados (deterministica) -- nao usa RANSAC/LMEDS.
"""
from __future__ import annotations

import cv2 as cv
import numpy as np

from ..config import FER_TAMANHO_ENTRADA, PONTOS_PADRAO_5

_PADRAO = np.array(PONTOS_PADRAO_5, dtype=np.float64)


class ErroAlinhamento(RuntimeError):
    """Pontos de referencia insuficientes ou degenerados."""


def _similaridade_nao_reflexiva(uv: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Matriz 3x3 T que leva `uv` (pontos detectados) em `xy` (pontos padrao)."""
    m = xy.shape[0]
    x = xy[:, 0].reshape((-1, 1))
    y = xy[:, 1].reshape((-1, 1))

    tmp1 = np.hstack((x, y, np.ones((m, 1)), np.zeros((m, 1))))
    tmp2 = np.hstack((y, -x, np.zeros((m, 1)), np.ones((m, 1))))
    big_x = np.vstack((tmp1, tmp2))

    u = uv[:, 0].reshape((-1, 1))
    v = uv[:, 1].reshape((-1, 1))
    big_u = np.vstack((u, v))

    if np.linalg.matrix_rank(big_x) < 4:
        raise ErroAlinhamento("pontos de referencia degenerados")

    r, _, _, _ = np.linalg.lstsq(big_x, big_u, rcond=None)
    sc, ss, tx, ty = np.squeeze(r)

    t_inv = np.array([[sc, -ss, 0.0], [ss, sc, 0.0], [tx, ty, 1.0]])
    try:
        t = np.linalg.inv(t_inv)
    except np.linalg.LinAlgError as exc:  # pragma: no cover - entrada degenerada
        raise ErroAlinhamento("transformacao nao inversivel") from exc
    t[:, 2] = np.array([0.0, 0.0, 1.0])
    return t


def matriz_alinhamento(pontos5: np.ndarray) -> np.ndarray:
    """Matriz afim 2x3 pronta para `cv2.warpAffine`."""
    pontos = np.asarray(pontos5, dtype=np.float64).reshape(-1, 2)
    if pontos.shape[0] != 5 or not np.isfinite(pontos).all():
        raise ErroAlinhamento("eram esperados 5 pontos faciais finitos")
    t = _similaridade_nao_reflexiva(pontos, _PADRAO.copy())
    return t[:, 0:2].T.astype(np.float32)


def alinhar_rosto(imagem: np.ndarray, pontos5: np.ndarray) -> np.ndarray:
    """Recorta e alinha o rosto para 112x112 conforme o modelo espera."""
    tfm = matriz_alinhamento(pontos5)
    return cv.warpAffine(imagem, tfm, FER_TAMANHO_ENTRADA, flags=cv.INTER_LINEAR)
