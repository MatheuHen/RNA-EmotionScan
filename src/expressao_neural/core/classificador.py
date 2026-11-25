"""Classificacao de expressoes faciais com rede neural (RF04 / RF05).

Pre-processamento e interpretacao de saida seguem exatamente a especificacao
oficial do modelo (OpenCV Zoo, `facial_fer_model.py`):

    entrada  : 'data'  -> blob NCHW 1x3x112x112, RGB, (pixel/255 - 0.5) / 0.5
    saida    : 'label' -> 7 logits brutos
    classes  : angry, disgust, fearful, happy, neutral, sad, surprised

Por serem logits, aplica-se softmax para obter probabilidades somando 1.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2 as cv
import numpy as np

from .. import config
from . import modelos
from .alinhamento import ErroAlinhamento, alinhar_rosto
from .detector import ErroModelo


def softmax(logits: np.ndarray) -> np.ndarray:
    """Softmax numericamente estavel sobre o ultimo eixo."""
    x = np.asarray(logits, dtype=np.float64)
    x = x - np.max(x, axis=-1, keepdims=True)
    exp = np.exp(x)
    return (exp / np.sum(exp, axis=-1, keepdims=True)).astype(np.float32)


@dataclass(frozen=True)
class Predicao:
    """Resultado da inferencia para um rosto."""

    indice: int
    probabilidades: np.ndarray  # (7,) float32, soma 1
    tempo_ms: float

    @property
    def rotulo_modelo(self) -> str:
        return config.CLASSES_MODELO[self.indice]

    @property
    def rotulo(self) -> str:
        return config.CLASSES_PT[self.indice]

    @property
    def confianca(self) -> float:
        return float(self.probabilidades[self.indice])

    def confiavel(self, limiar: float) -> bool:
        return self.confianca >= limiar

    def rotulo_exibicao(self, limiar: float) -> str:
        return self.rotulo if self.confiavel(limiar) else config.MSG_BAIXA_CONFIANCA


class ClassificadorExpressao:
    """Encapsula a rede MobileFaceNet/Progressive Teacher via `cv2.dnn`."""

    def __init__(self) -> None:
        info = modelos.CLASSIFICADOR
        if not info.caminho.is_file():
            raise ErroModelo(modelos.verificar(info, conferir_hash=False).mensagem)
        try:
            self._net = cv.dnn.readNet(str(info.caminho))
        except cv.error as exc:
            raise ErroModelo(
                f"Nao foi possivel carregar o classificador ({info.arquivo}): {exc}"
            ) from exc
        self._net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        self._net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
        self.info = info
        self._aquecido = False

    # ------------------------------------------------------- pre-processamento
    @staticmethod
    def preprocessar(rosto112: np.ndarray) -> np.ndarray:
        """Converte um recorte 112x112 BGR no blob de entrada do modelo."""
        if rosto112 is None or rosto112.ndim != 3 or rosto112.shape[2] != 3:
            raise ValueError("recorte facial invalido: esperado HxWx3 BGR")
        if rosto112.shape[:2] != (
            config.FER_TAMANHO_ENTRADA[1],
            config.FER_TAMANHO_ENTRADA[0],
        ):
            rosto112 = cv.resize(
                rosto112, config.FER_TAMANHO_ENTRADA, interpolation=cv.INTER_LINEAR
            )
        img = cv.cvtColor(rosto112, cv.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img -= config.FER_MEDIA
        img /= config.FER_DESVIO
        return cv.dnn.blobFromImage(img)

    # ---------------------------------------------------------------- inferir
    def aquecer(self) -> float:
        """Executa uma inferencia descartavel para pagar o custo de alocacao."""
        tela = np.zeros(
            (config.FER_TAMANHO_ENTRADA[1], config.FER_TAMANHO_ENTRADA[0], 3),
            dtype=np.uint8,
        )
        pred = self.inferir_recorte(tela)
        self._aquecido = True
        return pred.tempo_ms

    def inferir_recorte(self, rosto112: np.ndarray) -> Predicao:
        blob = self.preprocessar(rosto112)
        t0 = cv.getTickCount()
        try:
            self._net.setInput(blob, config.FER_NOME_ENTRADA)
            saida = self._net.forward(config.FER_NOME_SAIDA)
        except cv.error as exc:
            raise ErroModelo(f"Falha na inferencia da expressao: {exc}") from exc
        tempo_ms = (cv.getTickCount() - t0) / cv.getTickFrequency() * 1000.0

        logits = np.asarray(saida, dtype=np.float32).reshape(-1)
        if logits.size != len(config.CLASSES_MODELO):
            raise ErroModelo(
                f"Saida inesperada do modelo: {logits.size} valores, "
                f"esperado {len(config.CLASSES_MODELO)}."
            )
        probs = softmax(logits) if config.FER_SAIDA_E_LOGIT else logits
        return Predicao(int(np.argmax(probs)), probs, tempo_ms)

    def inferir(self, imagem: np.ndarray, pontos5: np.ndarray) -> Predicao:
        """Alinha o rosto pelos 5 pontos do detector e classifica."""
        try:
            alinhado = alinhar_rosto(imagem, pontos5)
        except ErroAlinhamento as exc:
            raise ErroModelo(f"Alinhamento facial falhou: {exc}") from exc
        return self.inferir_recorte(alinhado)
