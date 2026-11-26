"""Orquestracao deteccao -> alinhamento -> classificacao -> suavizacao.

`cv2.dnn.Net` e `cv2.FaceDetectorYN` NAO sao thread-safe: `setInput`/`forward`
e `setInputSize`/`detect` sao pares nao atomicos, e o OpenCV libera o GIL
durante a execucao. Como a analise de imagem estatica roda na thread do Tkinter
enquanto a captura roda na thread propria, todo acesso aos modelos e
serializado por um `RLock` desta classe.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np

from .. import config
from .classificador import ClassificadorExpressao, Predicao
from .detector import DetectorFacial, ErroModelo, Rosto
from .suavizacao import SuavizadorTemporal


@dataclass
class ResultadoRosto:
    rosto: Rosto
    predicao: Predicao
    probabilidades: np.ndarray  # vetor exibido (suavizado, se ativo)
    id_trilha: int
    estabilidade: float

    @property
    def indice(self) -> int:
        return int(np.argmax(self.probabilidades))

    @property
    def rotulo(self) -> str:
        return config.CLASSES_PT[self.indice]

    @property
    def confianca(self) -> float:
        return float(self.probabilidades[self.indice])

    def confiavel(self, limiar: float) -> bool:
        return self.confianca >= limiar

    def rotulo_exibicao(self, limiar: float) -> str:
        return self.rotulo if self.confiavel(limiar) else "Indefinida"


@dataclass
class ResultadoQuadro:
    """Tudo que a interface precisa saber sobre um quadro processado."""

    rostos: list[ResultadoRosto] = field(default_factory=list)
    tempo_deteccao_ms: float = 0.0
    tempo_inferencia_ms: float = 0.0
    tempo_total_ms: float = 0.0
    erro: str | None = None
    processado: bool = True

    @property
    def quantidade_rostos(self) -> int:
        return len(self.rostos)

    @property
    def principal(self) -> ResultadoRosto | None:
        return self.rostos[0] if self.rostos else None


class PipelineExpressao:
    """Mantem as instancias dos modelos e processa quadros sob demanda."""

    def __init__(self, ajustes: config.Ajustes | None = None) -> None:
        # Reentrante: `aplicar_ajustes` e `processar` podem se aninhar.
        self._lock = threading.RLock()
        self.ajustes = ajustes or config.Ajustes()
        self.detector = DetectorFacial(
            limiar_confianca=self.ajustes.limiar_deteccao,
            limiar_nms=self.ajustes.limiar_nms,
            max_rostos=self.ajustes.max_rostos if self.ajustes.multiplos_rostos else 1,
        )
        cal = None
        if self.ajustes.usar_calibracao:
            from .calibracao import Calibracao

            cal = Calibracao.carregar()
        self.classificador = ClassificadorExpressao(calibracao=cal)
        self.suavizador = SuavizadorTemporal(alfa=self.ajustes.suavizacao_alfa)
        self._ms_aquecimento = self.classificador.aquecer()

    # ----------------------------------------------------------------- ajustes
    @property
    def ms_aquecimento(self) -> float:
        return self._ms_aquecimento

    def aplicar_ajustes(self, ajustes: config.Ajustes) -> None:
        with self._lock:
            self.ajustes = ajustes
            self.detector.definir_limiares(ajustes.limiar_deteccao, ajustes.limiar_nms)
            self.detector.max_rostos = (
                ajustes.max_rostos if ajustes.multiplos_rostos else 1
            )
            self.suavizador.alfa = float(np.clip(ajustes.suavizacao_alfa, 0.05, 1.0))

    def reiniciar_estado(self) -> None:
        """Descarta o historico temporal (camera parou, trocou de fonte...)."""
        with self._lock:
            self.suavizador.limpar()

    # ---------------------------------------------------------------- execucao
    def processar(self, quadro: np.ndarray, suavizar: bool | None = None) -> ResultadoQuadro:
        with self._lock:
            return self._processar_sem_lock(quadro, suavizar)

    def _processar_sem_lock(
        self, quadro: np.ndarray, suavizar: bool | None
    ) -> ResultadoQuadro:
        if suavizar is None:
            suavizar = self.ajustes.suavizacao_ativa
        t_ini = time.perf_counter()
        resultado = ResultadoQuadro()

        try:
            t0 = time.perf_counter()
            rostos = self.detector.detectar(quadro)
            resultado.tempo_deteccao_ms = (time.perf_counter() - t0) * 1000.0
        except ErroModelo as exc:
            resultado.erro = str(exc)
            resultado.tempo_total_ms = (time.perf_counter() - t_ini) * 1000.0
            return resultado

        if not rostos:
            # Sem rosto: nenhuma classificacao e o historico e descartado (RF03/RF10).
            self.suavizador.esquecer_ausentes(set())
            resultado.tempo_total_ms = (time.perf_counter() - t_ini) * 1000.0
            return resultado

        usados: set[int] = set()
        total_infer = 0.0
        for rosto in rostos:
            try:
                pred = self.classificador.inferir(quadro, rosto.pontos)
            except ErroModelo as exc:
                resultado.erro = str(exc)
                continue
            total_infer += pred.tempo_ms

            escala = float(max(rosto.largura, rosto.altura))
            # A trilha e sempre atualizada (mantem o acompanhamento e a metrica de
            # estabilidade); o vetor exibido respeita a opcao de suavizacao.
            tid, vetor_suave = self.suavizador.suavizar(
                rosto.centro, escala, pred.probabilidades, usados
            )
            vetor = vetor_suave if suavizar else pred.probabilidades.copy()
            usados.add(tid)
            resultado.rostos.append(
                ResultadoRosto(
                    rosto=rosto,
                    predicao=pred,
                    probabilidades=vetor,
                    id_trilha=tid,
                    estabilidade=self.suavizador.estabilidade(tid),
                )
            )

        self.suavizador.esquecer_ausentes(usados)
        resultado.tempo_inferencia_ms = total_infer
        resultado.tempo_total_ms = (time.perf_counter() - t_ini) * 1000.0
        return resultado

    def liberar(self) -> None:
        with self._lock:
            self.suavizador.limpar()
