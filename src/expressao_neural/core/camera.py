"""Captura de webcam em thread dedicada (RF02 + secao 8 do escopo).

Projeto anti-travamento:
  * a leitura da camera e a inferencia ocorrem FORA da thread do Tkinter;
  * o resultado e publicado num slot unico (sempre o mais recente) -- nao existe
    fila que possa acumular quadros e crescer indefinidamente;
  * `parar()` sempre libera o `VideoCapture`, inclusive em caso de excecao.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass

import cv2 as cv
import numpy as np

from .. import config
from .pipeline import PipelineExpressao, ResultadoQuadro


@dataclass
class QuadroPronto:
    quadro: np.ndarray            # BGR, pronto para exibicao
    resultado: ResultadoQuadro
    fps: float
    indice: int


class ErroCamera(RuntimeError):
    """Camera ausente, ocupada ou incapaz de entregar quadros."""


def _abrir(indice: int) -> cv.VideoCapture:
    # CAP_DSHOW abre bem mais rapido no Windows; demais plataformas usam o padrao.
    tentativas = (
        [cv.CAP_DSHOW, cv.CAP_MSMF, cv.CAP_ANY]
        if hasattr(cv, "CAP_DSHOW")
        else [cv.CAP_ANY]
    )
    for api in tentativas:
        cap = cv.VideoCapture(indice, api)
        if cap.isOpened():
            return cap
        cap.release()
    raise ErroCamera(
        f"Não foi possível abrir a câmera {indice}. Verifique se ela está conectada, "
        "se outro programa (Teams, Zoom, Meet) está usando-a e se a permissão de "
        "câmera do Windows está liberada."
    )


def listar_cameras(maximo: int = 5) -> list[int]:
    """Indices de camera que respondem. Operacao lenta -- use fora do laco principal."""
    encontrados: list[int] = []
    for i in range(maximo):
        cap = None
        try:
            cap = cv.VideoCapture(i, cv.CAP_DSHOW if hasattr(cv, "CAP_DSHOW") else cv.CAP_ANY)
            if cap.isOpened():
                ok, _ = cap.read()
                if ok:
                    encontrados.append(i)
        except cv.error:
            pass
        finally:
            if cap is not None:
                cap.release()
    return encontrados


class TrabalhadorCamera(threading.Thread):
    """Le a camera, roda o pipeline e publica o quadro mais recente."""

    def __init__(self, pipeline: PipelineExpressao, ajustes: config.Ajustes) -> None:
        super().__init__(name="ExpressaoNeural-Captura", daemon=True)
        self._pipeline = pipeline
        self._ajustes = ajustes
        self._cap: cv.VideoCapture | None = None

        self._lock = threading.Lock()
        self._slot: QuadroPronto | None = None
        self._erro: str | None = None

        self._parar = threading.Event()
        self._pausado = threading.Event()
        self._pronto = threading.Event()

        self._trocar_para: int | None = None
        self._contador = 0
        self._marcas: deque[float] = deque(maxlen=30)
        self._ultimo_resultado = ResultadoQuadro(processado=False)

    # ------------------------------------------------------------------ estado
    @property
    def pausado(self) -> bool:
        return self._pausado.is_set()

    @property
    def ativo(self) -> bool:
        return self.is_alive() and not self._parar.is_set()

    @property
    def erro(self) -> str | None:
        with self._lock:
            return self._erro

    @property
    def indice_camera(self) -> int:
        return self._ajustes.indice_camera

    def aguardar_pronto(self, timeout: float = 8.0) -> bool:
        return self._pronto.wait(timeout)

    def obter(self) -> QuadroPronto | None:
        """Consome o quadro mais recente (nao bloqueia, nunca acumula)."""
        with self._lock:
            pronto, self._slot = self._slot, None
            return pronto

    # ----------------------------------------------------------------- comandos
    def pausar(self) -> None:
        self._pausado.set()

    def retomar(self) -> None:
        self._pausado.clear()

    def trocar_camera(self, indice: int) -> None:
        self._trocar_para = int(indice)

    def parar(self, timeout: float = 3.0) -> None:
        self._parar.set()
        self._pausado.clear()
        if self.is_alive():
            self.join(timeout)

    # ----------------------------------------------------------------- interno
    def _configurar(self, cap: cv.VideoCapture) -> None:
        cap.set(cv.CAP_PROP_FRAME_WIDTH, self._ajustes.largura_captura)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, self._ajustes.altura_captura)
        cap.set(cv.CAP_PROP_FPS, self._ajustes.fps_alvo)
        try:
            cap.set(cv.CAP_PROP_BUFFERSIZE, 1)  # nao suportado por todo backend
        except cv.error:
            pass

    def _registrar_erro(self, mensagem: str | None) -> None:
        with self._lock:
            self._erro = mensagem

    def _publicar(self, quadro: np.ndarray, resultado: ResultadoQuadro) -> None:
        agora = time.perf_counter()
        self._marcas.append(agora)
        fps = 0.0
        if len(self._marcas) >= 2:
            intervalo = self._marcas[-1] - self._marcas[0]
            if intervalo > 0:
                fps = (len(self._marcas) - 1) / intervalo
        with self._lock:
            self._slot = QuadroPronto(quadro, resultado, fps, self._contador)
        self._pronto.set()

    # -------------------------------------------------------------------- run
    def run(self) -> None:  # noqa: C901 - laco de captura concentra o fluxo
        falhas_leitura = 0
        try:
            try:
                self._cap = _abrir(self._ajustes.indice_camera)
                self._configurar(self._cap)
            except ErroCamera as exc:
                self._registrar_erro(str(exc))
                self._pronto.set()
                return

            intervalo_minimo = 1.0 / max(1, self._ajustes.fps_alvo)

            while not self._parar.is_set():
                inicio = time.perf_counter()

                if self._trocar_para is not None:
                    novo = self._trocar_para
                    self._trocar_para = None
                    if self._cap is not None:
                        self._cap.release()
                    try:
                        self._cap = _abrir(novo)
                        self._configurar(self._cap)
                        self._ajustes.indice_camera = novo
                        self._registrar_erro(None)
                        self._pipeline.reiniciar_estado()
                        falhas_leitura = 0
                    except ErroCamera as exc:
                        self._registrar_erro(str(exc))
                        time.sleep(0.5)
                        continue

                if self._pausado.is_set():
                    time.sleep(0.05)
                    continue

                assert self._cap is not None
                ok, quadro = self._cap.read()
                if not ok or quadro is None or quadro.size == 0:
                    falhas_leitura += 1
                    if falhas_leitura == 15:
                        self._registrar_erro(
                            "A câmera parou de entregar imagens. Tentando reconectar…"
                        )
                    if falhas_leitura >= 25:
                        # Tentativa de recuperacao: reabrir o dispositivo (RF02).
                        try:
                            self._cap.release()
                            self._cap = _abrir(self._ajustes.indice_camera)
                            self._configurar(self._cap)
                            self._registrar_erro(None)
                            falhas_leitura = 0
                        except ErroCamera as exc:
                            self._registrar_erro(str(exc))
                            time.sleep(1.0)
                    else:
                        time.sleep(0.03)
                    continue

                if falhas_leitura:
                    falhas_leitura = 0
                    self._registrar_erro(None)

                if self._ajustes.espelhar_imagem:
                    quadro = cv.flip(quadro, 1)

                self._contador += 1
                passo = max(1, int(self._ajustes.inferir_a_cada))
                if self._contador % passo == 0:
                    try:
                        self._ultimo_resultado = self._pipeline.processar(quadro)
                        if self._ultimo_resultado.erro:
                            self._registrar_erro(self._ultimo_resultado.erro)
                    except Exception as exc:  # nunca derrubar a thread de captura
                        self._registrar_erro(f"Falha no processamento: {exc}")
                        self._ultimo_resultado = ResultadoQuadro(
                            erro=str(exc), processado=False
                        )

                self._publicar(quadro, self._ultimo_resultado)

                # Limita a taxa para nao saturar a CPU com quadros inuteis.
                resto = intervalo_minimo - (time.perf_counter() - inicio)
                if resto > 0:
                    time.sleep(resto)
        except Exception as exc:  # pragma: no cover - rede de seguranca final
            self._registrar_erro(f"Erro inesperado na captura: {exc}")
            self._pronto.set()
        finally:
            if self._cap is not None:
                self._cap.release()
                self._cap = None
