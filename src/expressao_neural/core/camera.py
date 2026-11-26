"""Captura de webcam em DUAS threads (RF02 + secao 8 do escopo).

Por que duas, e nao uma
-----------------------
`cap.read()` nao devolve um quadro na hora: ele BLOQUEIA ate o proximo quadro
nascer. Uma webcam a 30 FPS produz quadros numa grade fixa de 33,3 ms, entao
`read()` sozinho ja consome todo o orcamento de um quadro.

Com leitura e inferencia na mesma thread, o ciclo virava:

    read (33 ms)  +  inferencia (~20-35 ms)  =  53 a 68 ms

Passando de 33,3 ms, o laco perde o ponto seguinte da grade e cai no proximo --
o resultado nao e uma queda proporcional, e sim um salto para METADE da taxa.
Medido no aplicativo real: camera entregando 30,0 FPS brutos, sistema exibindo
14,8 FPS.

Arquitetura atual:

    thread de captura    le a camera e publica o quadro IMEDIATAMENTE,
                         sem esperar classificacao nenhuma -> 30 FPS
    thread de analise    pega o quadro mais recente e infere no ritmo que
                         conseguir; o resultado alcanca o quadro seguinte

Projeto anti-travamento (mantido):
  * nada disso roda na thread do Tkinter;
  * cada ponto de troca usa um slot unico, sempre o mais recente -- nao existe
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

        # Ponte entre as duas threads: mais um slot unico, nunca uma fila.
        # Se a analise nao acompanhar, o quadro antigo e descartado -- e o
        # comportamento desejado, porque classificar um quadro velho nao tem
        # valor e a memoria nao pode crescer.
        self._lock_analise = threading.Lock()
        self._quadro_para_analisar: np.ndarray | None = None
        self._ha_quadro = threading.Event()
        self._thread_analise: threading.Thread | None = None

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

    def parar(self, timeout: float = 3.0) -> bool:
        """Sinaliza a parada e aguarda o encerramento.

        Devolve True se a thread terminou (camera garantidamente liberada pelo
        `finally` de `run`). Devolve False se o prazo expirou -- caso em que a
        thread continua detendo o dispositivo, tipicamente porque `cap.read()`
        travou. O chamador NAO deve tentar reabrir a camera nesse estado.
        """
        self._parar.set()
        self._pausado.clear()
        self._ha_quadro.set()          # libera a analise do wait()
        if self.is_alive():
            self.join(timeout)
        if self._thread_analise is not None and self._thread_analise.is_alive():
            self._thread_analise.join(min(timeout, 2.0))
        return not self.is_alive()

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

    # --------------------------------------------------------------- analise
    def _laco_de_analise(self) -> None:
        """Infere sobre o quadro mais recente, no ritmo que conseguir."""
        while not self._parar.is_set():
            if not self._ha_quadro.wait(timeout=0.2):
                continue
            self._ha_quadro.clear()
            with self._lock_analise:
                quadro, self._quadro_para_analisar = self._quadro_para_analisar, None
            if quadro is None or self._parar.is_set():
                continue
            try:
                resultado = self._pipeline.processar(quadro)
                if resultado.erro:
                    self._registrar_erro(resultado.erro)
            except Exception as exc:  # nunca derrubar a thread de analise
                self._registrar_erro(f"Falha no processamento: {exc}")
                resultado = ResultadoQuadro(erro=str(exc), processado=False)
            with self._lock:
                self._ultimo_resultado = resultado

    # -------------------------------------------------------------------- run
    def run(self) -> None:  # noqa: C901 - laco de captura concentra o fluxo
        falhas_leitura = 0
        self._thread_analise = threading.Thread(
            target=self._laco_de_analise, name="EmotionScan-Analise", daemon=True
        )
        self._thread_analise.start()
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
                        # Nunca manter referencia a um capture ja liberado: ler
                        # dele devolveria falhas ate a recuperacao reabrir o
                        # dispositivo ERRADO (o anterior) limpando o erro.
                        self._cap = None
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

                if self._cap is None:
                    # Troca de camera falhou e nao ha dispositivo valido: tenta
                    # reabrir o indice corrente sem bloquear a thread.
                    try:
                        self._cap = _abrir(self._ajustes.indice_camera)
                        self._configurar(self._cap)
                        self._registrar_erro(None)
                        falhas_leitura = 0
                    except ErroCamera as exc:
                        self._registrar_erro(str(exc))
                        time.sleep(0.5)
                    continue

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
                            self._cap = None
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

                # Entrega o quadro para a thread de inferencia e publica
                # IMEDIATAMENTE. Esta thread nao espera classificacao nenhuma:
                # se ela esperasse, perderia a cadencia da camera (ver o
                # comentario no topo da classe).
                passo = max(1, int(self._ajustes.inferir_a_cada))
                if self._contador % passo == 0:
                    with self._lock_analise:
                        self._quadro_para_analisar = quadro
                    self._ha_quadro.set()

                with self._lock:
                    resultado = self._ultimo_resultado
                self._publicar(quadro, resultado)

                # Sem sleep aqui: `cap.read()` ja bloqueia ate o proximo quadro
                # da camera. Dormir alem disso faria perder o ponto seguinte da
                # grade e derrubaria a taxa pela metade.
        except Exception as exc:  # pragma: no cover - rede de seguranca final
            self._registrar_erro(f"Erro inesperado na captura: {exc}")
            self._pronto.set()
        finally:
            self._ha_quadro.set()   # libera a thread de inferencia do wait()
            if self._cap is not None:
                self._cap.release()
                self._cap = None
