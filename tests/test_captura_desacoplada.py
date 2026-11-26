"""Separação entre leitura da câmera e inferência.

Problema medido no aplicativo real: `cap.read()` bloqueia ~33 ms esperando o
próximo quadro de uma câmera a 30 FPS — isso já consome todo o orçamento do
quadro. Quando a inferência (~20 ms) roda na MESMA thread, o ciclo passa de
53 ms, perde a cadência da câmera e trava na metade da taxa: 15 FPS.

Medição que motivou a correção:
    taxa bruta da câmera ....... 30,0 FPS
    FPS exibido pelo sistema ... 14,8 a 15,1

A correção separa as duas coisas: uma thread só lê (acompanha os 30 FPS) e
outra infere sobre o quadro mais recente. O vídeo roda na taxa da câmera e a
inferência no ritmo que conseguir.
"""
from __future__ import annotations

import math
import time

import numpy as np
import pytest

from expressao_neural.config import Ajustes
from expressao_neural.core import camera as mod_camera
from expressao_neural.core.pipeline import ResultadoQuadro


class _CameraCadenciada:
    """Dublê de webcam real: os quadros nascem numa GRADE FIXA de tempo.

    Este detalhe é o que reproduz o defeito. Uma webcam a 30 FPS produz quadros
    em instantes fixos (0, 33, 66, 99 ms…), e `read()` bloqueia até o PRÓXIMO
    instante da grade. Quem demora 53 ms para voltar perde um ponto da grade e
    cai no seguinte — 66 ms por ciclo, ou seja, 15 FPS.

    Um dublê que apenas devolvesse o quadro sob demanda NÃO reproduziria o
    problema, e o teste passaria mesmo com o defeito presente.
    """

    def __init__(self, fps: float = 30.0) -> None:
        self._intervalo = 1.0 / fps
        self._origem = time.perf_counter()
        self.liberada = False
        self.leituras = 0

    def isOpened(self):  # noqa: N802
        return True

    def read(self):
        agora = time.perf_counter()
        decorrido = agora - self._origem
        k = math.ceil(decorrido / self._intervalo)
        if abs(k * self._intervalo - decorrido) < 1e-9:
            k += 1                                  # já estamos no limite
        alvo = self._origem + k * self._intervalo
        espera = alvo - agora
        if espera > 0:
            time.sleep(espera)
        self.leituras += 1
        return True, np.zeros((480, 640, 3), dtype=np.uint8)

    def set(self, *_a):
        return True

    def release(self):
        self.liberada = True


# Custo de uma passada de inferência no aplicativo vivo. O pipeline isolado
# mede ~21 ms a 640×480, mas na aplicação ele disputa o GIL com a thread que
# renderiza (PhotoImage, redimensionamento), e o custo efetivo sobe. 35 ms é o
# valor que reproduz o comportamento observado: 15 FPS com câmera de 30 FPS.
CUSTO_INFERENCIA_S = 0.035


class _PipelineLento:
    """Pipeline que custa o mesmo que a inferência real sob contenção."""

    def __init__(self, custo_s: float = CUSTO_INFERENCIA_S) -> None:
        self.custo = custo_s
        self.chamadas = 0

    def processar(self, _quadro, suavizar=None):
        time.sleep(self.custo)
        self.chamadas += 1
        return ResultadoQuadro(tempo_inferencia_ms=self.custo * 1000)

    def reiniciar_estado(self) -> None:
        pass


def _medir_fps(monkeypatch, segundos: float = 4.0, inferir_a_cada: int = 2):
    """Roda o trabalhador com câmera cadenciada e devolve (fps, pipeline, cam)."""
    cameras: list[_CameraCadenciada] = []

    def fabricar(*_a, **_k):
        c = _CameraCadenciada(30.0)
        cameras.append(c)
        return c

    monkeypatch.setattr(mod_camera.cv, "VideoCapture", fabricar)
    pipeline = _PipelineLento()
    ajustes = Ajustes(fps_alvo=30, inferir_a_cada=inferir_a_cada,
                      espelhar_imagem=False)
    t = mod_camera.TrabalhadorCamera(pipeline, ajustes)
    t.start()
    try:
        assert t.aguardar_pronto(10.0)
        time.sleep(0.8)                     # deixa o medidor de FPS encher
        consumidos, fim = 0, time.time() + segundos
        fps_lido = 0.0
        while time.time() < fim:
            pronto = t.obter()
            if pronto is not None:
                consumidos += 1
                fps_lido = pronto.fps
            time.sleep(0.005)               # consumidor rápido, como a interface
        return fps_lido, pipeline, cameras[0], consumidos
    finally:
        t.parar(timeout=5.0)


# ===================================================== o ganho que justifica
def test_captura_sustenta_a_taxa_da_camera(monkeypatch) -> None:
    """Com a leitura desacoplada, o vídeo acompanha os 30 FPS da câmera.

    Na arquitetura anterior (leitura e inferência na mesma thread) este teste
    media ~15 FPS, exatamente metade da taxa da câmera.
    """
    fps, pipeline, cam, consumidos = _medir_fps(monkeypatch, 4.0)
    print(f"\n    FPS publicado: {fps:.1f} | inferências: {pipeline.chamadas} "
          f"| leituras: {cam.leituras} | consumidos: {consumidos}")
    assert fps >= 25.0, (
        f"o vídeo está a {fps:.1f} FPS com uma câmera de 30 FPS — a leitura "
        "ainda está presa à inferência"
    )


def test_inferencia_nao_bloqueia_a_leitura(monkeypatch) -> None:
    """A câmera deve ser lida ~30 vezes por segundo, independente da inferência."""
    _, _, cam, _ = _medir_fps(monkeypatch, 3.0)
    leituras_por_segundo = cam.leituras / 3.8   # inclui o aquecimento
    print(f"\n    leituras/s: {leituras_por_segundo:.1f}")
    assert leituras_por_segundo >= 22.0, (
        f"só {leituras_por_segundo:.1f} leituras/s — a inferência está "
        "atrasando a captura"
    )


def test_inferencia_continua_acontecendo(monkeypatch) -> None:
    """Desacoplar não pode significar parar de inferir."""
    _, pipeline, _, _ = _medir_fps(monkeypatch, 3.0)
    assert pipeline.chamadas >= 15, (
        f"apenas {pipeline.chamadas} inferências em ~3,8 s"
    )


def test_quadro_publicado_carrega_resultado_de_inferencia(monkeypatch) -> None:
    """O quadro exibido precisa vir acompanhado da classificação mais recente."""
    cameras: list[_CameraCadenciada] = []
    monkeypatch.setattr(
        mod_camera.cv, "VideoCapture",
        lambda *_a, **_k: cameras.append(_CameraCadenciada(30.0)) or cameras[-1],
    )
    pipeline = _PipelineLento()
    t = mod_camera.TrabalhadorCamera(
        pipeline, Ajustes(fps_alvo=30, inferir_a_cada=1, espelhar_imagem=False)
    )
    t.start()
    try:
        assert t.aguardar_pronto(10.0)
        time.sleep(0.5)
        encontrou = False
        fim = time.time() + 3.0
        while time.time() < fim and not encontrou:
            pronto = t.obter()
            if pronto is not None and pronto.resultado is not None:
                assert pronto.quadro.shape == (480, 640, 3)
                encontrou = True
            time.sleep(0.005)
        assert encontrou, "nenhum quadro publicado trazia resultado"
    finally:
        t.parar(timeout=5.0)


def test_recursos_liberados_com_as_duas_threads(monkeypatch) -> None:
    """Separar em duas threads não pode vazar a câmera."""
    cameras: list[_CameraCadenciada] = []

    def fabricar(*_a, **_k):
        c = _CameraCadenciada(30.0)
        cameras.append(c)
        return c

    monkeypatch.setattr(mod_camera.cv, "VideoCapture", fabricar)
    t = mod_camera.TrabalhadorCamera(
        _PipelineLento(), Ajustes(fps_alvo=30, inferir_a_cada=1)
    )
    t.start()
    assert t.aguardar_pronto(10.0)
    time.sleep(0.3)
    assert t.parar(timeout=5.0) is True
    assert not t.is_alive()
    assert all(c.liberada for c in cameras), "câmera não foi liberada"
