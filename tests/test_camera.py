"""Ciclo de vida da captura: indisponibilidade, pausa, reinicio, liberacao."""
from __future__ import annotations

import time

import numpy as np
import pytest

from expressao_neural.config import Ajustes
from expressao_neural.core import camera as mod_camera
from expressao_neural.core.camera import ErroCamera, TrabalhadorCamera
from expressao_neural.core.pipeline import ResultadoQuadro


class CameraFalsa:
    """Dublê de `cv2.VideoCapture` para exercitar o laço sem hardware."""

    def __init__(self, quadros: int = 10**9, abre: bool = True, falhar_apos: int | None = None):
        self._abre = abre
        self._restantes = quadros
        self._falhar_apos = falhar_apos
        self._lidos = 0
        self.liberada = False

    def isOpened(self) -> bool:  # noqa: N802 - assinatura do OpenCV
        return self._abre

    def read(self):
        self._lidos += 1
        if self._falhar_apos is not None and self._lidos > self._falhar_apos:
            return False, None
        if self._restantes <= 0:
            return False, None
        self._restantes -= 1
        quadro = np.full((48, 64, 3), 40, dtype=np.uint8)
        return True, quadro

    def set(self, *_args) -> bool:
        return True

    def release(self) -> None:
        self.liberada = True


class PipelineFalso:
    """Pipeline sem rede neural: conta chamadas e devolve resultado vazio."""

    def __init__(self) -> None:
        self.chamadas = 0
        self.reinicios = 0

    def processar(self, _quadro, suavizar=None) -> ResultadoQuadro:
        self.chamadas += 1
        return ResultadoQuadro(tempo_inferencia_ms=1.0)

    def reiniciar_estado(self) -> None:
        self.reinicios += 1


# -------------------------------------------------------------- indisponivel
def test_camera_indisponivel_gera_mensagem_acionavel(monkeypatch) -> None:
    monkeypatch.setattr(
        mod_camera.cv, "VideoCapture", lambda *_a, **_k: CameraFalsa(abre=False)
    )
    with pytest.raises(ErroCamera) as exc:
        mod_camera._abrir(0)
    texto = str(exc.value)
    assert "câmera" in texto.lower()
    assert "permiss" in texto.lower()


def test_trabalhador_reporta_erro_sem_derrubar_a_thread(monkeypatch) -> None:
    monkeypatch.setattr(
        mod_camera.cv, "VideoCapture", lambda *_a, **_k: CameraFalsa(abre=False)
    )
    t = TrabalhadorCamera(PipelineFalso(), Ajustes())
    t.start()
    assert t.aguardar_pronto(5.0)
    t.join(3.0)
    assert t.erro is not None
    assert not t.is_alive()
    assert t.obter() is None


def test_listar_cameras_tolera_falha_do_backend(monkeypatch) -> None:
    def explode(*_a, **_k):
        raise mod_camera.cv.error("backend indisponível")

    monkeypatch.setattr(mod_camera.cv, "VideoCapture", explode)
    assert mod_camera.listar_cameras(2) == []


# ---------------------------------------------------------------- ciclo de vida
def _trabalhador(monkeypatch, **kw) -> tuple[TrabalhadorCamera, PipelineFalso, list]:
    criadas: list[CameraFalsa] = []

    def fabricar(*_a, **_k):
        cam = CameraFalsa(**kw)
        criadas.append(cam)
        return cam

    monkeypatch.setattr(mod_camera.cv, "VideoCapture", fabricar)
    pipeline = PipelineFalso()
    ajustes = Ajustes(fps_alvo=60, inferir_a_cada=1, espelhar_imagem=False)
    return TrabalhadorCamera(pipeline, ajustes), pipeline, criadas


def test_captura_publica_quadros_e_libera_a_camera(monkeypatch) -> None:
    t, pipeline, criadas = _trabalhador(monkeypatch)
    t.start()
    try:
        assert t.aguardar_pronto(5.0)
        pronto = None
        prazo = time.time() + 3.0
        while pronto is None and time.time() < prazo:
            pronto = t.obter()
            time.sleep(0.01)
        assert pronto is not None
        assert pronto.quadro.shape == (48, 64, 3)
        assert pipeline.chamadas >= 1
    finally:
        t.parar()
    assert not t.is_alive()
    assert criadas and all(c.liberada for c in criadas)


def test_pausa_interrompe_o_processamento_e_retoma(monkeypatch) -> None:
    t, pipeline, _ = _trabalhador(monkeypatch)
    t.start()
    try:
        assert t.aguardar_pronto(5.0)
        time.sleep(0.2)
        t.pausar()
        assert t.pausado
        time.sleep(0.15)
        antes = pipeline.chamadas
        time.sleep(0.25)
        assert pipeline.chamadas == antes, "pausa deve cessar o processamento"
        t.retomar()
        assert not t.pausado
        time.sleep(0.25)
        assert pipeline.chamadas > antes
    finally:
        t.parar()


def test_passo_de_inferencia_reduz_chamadas(monkeypatch) -> None:
    t, pipeline, _ = _trabalhador(monkeypatch)
    t._ajustes.inferir_a_cada = 3
    t.start()
    try:
        assert t.aguardar_pronto(5.0)
        time.sleep(0.6)
        quadros = t._contador
        assert quadros >= 3
        # Tolerancia de 1 pela janela de amostragem.
        assert pipeline.chamadas <= quadros // 3 + 1
    finally:
        t.parar()


def test_recuperacao_apos_falhas_de_leitura(monkeypatch) -> None:
    """RF02: apos falhas sucessivas o dispositivo e reaberto."""
    t, _, criadas = _trabalhador(monkeypatch, falhar_apos=3)
    t.start()
    try:
        assert t.aguardar_pronto(5.0)
        prazo = time.time() + 6.0
        while len(criadas) < 2 and time.time() < prazo:
            time.sleep(0.05)
        assert len(criadas) >= 2, "a captura deveria ter tentado reabrir a câmera"
    finally:
        t.parar()
    assert all(c.liberada for c in criadas[:-1])


def test_troca_de_camera_reinicia_o_estado_temporal(monkeypatch) -> None:
    t, pipeline, criadas = _trabalhador(monkeypatch)
    t.start()
    try:
        assert t.aguardar_pronto(5.0)
        t.trocar_camera(1)
        prazo = time.time() + 4.0
        while pipeline.reinicios < 1 and time.time() < prazo:
            time.sleep(0.03)
        assert pipeline.reinicios >= 1
        assert t.indice_camera == 1
    finally:
        t.parar()


def test_parar_duas_vezes_e_seguro(monkeypatch) -> None:
    t, _, _ = _trabalhador(monkeypatch)
    t.start()
    assert t.aguardar_pronto(5.0)
    t.parar()
    t.parar()
    assert not t.is_alive()


def test_falha_no_pipeline_nao_mata_a_captura(monkeypatch) -> None:
    class PipelineQuebrado(PipelineFalso):
        def processar(self, _quadro, suavizar=None):
            self.chamadas += 1
            raise RuntimeError("falha sintética de inferência")

    criadas: list[CameraFalsa] = []

    def fabricar(*_a, **_k):
        cam = CameraFalsa()
        criadas.append(cam)
        return cam

    monkeypatch.setattr(mod_camera.cv, "VideoCapture", fabricar)
    pipeline = PipelineQuebrado()
    t = TrabalhadorCamera(pipeline, Ajustes(fps_alvo=60, inferir_a_cada=1))
    t.start()
    try:
        assert t.aguardar_pronto(5.0)
        time.sleep(0.3)
        assert t.is_alive(), "a thread de captura não deve morrer por erro de inferência"
        assert t.erro is not None and "sintética" in t.erro
    finally:
        t.parar()
    assert all(c.liberada for c in criadas)


def test_slot_nao_acumula_quadros(monkeypatch) -> None:
    """Controle de fila: so existe o quadro mais recente."""
    t, _, _ = _trabalhador(monkeypatch)
    t.start()
    try:
        assert t.aguardar_pronto(5.0)
        time.sleep(0.4)
        primeiro = t.obter()
        assert primeiro is not None
        assert t.obter() is None, "o slot deve ficar vazio após o consumo"
    finally:
        t.parar()
