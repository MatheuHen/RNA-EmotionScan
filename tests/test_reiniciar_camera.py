"""Regressão do botão Reiniciar — bug relatado em uso real.

Mecanismo do defeito (confirmado por leitura do código antes da correção):

    reiniciar_camera() encerrava o trabalhador e agendava iniciar_camera()
    para 300 ms depois, mas NUNCA alterava self._modo. Como iniciar_camera()
    começa com `if self._modo in ("ativo", "pausado"): return`, a reabertura
    retornava na primeira linha e nenhum trabalhador novo era criado.

    Resultado: interface viva, botões habilitados, _modo == "ativo",
    _trabalhador is None — e nenhuma imagem, para sempre. Só "Encerrar"
    devolvia o sistema a um estado iniciável.

Segundo defeito no mesmo caminho: parar() faz join(3 s) na thread do Tk,
bloqueando o laço de eventos e podendo marcar a janela como "Não respondendo".
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from expressao_neural.config import Ajustes
from expressao_neural.core import camera as mod_camera

tk = pytest.importorskip("tkinter")


class _CameraFake:
    """Dublê que entrega quadros válidos indefinidamente."""

    def __init__(self, atraso: float = 0.0) -> None:
        self._atraso = atraso
        self.liberada = False

    def isOpened(self):  # noqa: N802
        return True

    def read(self):
        if self._atraso:
            time.sleep(self._atraso)
        return True, np.zeros((48, 64, 3), dtype=np.uint8)

    def set(self, *_a):
        return True

    def release(self):
        self.liberada = True


@pytest.fixture()
def janela(monkeypatch):
    """Janela real com câmera dublê — não aciona hardware."""
    criadas: list[_CameraFake] = []

    def fabricar(*_a, **_k):
        cam = _CameraFake()
        criadas.append(cam)
        return cam

    monkeypatch.setattr(mod_camera.cv, "VideoCapture", fabricar)
    monkeypatch.setattr("tkinter.messagebox.showerror", lambda *_a, **_k: None)

    from expressao_neural.ui.janela import JanelaPrincipal

    try:
        j = JanelaPrincipal()
    except tk.TclError:  # pragma: no cover
        pytest.skip("ambiente sem interface gráfica disponível")
    j.ajustes = Ajustes(fps_alvo=60, inferir_a_cada=1)
    j._cameras_criadas = criadas  # type: ignore[attr-defined]
    yield j
    j.encerrar()


def _esperar(janela, condicao, prazo=8.0):
    """Processa eventos do Tk até a condição valer ou o prazo acabar."""
    fim = time.time() + prazo
    while time.time() < fim:
        janela.update()
        if condicao():
            return True
        time.sleep(0.02)
    return False


# ============================================================ o bug central
def test_reiniciar_recria_o_trabalhador(janela) -> None:
    """Depois de Reiniciar, a captura precisa voltar a existir e publicar."""
    if not janela._garantir_pipeline():
        pytest.skip("modelos indisponíveis")

    janela.iniciar_camera()
    assert _esperar(janela, lambda: janela._trabalhador is not None)
    primeiro = janela._trabalhador
    assert _esperar(janela, lambda: primeiro.obter() is not None), (
        "a captura inicial não publicou quadro"
    )

    janela.reiniciar_camera()

    # O after(300) precisa ser processado.
    assert _esperar(
        janela,
        lambda: janela._trabalhador is not None and janela._trabalhador is not primeiro,
    ), "Reiniciar não recriou o trabalhador — a câmera não volta"

    segundo = janela._trabalhador
    assert segundo.ativo
    assert _esperar(janela, lambda: segundo.obter() is not None), (
        "a captura reiniciada não voltou a publicar quadros"
    )
    assert janela._modo == "ativo"


def test_reiniciar_nao_deixa_modo_incoerente_com_trabalhador(janela) -> None:
    """Enquanto a captura não volta, _modo não pode afirmar que está ativa.

    Era exatamente essa incoerência que bloqueava a reabertura: _modo dizia
    "ativo" com _trabalhador nulo, e a guarda de iniciar_camera abortava.
    """
    if not janela._garantir_pipeline():
        pytest.skip("modelos indisponíveis")

    janela.iniciar_camera()
    assert _esperar(janela, lambda: janela._trabalhador is not None)

    janela.reiniciar_camera()
    # Logo após o encerramento e antes do after(300) ser processado:
    if janela._trabalhador is None:
        assert janela._modo != "ativo", (
            '_modo não pode ser "ativo" sem trabalhador — a guarda de '
            "iniciar_camera abortaria a reabertura"
        )

    assert _esperar(janela, lambda: janela._trabalhador is not None)


def test_reiniciar_duas_vezes_seguidas(janela) -> None:
    """Clicar duas vezes em Reiniciar não pode deixar o sistema sem câmera."""
    if not janela._garantir_pipeline():
        pytest.skip("modelos indisponíveis")

    janela.iniciar_camera()
    assert _esperar(janela, lambda: janela._trabalhador is not None)

    janela.reiniciar_camera()
    janela.update()
    janela.reiniciar_camera()

    assert _esperar(janela, lambda: janela._trabalhador is not None), (
        "duplo Reiniciar deixou o sistema sem captura"
    )
    assert janela._trabalhador.ativo


def test_reiniciar_com_camera_parada_apenas_inicia(janela) -> None:
    """Sem trabalhador, Reiniciar deve se comportar como Iniciar."""
    if not janela._garantir_pipeline():
        pytest.skip("modelos indisponíveis")

    assert janela._trabalhador is None
    janela.reiniciar_camera()
    assert _esperar(janela, lambda: janela._trabalhador is not None)


# =================================== segundo defeito: bloqueio da thread do Tk
def test_reiniciar_nao_bloqueia_a_interface(janela, monkeypatch) -> None:
    """A interface não pode congelar por segundos durante o Reiniciar.

    Com uma câmera cujo `read()` demora, `parar()` estoura o join. Se esse
    join rodasse na thread do Tk, a janela deixaria de responder — era o
    comportamento anterior (medido em 1,43 s de congelamento).

    Usa a MESMA janela da fixture: o Tkinter só admite uma raiz por processo.
    """
    if not janela._garantir_pipeline():
        pytest.skip("modelos indisponíveis")

    # A partir daqui, toda câmera aberta tem read() lento de propósito.
    monkeypatch.setattr(
        mod_camera.cv, "VideoCapture", lambda *_a, **_k: _CameraFake(atraso=1.5)
    )
    janela.iniciar_camera()
    assert _esperar(janela, lambda: janela._trabalhador is not None, 5.0)

    inicio = time.perf_counter()
    janela.reiniciar_camera()
    bloqueio = time.perf_counter() - inicio

    assert bloqueio < 1.0, (
        f"reiniciar_camera bloqueou a thread da interface por {bloqueio:.2f} s "
        "— a janela congela nesse período"
    )
    # E o reinício precisa concluir mesmo com a câmera lenta.
    assert _esperar(janela, lambda: not janela._reiniciando, 12.0)
