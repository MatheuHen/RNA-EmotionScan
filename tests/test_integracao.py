"""Integracao entre componentes, robustez a condicoes do workshop e estabilidade.

Os testes de condicao usam transformacoes controladas sobre a imagem de
referencia (brilho, escala, rotacao, oclusao). Eles medem se o PIPELINE
continua funcional e numericamente valido -- NAO afirmam acuracia sobre
pessoas reais, que depende de validacao presencial.
"""
from __future__ import annotations

import gc
import os
import time

import cv2 as cv
import numpy as np
import pytest

from expressao_neural.config import Ajustes
from expressao_neural.core.pipeline import PipelineExpressao

from .conftest import modelos_disponiveis

pytestmark = modelos_disponiveis


# ======================================================== integracao ponta a ponta
def test_integracao_pipeline_completo(pipeline, imagem_rosto) -> None:
    pipeline.reiniciar_estado()
    res = pipeline.processar(imagem_rosto)
    assert res.erro is None
    assert res.quantidade_rostos >= 1
    alvo = res.principal
    assert alvo is not None
    # Cada etapa alimentou a seguinte:
    assert alvo.rosto.pontos.shape == (5, 2)          # detecção
    assert alvo.predicao.probabilidades.shape == (7,)  # classificação
    assert alvo.probabilidades.shape == (7,)           # suavização
    assert alvo.id_trilha >= 1                         # acompanhamento
    assert res.tempo_deteccao_ms > 0 and res.tempo_inferencia_ms > 0
    assert res.tempo_total_ms >= res.tempo_deteccao_ms


def test_integracao_com_captura_simulada(imagem_rosto) -> None:
    """Thread de captura + pipeline real, sem hardware de câmera."""
    from expressao_neural.core import camera as mod_camera
    from expressao_neural.core.camera import TrabalhadorCamera

    class CameraDeImagem:
        def __init__(self, quadro):
            self._quadro = quadro
            self.liberada = False

        def isOpened(self):  # noqa: N802
            return True

        def read(self):
            return True, self._quadro.copy()

        def set(self, *_a):
            return True

        def release(self):
            self.liberada = True

    criadas = []
    original = mod_camera.cv.VideoCapture

    def fabricar(*_a, **_k):
        cam = CameraDeImagem(imagem_rosto)
        criadas.append(cam)
        return cam

    mod_camera.cv.VideoCapture = fabricar
    try:
        p = PipelineExpressao(Ajustes(inferir_a_cada=1, fps_alvo=30, espelhar_imagem=False))
        t = TrabalhadorCamera(p, Ajustes(inferir_a_cada=1, fps_alvo=30, espelhar_imagem=False))
        t.start()
        try:
            assert t.aguardar_pronto(10.0)
            pronto = None
            prazo = time.time() + 6.0
            while time.time() < prazo:
                pronto = t.obter()
                if pronto is not None and pronto.resultado.quantidade_rostos >= 1:
                    break
                time.sleep(0.02)
            assert pronto is not None, "nenhum quadro foi publicado"
            assert pronto.resultado.quantidade_rostos >= 1
            assert pronto.fps >= 0.0
        finally:
            t.parar()
        p.liberar()
    finally:
        mod_camera.cv.VideoCapture = original
    assert criadas and all(c.liberada for c in criadas)


def test_reinicio_do_estado_nao_afeta_a_inferencia(pipeline, imagem_rosto) -> None:
    antes = pipeline.processar(imagem_rosto, suavizar=False).principal
    pipeline.reiniciar_estado()
    assert pipeline.suavizador.quantidade_trilhas == 0
    depois = pipeline.processar(imagem_rosto, suavizar=False).principal
    assert antes is not None and depois is not None
    assert np.allclose(antes.predicao.probabilidades, depois.predicao.probabilidades)


def test_ajustes_aplicados_em_tempo_de_execucao(pipeline, imagem_rosto) -> None:
    pipeline.aplicar_ajustes(Ajustes(multiplos_rostos=False, suavizacao_alfa=0.9))
    assert pipeline.detector.max_rostos == 1
    assert pipeline.suavizador.alfa == pytest.approx(0.9)
    res = pipeline.processar(imagem_rosto)
    assert res.quantidade_rostos <= 1
    pipeline.aplicar_ajustes(Ajustes(multiplos_rostos=True, max_rostos=5))
    assert pipeline.detector.max_rostos == 5


# ================================================= condicoes do workshop
def _descreve(res) -> str:
    alvo = res.principal
    if alvo is None:
        return "nenhum rosto"
    return f"{alvo.rotulo} {alvo.confianca * 100:.1f}%"


@pytest.mark.parametrize(
    "nome,alpha,beta",
    [
        ("iluminação muito reduzida", 0.35, -10),
        ("iluminação reduzida", 0.60, 0),
        ("iluminação normal", 1.00, 0),
        ("iluminação elevada", 1.45, 25),
        ("iluminação excessiva (estouro)", 1.90, 60),
    ],
)
def test_condicoes_de_iluminacao(pipeline, imagem_rosto, nome, alpha, beta) -> None:
    """O sistema deve permanecer estável e numericamente válido, ou não detectar."""
    img = cv.convertScaleAbs(imagem_rosto, alpha=alpha, beta=beta)
    pipeline.reiniciar_estado()
    res = pipeline.processar(img, suavizar=False)
    assert res.erro is None, f"{nome}: {res.erro}"
    for r in res.rostos:
        assert pytest.approx(1.0, abs=1e-4) == float(r.predicao.probabilidades.sum())
        assert np.isfinite(r.predicao.probabilidades).all()
    print(f"    {nome}: {_descreve(res)}")


@pytest.mark.parametrize("fator", [0.35, 0.5, 0.75, 1.0, 1.5])
def test_diferentes_distancias_da_camera(pipeline, imagem_rosto, fator) -> None:
    """Escala da imagem simula aproximar/afastar o rosto da câmera."""
    h, w = imagem_rosto.shape[:2]
    img = cv.resize(imagem_rosto, (int(w * fator), int(h * fator)))
    pipeline.reiniciar_estado()
    res = pipeline.processar(img, suavizar=False)
    assert res.erro is None
    for r in res.rostos:
        assert pytest.approx(1.0, abs=1e-4) == float(r.predicao.probabilidades.sum())
    print(f"    escala {fator:.2f} ({img.shape[1]}x{img.shape[0]}): {_descreve(res)}")


@pytest.mark.parametrize("graus", [-20, -10, 0, 10, 20])
def test_rosto_inclinado(pipeline, imagem_rosto, graus) -> None:
    """Movimentação/rotação da cabeça: o alinhamento deve absorver a inclinação."""
    h, w = imagem_rosto.shape[:2]
    m = cv.getRotationMatrix2D((w / 2, h / 2), graus, 1.0)
    img = cv.warpAffine(imagem_rosto, m, (w, h), borderMode=cv.BORDER_REPLICATE)
    pipeline.reiniciar_estado()
    res = pipeline.processar(img, suavizar=False)
    assert res.erro is None
    for r in res.rostos:
        assert np.isfinite(r.predicao.probabilidades).all()
    print(f"    inclinação {graus:+d}°: {_descreve(res)}")


def test_oclusao_parcial_tipo_oculos(pipeline, imagem_rosto) -> None:
    """Faixa escura sobre os olhos aproxima o efeito de óculos escuros."""
    img = imagem_rosto.copy()
    rostos = pipeline.detector.detectar(imagem_rosto)
    if not rostos:
        pytest.skip("rosto de referência não detectado")
    r = rostos[0]
    y = r.y + int(r.altura * 0.28)
    cv.rectangle(
        img, (r.x, y), (r.x + r.largura, y + int(r.altura * 0.16)), (30, 30, 30), -1
    )
    pipeline.reiniciar_estado()
    res = pipeline.processar(img, suavizar=False)
    assert res.erro is None
    print(f"    oclusão dos olhos: {_descreve(res)}")


def test_ruido_e_desfoque_nao_quebram_o_pipeline(pipeline, imagem_rosto) -> None:
    rng = np.random.default_rng(3)
    ruidosa = np.clip(
        imagem_rosto.astype(np.int16) + rng.integers(-35, 36, imagem_rosto.shape),
        0,
        255,
    ).astype(np.uint8)
    desfocada = cv.GaussianBlur(imagem_rosto, (11, 11), 0)
    for nome, img in (("ruído", ruidosa), ("desfoque", desfocada)):
        pipeline.reiniciar_estado()
        res = pipeline.processar(img, suavizar=False)
        assert res.erro is None, f"{nome}: {res.erro}"
        print(f"    {nome}: {_descreve(res)}")


@pytest.mark.parametrize(
    "nome,imagem_fn",
    [
        ("totalmente preta", lambda: np.zeros((360, 480, 3), dtype=np.uint8)),
        ("totalmente branca", lambda: np.full((360, 480, 3), 255, dtype=np.uint8)),
        ("faixa estreita", lambda: np.zeros((40, 480, 3), dtype=np.uint8)),
        ("quadro minúsculo", lambda: np.zeros((8, 8, 3), dtype=np.uint8)),
    ],
)
def test_entradas_degeneradas_nao_derrubam(pipeline, nome, imagem_fn) -> None:
    pipeline.reiniciar_estado()
    res = pipeline.processar(imagem_fn(), suavizar=False)
    assert res.erro is None, f"{nome}: {res.erro}"
    assert res.quantidade_rostos == 0


def test_multiplos_rostos_nao_misturam_resultados(
    pipeline, imagem_multiplos_rostos
) -> None:
    """RF08: cada rosto mantém a própria trilha ao longo dos quadros."""
    pipeline.aplicar_ajustes(Ajustes(multiplos_rostos=True, max_rostos=5))
    pipeline.reiniciar_estado()
    ids_por_quadro = []
    for _ in range(6):
        res = pipeline.processar(imagem_multiplos_rostos)
        if res.quantidade_rostos >= 2:
            ids_por_quadro.append(tuple(sorted(r.id_trilha for r in res.rostos)))
    assert ids_por_quadro, "não foi possível detectar dois rostos na fixture"
    assert len(set(ids_por_quadro)) == 1, "as trilhas devem permanecer estáveis"
    assert len(set(ids_por_quadro[0])) == len(ids_por_quadro[0])


# ====================================================== desempenho e memoria
def test_desempenho_cpu_medido(pipeline, imagem_rosto) -> None:
    """Mede o ciclo real nesta máquina. Não define meta: apenas registra."""
    for _ in range(3):
        pipeline.processar(imagem_rosto)
    amostras = []
    for _ in range(25):
        t = time.perf_counter()
        pipeline.processar(imagem_rosto)
        amostras.append((time.perf_counter() - t) * 1000.0)
    media = sum(amostras) / len(amostras)
    print(
        f"\n    ciclo completo: média {media:.1f} ms | mín {min(amostras):.1f} | "
        f"máx {max(amostras):.1f} ms  (~{1000 / media:.1f} inf/s, 1 rosto)"
    )
    assert np.isfinite(media) and media > 0
    # Limite generoso: acima disso o tempo real fica inviável mesmo em CPU modesta.
    assert media < 1000, f"ciclo excessivamente lento: {media:.0f} ms"


def test_execucao_repetida_nao_vaza_memoria(pipeline, imagem_rosto) -> None:
    """Proxy rápido do teste prolongado: nada deve crescer sem limite."""
    for _ in range(50):
        pipeline.processar(imagem_rosto)
    gc.collect()
    assert pipeline.suavizador.quantidade_trilhas <= 5
    trilhas = pipeline.suavizador._trilhas
    for trilha in trilhas.values():
        assert len(trilha.historico) <= 60


def test_rostos_que_desaparecem_nao_acumulam_trilhas(pipeline, imagem_rosto) -> None:
    vazio = np.zeros((360, 480, 3), dtype=np.uint8)
    for _ in range(12):
        pipeline.processar(imagem_rosto)
        for _ in range(6):
            pipeline.processar(vazio)
    assert pipeline.suavizador.quantidade_trilhas <= 3


@pytest.mark.lento
def test_execucao_prolongada(pipeline, imagem_rosto) -> None:
    """Execução contínua (RF: estabilidade por >= 30 min na apresentação).

    Duração configurável por variável de ambiente, para permitir tanto a
    verificação rápida do dia a dia quanto o ensaio completo:

        EXPRESSAO_NEURAL_DURACAO_MIN=30  pytest -m lento
    """
    minutos = float(os.environ.get("EXPRESSAO_NEURAL_DURACAO_MIN", "2"))
    limite = time.time() + minutos * 60
    ciclos = 0
    tempos: list[float] = []
    vazio = np.zeros(imagem_rosto.shape, dtype=np.uint8)

    while time.time() < limite:
        t = time.perf_counter()
        pipeline.processar(imagem_rosto if ciclos % 10 != 9 else vazio)
        tempos.append((time.perf_counter() - t) * 1000.0)
        ciclos += 1

    assert ciclos > 0
    metade = len(tempos) // 2
    inicio = sum(tempos[:metade]) / max(1, metade)
    fim = sum(tempos[metade:]) / max(1, len(tempos) - metade)
    print(
        f"\n    {minutos:.0f} min | {ciclos} ciclos | primeira metade {inicio:.1f} ms | "
        f"segunda metade {fim:.1f} ms | trilhas vivas "
        f"{pipeline.suavizador.quantidade_trilhas}"
    )
    # Sem degradação progressiva: a segunda metade não pode ser muito mais lenta.
    assert fim < inicio * 2.0 + 20, "houve degradação de desempenho ao longo do tempo"
    assert pipeline.suavizador.quantidade_trilhas <= 5, "vazamento de trilhas"
