"""Pre-processamento, interpretacao de saida, limiares e inferencia real."""
from __future__ import annotations

import numpy as np
import pytest

from expressao_neural import config
from expressao_neural.core.alinhamento import ErroAlinhamento, alinhar_rosto, matriz_alinhamento
from expressao_neural.core.classificador import softmax

from .conftest import modelos_disponiveis


# ------------------------------------------------------------------ mapeamento
def test_sete_categorias_consistentes() -> None:
    assert len(config.CLASSES_MODELO) == 7
    assert len(config.CLASSES_PT) == 7
    assert len(config.CORES_CLASSES) == 7


def test_ordem_das_classes_bate_com_o_modelo_oficial() -> None:
    """Ordem exata de `FacialExpressionRecog.getDesc` (OpenCV Zoo)."""
    assert config.CLASSES_MODELO == (
        "angry", "disgust", "fearful", "happy", "neutral", "sad", "surprised",
    )


def test_traducao_alinhada_indice_por_indice() -> None:
    esperado = {
        "angry": "Raiva",
        "disgust": "Nojo",
        "fearful": "Medo",
        "happy": "Alegria",
        "neutral": "Neutra",
        "sad": "Tristeza",
        "surprised": "Surpresa",
    }
    for i, chave in enumerate(config.CLASSES_MODELO):
        assert config.CLASSES_PT[i] == esperado[chave]


# --------------------------------------------------------------------- softmax
def test_softmax_normaliza_e_preserva_argmax() -> None:
    logits = np.array([-3.072, -1.436, -4.826, 4.568, 6.266, -4.749, 3.186], dtype=np.float32)
    p = softmax(logits)
    assert p.shape == (7,)
    assert pytest.approx(1.0, abs=1e-5) == float(p.sum())
    assert (p >= 0).all() and (p <= 1).all()
    assert int(np.argmax(p)) == int(np.argmax(logits)) == 4  # neutral


def test_softmax_estavel_com_valores_extremos() -> None:
    p = softmax(np.array([1000.0, 1001.0, -1000.0], dtype=np.float32))
    assert np.isfinite(p).all()
    assert pytest.approx(1.0, abs=1e-5) == float(p.sum())


# ---------------------------------------------------------------- alinhamento
def test_matriz_de_alinhamento_tem_forma_afim() -> None:
    pontos = np.array(config.PONTOS_PADRAO_5, dtype=np.float32)
    m = matriz_alinhamento(pontos)
    assert m.shape == (2, 3)
    # Pontos ja na posicao padrao -> transformacao praticamente identidade.
    assert np.allclose(m[:, :2], np.eye(2), atol=1e-3)
    assert np.allclose(m[:, 2], 0.0, atol=1e-3)


def test_alinhamento_gera_112x112(imagem_rosto: np.ndarray) -> None:
    pontos = np.array(
        [[230, 260], [300, 260], [265, 300], [240, 340], [295, 340]], dtype=np.float32
    )
    recorte = alinhar_rosto(imagem_rosto, pontos)
    assert recorte.shape == (112, 112, 3)


@pytest.mark.parametrize(
    "pontos",
    [
        np.zeros((3, 2), dtype=np.float32),
        np.full((5, 2), np.nan, dtype=np.float32),
    ],
)
def test_alinhamento_rejeita_pontos_invalidos(imagem_rosto, pontos) -> None:
    with pytest.raises(ErroAlinhamento):
        alinhar_rosto(imagem_rosto, pontos)


# ------------------------------------------------------- pre-processamento real
@modelos_disponiveis
def test_blob_de_entrada_respeita_a_especificacao() -> None:
    from expressao_neural.core.classificador import ClassificadorExpressao

    recorte = np.full((112, 112, 3), 255, dtype=np.uint8)
    blob = ClassificadorExpressao.preprocessar(recorte)
    assert blob.shape == (1, 3, 112, 112)
    # (255/255 - 0.5)/0.5 == 1.0
    assert pytest.approx(1.0, abs=1e-5) == float(blob.max())

    recorte_zero = np.zeros((112, 112, 3), dtype=np.uint8)
    blob_zero = ClassificadorExpressao.preprocessar(recorte_zero)
    assert pytest.approx(-1.0, abs=1e-5) == float(blob_zero.min())


@modelos_disponiveis
def test_preprocessar_redimensiona_entrada_fora_de_112() -> None:
    from expressao_neural.core.classificador import ClassificadorExpressao

    blob = ClassificadorExpressao.preprocessar(np.zeros((200, 150, 3), dtype=np.uint8))
    assert blob.shape == (1, 3, 112, 112)


@modelos_disponiveis
def test_preprocessar_rejeita_entrada_invalida() -> None:
    from expressao_neural.core.classificador import ClassificadorExpressao

    with pytest.raises(ValueError):
        ClassificadorExpressao.preprocessar(np.zeros((112, 112), dtype=np.uint8))


# ---------------------------------------------------------- inferencia de fato
@modelos_disponiveis
def test_deteccao_em_imagem_com_rosto(pipeline, imagem_rosto) -> None:
    rostos = pipeline.detector.detectar(imagem_rosto)
    assert len(rostos) >= 1
    r = rostos[0]
    assert r.largura > 0 and r.altura > 0
    assert 0.0 <= r.confianca <= 1.0
    assert r.pontos.shape == (5, 2)


@modelos_disponiveis
def test_ausencia_de_rosto_nao_classifica(pipeline, imagem_sem_rosto) -> None:
    res = pipeline.processar(imagem_sem_rosto)
    assert res.quantidade_rostos == 0
    assert res.rostos == []
    assert res.erro is None


@modelos_disponiveis
def test_inferencia_real_produz_distribuicao_valida(pipeline, imagem_rosto) -> None:
    res = pipeline.processar(imagem_rosto, suavizar=False)
    assert res.quantidade_rostos >= 1
    alvo = res.principal
    assert alvo is not None
    probs = alvo.predicao.probabilidades
    assert probs.shape == (7,)
    assert pytest.approx(1.0, abs=1e-4) == float(probs.sum())
    assert 0 <= alvo.indice < 7
    assert alvo.rotulo in config.CLASSES_PT
    assert res.tempo_inferencia_ms > 0.0


@modelos_disponiveis
def test_inferencia_e_deterministica(pipeline, imagem_rosto) -> None:
    """Rede neural real com peso fixo: mesma entrada -> mesma saida."""
    pipeline.reiniciar_estado()
    a = pipeline.processar(imagem_rosto, suavizar=False).principal
    pipeline.reiniciar_estado()
    b = pipeline.processar(imagem_rosto, suavizar=False).principal
    assert a is not None and b is not None
    assert np.allclose(a.predicao.probabilidades, b.predicao.probabilidades, atol=1e-6)


@modelos_disponiveis
def test_saidas_diferem_entre_rostos_diferentes(pipeline, imagem_rosto) -> None:
    """Garante que ha inferencia real, e nao valor fixo devolvido sempre."""
    import cv2 as cv

    pipeline.reiniciar_estado()
    normal = pipeline.processar(imagem_rosto, suavizar=False).principal
    escuro = cv.convertScaleAbs(imagem_rosto, alpha=0.45, beta=0)
    pipeline.reiniciar_estado()
    alterado = pipeline.processar(escuro, suavizar=False).principal
    assert normal is not None and alterado is not None
    assert not np.allclose(
        normal.predicao.probabilidades, alterado.predicao.probabilidades, atol=1e-4
    )


@modelos_disponiveis
def test_limiar_de_confianca(pipeline, imagem_rosto) -> None:
    alvo = pipeline.processar(imagem_rosto, suavizar=False).principal
    assert alvo is not None
    assert alvo.confiavel(0.0) is True
    assert alvo.confiavel(1.01) is False
    assert alvo.rotulo_exibicao(1.01) == "Indefinida"
    assert alvo.rotulo_exibicao(0.0) == alvo.rotulo


@modelos_disponiveis
def test_multiplos_rostos_classificados_individualmente(
    pipeline, imagem_multiplos_rostos
) -> None:
    res = pipeline.processar(imagem_multiplos_rostos, suavizar=False)
    assert res.quantidade_rostos >= 2
    ids = {r.id_trilha for r in res.rostos}
    assert len(ids) == res.quantidade_rostos
    for r in res.rostos:
        assert pytest.approx(1.0, abs=1e-4) == float(r.predicao.probabilidades.sum())
