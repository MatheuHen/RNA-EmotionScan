"""Treinamento da camada de calibração: aprendizado, métricas e persistência."""
from __future__ import annotations

import numpy as np
import pytest

from expressao_neural.config import CLASSES_MODELO
from expressao_neural.core import calibracao as cal


def _conjunto_sintetico(n_por_classe=40, semente=7):
    """Logits com viés sistemático que uma camada linear consegue corrigir.

    A classe 4 (neutral) recebe um bônus constante — exatamente o tipo de
    desvio que a calibração deve aprender a remover.
    """
    rng = np.random.default_rng(semente)
    X, y = [], []
    for classe in range(len(CLASSES_MODELO)):
        base = rng.normal(0.0, 1.0, (n_por_classe, len(CLASSES_MODELO)))
        base[:, classe] += 2.0        # sinal verdadeiro
        base[:, 4] += 2.6             # viés que atrapalha todas as outras
        X.append(base)
        y.extend([classe] * n_por_classe)
    return np.vstack(X), np.array(y, dtype=np.int64)


# ------------------------------------------------------------- identidade
def test_identidade_reproduz_o_modelo_base() -> None:
    c = cal.Calibracao.identidade()
    z = np.array([-3.0, -1.4, -4.8, 4.5, 6.2, -4.7, 3.1], dtype=np.float32)
    assert np.allclose(c.aplicar(z), z, atol=1e-5)
    assert c.metadados["treinada"] is False


def test_aplicar_rejeita_tamanho_errado() -> None:
    c = cal.Calibracao.identidade()
    with pytest.raises(ValueError):
        c.aplicar(np.zeros(5, dtype=np.float32))


# ------------------------------------------------------------ treinamento
def test_treinamento_reduz_a_perda() -> None:
    X, y = _conjunto_sintetico()
    _, historico = cal.treinar(X, y, epocas=200, taxa=0.1)
    assert len(historico) == 200
    assert historico[-1] < historico[0], "a perda deve cair ao longo das épocas"
    assert all(np.isfinite(v) for v in historico)


def test_treinamento_melhora_a_acuracia_sobre_vies_sistematico() -> None:
    """O ponto central: a camada treinada supera o modelo base."""
    X, y = _conjunto_sintetico()
    n = len(y)
    rng = np.random.default_rng(1)
    ordem = rng.permutation(n)
    corte = int(n * 0.75)
    i_tr, i_te = ordem[:corte], ordem[corte:]

    antes = cal.metricas(y[i_te], np.argmax(X[i_te], axis=1))
    c, _ = cal.treinar(X[i_tr], y[i_tr], epocas=400, taxa=0.1)
    depois = cal.metricas(y[i_te], np.argmax(X[i_te] @ c.W.T + c.b, axis=1))

    assert depois["acuracia"] > antes["acuracia"], (
        f"calibração não melhorou: {antes['acuracia']} -> {depois['acuracia']}"
    )
    assert depois["macro_f1"] > antes["macro_f1"]


def test_inicializacao_e_a_identidade() -> None:
    """Com 0 épocas a camada tem de ser exatamente o modelo base."""
    X, y = _conjunto_sintetico(n_por_classe=5)
    c, historico = cal.treinar(X, y, epocas=0)
    assert historico == []
    assert np.allclose(c.W, np.eye(len(CLASSES_MODELO)))
    assert np.allclose(c.b, 0.0)


def test_treinamento_rejeita_entrada_malformada() -> None:
    with pytest.raises(ValueError):
        cal.treinar(np.zeros((10, 3)), np.zeros(10, dtype=np.int64))
    with pytest.raises(ValueError):
        cal.treinar(np.zeros((10, 7)), np.zeros(4, dtype=np.int64))


def test_metadados_registram_a_procedencia() -> None:
    X, y = _conjunto_sintetico(n_por_classe=10)
    c, _ = cal.treinar(X, y, epocas=50, taxa=0.05)
    m = c.metadados
    assert m["treinada"] is True
    assert m["parametros_treinaveis"] == 56
    assert m["epocas"] == 50
    assert m["amostras_treino"] == len(y)
    assert "CONGELADO" in m["observacao"] or "NÃO foram alterados" in m["observacao"]


# --------------------------------------------------------------- metricas
def test_matriz_de_confusao_conta_corretamente() -> None:
    v = np.array([0, 0, 1, 1, 2])
    p = np.array([0, 1, 1, 1, 2])
    mc = cal.matriz_confusao(v, p)
    assert mc.shape == (7, 7)
    assert mc[0, 0] == 1 and mc[0, 1] == 1
    assert mc[1, 1] == 2 and mc[2, 2] == 1
    assert mc.sum() == 5


def test_metricas_perfeitas_quando_tudo_acerta() -> None:
    y = np.arange(7)
    m = cal.metricas(y, y)
    assert m["acuracia"] == 1.0
    assert m["macro_f1"] == 1.0
    assert np.trace(np.array(m["matriz_confusao"])) == 7


def test_metricas_coerentes_em_caso_misto() -> None:
    v = np.array([3, 3, 3, 4, 4])
    p = np.array([3, 3, 4, 4, 4])
    m = cal.metricas(v, p)
    assert m["acuracia"] == pytest.approx(4 / 5)
    alegria = next(c for c in m["por_classe"] if c["classe"] == "happy")
    assert alegria["revocacao"] == pytest.approx(2 / 3, abs=1e-4)
    assert alegria["precisao"] == pytest.approx(1.0)


# ------------------------------------------------------------ persistencia
def test_salvar_e_carregar_preserva_os_pesos(tmp_path) -> None:
    X, y = _conjunto_sintetico(n_por_classe=12)
    c, _ = cal.treinar(X, y, epocas=80, taxa=0.08)
    destino = tmp_path / "calibracao.npz"
    c.salvar(destino)
    lido = cal.Calibracao.carregar(destino)
    assert lido is not None
    assert np.allclose(lido.W, c.W, atol=1e-6)
    assert np.allclose(lido.b, c.b, atol=1e-6)
    assert lido.metadados["epocas"] == 80


def test_carregar_arquivo_inexistente_devolve_none(tmp_path) -> None:
    assert cal.Calibracao.carregar(tmp_path / "nao_existe.npz") is None


def test_carregar_arquivo_corrompido_devolve_none(tmp_path) -> None:
    ruim = tmp_path / "ruim.npz"
    ruim.write_bytes(b"isto nao e um npz")
    assert cal.Calibracao.carregar(ruim) is None


def test_carregar_formato_errado_devolve_none(tmp_path) -> None:
    import json as _json
    ruim = tmp_path / "forma.npz"
    np.savez(ruim, W=np.zeros((3, 3)), b=np.zeros(3),
             metadados=_json.dumps({"treinada": True}))
    assert cal.Calibracao.carregar(ruim) is None


# ---------------------------------------------- integracao com o classificador
def test_classificador_sem_calibracao_nao_altera_a_saida() -> None:
    """Padrão do sistema: saída idêntica à do modelo pré-treinado."""
    from expressao_neural.core.classificador import ClassificadorExpressao

    assert ClassificadorExpressao.__init__.__defaults__ == (None,)


def test_pipeline_nao_usa_calibracao_por_padrao() -> None:
    from expressao_neural.config import Ajustes

    assert Ajustes().usar_calibracao is False
