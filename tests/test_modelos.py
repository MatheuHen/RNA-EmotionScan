"""Dependencias, integridade dos arquivos ONNX e carregamento real."""
from __future__ import annotations

import importlib

import pytest

from expressao_neural.core import modelos
from expressao_neural.utils import ambiente

from .conftest import modelos_disponiveis


@pytest.mark.parametrize("modulo", ["cv2", "numpy", "PIL", "tkinter"])
def test_dependencias_importam(modulo: str) -> None:
    assert importlib.import_module(modulo) is not None


def test_nao_usa_bibliotecas_pesadas() -> None:
    """O escopo proibe TensorFlow/PyTorch: a inferencia usa apenas cv2.dnn."""
    for proibido in ("tensorflow", "torch", "onnxruntime"):
        assert importlib.util.find_spec(proibido) is None, (
            f"{proibido} nao deve ser necessario para a inferencia"
        )


def test_registro_tem_metadados_completos() -> None:
    for info in modelos.TODOS:
        assert info.url.startswith("https://")
        assert len(info.sha256) == 64
        assert info.tamanho_bytes > 0
        assert "Apache" in info.licenca
        assert info.treinamento and info.origem and info.arquitetura


@modelos_disponiveis
def test_integridade_dos_arquivos_onnx() -> None:
    for status in modelos.verificar_todos(conferir_hash=True):
        assert status.ok, status.mensagem
        assert status.sha256_encontrado == status.info.sha256


@modelos_disponiveis
def test_detector_carrega_de_verdade() -> None:
    from expressao_neural.core.detector import DetectorFacial

    det = DetectorFacial()
    assert det.info.chave == "detector"
    assert det.max_rostos >= 1


@modelos_disponiveis
def test_classificador_carrega_e_aquece() -> None:
    from expressao_neural.core.classificador import ClassificadorExpressao

    clf = ClassificadorExpressao()
    ms = clf.aquecer()
    assert ms >= 0.0


def test_modelo_ausente_gera_mensagem_acionavel(monkeypatch, tmp_path) -> None:
    """RF01: modelo ausente deve produzir erro claro, nao stack trace cru."""
    from expressao_neural.core import detector

    monkeypatch.setattr(modelos, "DIR_MODELOS", tmp_path, raising=False)
    falso = modelos.ModeloInfo(
        chave="detector",
        nome="inexistente",
        arquivo="nao_existe.onnx",
        url="https://exemplo.invalido/nao_existe.onnx",
        sha256="0" * 64,
        tamanho_bytes=1,
        licenca="Apache License 2.0",
        origem="teste",
        arquitetura="teste",
        treinamento="teste",
        papel="teste",
    )
    monkeypatch.setattr(modelos, "DETECTOR", falso)
    with pytest.raises(detector.ErroModelo) as exc:
        detector.DetectorFacial()
    assert "configurar.bat" in str(exc.value)


def test_relatorio_de_ambiente_cobre_itens_essenciais() -> None:
    rel = ambiente.verificar_ambiente(conferir_hash=False)
    nomes = " ".join(i.nome for i in rel.itens)
    for esperado in ("Python", "opencv-python", "numpy", "pillow", "Modelo detector"):
        assert esperado in nomes
    assert rel.texto()
