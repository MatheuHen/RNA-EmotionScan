"""Configuracao comum dos testes."""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Nenhum teste pode sobrescrever as preferencias reais do usuario.
os.environ["EXPRESSAO_NEURAL_SEM_PERSISTENCIA"] = "1"

RAIZ = Path(__file__).resolve().parents[1]
SRC = RAIZ / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import cv2 as cv  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from expressao_neural.core import modelos  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"

_motivo_modelos = "modelos ONNX ausentes ou corrompidos — execute configurar.bat"
modelos_disponiveis = pytest.mark.skipif(
    not all(s.existe and s.tamanho_ok for s in modelos.verificar_todos(conferir_hash=False)),
    reason=_motivo_modelos,
)


@pytest.fixture(scope="session")
def caminho_rosto() -> Path:
    caminho = FIXTURES / "lena.jpg"
    if not caminho.is_file():
        pytest.skip("fixture lena.jpg ausente")
    return caminho


@pytest.fixture(scope="session")
def imagem_rosto(caminho_rosto: Path) -> np.ndarray:
    img = cv.imread(str(caminho_rosto))
    assert img is not None, "falha ao ler a fixture de rosto"
    return img


@pytest.fixture(scope="session")
def imagem_sem_rosto() -> np.ndarray:
    """Ruido deterministico: nao contem rosto algum."""
    rng = np.random.default_rng(20261130)
    return rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)


@pytest.fixture(scope="session")
def imagem_multiplos_rostos(imagem_rosto: np.ndarray) -> np.ndarray:
    """Dois rostos lado a lado, construido a partir da fixture (offline)."""
    espelhado = cv.flip(imagem_rosto, 1)
    return cv.hconcat([imagem_rosto, espelhado])


@pytest.fixture(scope="session")
def pipeline():
    from expressao_neural.config import Ajustes
    from expressao_neural.core.pipeline import PipelineExpressao

    for status in modelos.verificar_todos(conferir_hash=False):
        if not (status.existe and status.tamanho_ok):
            pytest.skip(_motivo_modelos)
    ajustes = Ajustes(multiplos_rostos=True, max_rostos=5, limiar_deteccao=0.80)
    p = PipelineExpressao(ajustes)
    yield p
    p.liberar()
