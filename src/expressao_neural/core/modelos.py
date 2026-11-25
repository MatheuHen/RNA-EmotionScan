"""Registro dos modelos neurais ONNX: origem, licenca, integridade.

Os arquivos sao obtidos UMA vez por `scripts/baixar_modelos.py` (ou configurar.bat).
A aplicacao nunca baixa modelos durante a execucao -- isso garante o modo offline.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from ..config import DIR_MODELOS


@dataclass(frozen=True)
class ModeloInfo:
    chave: str
    nome: str
    arquivo: str
    url: str
    sha256: str
    tamanho_bytes: int
    licenca: str
    origem: str
    arquitetura: str
    treinamento: str
    papel: str

    @property
    def caminho(self) -> Path:
        return DIR_MODELOS / self.arquivo


DETECTOR = ModeloInfo(
    chave="detector",
    nome="YuNet (face_detection_yunet_2023mar)",
    arquivo="face_detection_yunet_2023mar.onnx",
    url=(
        "https://huggingface.co/opencv/face_detection_yunet/resolve/main/"
        "face_detection_yunet_2023mar.onnx"
    ),
    sha256="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    tamanho_bytes=232589,
    licenca="Apache License 2.0 (OpenCV Zoo)",
    origem="OpenCV Zoo / Hugging Face — opencv/face_detection_yunet",
    arquitetura=(
        "CNN anchor-free leve para deteccao de rostos; devolve caixa, 5 pontos "
        "faciais de referencia e pontuacao de confianca por rosto."
    ),
    treinamento=(
        "Modelo pre-treinado distribuido pelo OpenCV Zoo. O repositorio oficial "
        "reporta avaliacao no benchmark WIDER Face. Nao houve treinamento proprio "
        "neste trabalho."
    ),
    papel="Deteccao e alinhamento dos rostos (RF03)",
)

CLASSIFICADOR = ModeloInfo(
    chave="classificador",
    nome="Progressive Teacher / MobileFaceNet (facial_expression_recognition_mobilefacenet_2022july)",
    arquivo="facial_expression_recognition_mobilefacenet_2022july.onnx",
    url=(
        "https://huggingface.co/opencv/facial_expression_recognition/resolve/main/"
        "facial_expression_recognition_mobilefacenet_2022july.onnx"
    ),
    sha256="4f61307602fc089ce20488a31d4e4614e3c9753a7d6c41578c854858b183e1a9",
    tamanho_bytes=4791892,
    licenca="Apache License 2.0 (OpenCV Zoo)",
    origem="OpenCV Zoo / Hugging Face — opencv/facial_expression_recognition",
    arquitetura=(
        "MobileFaceNet como backbone, treinado pelo metodo semi-supervisionado "
        "Progressive Teacher. Entrada 112x112x3 RGB normalizada; saida 'label' "
        "com 7 logits (uma por expressao)."
    ),
    treinamento=(
        "Modelo pre-treinado. O README oficial do OpenCV Zoo reporta acuracia de "
        "88,27% na avaliacao sobre o conjunto RAF-DB. O repositorio NAO declara uso "
        "de FER2013. Nao houve treinamento proprio neste trabalho."
    ),
    papel="Classificacao das 7 expressoes faciais (RF04)",
)

TODOS: tuple[ModeloInfo, ...] = (DETECTOR, CLASSIFICADOR)


def sha256_do_arquivo(caminho: Path, bloco: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(caminho, "rb") as fh:
        for pedaco in iter(lambda: fh.read(bloco), b""):
            h.update(pedaco)
    return h.hexdigest()


@dataclass
class StatusModelo:
    info: ModeloInfo
    existe: bool
    tamanho_ok: bool
    hash_ok: bool
    sha256_encontrado: str | None

    @property
    def ok(self) -> bool:
        return self.existe and self.tamanho_ok and self.hash_ok

    @property
    def mensagem(self) -> str:
        if not self.existe:
            return (
                f"Modelo ausente: {self.info.arquivo}. "
                f"Execute configurar.bat (ou scripts/baixar_modelos.py) com internet."
            )
        if not self.tamanho_ok:
            return (
                f"Modelo {self.info.arquivo} com tamanho inesperado "
                f"(esperado {self.info.tamanho_bytes} bytes). Baixe novamente."
            )
        if not self.hash_ok:
            return (
                f"Integridade do modelo {self.info.arquivo} nao confere (SHA-256 "
                f"divergente). Apague o arquivo e execute configurar.bat novamente."
            )
        return f"{self.info.arquivo}: integridade verificada."


def verificar(info: ModeloInfo, conferir_hash: bool = True) -> StatusModelo:
    caminho = info.caminho
    if not caminho.is_file():
        return StatusModelo(info, False, False, False, None)
    tamanho_ok = caminho.stat().st_size == info.tamanho_bytes
    encontrado = None
    hash_ok = True
    if conferir_hash:
        encontrado = sha256_do_arquivo(caminho)
        hash_ok = encontrado == info.sha256
    return StatusModelo(info, True, tamanho_ok, hash_ok, encontrado)


def verificar_todos(conferir_hash: bool = True) -> list[StatusModelo]:
    return [verificar(m, conferir_hash) for m in TODOS]
