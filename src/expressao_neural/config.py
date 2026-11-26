"""Configuracao central do EmotionScan.

Toda constante de caminho, limiar, rotulo e identificacao academica fica aqui.
Nenhum segredo, credencial ou endereco de servico externo de execucao.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path

# ---------------------------------------------------------------- caminhos
RAIZ = Path(__file__).resolve().parents[2]
DIR_MODELOS = RAIZ / "models"
DIR_TESTES = RAIZ / "tests"
DIR_EVIDENCIAS = RAIZ / "evidencias"
DIR_DOCS = RAIZ / "docs"
ARQ_CONFIG_USUARIO = RAIZ / "config.local.json"

# ---------------------------------------------------- identidade academica
APP_NOME = "EmotionScan"
APP_DESCRICAO = "Reconhecimento de expressões faciais em tempo real com redes neurais"
APP_VERSAO = "1.3"
ACADEMICO = "Matheus Henrique de Araújo Carneiro"
EMAIL = "matheus.carneiro@academico.unirv.edu.br"
CURSO = "Engenharia de Software — UniRV"
DISCIPLINA = "Redes Neurais Artificiais (RNA)"
PROFESSOR = "Sandro Silva Moreira"
EVENTO = "5º Workshop de Trabalhos de Inteligência Artificial da FASOFT — UniRV"

AVISO_PRIVACIDADE = (
    "Este sistema classifica padrões visuais de expressões faciais. "
    "Os resultados não representam diagnósticos psicológicos nem determinam "
    "com certeza as emoções reais de uma pessoa."
)

# ------------------------------------------------------------- expressoes
# Ordem EXATA das 7 saidas do modelo facial_expression_recognition_mobilefacenet_2022july.onnx
# Fonte: opencv_zoo/models/facial_expression_recognition/facial_fer_model.py (getDesc)
CLASSES_MODELO: tuple[str, ...] = (
    "angry", "disgust", "fearful", "happy", "neutral", "sad", "surprised",
)

# Traducao apresentada na interface (mesma ordem dos indices do modelo)
CLASSES_PT: tuple[str, ...] = (
    "Raiva", "Nojo", "Medo", "Alegria", "Neutra", "Tristeza", "Surpresa",
)

# Cor de destaque por classe (hex), mesma ordem
CORES_CLASSES: tuple[str, ...] = (
    "#ef4444",  # Raiva
    "#84cc16",  # Nojo
    "#a855f7",  # Medo
    "#22c55e",  # Alegria
    "#64748b",  # Neutra
    "#3b82f6",  # Tristeza
    "#f59e0b",  # Surpresa
)

MSG_BAIXA_CONFIANCA = "Expressão não identificada com confiança suficiente."

# ---------------------------------------------------- parametros do modelo
# Pre-processamento do classificador (fonte: facial_fer_model.py)
FER_TAMANHO_ENTRADA = (112, 112)
FER_MEDIA = 0.5
FER_DESVIO = 0.5
FER_NOME_ENTRADA = "data"
FER_NOME_SAIDA = "label"
# A saida 'label' sao logits brutos -> softmax e necessario para obter probabilidades.
FER_SAIDA_E_LOGIT = True

# Pontos de referencia padrao (ArcFace 112x112) usados no alinhamento facial
PONTOS_PADRAO_5 = (
    (38.2946, 51.6963),
    (73.5318, 51.5014),
    (56.0252, 71.7366),
    (41.5493, 92.3655),
    (70.7299, 92.2041),
)


@dataclass
class Ajustes:
    """Parametros ajustaveis em tempo de execucao (persistidos em config.local.json)."""

    indice_camera: int = 0
    largura_captura: int = 640
    altura_captura: int = 480
    fps_alvo: int = 30
    # Executa a inferencia a cada N quadros capturados (1 = todos os quadros).
    # 1 por padrao: o ciclo completo medido (~26 ms) cabe no intervalo de 33 ms
    # de uma camera a 30 FPS, entao nao ha motivo para pular quadros. Pular
    # introduzia atraso perceptivel em expressoes rapidas.
    inferir_a_cada: int = 1
    # 0,35 em vez de 0,50: expressoes NATURAIS (sorriso leve, surpresa discreta)
    # costumam vencer com 35-55% de confianca. Com o limiar em 0,50 a previsao
    # correta do modelo era rotulada "Indefinida", obrigando o usuario a exagerar
    # a expressao -- o que nao representa uso real nem apresentacao ao publico.
    # O valor continua ajustavel na interface.
    limiar_confianca: float = 0.35
    limiar_deteccao: float = 0.85
    limiar_nms: float = 0.30
    max_rostos: int = 5
    multiplos_rostos: bool = True
    # Sempre exibir a classe prevista, mesmo abaixo do limiar, sinalizando a
    # baixa confianca em vez de esconder o resultado atras de "Indefinida".
    # Expressoes sutis sao previsoes legitimas do modelo; oculta-las dava a
    # impressao de que o sistema nao estava funcionando.
    sempre_exibir_previsao: bool = True
    suavizacao_ativa: bool = True
    # Peso do quadro novo na media exponencial (1.0 = sem suavizacao).
    # 0,55 responde mais rapido a mudanca de expressao sem voltar a oscilar:
    # com 0,40 uma expressao sutil levava varios quadros para aparecer.
    suavizacao_alfa: float = 0.55
    espelhar_imagem: bool = True
    # Aplica a camada de calibracao treinada (models/calibracao_pessoal.npz),
    # quando existir. Desligado por padrao: o sistema entrega a saida do modelo
    # pre-treinado a menos que voce opte conscientemente pela calibracao.
    usar_calibracao: bool = False

    def salvar(self, caminho: Path = ARQ_CONFIG_USUARIO) -> None:
        # Testes e geracao de evidencias nao devem alterar as preferencias reais.
        if os.environ.get("EXPRESSAO_NEURAL_SEM_PERSISTENCIA") == "1":
            return
        try:
            caminho.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        except OSError:
            pass  # preferencias sao opcionais; falha nao deve derrubar a aplicacao

    @classmethod
    def carregar(cls, caminho: Path = ARQ_CONFIG_USUARIO) -> "Ajustes":
        base = cls()
        if not caminho.is_file():
            return base
        try:
            dados = json.loads(caminho.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return base
        if not isinstance(dados, dict):
            return base
        validos = {c: getattr(base, c) for c in base.__dataclass_fields__}
        for chave, valor in dados.items():
            if chave in validos and isinstance(valor, type(validos[chave])):
                setattr(base, chave, valor)
        return base


# ------------------------------------------------------- imagem estatica
EXTENSOES_IMAGEM = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
TAMANHO_MAX_IMAGEM_BYTES = 25 * 1024 * 1024  # 25 MB
LADO_MAX_IMAGEM = 6000  # pixels por lado

# ------------------------------------------------------------ diagnostico
MODO_DEBUG = os.environ.get("EXPRESSAO_NEURAL_DEBUG") == "1"
