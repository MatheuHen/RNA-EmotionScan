"""Camada de calibracao supervisionada treinada sobre as saidas da rede base.

O que e, com precisao
---------------------
O modelo de expressoes continua sendo o pre-treinado do OpenCV Zoo, com os
pesos originais intactos. Sobre os 7 logits que ele produz, aplica-se uma
camada linear propria:

    z' = W · z + b          (W: 7x7, b: 7 -> 56 parametros)
    p  = softmax(z')

`W` e `b` sao APRENDIDOS por descida de gradiente sobre um conjunto rotulado
coletado pelo proprio usuario (`scripts/treinar_calibracao.py`).

Inicializacao: W = identidade, b = 0. Nesse ponto a camada e exatamente a
funcao identidade, e o sistema reproduz o modelo base. O treinamento so pode
afastar-se disso se os dados justificarem -- o que torna o ganho mensuravel.

Isto NAO e:
  * treinamento da rede convolucional (os pesos dela nao sao tocados);
  * ajuste fino (fine-tuning) do backbone;
  * um modelo novo de reconhecimento facial.

Isto E: aprendizado supervisionado de uma camada de saida propria, com dataset
proprio, divisao treino/teste e metricas proprias.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from ..config import CLASSES_MODELO, DIR_MODELOS

N_CLASSES = len(CLASSES_MODELO)
ARQUIVO_PADRAO = DIR_MODELOS / "calibracao_pessoal.npz"


@dataclass
class Calibracao:
    """Pesos treinados e a procedencia do treinamento."""

    W: np.ndarray          # (7, 7)
    b: np.ndarray          # (7,)
    metadados: dict

    # ------------------------------------------------------------- aplicacao
    def aplicar(self, logits: np.ndarray) -> np.ndarray:
        """Transforma os logits do modelo base nos logits calibrados."""
        z = np.asarray(logits, dtype=np.float64).reshape(-1)
        if z.size != N_CLASSES:
            raise ValueError(f"esperado {N_CLASSES} logits, recebido {z.size}")
        return (self.W @ z + self.b).astype(np.float32)

    # ---------------------------------------------------------------- disco
    def salvar(self, caminho: Path = ARQUIVO_PADRAO) -> Path:
        caminho.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            caminho,
            W=self.W.astype(np.float32),
            b=self.b.astype(np.float32),
            metadados=json.dumps(self.metadados, ensure_ascii=False),
        )
        return caminho

    @classmethod
    def carregar(cls, caminho: Path = ARQUIVO_PADRAO) -> "Calibracao | None":
        if not caminho.is_file():
            return None
        try:
            dados = np.load(caminho, allow_pickle=False)
            W = np.asarray(dados["W"], dtype=np.float64)
            b = np.asarray(dados["b"], dtype=np.float64)
            if W.shape != (N_CLASSES, N_CLASSES) or b.shape != (N_CLASSES,):
                return None
            if not (np.isfinite(W).all() and np.isfinite(b).all()):
                return None
            meta = json.loads(str(dados["metadados"]))
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            return None
        return cls(W=W, b=b, metadados=meta)

    @classmethod
    def identidade(cls) -> "Calibracao":
        """Camada neutra: reproduz exatamente o modelo base."""
        return cls(
            W=np.eye(N_CLASSES, dtype=np.float64),
            b=np.zeros(N_CLASSES, dtype=np.float64),
            metadados={"treinada": False, "descricao": "identidade (modelo base)"},
        )

    @property
    def resumo(self) -> str:
        m = self.metadados
        if not m.get("treinada"):
            return "sem calibração (modelo base)"
        return (
            f"calibração própria · {m.get('amostras_treino', '?')} amostras de "
            f"treino · acurácia no teste {m.get('acuracia_teste', 0) * 100:.1f}%"
        )


# ============================================================== treinamento
def softmax_linhas(Z: np.ndarray) -> np.ndarray:
    """Softmax estavel aplicado linha a linha."""
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def treinar(
    logits: np.ndarray,
    rotulos: np.ndarray,
    epocas: int = 400,
    taxa: float = 0.05,
    regularizacao: float = 1e-3,
    semente: int = 20261130,
) -> tuple[Calibracao, list[float]]:
    """Treina W e b por descida de gradiente com entropia cruzada.

    Parametros
    ----------
    logits : (N, 7) saidas brutas do modelo base
    rotulos: (N,)   indice da classe verdadeira
    epocas : numero de passagens completas pelo conjunto de treino
    taxa   : taxa de aprendizado
    regularizacao : peso da penalidade L2 sobre o AFASTAMENTO da identidade,
        nao sobre W em si. Assim o treino so se afasta do modelo base quando os
        dados pagam por isso -- importante com poucas amostras.

    Devolve a calibracao treinada e o historico de perda por epoca.
    """
    X = np.asarray(logits, dtype=np.float64)
    y = np.asarray(rotulos, dtype=np.int64)
    if X.ndim != 2 or X.shape[1] != N_CLASSES:
        raise ValueError(f"logits devem ter forma (N, {N_CLASSES})")
    if y.shape[0] != X.shape[0]:
        raise ValueError("quantidade de rótulos diferente da de amostras")

    rng = np.random.default_rng(semente)
    n = X.shape[0]
    I = np.eye(N_CLASSES)
    W = I.copy()
    b = np.zeros(N_CLASSES)

    # Alvo em codificacao one-hot
    Y = np.zeros((n, N_CLASSES))
    Y[np.arange(n), y] = 1.0

    historico: list[float] = []
    for _ in range(epocas):
        ordem = rng.permutation(n)
        Xe, Ye = X[ordem], Y[ordem]

        P = softmax_linhas(Xe @ W.T + b)
        # Entropia cruzada média + penalidade de afastamento da identidade
        perda = float(-np.sum(Ye * np.log(P + 1e-12)) / n
                      + regularizacao * np.sum((W - I) ** 2))
        historico.append(perda)

        dZ = (P - Ye) / n                      # (n, 7)
        gW = dZ.T @ Xe + 2.0 * regularizacao * (W - I)
        gb = dZ.sum(axis=0)
        W -= taxa * gW
        b -= taxa * gb

    cal = Calibracao(
        W=W, b=b,
        metadados={
            "treinada": True,
            "treinado_em": datetime.now().isoformat(timespec="seconds"),
            "amostras_treino": int(n),
            "epocas": int(epocas),
            "taxa_aprendizado": float(taxa),
            "regularizacao_l2_ate_identidade": float(regularizacao),
            "perda_inicial": historico[0] if historico else None,
            "perda_final": historico[-1] if historico else None,
            "parametros_treinaveis": int(N_CLASSES * N_CLASSES + N_CLASSES),
            "observacao": (
                "Camada linear treinada sobre os logits do modelo pré-treinado. "
                "Os pesos da rede convolucional NÃO foram alterados."
            ),
        },
    )
    return cal, historico


# ================================================================= metricas
def matriz_confusao(verdadeiro: np.ndarray, previsto: np.ndarray) -> np.ndarray:
    """Matriz 7x7: linhas = classe verdadeira, colunas = classe prevista."""
    m = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    for v, p in zip(np.asarray(verdadeiro).ravel(), np.asarray(previsto).ravel()):
        if 0 <= v < N_CLASSES and 0 <= p < N_CLASSES:
            m[int(v), int(p)] += 1
    return m


def metricas(verdadeiro: np.ndarray, previsto: np.ndarray) -> dict:
    """Acuracia, e precisao/revocacao/F1 por classe e macro."""
    mc = matriz_confusao(verdadeiro, previsto)
    total = int(mc.sum())
    acertos = int(np.trace(mc))
    por_classe = []
    for i in range(N_CLASSES):
        vp = int(mc[i, i])
        fp = int(mc[:, i].sum()) - vp
        fn = int(mc[i, :].sum()) - vp
        precisao = vp / (vp + fp) if (vp + fp) else 0.0
        revocacao = vp / (vp + fn) if (vp + fn) else 0.0
        f1 = (2 * precisao * revocacao / (precisao + revocacao)
              if (precisao + revocacao) else 0.0)
        por_classe.append({
            "classe": CLASSES_MODELO[i],
            "amostras": int(mc[i, :].sum()),
            "precisao": round(precisao, 4),
            "revocacao": round(revocacao, 4),
            "f1": round(f1, 4),
        })
    presentes = [c for c in por_classe if c["amostras"] > 0]
    return {
        "acuracia": round(acertos / total, 4) if total else 0.0,
        "amostras": total,
        "acertos": acertos,
        "por_classe": por_classe,
        "macro_precisao": round(
            float(np.mean([c["precisao"] for c in presentes])), 4) if presentes else 0.0,
        "macro_revocacao": round(
            float(np.mean([c["revocacao"] for c in presentes])), 4) if presentes else 0.0,
        "macro_f1": round(
            float(np.mean([c["f1"] for c in presentes])), 4) if presentes else 0.0,
        "matriz_confusao": mc.tolist(),
    }
