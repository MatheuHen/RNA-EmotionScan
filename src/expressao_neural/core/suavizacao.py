"""Estabilidade temporal das previsoes (RF10).

Estrategia: media movel exponencial (EMA) sobre o VETOR COMPLETO de
probabilidades de cada rosto acompanhado. A EMA nao esconde erro -- ela e
aplicada igualmente a todas as classes e o grafico exibe o vetor suavizado,
que continua sendo uma distribuicao valida (soma 1). O rotulo final e sempre
o argmax do vetor efetivamente exibido.

Quando um rosto desaparece, seu estado e descartado em
`esquecer_ausentes`/`limpar`, de modo que nenhum resultado antigo permaneca
na tela (RF10).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import CLASSES_MODELO

_N = len(CLASSES_MODELO)


@dataclass
class _Trilha:
    centro: tuple[float, float]
    escala: float
    vetor: np.ndarray
    quadros_sem_ver: int = 0
    amostras: int = 1
    historico: list[int] = field(default_factory=list)


class SuavizadorTemporal:
    """Acompanha rostos por proximidade e suaviza as probabilidades."""

    def __init__(self, alfa: float = 0.40, tolerancia_quadros: int = 5) -> None:
        self.alfa = float(np.clip(alfa, 0.05, 1.0))
        self.tolerancia_quadros = max(1, int(tolerancia_quadros))
        self._trilhas: dict[int, _Trilha] = {}
        self._proximo_id = 1

    # ------------------------------------------------------------------ estado
    def limpar(self) -> None:
        self._trilhas.clear()
        self._proximo_id = 1

    @property
    def quantidade_trilhas(self) -> int:
        return len(self._trilhas)

    # ----------------------------------------------------------------- interno
    def _casar(self, centro: tuple[float, float], escala: float) -> int | None:
        melhor_id, melhor_dist = None, float("inf")
        limite = max(24.0, escala * 0.6)
        for tid, trilha in self._trilhas.items():
            dx = trilha.centro[0] - centro[0]
            dy = trilha.centro[1] - centro[1]
            dist = (dx * dx + dy * dy) ** 0.5
            if dist < limite and dist < melhor_dist:
                melhor_id, melhor_dist = tid, dist
        return melhor_id

    # ------------------------------------------------------------------ uso
    def suavizar(
        self,
        centro: tuple[float, float],
        escala: float,
        probabilidades: np.ndarray,
        usados: set[int] | None = None,
    ) -> tuple[int, np.ndarray]:
        """Devolve (id_da_trilha, vetor_suavizado) e atualiza o estado."""
        vetor = np.asarray(probabilidades, dtype=np.float32).reshape(-1)
        if vetor.size != _N:
            raise ValueError(f"vetor de probabilidades deve ter {_N} posicoes")

        tid = self._casar(centro, escala)
        if usados is not None and tid in usados:
            tid = None  # trilha ja consumida por outro rosto neste quadro

        if tid is None:
            tid = self._proximo_id
            self._proximo_id += 1
            self._trilhas[tid] = _Trilha(centro, escala, vetor.copy(), 0, 1, [int(np.argmax(vetor))])
            return tid, vetor.copy()

        trilha = self._trilhas[tid]
        a = self.alfa
        novo = (a * vetor + (1.0 - a) * trilha.vetor).astype(np.float32)
        soma = float(novo.sum())
        if soma > 0:
            novo /= soma  # mantem a distribuicao normalizada
        trilha.vetor = novo
        trilha.centro = centro
        trilha.escala = escala
        trilha.quadros_sem_ver = 0
        trilha.amostras += 1
        trilha.historico.append(int(np.argmax(novo)))
        if len(trilha.historico) > 60:
            del trilha.historico[:-60]
        return tid, novo.copy()

    def esquecer_ausentes(self, vistos: set[int]) -> None:
        """Remove trilhas que nao aparecem ha mais quadros que a tolerancia."""
        for tid in list(self._trilhas):
            if tid in vistos:
                continue
            self._trilhas[tid].quadros_sem_ver += 1
            if self._trilhas[tid].quadros_sem_ver >= self.tolerancia_quadros:
                del self._trilhas[tid]

    def estabilidade(self, tid: int) -> float:
        """Fracao dos ultimos quadros em que o rotulo dominante se manteve."""
        trilha = self._trilhas.get(tid)
        if trilha is None or not trilha.historico:
            return 0.0
        ultimos = trilha.historico[-20:]
        dominante = max(set(ultimos), key=ultimos.count)
        return ultimos.count(dominante) / len(ultimos)
