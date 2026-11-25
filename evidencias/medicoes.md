# Medições reais — Expressão Neural

Geradas em 2026-11-25T16:03:30 por `scripts/gerar_evidencias.py`.
Todos os números vêm de execução real nesta máquina; nenhum valor é estimado.

## Ambiente de medição

| Item | Valor |
|---|---|
| Sistema | Windows 10 |
| Processador | AMD64 Family 25 Model 68 Stepping 1, AuthenticAMD |
| Arquitetura | AMD64 |
| Python | 3.11.9 |
| Opencv | 4.12.0 |
| Numpy | 2.2.6 |
| Aceleracao | CPU (cv2.dnn, DNN_TARGET_CPU) |

## Desempenho (1 rosto, imagem 512x512, 120 amostras)

| Etapa | Média | Mediana | Mínimo | Máximo | Desvio padrão |
|---|---|---|---|---|---|
| Detecção (YuNet) | 13.52 ms | 13.4 ms | 11.84 ms | 17.35 ms | 0.83 ms |
| Classificação (MobileFaceNet) | 2.87 ms | 2.77 ms | 2.48 ms | 3.63 ms | 0.3 ms |
| Ciclo completo | 17.15 ms | 17.03 ms | 15.31 ms | 21.32 ms | 1.0 ms |

Equivalente a **58.3 ciclos de inferência por segundo** em CPU, sem GPU dedicada.
Carregamento dos modelos: 66.8 ms (aquecimento 47.86 ms).

## Classificação da imagem de referência

Expressão prevista: **Neutra** (classe do modelo: `neutral`), confiança 62.0%.
Soma das probabilidades: 1.0.

| Categoria | Probabilidade |
|---|---|
| Raiva | 0.01% |
| Nojo | 0.06% |
| Medo | 0.00% |
| Alegria | 27.74% |
| Neutra | 61.95% |
| Tristeza | 0.00% |
| Surpresa | 10.24% |

> Estes números descrevem **desempenho computacional** e **confiança da previsão**, não acurácia do sistema sobre pessoas reais. Acurácia própria exige o protocolo de avaliação descrito em `docs/testes.md`.
