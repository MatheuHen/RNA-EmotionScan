# Treinamento da camada de calibração

Executado em 2026-11-26T08:32:21 por `scripts/treinar_calibracao.py`.

## O que foi treinado

Aprendizado supervisionado de uma camada linear de saída (W 7×7 e b 7, 56 parâmetros) sobre os logits do modelo pré-treinado, que permanece CONGELADO. Não houve treinamento nem ajuste fino da rede convolucional.

| Item | Valor |
|---|---|
| Parâmetros treináveis | **56** (W 7×7 + b 7) |
| Função de perda | entropia cruzada |
| Otimizador | descida de gradiente em lote completo |
| Épocas | **400** |
| Taxa de aprendizado | 0.05 |
| Regularização L2 (até a identidade) | 0.001 |
| Inicialização | W = identidade, b = 0 (reproduz o modelo base) |
| Perda inicial → final | 2.0322 → 0.2088 |

## Conjunto de dados

| Item | Valor |
|---|---|
| Origem | amostras coletadas pelo próprio autor via webcam |
| Rótulo verdadeiro | a expressão SOLICITADA (expressões encenadas) |
| Total de amostras | **210** |
| Treino | 154 |
| Teste | 56 |
| Divisão | estratificada por classe |

Amostras por classe:

| Classe | Amostras |
|---|---|
| Alegria | 30 |
| Medo | 30 |
| Neutra | 30 |
| Nojo | 30 |
| Raiva | 30 |
| Surpresa | 30 |
| Tristeza | 30 |

## Resultados no conjunto de teste

| Métrica | Modelo base | Com calibração |
|---|---|---|
| **Acurácia** | 67.9% | **94.6%** |
| Precisão (macro) | 0.515 | 0.946 |
| Revocação (macro) | 0.679 | 0.946 |
| F1-score (macro) | 0.578 | 0.945 |

Variação de acurácia: **+26.78 pontos percentuais**.

### Por classe (com calibração)

| Classe | Amostras | Precisão | Revocação | F1 |
|---|---|---|---|---|
| angry | 8 | 1.000 | 1.000 | 1.000 |
| disgust | 8 | 1.000 | 1.000 | 1.000 |
| fearful | 8 | 0.889 | 1.000 | 0.941 |
| happy | 8 | 1.000 | 1.000 | 1.000 |
| neutral | 8 | 0.875 | 0.875 | 0.875 |
| sad | 8 | 1.000 | 1.000 | 1.000 |
| surprised | 8 | 0.857 | 0.750 | 0.800 |

### Matriz de confusão (com calibração)

Linhas: classe verdadeira · Colunas: classe prevista

| | Raiv | Nojo | Medo | Aleg | Neut | Tris | Surp |
|---|---|---|---|---|---|---|---|
| **Raiva** | 8 | 0 | 0 | 0 | 0 | 0 | 0 |
| **Nojo** | 0 | 8 | 0 | 0 | 0 | 0 | 0 |
| **Medo** | 0 | 0 | 8 | 0 | 0 | 0 | 0 |
| **Alegria** | 0 | 0 | 0 | 8 | 0 | 0 | 0 |
| **Neutra** | 0 | 0 | 0 | 0 | 7 | 0 | 1 |
| **Tristeza** | 0 | 0 | 0 | 0 | 0 | 8 | 0 |
| **Surpresa** | 0 | 0 | 1 | 0 | 1 | 0 | 6 |

## Limitações deste treinamento

Declaradas para que o resultado não seja superinterpretado:

- O conjunto tem **210 amostras de uma única pessoa**. A acurácia medida vale para esse contexto e **não generaliza** para público diverso.
- As expressões são **encenadas sob solicitação**, não espontâneas.
- O conjunto de teste é pequeno; a acurácia tem incerteza alta.
- A rede convolucional **não foi treinada**: apenas a camada de saída de 56 parâmetros. O ganho possível é de **calibração**, não de capacidade de representação.
- Os 88,27% em RAF-DB continuam sendo do **modelo pré-treinado**, e não devem ser confundidos com a acurácia medida aqui.
