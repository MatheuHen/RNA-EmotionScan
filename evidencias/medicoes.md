# Medições reais — Expressão Neural

Geradas em 2026-11-25T19:44:06 por `scripts/gerar_evidencias.py`.
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
| Processador | AMD Ryzen 7 7435HS |
| Clock atual mhz | 2496 |
| Clock maximo mhz | 3100 |
| Carga cpu percent | 11 |
| Nucleos logicos | 16 |
| Fracao do clock maximo | 0.805 |

## Desempenho (1 rosto, imagem 512x512, 120 amostras)

| Etapa | Média | Mediana | Mínimo | Máximo | Desvio padrão |
|---|---|---|---|---|---|
| Detecção (YuNet) | 23.23 ms | 23.07 ms | 20.39 ms | 28.58 ms | 1.86 ms |
| Classificação (MobileFaceNet) | 6.67 ms | 6.42 ms | 5.94 ms | 12.4 ms | 0.89 ms |
| Ciclo completo | 31.18 ms | 30.96 ms | 27.62 ms | 42.13 ms | 2.45 ms |

Equivalente a **32.1 ciclos de inferência por segundo** em CPU, sem GPU dedicada.
Carregamento dos modelos: 134.0 ms (aquecimento 98.38 ms).

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
