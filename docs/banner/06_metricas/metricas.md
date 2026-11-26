# Métricas reais — Expressão Neural

Coletadas em 2026-11-26T13:05:03 por `scripts/banner_metricas.py`.
**Todos os números vêm de execução real. Nada é estimado ou simulado.**

## Hardware e condições do teste

| Item | Valor |
|---|---|
| Processador | AMD Ryzen 7 7435HS |
| Núcleos lógicos | 16 |
| Frequência na medição | 3100 MHz de 3100 MHz máximos |
| Carga da CPU | 7% |
| Memória RAM total | 7.8 GB |
| GPU | nenhuma — inferência somente em CPU (DNN_TARGET_CPU) |
| Sistema | Windows 10 |
| Python | 3.11.9 |
| OpenCV | 4.12.0 |
| NumPy | 2.2.6 |
| Motor de inferência | cv2.dnn (OpenCV DNN) — sem TensorFlow/PyTorch/onnxruntime |

## Desempenho

Amostras: **150** execuções consecutivas, 1 rosto, após aquecimento.

| Etapa | Média | Mediana | Mínimo | Máximo | Desvio padrão |
|---|---|---|---|---|---|
| Detecção (YuNet) | 13.63 ms | 13.48 ms | 12.42 ms | 16.22 ms | 0.87 ms |
| Classificação (MobileFaceNet) | 2.91 ms | 2.77 ms | 2.51 ms | 4.93 ms | 0.37 ms |
| Ciclo completo | 17.35 ms | 17.06 ms | 15.72 ms | 22.71 ms | 1.14 ms |

- **FPS equivalente de inferência:** 57.7 por segundo (1 rosto).
- **Tempo de inicialização:** 65.0 ms para carregar os dois modelos + 47.37 ms de aquecimento.
- **Memória do processo:** 77.8 MB.
- **Classes reconhecidas:** 7.

## Testes automatizados

- Total executado: **142**
- Aprovados: **142**
- Reprovados: **0**
- Ignorados: 0
- Duração: 20.9 s

| Arquivo | Testes | Cobre |
|---|---|---|
| `test_integracao.py` | 29 | integração, condições adversas, desempenho |
| `test_imagem_e_suavizacao.py` | 27 | validação de imagem e estabilidade temporal |
| `test_inferencia.py` | 19 | pré-processamento, classes, softmax, inferência real |
| `test_calibracao.py` | 16 |  |
| `test_robustez.py` | 13 | regressões da auditoria de segurança |
| `test_camera.py` | 11 | ciclo de vida da captura e liberação de recursos |
| `test_interface.py` | 11 | tema, componentes, janela e encerramento |
| `test_modelos.py` | 11 | dependências, integridade SHA-256, carregamento |
| `test_reiniciar_camera.py` | 5 |  |

> Mocks são usados apenas no ciclo de vida da câmera; toda inferência nos testes usa os modelos ONNX reais.

## Execução prolongada

- Duração: **30 minutos contínuos**
- Ciclos executados: **95.811**
- Primeira metade: 18.5 ms · Segunda metade: 19.1 ms
- **Variação: +3.2%** (sem degradação progressiva)
- Trilhas de rosto vivas ao final: 1 (sem vazamento)
- Memória estável em ~93 MB

## Comportamento sob condições controladas

Variações aplicadas sobre a mesma imagem de referência. Mede **estabilidade**, não acurácia.

| Condição | Detectou rosto | Classe prevista | Confiança |
|---|---|---|---|
| Iluminação muito reduzida | sim | Neutra | 91.0% |
| Iluminação reduzida | sim | Neutra | 78.6% |
| Iluminação normal (referência) | sim | Neutra | 62.0% |
| Iluminação elevada | sim | Alegria | 58.3% |
| Iluminação excessiva (estouro) | sim | Neutra | 97.2% |
| Rosto distante (escala 0,35×) | sim | Surpresa | 77.5% |
| Rosto afastado (escala 0,50×) | sim | Neutra | 89.1% |
| Rosto próximo (escala 1,50×) | sim | Neutra | 67.4% |
| Cabeça inclinada −20° | sim | Neutra | 89.5% |
| Cabeça inclinada +20° | sim | Alegria | 50.5% |
| Ruído de sensor | sim | Surpresa | 70.6% |
| Desfoque de movimento | sim | Neutra | 59.2% |
| Oclusão dos olhos (óculos escuros) | não | — | — |
| Ausência de rosto (ruído puro) | não | — | — |

- Condições avaliadas: **14**
- Detectaram rosto: **12**
- Mudaram a classe vencedora: **4**
- Houve erro ou valor inválido: **não**
- Todas as distribuições somaram 1,0: **sim**

## Métricas de classificação

Avaliadas sobre 210 amostras próprias (154 treino / 56 teste).

| Métrica | Modelo base | Com calibração própria |
|---|---|---|
| **Acurácia** | 67.9% | **94.6%** |
| Precisão (macro) | — | 0.946 |
| Recall (macro) | — | 0.946 |
| F1-score (macro) | — | 0.945 |

Ganho: **+26.78 pontos percentuais**.

Matriz de confusão em `docs/banner/04_graficos/matriz_confusao.png`.

> ⚠️ Conjunto de UMA pessoa, com expressões encenadas. A acurácia vale para esse contexto e NÃO generaliza para público diverso.

> O valor **88,27% em RAF-DB** pertence ao **modelo pré-treinado** (fonte: README oficial do OpenCV Zoo), **não** ao sistema aqui desenvolvido. Citá-lo como acurácia deste trabalho seria incorreto.

## Funcionamento offline

| Verificação | Resultado |
|---|---|
| Chamadas de rede em `src/` | **0** (auditoria de código) |
| Modelos disponíveis localmente | sim |
| Teste com a rede desligada | **NÃO EXECUTADO** — depende de validação manual |
