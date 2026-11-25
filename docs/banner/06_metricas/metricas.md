# Métricas reais — Expressão Neural

Coletadas em 2026-11-25T20:05:24 por `scripts/banner_metricas.py`.
**Todos os números vêm de execução real. Nada é estimado ou simulado.**

## Hardware e condições do teste

| Item | Valor |
|---|---|
| Processador | AMD Ryzen 7 7435HS |
| Núcleos lógicos | 16 |
| Frequência na medição | 2496 MHz de 3100 MHz máximos |
| Carga da CPU | 2% |
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
| Detecção (YuNet) | 20.29 ms | 20.54 ms | 13.55 ms | 25.82 ms | 2.67 ms |
| Classificação (MobileFaceNet) | 4.83 ms | 4.79 ms | 3.05 ms | 7.74 ms | 0.91 ms |
| Ciclo completo | 26.19 ms | 26.58 ms | 18.04 ms | 33.1 ms | 3.45 ms |

- **FPS equivalente de inferência:** 38.2 por segundo (1 rosto).
- **Tempo de inicialização:** 119.2 ms para carregar os dois modelos + 88.53 ms de aquecimento.
- **Memória do processo:** 78.0 MB.
- **Classes reconhecidas:** 7.

## Testes automatizados

- Total executado: **119**
- Aprovados: **119**
- Reprovados: **0**
- Ignorados: 0
- Duração: 19.0 s

| Arquivo | Testes | Cobre |
|---|---|---|
| `test_integracao.py` | 29 | integração, condições adversas, desempenho |
| `test_imagem_e_suavizacao.py` | 27 | validação de imagem e estabilidade temporal |
| `test_inferencia.py` | 19 | pré-processamento, classes, softmax, inferência real |
| `test_camera.py` | 11 | ciclo de vida da captura e liberação de recursos |
| `test_interface.py` | 11 | tema, componentes, janela e encerramento |
| `test_modelos.py` | 11 | dependências, integridade SHA-256, carregamento |
| `test_robustez.py` | 11 | regressões da auditoria de segurança |

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

| Métrica | Situação |
|---|---|
| Acurácia | **não avaliada** |
| Precisão | **não avaliada** |
| Recall | **não avaliada** |
| F1-score | **não avaliada** |
| Matriz de confusão | **não avaliada** |

**Motivo:** Não há conjunto de imagens rotuladas disponível neste projeto. Métricas de classificação exigem rótulo verdadeiro por amostra, que só pode ser obtido com voluntários e consentimento registrado.

**Como obter:** Execute scripts/avaliar_metricas.py apontando para uma pasta com subpastas nomeadas pelas 7 classes. O script calcula acurácia, precisão, recall, F1 e matriz de confusão reais.

> O valor **88,27% em RAF-DB** pertence ao **modelo pré-treinado** (fonte: README oficial do OpenCV Zoo), **não** ao sistema aqui desenvolvido. Citá-lo como acurácia deste trabalho seria incorreto.

## Funcionamento offline

| Verificação | Resultado |
|---|---|
| Chamadas de rede em `src/` | **0** (auditoria de código) |
| Modelos disponíveis localmente | sim |
| Teste com a rede desligada | **NÃO EXECUTADO** — depende de validação manual |
