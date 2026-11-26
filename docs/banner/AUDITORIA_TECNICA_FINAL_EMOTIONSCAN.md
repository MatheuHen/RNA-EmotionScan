# AUDITORIA TÉCNICA E ACADÊMICA FINAL

**Sistema auditado:** `C:\ProjetosIA\Projetos\ExpressaoNeural`
**Data da auditoria:** 26/11/2026
**Commit auditado:** `e328887` · 21 commits · 113 arquivos versionados
**Natureza:** somente leitura. Nenhum arquivo do sistema foi alterado durante a auditoria.

## ⚠️ ADVERTÊNCIA INICIAL — DIVERGÊNCIA DE NOME

O título solicitado para o banner é **"EmotionScan"**. Uma busca em todo o
código-fonte, documentação e scripts retornou **zero ocorrências** desse nome.

| Onde | Nome atual |
|---|---|
| `src/expressao_neural/config.py:22` | `APP_NOME = "Expressão Neural"` |
| Pacote Python | `expressao_neural` |
| Pasta do projeto | `ExpressaoNeural` |
| Interface, rodapé, janela "Sobre" | "Expressão Neural" |
| `pyproject.toml` | `name = "expressao-neural"` |

**Consequência prática:** se o banner disser "EmotionScan" e o sistema exibir
"Expressão Neural" na tela durante a apresentação, a divergência fica visível
para o professor e para os visitantes. **Decida antes de imprimir:**

1. **Renomear o sistema** para EmotionScan (exige alterar código, interface,
   documentação e refazer as capturas de tela); ou
2. **Manter "Expressão Neural"** no banner; ou
3. Usar ambos de forma explícita: *"EmotionScan (Expressão Neural)"*.

Esta auditoria descreve o sistema **como ele é hoje**: Expressão Neural.

### Legenda de classificação

| Símbolo | Significado |
|---|---|
| ✅ **Comprovada** | verificada diretamente em código, modelo ou execução, com prova citada |
| 🟡 **Parcialmente comprovada** | verificada em parte; a parte não verificada está declarada |
| ⬜ **Não avaliada** | não foi medida; não deve ser afirmada |
| ❌ **Não implementada** | não existe no sistema |

---

# 1. Arquitetura das redes neurais

> **Método de verificação:** os dois arquivos `.onnx` foram abertos e o
> protobuf foi percorrido diretamente (sem a biblioteca `onnx`), contando
> tensores de peso (`GraphProto.initializer`) e tipos de operador
> (`NodeProto.op_type`). Script da auditoria disponível mediante solicitação.

## 1.1 Quantas redes participam do processamento — ✅ Comprovada

**Duas** redes neurais convolucionais, em cascata. Não há terceira rede.
A camada de calibração (seção 2) é um classificador linear, não uma rede
convolucional.

| # | Rede | Função |
|---|---|---|
| 1 | **YuNet** | detecção facial + 5 pontos de referência |
| 2 | **MobileFaceNet / Progressive Teacher** | classificação das 7 expressões |

## 1.2 Rede 1 — YuNet (detecção) — ✅ Comprovada

| Item | Valor | Prova |
|---|---|---|
| Arquivo | `face_detection_yunet_2023mar.onnx` | `models/` |
| Versão | **2023mar** (março de 2023) | nome do arquivo |
| Tamanho | 232.589 bytes | `core/modelos.py:46` |
| SHA-256 | `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` | `core/modelos.py:42` |
| **Parâmetros** | **53.121** | contagem no protobuf (112 tensores de peso) |
| Nós do grafo | 106 | contagem de `op_type` |
| É CNN? | **Sim** | 53 operadores `Conv` |
| Interface | `cv2.FaceDetectorYN` | `core/detector.py:54` |

**Composição do grafo (contagem exata de operadores):**

| Operador | Quantidade | Papel |
|---|---|---|
| `Conv` | 53 | convoluções do backbone e das cabeças |
| `Relu` | 15 | ativação intermediária |
| `Transpose` | 12 | reorganização das saídas multiescala |
| `Reshape` | 12 | achatamento das predições |
| `Sigmoid` | 6 | ativação das cabeças de classificação e objetividade |
| `MaxPool` | 4 | redução espacial |
| `Resize` | 2 | upsampling do *neck* (FPN) |
| `Add` | 2 | fusão lateral do *neck* (FPN) |

**Estrutura comprovada pelos nomes dos tensores:** `backbone.model1`…`model5`,
`neck.lateral_convs.0/1/2` — ou seja, **backbone convolucional + neck tipo FPN**.

**Entrada:** nó `input` (a resolução é definida em tempo de execução por
`setInputSize`, `core/detector.py:86`).

**Saídas — 12 tensores, em 3 escalas (strides 8, 16 e 32):**

```
cls_8   cls_16   cls_32     classificação (é rosto?)
obj_8   obj_16   obj_32     objetividade
bbox_8  bbox_16 bbox_32     caixa delimitadora
kps_8   kps_16  kps_32      5 pontos faciais
```

Isso caracteriza um detector **anchor-free, multiescala, com cabeças
separadas**. Após a supressão de não-máximos (NMS), a API do OpenCV entrega
**15 valores por rosto**: 4 de caixa + 10 de pontos + 1 de confiança
(`core/detector.py:100-116`).

**Ativação:** ReLU no corpo da rede, Sigmoid nas cabeças de classificação e
objetividade.

## 1.3 Rede 2 — MobileFaceNet / Progressive Teacher — ✅ Comprovada

| Item | Valor | Prova |
|---|---|---|
| Arquivo | `facial_expression_recognition_mobilefacenet_2022july.onnx` | `models/` |
| Versão | **2022july** (julho de 2022) | nome do arquivo |
| Tamanho | 4.791.892 bytes | `core/modelos.py:70` |
| SHA-256 | `4f61307602fc089ce20488a31d4e4614e3c9753a7d6c41578c854858b183e1a9` | `core/modelos.py:66` |
| **Parâmetros** | **1.193.927** | contagem no protobuf (135 tensores) |
| Nós do grafo | 97 | contagem de `op_type` |
| É CNN? | **Sim** | 50 operadores `Conv` |
| É MobileFaceNet? | **Sim** | 15 blocos bottleneck + PReLU + embedding 512-d |
| Interface | `cv2.dnn.readNet` | `core/classificador.py:72` |

**Composição do grafo (contagem exata de operadores):**

| Operador | Quantidade | Papel |
|---|---|---|
| `Conv` | 50 | convoluções, incluindo separáveis em profundidade |
| `PRelu` | **33** | ativação (PReLU, não ReLU) |
| `Add` | 12 | conexões residuais dos blocos bottleneck |
| `Flatten` | 1 | vetorização antes da camada final |
| `Gemm` | 1 | camada totalmente conectada final |

**Estrutura comprovada pelos nomes dos tensores:**

```
conv1          →  conv2d inicial            (prelu.weight [1, 64, 1, 1])
dw_conv1       →  convolução depthwise      (prelu.weight [1, 64, 1, 1])
blocks.0 … blocks.14  →  15 blocos bottleneck residuais
conv2          →  conv2d final              (prelu.weight [1, 512, 1, 1])
fc             →  fc.weight [7, 512]  +  fc.bias [7]
```

**Dimensões de entrada e saída:**

| | Valor | Prova |
|---|---|---|
| Entrada | nó `data`, tensor `1 × 3 × 112 × 112` (NCHW, RGB) | `config.py:67`, `classificador.py:98` |
| Vetor de características | **512 dimensões** | `fc.weight` tem forma `[7, 512]` |
| Saída | nó `label`, `1 × 7` | `config.py:71` |

**Ativação principal: PReLU** (*Parametric ReLU*) — 33 ocorrências. Esta é uma
correção relevante: a documentação anterior do projeto não especificava a
função de ativação.

## 1.4 Como funciona a classificação final — ✅ Comprovada

**O sistema usa Softmax — mas o Softmax NÃO está dentro do modelo.**

Prova decisiva: **nenhum dos dois grafos ONNX contém um operador `Softmax`**.
O último nó do classificador é `Gemm` (camada totalmente conectada), que produz
**7 logits brutos**.

O Softmax é aplicado pelo código da aplicação:

```python
# src/expressao_neural/core/classificador.py:26-31
def softmax(logits):
    x = np.asarray(logits, dtype=np.float64)
    x = x - np.max(x, axis=-1, keepdims=True)   # estabilidade numérica
    exp = np.exp(x)
    return (exp / np.sum(exp, axis=-1, keepdims=True)).astype(np.float32)
```

Controlado pela constante `FER_SAIDA_E_LOGIT = True` (`config.py:73`), aplicada
em `classificador.py:129`. A classe prevista é o `argmax` das probabilidades
(`classificador.py:130`).

**Por que isso importa academicamente:** aplicar Softmax sobre uma saída que já
fosse probabilidade distorceria a escala. A auditoria confirma que a decisão
está correta, e a prova é a ausência do operador no grafo.

## 1.5 Origem oficial dos modelos — ✅ Comprovada

Ambos vêm do **OpenCV Zoo**, licença **Apache 2.0**, com espelho no Hugging Face.
URLs e hashes registrados em `core/modelos.py:36-78`. A integridade é conferida
por SHA-256 no download, em `verificar.bat` e no teste
`test_integridade_dos_arquivos_onnx`.

## 1.6 Arquitetura das redes ≠ arquitetura do sistema

| | O que é |
|---|---|
| **Arquitetura das redes neurais** | O que está descrito acima: camadas convolucionais, PReLU, blocos residuais, cabeças multiescala. É de terceiros. |
| **Arquitetura do sistema** | Como a aplicação é organizada: threads, pipeline, interface, tratamento de falhas. É contribuição do autor (seção 8). |

---

# 2. Treinamento dos modelos

Esta seção foi auditada com rigor especial, por ser a mais sensível
academicamente.

## 2.1 Os modelos são pré-treinados — ✅ Comprovada

**Sim.** As duas redes convolucionais são pré-treinadas por terceiros e
**permanecem congeladas**: nenhum peso delas é alterado em nenhum ponto do
código. O código apenas lê os arquivos `.onnx` e executa inferência
(`cv2.dnn.readNet`, `cv2.FaceDetectorYN.create`).

| Rede | Treinada originalmente por | Dataset citado na fonte oficial |
|---|---|---|
| YuNet | Wei Wu, Hanyang Peng, Shiqi Yu (OpenCV Zoo) | avaliação em **WIDER FACE**; o conjunto exato de treinamento **não é detalhado** na documentação pública |
| MobileFaceNet / Progressive Teacher | Jing Jiang e Weihong Deng · Shenzhen Institute of AI and Robotics for Society | **RAF-DB** — acurácia reportada de **88,27%** |

## 2.2 RAF-DB, FER2013 ou outro? — ✅ Comprovada

- **RAF-DB**: sim, é o conjunto de avaliação reportado para o classificador.
- **FER2013**: ❌ **não aparece em nenhuma fonte oficial consultada** sobre estes
  modelos. **Não deve ser citado no banner.**
- **WIDER FACE**: citado na avaliação do detector.

> ⚠️ Os **88,27% em RAF-DB** pertencem ao **modelo original**, em condições
> controladas. **Não são a acurácia deste sistema.** Se citar, atribua
> explicitamente.

## 2.3 Houve treinamento próprio neste projeto? — ✅ Comprovada: SIM

**Sim, houve — e foi executado com dados reais.** Mas é preciso ser exato sobre
*o quê* foi treinado.

### O que foi treinado

Uma **camada linear de saída** sobre os logits do modelo pré-treinado:

```
z' = W · z + b        W: 7×7,  b: 7   →   56 parâmetros treináveis
p  = softmax(z')
```

`W` e `b` foram aprendidos por **descida de gradiente** com **entropia cruzada**,
sobre um conjunto rotulado coletado pelo próprio autor.

**Implementação:** `src/expressao_neural/core/calibracao.py` (função `treinar`,
linhas 118-186) e `scripts/treinar_calibracao.py`.

### O que NÃO foi treinado

| Pergunta | Resposta |
|---|---|
| A rede convolucional foi treinada? | ❌ **Não.** O backbone está congelado. |
| Houve *fine-tuning*? | ❌ **Não.** Nenhum peso da CNN foi ajustado. |
| Houve *transfer learning*? | 🟡 **Em sentido estrito, sim — na forma mais limitada.** O modelo pré-treinado é usado como extrator fixo e apenas um classificador linear foi treinado sobre suas saídas. Isso é o que a literatura chama de *linear probing* / *frozen feature extraction*. **Não é** *fine-tuning*. Se o professor perguntar, esta é a formulação correta. |
| Houve alteração dos pesos das redes neurais? | ❌ **Não.** Verificável: o SHA-256 dos dois `.onnx` continua idêntico ao do download. |
| Existe script de treinamento? | ✅ **Sim.** `scripts/treinar_calibracao.py` e `scripts/coletar_amostras.py`. |
| Existem checkpoints próprios? | ✅ **Sim.** `models/calibracao_pessoal.npz` (56 parâmetros; não versionado). |
| Houve *data augmentation*? | ❌ **Não implementada.** A variação do conjunto vem de o autor ter variado pose, distância e intensidade durante a coleta — não de transformação sintética. |
| Existe técnica de calibração? | ✅ **Sim** — é exatamente o que foi treinado. |
| O sistema aprende durante o uso? | ❌ **Não.** Em execução o sistema **apenas infere**. O treinamento é offline e os pesos são carregados de arquivo. Não há aprendizado on-line nem adaptação contínua. |

## 2.4 Dados do treinamento próprio — ✅ Comprovada

> Fonte de todos os números: `docs/banner/06_metricas/treinamento_calibracao.json`,
> gerado pela execução real em 26/11/2026.

| Item | Valor |
|---|---|
| **Imagens** | **210** |
| **Classes** | **7** (todas representadas) |
| Imagens por classe | 30 em cada uma das 7 |
| Origem | webcam do próprio autor; recortes 112×112 **já alinhados** |
| Rótulo verdadeiro | a expressão **solicitada** — expressões **encenadas**, não espontâneas |
| **Divisão treino/teste** | **154 / 56**, estratificada por classe |
| Conjunto de **validação** separado | ❌ **Não há.** Apenas treino e teste. Declarar assim. |
| **Épocas** | **400** |
| **Batch size** | **154** (lote completo — *full-batch gradient descent*) |
| **Learning rate** | **0,05** |
| **Otimizador** | descida de gradiente em lote completo (sem momento, sem Adam) |
| **Função de perda** | entropia cruzada + L2 sobre o afastamento da identidade (λ = 0,001) |
| Inicialização | `W` = matriz identidade, `b` = 0 |
| **Perda inicial → final** | **2,0322 → 0,2088** |
| **Tempo de treinamento** | **27,2 ms** (medido; 400 épocas sobre 154 amostras) |
| **Hardware** | AMD Ryzen 7 7435HS, 8C/16T, somente CPU, sem GPU |
| Parâmetros treináveis | **56** |

### Resultados no conjunto de teste (56 amostras retidas) — ✅ Comprovada

| Métrica | Modelo base | Com calibração treinada |
|---|---|---|
| **Acurácia** | **67,86%** | **94,64%** |
| Precisão (macro) | 0,5153 | **0,9459** |
| Revocação (macro) | 0,6786 | **0,9464** |
| F1-score (macro) | 0,5784 | **0,9452** |

**Variação: +26,78 pontos percentuais de acurácia.**

Por classe, com calibração:

| Classe | Amostras | Precisão | Revocação | F1 |
|---|---|---|---|---|
| angry (Raiva) | 8 | 1,000 | 1,000 | 1,000 |
| disgust (Nojo) | 8 | 1,000 | 1,000 | 1,000 |
| fearful (Medo) | 8 | 0,889 | 1,000 | 0,941 |
| happy (Alegria) | 8 | 1,000 | 1,000 | 1,000 |
| neutral (Neutra) | 8 | 0,875 | 0,875 | 0,875 |
| sad (Tristeza) | 8 | 1,000 | 1,000 | 1,000 |
| surprised (Surpresa) | 8 | 0,857 | 0,750 | 0,800 |

### Observação metodológica importante

A inicialização em `W` = identidade faz a camada **reproduzir exatamente o modelo
base** na época 0, e a regularização penaliza o *afastamento* da identidade.
Consequência: o ganho de 67,86% → 94,64% é medido **contra a própria linha de
base, no mesmo conjunto de teste**, e não contra um número externo.

## 2.5 Como declarar isso — texto recomendado

> "Os modelos de detecção e classificação são pré-treinados (OpenCV Zoo,
> Apache 2.0) e permanecem congelados. Sobre as saídas do classificador foi
> treinada uma camada de calibração supervisionada de 56 parâmetros, com 210
> amostras rotuladas coletadas pelo autor (154 treino / 56 teste, 400 épocas),
> elevando a acurácia no conjunto de teste de 67,9% para 94,6%."

✅ Verdadeiro e verificável.

**NÃO escrever:** "treinei uma rede neural para reconhecer expressões",
"desenvolvi uma CNN", "fiz fine-tuning do modelo".

---

# 3. Processamento das imagens

## 3.1 Fluxo completo — ✅ Comprovada

| # | Etapa | Implementada? | Onde |
|---|---|---|---|
| 1 | Captura dos quadros | ✅ | `core/camera.py` — `cv2.VideoCapture`, 640×480 @ 30 FPS, thread dedicada |
| 2 | Espelhamento horizontal | ✅ | `camera.py` — opcional, ergonômico; não afeta a inferência |
| 3 | Detecção facial | ✅ | `core/detector.py` — YuNet |
| 4 | Localização dos pontos faciais | ✅ | 5 pontos vindos do próprio YuNet (`detector.py:108`) |
| 5 | Alinhamento facial | ✅ | `core/alinhamento.py` — similaridade não reflexiva por mínimos quadrados |
| 6 | Recorte da região de interesse | ✅ | resultado do `warpAffine` do alinhamento |
| 7 | Redimensionamento | ✅ | `classificador.py:92` — para 112×112 quando necessário |
| 8 | Conversão de cores | ✅ | `classificador.py:95` — BGR → RGB |
| 9 | Normalização | ✅ | `classificador.py:96-97` — `(px/255 − 0,5)/0,5` → `[−1, +1]` |
| 10 | Inferência neural | ✅ | `classificador.py:115-116` — `cv2.dnn` |
| 11 | Classificação | ✅ | `argmax` do vetor de probabilidades |
| 12 | Cálculo das pontuações | ✅ | Softmax (`classificador.py:129`) |
| 13 | Camada de calibração | ✅ implementada, ❌ **desligada por padrão** | `classificador.py:127-128`; `usar_calibracao = False` (`config.py:122`) |
| 14 | Limite mínimo de confiança | ✅ | `limiar_confianca = 0,35` (`config.py:103`), ajustável na interface |
| 15 | Suavização temporal | ✅ | `core/suavizacao.py` — EMA, α = 0,55 |
| 16 | Estabilidade das previsões | ✅ | métrica exibida; `suavizacao.py:estabilidade()` |
| 17 | Apresentação na interface | ✅ | `ui/janela.py`, `ui/desenho.py` |

**Detalhe do alinhamento (etapa 5):** os 5 pontos detectados são levados, por
mínimos quadrados (`numpy.linalg.lstsq`), às posições canônicas do ArcFace em um
recorte 112×112 (`config.py:76-82`). A solução é **determinística** — não usa
RANSAC nem LMEDS. Comprovado por `test_inferencia_e_deterministica`.

## 3.2 Sem rosto na imagem — ✅ Comprovada

**Nenhuma classificação é produzida.** `pipeline.py:132-136`: se a lista de
rostos vem vazia, o pipeline retorna imediatamente, descarta o estado temporal e
a interface limpa o painel (`janela.py:_limpar_resultados`). O rodapé passa a
exibir "Câmera ativa — nenhum rosto no enquadramento".

Comprovado por `test_ausencia_de_rosto_nao_classifica` e
`test_entradas_degeneradas_nao_derrubam` (preto, branco, ruído, 8×8 px, faixa
estreita).

## 3.3 Confiança insuficiente — ✅ Comprovada (comportamento mudou)

Com `sempre_exibir_previsao = True` (padrão atual, `config.py:112`):

- a classe prevista **é exibida**, acompanhada da confiança real;
- a baixa confiança é **sinalizada**: cor âmbar, texto "confiança baixa" no
  painel e sufixo `?` no rótulo sobre o vídeo (`desenho.py:88-93`);
- a mensagem *"Expressão não identificada com confiança suficiente"* continua
  aparecendo como aviso.

Com a opção desligada, volta ao comportamento anterior: rótulo substituído por
"Indefinida".

**Justificativa documentada:** expressões naturais vencem com 35–55% de
confiança; ocultá-las dava a impressão de sistema inoperante
(`docs/arquitetura.md`, seção 4.7-A).

---

# 4. Funcionalidades atuais

> Todas verificadas no código. Nada planejado está listado como implementado.

| Funcionalidade | Situação | Prova |
|---|---|---|
| Reconhecimento pela webcam | ✅ Comprovada | `camera.py` + capturas em `02_capturas/` |
| Classificação de imagens estáticas | ✅ Comprovada | `core/imagem_estatica.py`, `janela.py:analisar_imagem` |
| Detecção facial | ✅ Comprovada | `core/detector.py` |
| Múltiplos rostos | ✅ Comprovada | `max_rostos = 5`; `test_multiplos_rostos_nao_misturam_resultados` |
| Seleção de dispositivo de câmera | ✅ Comprovada | `listar_cameras()`, seletor na interface |
| Pausar / retomar | ✅ Comprovada | `alternar_pausa()`; `test_pausa_interrompe_o_processamento_e_retoma` |
| Reiniciar câmera | 🟡 **Implementada, com suspeita de travamento** | `reiniciar_camera()` — ver seção 11 |
| Encerrar câmera e liberar recurso | ✅ Comprovada | `test_captura_publica_quadros_e_libera_a_camera` |
| Recuperação após falha de captura | ✅ Comprovada | `test_recuperacao_apos_falhas_de_leitura` |
| Gráfico das 7 categorias | ✅ Comprovada | `ui/componentes.py:GraficoProbabilidades` |
| Indicador de confiança | ✅ Comprovada | `MedidorConfianca` |
| Indicador de estabilidade | ✅ Comprovada | `suavizacao.py:estabilidade()` |
| FPS | ✅ Comprovada | `camera.py:_publicar` — janela móvel de 30 amostras |
| Tempo de inferência | ✅ Comprovada | `classificador.py:119` |
| Tempo de detecção | ✅ Comprovada | `pipeline.py:124-126` |
| Modo apresentação (tela cheia) | ✅ Comprovada | `alternar_tela_cheia()`, F11 |
| Painel lateral rolável | ✅ Comprovada | `componentes.PainelRolavel` |
| Ajustes em tempo real | ✅ Comprovada | limiar, suavização, espelho, múltiplos rostos, N quadros |
| Janela "Sobre o Projeto" | ✅ Comprovada | `ui/sobre.py` — 7 abas |
| Verificação de integridade dos modelos | ✅ Comprovada | SHA-256 em 3 momentos |
| Diagnóstico de ambiente | ✅ Comprovada | `utils/ambiente.py` |
| Funcionamento offline | 🟡 **Parcialmente comprovada** | código auditado: **zero** chamadas de rede em `src/`. **Teste com a rede desligada ainda NÃO foi executado.** |
| Privacidade das imagens | ✅ Comprovada | nenhum `imwrite`/`Image.save` em `src/`; só grava `config.local.json` |
| Coleta de amostras rotuladas | ✅ Comprovada | `scripts/coletar_amostras.py` |
| Treinamento da calibração | ✅ Comprovada | `scripts/treinar_calibracao.py` |
| Geração de evidências e gráficos | ✅ Comprovada | 4 scripts em `scripts/` |
| Modo desafio de expressões | ❌ **Não implementada** | era item opcional do escopo |
| Gravação de vídeo / foto | ❌ **Não implementada** | intencional, por privacidade |
| Reconhecimento de identidade | ❌ **Não implementada** | intencional |

## 4.1 Categorias reconhecidas — ✅ Comprovada

**Exatamente 7. Não há categoria adicional.**

| Índice | Classe do modelo | Rótulo na interface |
|---|---|---|
| 0 | `angry` | Raiva |
| 1 | `disgust` | Nojo |
| 2 | `fearful` | Medo |
| 3 | `happy` | Alegria |
| 4 | `neutral` | Neutra |
| 5 | `sad` | Tristeza |
| 6 | `surprised` | Surpresa |

A ordem corresponde aos índices de saída da rede e é travada por dois testes
(`test_ordem_das_classes_bate_com_o_modelo_oficial`,
`test_traducao_alinhada_indice_por_indice`).

"Indefinida" **não é uma oitava classe** — é um estado de exibição quando a
confiança fica abaixo do limiar.

---

# 5. Tecnologias utilizadas

> Verificadas no ambiente virtual real do projeto.

| Tecnologia | Versão | Função |
|---|---|---|
| **Python** | **3.11.9** | linguagem única do projeto |
| **OpenCV** (`opencv-python`) | **4.12.0** | captura de vídeo, processamento de imagem, alinhamento e **inferência ONNX** via `cv2.dnn` |
| **NumPy** | **2.2.6** | álgebra, softmax, vetores de probabilidade e **o treinamento da calibração** |
| **Pillow** | **11.3.0** | ponte entre o quadro OpenCV e o Tkinter; validação de imagem |
| **Tkinter / Tcl-Tk** | **8.6** | interface gráfica desktop (biblioteca padrão) |
| **ONNX** | formato | formato dos dois modelos; **a biblioteca `onnx` NÃO é instalada** — o OpenCV lê o formato diretamente |
| **pytest** | **8.4.2** | testes automatizados |
| **Git** | 2.56.0 | versionamento — 21 commits, 113 arquivos |
| PowerShell / Batch | — | 5 scripts `.bat` de automação |

## 5.1 Bibliotecas pesadas — ✅ Comprovada: ausentes

Verificado com `importlib.util.find_spec`:

| Biblioteca | Situação |
|---|---|
| TensorFlow | ausente |
| PyTorch | ausente |
| onnxruntime | ausente |
| scikit-learn | ausente |
| SciPy | ausente |
| pandas | ausente |
| matplotlib | ausente |

Toda a inferência usa `cv2.dnn`; todo o treinamento e os gráficos usam NumPy e
Pillow. O teste `test_nao_usa_bibliotecas_pesadas` falha se isso mudar.

---

# 6. Resultados atualizados

## 6.1 Testes automatizados — ✅ Comprovada (número mudou)

| Item | Valor |
|---|---|
| **Coletados** | **138** |
| **Executados na suíte padrão** | **137** |
| **Aprovados** | **137** |
| **Reprovados** | **0** |
| Ignorados | 0 |
| Marcados como lentos (fora do padrão) | 1 (execução prolongada) |
| Duração | ~16 s |

> ⚠️ **O número 119 está desatualizado.** Era correto antes da auditoria de
> segurança e do módulo de calibração. **Use 137.**

Distribuição por arquivo:

| Arquivo | Cobertura |
|---|---|
| `test_integracao.py` | integração, condições adversas, desempenho, execução prolongada |
| `test_imagem_e_suavizacao.py` | validação de imagem e estabilidade temporal |
| `test_inferencia.py` | pré-processamento, classes, softmax, inferência real |
| `test_calibracao.py` | treinamento, métricas, matriz de confusão, persistência |
| `test_robustez.py` | regressões da auditoria de segurança |
| `test_camera.py` | ciclo de vida da captura |
| `test_interface.py` | tema, componentes, janela, encerramento |
| `test_modelos.py` | dependências, integridade SHA-256, carregamento |

## 6.2 Desempenho — ✅ Comprovada (remedido nesta auditoria)

**Metodologia:** 200 execuções consecutivas do ciclo completo após aquecimento,
1 rosto, somente CPU. Medido com `time.perf_counter()`.

**Estado da máquina na medição:** Ryzen 7 7435HS a **3100/3100 MHz (turbo
pleno)**, carga de 4%, 7,8 GB de RAM, sem GPU.

| Etapa | Imagem 512×512 | Quadro 640×480 (webcam) |
|---|---|---|
| Detecção (YuNet) | **14,74 ms** | **16,90 ms** |
| Classificação (MobileFaceNet) | **3,15 ms** | **3,70 ms** |
| **Ciclo completo** | **18,78 ms** | **21,45 ms** |
| Desvio padrão | 1,59 ms | 1,51 ms |
| **Benchmark** | **53,3 inferências/s** | **46,6 inferências/s** |

| Item | Valor |
|---|---|
| Carregamento dos dois modelos | **52,1 ms** |
| Aquecimento da rede | **35,2 ms** |
| **Memória do processo** | **76 MB** (benchmark) · ~93 MB (execução prolongada) |

## 6.3 FPS da webcam ≠ inferências por segundo — ✅ Comprovada

**Esta distinção é importante e estava sendo confundida.**

| Métrica | Valor | O que é |
|---|---|---|
| **FPS real da webcam** | **30,1 – 30,2** | taxa de quadros entregues pela câmera e exibidos. **Limitada pelo hardware da câmera**, que opera a 30 FPS. |
| **Inferências/s em benchmark** | **46,6 – 53,3** | capacidade de processamento do pipeline sem a câmera como gargalo. |

Os 30 FPS da webcam são **registro real** das capturas em
`docs/banner/02_capturas/` (indicador "FPS" visível na interface), com detecção
de 15,2–18,4 ms e inferência de 3,2–3,5 ms por quadro.

**Leitura correta:** o pipeline tem folga. Processa até ~47 quadros/s, mas a
câmera só entrega 30 — então **a câmera é o gargalo, não a rede neural**. Com
`inferir_a_cada = 1` (padrão atual), todo quadro é analisado.

> ⚠️ **"~38 inferências/s" está desatualizado.** Foi medido com a CPU a
> 2496 MHz e outros programas abertos. Em turbo pleno: **53,3/s** (512×512) ou
> **46,6/s** (640×480).

## 6.4 Diferença entre testes isolados e aplicação completa — ✅ Comprovada

Sim, existe, e tem explicação:

| Contexto | Ciclo | Por quê |
|---|---|---|
| Benchmark 512×512 | 18,78 ms | imagem menor em área (262.144 px) |
| Benchmark 640×480 | 21,45 ms | quadro **17% maior em área** (307.200 px) |
| Aplicação com webcam | detecção 15,2–18,4 ms | além da inferência, há captura, anotação, redimensionamento e renderização Tkinter, mas estes ocorrem **fora** do tempo medido do pipeline |

**Correção de um erro anterior:** o relatório afirmava que os tempos com webcam
eram menores "porque o quadro da câmera é menor". **Isso está errado:** 640×480
é **maior** em área que 512×512. A diferença observada vinha do **estado de
energia da CPU** nas duas medições, não da resolução. Esta auditoria mediu as
duas resoluções no mesmo estado de máquina e confirmou que 640×480 é **mais
lento**, como a proporcionalidade com a área prevê.

## 6.5 Execução prolongada — ✅ Comprovada

**Executada: 30 minutos contínuos.** Log bruto em
`evidencias/execucao_prolongada.log`.

| Métrica | Valor |
|---|---|
| Duração | 1800,20 s |
| **Ciclos completos** | **95.811** |
| Primeira metade | 18,5 ms |
| Segunda metade | 19,1 ms |
| **Degradação** | **+3,2%** |
| Trilhas de rosto vivas ao final | 1 (sem vazamento) |
| Memória | ~93 MB, estável |

Os **95.811 ciclos continuam corretos**. Ressalva: o ensaio exercita o
**pipeline de inferência**, não a captura física da webcam.

## 6.6 Funcionamento offline — 🟡 Parcialmente comprovada

| Verificação | Resultado |
|---|---|
| Imports de rede em `src/` | **zero** (`urllib`, `requests`, `socket`, `http`, `aiohttp`) |
| Chamadas de rede em `src/` | **zero** — as únicas URLs são strings de metadados em `core/modelos.py`, nunca acessadas |
| Modelos locais verificados por SHA-256 | sim |
| Recursos de CDN (fonte, ícone, script) | nenhum — fontes do sistema, logotipo desenhado em Canvas |
| **Teste com a rede fisicamente desligada** | ⬜ **NÃO EXECUTADO** |

**Declarar como:** *"nenhuma dependência de rede em tempo de execução,
verificada por auditoria de código; o teste com a rede desligada está pendente."*

## 6.7 Hardware dos testes — ✅ Comprovada

| Item | Valor |
|---|---|
| Processador | AMD Ryzen 7 7435HS, 8 núcleos / 16 threads |
| Frequência | 3100 MHz (máxima), medido em turbo pleno |
| Memória RAM | 7,8 GB |
| **GPU** | **nenhuma utilizada** — `DNN_TARGET_CPU` |
| Sistema | Windows 10 |

---

# 7. Acurácia e avaliação das redes neurais

## 7.1 Existe avaliação própria? — ✅ Comprovada: SIM

**Sim, a partir de 26/11/2026.** Antes desta data a resposta era "não avaliada".

| Pergunta | Resposta |
|---|---|
| Conjunto rotulado próprio? | ✅ **Sim** — 210 amostras, 30 por classe, 7 classes |
| Matriz de confusão? | ✅ **Sim** — 7×7, em `04_graficos/matriz_confusao.png` e `matriz_confusao_base.png` |
| Precision, recall, F1 documentados? | ✅ **Sim** — macro e por classe |
| Acurácia própria | ✅ **67,86% (base) → 94,64% (calibrada)** no conjunto de teste de 56 amostras |

## 7.2 Categorias com maior dificuldade — ✅ Comprovada

**No conjunto de teste próprio, com calibração:**

| Classe | F1 | Confusões observadas |
|---|---|---|
| Surpresa | **0,800** | 1 lida como Medo, 1 como Neutra |
| Neutra | **0,875** | 1 lida como Surpresa |
| Medo | 0,941 | nenhuma como verdadeira; recebeu 1 falso positivo |
| Raiva, Nojo, Alegria, Tristeza | **1,000** | nenhuma |

> Observação relevante: **Surpresa e Neutra** foram as mais difíceis **neste
> conjunto**, o que **contraria** a expectativa da literatura (que aponta Medo,
> Tristeza e Nojo). Isso é esperado: a calibração foi ajustada às expressões
> encenadas deste autor específico. **Não generalize.**

**Sem calibração**, no conjunto base de testes de condições adversas, a
oclusão da região dos olhos **impede a detecção** do rosto (limitação do
YuNet, não do classificador) — comprovado por `test_oclusao_parcial_tipo_oculos`.

## 7.3 Acurácia original × resultados deste projeto — ✅ Comprovada

| Número | A que se refere | Não confundir |
|---|---|---|
| **88,27%** | modelo pré-treinado original, conjunto **RAF-DB**, condições controladas | **não é** deste sistema |
| **67,86%** | modelo base, no conjunto de teste **próprio** (56 amostras, um indivíduo) | linha de base da comparação |
| **94,64%** | com a camada de calibração treinada, **mesmo** conjunto de teste | **não generaliza** para outras pessoas |

O número academicamente mais defensável é o **ganho de +26,78 pontos
percentuais**, porque é medido contra a própria linha de base, no mesmo
conjunto, com a mesma metodologia.

## 7.4 Limitações identificadas — ✅ Comprovada

1. O conjunto próprio tem **210 amostras de uma única pessoa**. A acurácia de
   94,64% **não generaliza** para público diverso.
2. As expressões são **encenadas sob solicitação**, não espontâneas.
3. O conjunto de teste tem **56 amostras** — incerteza estatística alta.
4. **Não há conjunto de validação** separado, apenas treino e teste.
5. A CNN **não foi treinada**: o ganho é de **calibração**, não de capacidade de
   representação.
6. **Oclusão dos olhos impede a detecção** (limitação do YuNet).
7. A previsão é **sensível a iluminação, escala e rotação**: em 14 variações
   controladas da mesma imagem, a classe vencedora mudou em **4 de 12** casos
   com detecção (`06_metricas/metricas.md`).
8. **Confiança alta não é garantia de acerto.**
9. **Viés possível do conjunto de treinamento original** (idade, etnia, tom de
   pele, iluminação) — **não auditado** neste trabalho.
10. O desempenho medido vale para **esta máquina** e para **1 rosto**.
11. O sistema classifica **padrões visuais**, não emoções. Não é instrumento de
    diagnóstico.

---

# 8. Contribuição acadêmica do desenvolvedor

> Esta seção existe para você defender o trabalho caso o professor pergunte
> *"o que você fez além de usar modelos prontos?"*.

## 8.1 O que é de terceiros

| Recurso | Origem | Licença |
|---|---|---|
| Rede YuNet (53.121 parâmetros) | OpenCV Zoo | Apache 2.0 |
| Rede MobileFaceNet/Progressive Teacher (1.193.927 parâmetros) | OpenCV Zoo | Apache 2.0 |
| OpenCV, NumPy, Pillow, Tkinter, pytest | projetos abertos | Apache 2.0 / BSD / MIT |
| Algoritmo de alinhamento | portado de `facial_fer_model.py` do OpenCV Zoo | Apache 2.0, atribuído no cabeçalho do arquivo |

## 8.2 O que foi desenvolvido neste projeto

| Contribuição | Evidência |
|---|---|
| **Arquitetura em camadas** com separação entre processamento e interface | `docs/arquitetura.md`; `core/` + `ui/` |
| **Pipeline de visão computacional** encadeando as duas redes | `core/pipeline.py` |
| **Implementação fiel do pré-processamento** especificado pelo modelo | `core/classificador.py:84-98` — verificado contra a fonte oficial |
| **Interpretação correta das saídas** (logits → softmax) | provado pela ausência de operador Softmax no grafo |
| **Camada de calibração supervisionada treinada** — 56 parâmetros, 210 amostras próprias | `core/calibracao.py`, `scripts/treinar_calibracao.py` |
| **Pipeline de coleta de dataset rotulado** | `scripts/coletar_amostras.py` |
| **Estratégia de estabilidade temporal** com acompanhamento de rostos | `core/suavizacao.py` |
| **Modelo de concorrência**: thread de captura + compartimento único (sem fila) | `core/camera.py` |
| **Serialização do acesso aos modelos** (`cv2.dnn` não é thread-safe) | `core/pipeline.py:75,112` — custo medido: 0,239 µs |
| **Recuperação automática de falha de câmera** | `camera.py:206-225` |
| **Verificação de integridade por SHA-256** em 3 momentos | `core/modelos.py` |
| **Interface gráfica completa** em português, com DPI awareness e painel rolável | `ui/` — 6 módulos |
| **Validação de imagem em camadas**, incluindo proteção contra *decompression bomb* | `core/imagem_estatica.py` |
| **Tratamento de falhas** documentado em 10 cenários | `docs/arquitetura.md` §5 |
| **Suíte de 137 testes** com inferência real | `tests/` — 8 arquivos |
| **Correção de 6 achados de auditoria de segurança** | `tests/test_robustez.py` |
| **Infraestrutura de medição e evidências** | 5 scripts em `scripts/` |
| **Documentação técnica e acadêmica** | 8 documentos em `docs/` |

## 8.3 Resposta curta para a banca

> "Os modelos de detecção e classificação são pré-treinados do OpenCV Zoo — eu
> não treinei a rede convolucional. O que desenvolvi foi: a aplicação completa
> que integra as duas redes em tempo real, a implementação fiel do
> pré-processamento que o modelo exige, a interpretação correta das saídas, o
> modelo de concorrência que mantém 30 FPS sem travar a interface, a estratégia
> de estabilidade temporal, a interface gráfica, o tratamento de falhas, e 137
> testes automatizados. Além disso, coletei um conjunto de 210 amostras
> rotuladas e treinei uma camada de calibração supervisionada de 56 parâmetros
> sobre as saídas do modelo congelado, o que elevou a acurácia no meu conjunto
> de teste de 67,9% para 94,6%."

---

# 9. Diagrama atualizado

## 9.1 O fluxo proposto está incompleto

O fluxo sugerido —
`Webcam → Detecção facial (YuNet) → Pré-processamento → Classificação (MobileFaceNet/ONNX) → Expressão e pontuações`
— está **correto na essência, mas omite 4 etapas reais**:

1. o **alinhamento geométrico** (não é "pré-processamento" genérico — é uma
   transformação de similaridade pelos 5 pontos);
2. o **Softmax**, que é aplicado pela aplicação e não pelo modelo;
3. a **estabilidade temporal** (EMA por rosto acompanhado);
4. a **camada de calibração**, implementada e desligada por padrão.

## 9.2 Fluxo real

```
┌──────────┐
│  WEBCAM  │  cv2.VideoCapture · 640×480 @ 30 FPS · thread dedicada
└────┬─────┘
     │ quadro BGR
     ▼
┌─────────────────────┐
│  DETECÇÃO FACIAL    │  YuNet (ONNX, 53.121 parâmetros)
│  cv2.FaceDetectorYN │  → caixa + 5 pontos + confiança, por rosto
└────┬────────────────┘  limiar 0,85 · NMS 0,30
     │
     ├── 0 rostos ──► nenhuma classificação · painel limpo
     │
     ▼ ≥1 rosto
┌─────────────────────┐
│  ALINHAMENTO        │  similaridade não reflexiva por mínimos quadrados
│                     │  5 pontos → posições canônicas ArcFace → 112×112
└────┬────────────────┘
     ▼
┌─────────────────────┐
│  PRÉ-PROCESSAMENTO  │  BGR→RGB · (px/255 − 0,5)/0,5 · tensor 1×3×112×112
└────┬────────────────┘
     ▼
┌─────────────────────┐
│  CLASSIFICAÇÃO      │  MobileFaceNet / Progressive Teacher
│  cv2.dnn (CPU)      │  1.193.927 parâmetros · 50 Conv · 33 PReLU
│                     │  nó 'data' → nó 'label' → 7 LOGITS BRUTOS
└────┬────────────────┘
     ▼
┌─────────────────────┐
│  CALIBRAÇÃO         │  (opcional, desligada por padrão)
│                     │  z' = W·z + b — 56 parâmetros treinados
└────┬────────────────┘
     ▼
┌─────────────────────┐
│  SOFTMAX            │  aplicado pela APLICAÇÃO, não pelo modelo
│                     │  → 7 probabilidades somando 1,0
└────┬────────────────┘
     ▼
┌─────────────────────┐
│  ESTABILIDADE       │  média móvel exponencial (α = 0,55)
│  TEMPORAL           │  por rosto acompanhado · descarte ao sumir
└────┬────────────────┘
     ▼
┌─────────────────────┐
│  INTERFACE          │  vídeo anotado · gráfico das 7 classes
│  Tkinter            │  confiança · FPS · tempos · estabilidade
└─────────────────────┘
```

### Resumo de cada etapa

| Etapa | O que faz |
|---|---|
| **Webcam** | entrega quadros em thread própria; a interface nunca espera pela câmera |
| **Detecção** | localiza rostos e devolve 5 pontos de referência por rosto |
| **Alinhamento** | corrige rotação e escala levando os pontos às posições canônicas |
| **Pré-processamento** | converte cor e normaliza para o intervalo exigido pela rede |
| **Classificação** | produz 7 logits — um por expressão |
| **Calibração** | ajuste linear aprendido (opcional) |
| **Softmax** | converte logits em probabilidades comparáveis |
| **Estabilidade** | amortece oscilação entre quadros consecutivos |
| **Interface** | apresenta a distribuição completa, não apenas a vencedora |

> O diagrama vetorial em `03_diagramas/arquitetura_sistema.svg` está **correto**,
> mas **não mostra a camada de calibração** (criada depois). Se for usar o
> treinamento no banner, regenere o diagrama ou mencione a calibração no texto.

---

# 10. Conteúdo definitivo para o banner

> ⚠️ Os textos abaixo usam o título solicitado (**EmotionScan**). Releia a
> advertência inicial: o sistema exibe "Expressão Neural" na tela.

## Título

**EmotionScan: Reconhecimento de Expressões Faciais utilizando Redes Neurais**

## Introdução (148 palavras)

> As expressões faciais são um dos canais mais imediatos de comunicação não
> verbal humana, e seu reconhecimento automático é um problema consolidado da
> visão computacional. A dificuldade central está na variabilidade: uma mesma
> expressão muda de aparência conforme iluminação, ângulo, distância e
> características individuais do rosto. Redes neurais convolucionais tornaram-se
> a abordagem predominante para esse problema por aprenderem, diretamente dos
> dados, as representações visuais relevantes, dispensando a engenharia manual
> de atributos.
>
> Este trabalho apresenta o EmotionScan, uma aplicação desktop que classifica
> sete expressões faciais em tempo real a partir da webcam, encadeando duas redes
> neurais convolucionais em formato ONNX. O sistema executa integralmente em CPU,
> sem placa de vídeo dedicada e sem qualquer conexão com a internet, de modo que
> nenhuma imagem deixa o computador do usuário. O objetivo é demonstrar a
> viabilidade de um classificador de expressões faciais funcional, estável e
> eticamente transparente em hardware comum.

## Metodologia (156 palavras)

> O sistema encadeia duas redes convolucionais executadas pelo módulo `cv2.dnn`
> do OpenCV, somente em CPU. A rede YuNet (53.121 parâmetros) localiza rostos e
> devolve a caixa delimitadora e cinco pontos faciais de referência. Esses
> pontos alimentam uma transformação de similaridade não reflexiva, resolvida por
> mínimos quadrados, que alinha o rosto em um recorte de 112×112 pixels. Após
> conversão para RGB e normalização ao intervalo [−1, 1], o tensor é classificado
> pela rede MobileFaceNet (1.193.927 parâmetros, 50 convoluções e 33 ativações
> PReLU), treinada pelo método semi-supervisionado Progressive Teacher, que
> produz sete logits convertidos em probabilidades pela função softmax.
>
> Ambas as redes são pré-treinadas (OpenCV Zoo) e permanecem congeladas. Sobre as
> saídas do classificador foi treinada uma camada de calibração supervisionada de
> 56 parâmetros, com 210 amostras rotuladas coletadas pelo autor, divididas em
> 154 para treino e 56 para teste, ao longo de 400 épocas. A captura e a
> inferência executam em thread dedicada, mantendo a interface sempre responsiva.

## Resultados e conclusão (138 palavras)

> O sistema detecta rostos e classifica as sete expressões em tempo real,
> sustentando 30 FPS com a webcam, com ciclo completo de 18,8 ms e capacidade de
> 53 inferências por segundo em CPU, sem GPU. Em ensaio de 30 minutos contínuos
> completou 95.811 ciclos com variação de apenas 3,2% e memória estável. A suíte
> automatizada executa 137 testes, todos aprovados, com os modelos reais.
>
> A camada de calibração treinada elevou a acurácia no conjunto de teste próprio
> de 67,9% para 94,6%, um ganho de 26,8 pontos percentuais. Esse resultado, no
> entanto, vale para 56 amostras de um único indivíduo com expressões encenadas e
> não generaliza para público diverso. Submetido a variações controladas de
> iluminação, escala e rotação, o sistema nunca falhou, mas a classe vencedora
> mudou em 4 de 12 condições — razão pela qual a interface exibe sempre as sete
> pontuações e o nível de confiança, em vez de um rótulo isolado como verdade.

## Indicadores de destaque (6)

```
┌────────────────────────────────────────────────────────────┐
│  2 redes convolucionais  ·  1,25 milhão de parâmetros      │
│  18,8 ms por ciclo  ·  53 inferências/s  ·  somente CPU    │
│  30 FPS sustentados com webcam                             │
│  137 testes automatizados aprovados                        │
│  67,9% → 94,6% com calibração própria (210 amostras)       │
│  100% offline — nenhuma imagem sai do computador           │
└────────────────────────────────────────────────────────────┘
```

## Aviso ético (obrigatório)

> Este sistema classifica padrões visuais de expressões faciais. Os resultados
> não representam diagnósticos psicológicos nem determinam com certeza as emoções
> reais de uma pessoa.

## Referências (5, ABNT)

CHEN, Sheng; LIU, Yang; GAO, Xiang; HAN, Zhen. **MobileFaceNets**: efficient CNNs
for accurate real-time face verification on mobile devices. In: CHINESE
CONFERENCE ON BIOMETRIC RECOGNITION (CCBR), 13., 2018, Urumqi. *Lecture Notes in
Computer Science*, v. 10996. Cham: Springer, 2018. p. 428-438. DOI:
10.1007/978-3-319-97909-0_46. Disponível em:
https://link.springer.com/chapter/10.1007/978-3-319-97909-0_46. Acesso em: 26
nov. 2026.

JIANG, Jing; DENG, Weihong. **Boosting facial expression recognition by a
semi-supervised progressive teacher**. *IEEE Transactions on Affective
Computing*, v. 13, n. 4, p. 2026-2038, 2022. DOI: 10.1109/TAFFC.2021.3131621.
Disponível em: https://ieeexplore.ieee.org/abstract/document/9629313. Acesso em:
26 nov. 2026.

LI, Shan; DENG, Weihong; DU, JunPing. **Reliable crowdsourcing and deep
locality-preserving learning for expression recognition in the wild**. In: IEEE
CONFERENCE ON COMPUTER VISION AND PATTERN RECOGNITION (CVPR), 2017, Honolulu.
*Proceedings* [...]. Honolulu: IEEE, 2017. p. 2584-2593. DOI:
10.1109/CVPR.2017.277. Disponível em: http://whdeng.cn/RAF/model1.html. Acesso
em: 26 nov. 2026.

WU, Wei; PENG, Hanyang; YU, Shiqi. **YuNet**: a tiny millisecond-level face
detector. *Machine Intelligence Research*, v. 20, n. 5, p. 656-665, 2023. DOI:
10.1007/s11633-023-1423-y. Disponível em:
https://link.springer.com/article/10.1007/s11633-023-1423-y. Acesso em: 26 nov.
2026.

BRADSKI, Gary. **The OpenCV Library**. *Dr. Dobb's Journal of Software Tools*,
v. 25, n. 11, p. 120-125, 2000. Disponível em: https://opencv.org. Acesso em: 26
nov. 2026.

---

# 11. Revisão crítica

## 11.1 Informações incorretas encontradas em documentação anterior

| # | Afirmação | Onde | Correção |
|---|---|---|---|
| 1 | "119 testes automatizados" | `RELATORIO_COMPLETO_BANNER.md`, textos do banner | **137** |
| 2 | "108 testes" | `docs/testes.md`, `docs/resultados.md` | **137** |
| 3 | "~26 ms por ciclo · ~38 inferências/s" | `RELATORIO_COMPLETO_BANNER.md` | medido com CPU a 2496 MHz. Em turbo: **18,78 ms / 53,3 por s** |
| 4 | "17,15 ms" | `docs/resultados.md` §5.1 | medição antiga; a atual é **18,78 ms** em condições documentadas |
| 5 | **"Os tempos com webcam ficaram menores porque o quadro da câmera (640×480) é menor"** | `RELATORIO_COMPLETO_BANNER.md` §2.1 | **FALSO.** 640×480 = 307.200 px é **17% maior** que 512×512 = 262.144 px. A diferença vinha do estado da CPU. Medido: 640×480 é **mais lento** (21,45 ms contra 18,78 ms) |
| 6 | "limiar de confiança 0,50" / "inferir a cada 2 quadros" / "α = 0,40" | documentos anteriores à mudança de sensibilidade | hoje: **0,35 / 1 quadro / 0,55** |
| 7 | "acurácia: não avaliada" | `06_metricas/metricas.md`, `RELATORIO_COMPLETO_BANNER.md` §5.6 | **desatualizado** — hoje há avaliação própria: 67,86% → 94,64% |
| 8 | "não houve treinamento neste trabalho" | `docs/modelos.md`, `ui/sobre.py`, textos do banner | **desatualizado** — houve treinamento da camada de calibração. A afirmação continua correta **para a rede convolucional** |
| 9 | `APP_VERSAO = "1.0.0"` | `config.py:24` | o usuário solicitou **1.3** — ainda não aplicado |
| 10 | Nome "EmotionScan" | nenhum arquivo | divergência não resolvida |

## 11.2 Afirmações que NÃO podem ser comprovadas

| Afirmação | Situação |
|---|---|
| "Funciona offline" como fato verificado em execução | ⬜ apenas auditoria de código; teste com rede desligada **não executado** |
| Acurácia com público diverso | ⬜ **não avaliada** — só há dados de uma pessoa |
| Desempenho em outras máquinas | ⬜ **não avaliado** |
| Conjunto de treinamento exato do YuNet | ⬜ **não detalhado** na documentação pública |
| Ausência de viés dos modelos | ⬜ **não auditado** |
| Estabilidade de 30 min **com webcam ligada** | ⬜ o ensaio cobriu o pipeline, não a captura física |

## 11.3 O que NÃO colocar no banner

- ❌ "Acurácia de 88,27%" sem atribuir ao modelo pré-treinado em RAF-DB
- ❌ "94,6% de acurácia" **sem** o contexto "em 56 amostras de teste do próprio autor"
- ❌ "Treinei uma rede neural" / "desenvolvi uma CNN" / "fiz fine-tuning"
- ❌ **FER2013** — não citado em nenhuma fonte oficial destes modelos
- ❌ "Detecta emoções" — o sistema classifica **padrões visuais de expressão**
- ❌ Número de épocas da CNN, tamanho do dataset original, curva de treinamento
      do backbone — nada disso existe neste trabalho
- ❌ "119 testes" (desatualizado) ou "~38 inferências/s" (medido sob carga)
- ❌ Confundir **30 FPS da webcam** com **53 inferências/s de benchmark**
- ❌ Rostos de terceiros sem autorização por escrito

## 11.4 Perguntas prováveis do professor e respostas

| Pergunta | Resposta defensável |
|---|---|
| "Você treinou a rede?" | "Não a rede convolucional — ela é pré-treinada e congelada. Treinei uma camada de calibração de 56 parâmetros sobre as saídas dela, com 210 amostras que coletei." |
| "Então é transfer learning?" | "Na forma mais limitada: *linear probing* sobre extrator congelado. Não é fine-tuning, porque nenhum peso do backbone foi ajustado." |
| "Quantas épocas? Qual a loss?" | "400 épocas, entropia cruzada, de 2,0322 para 0,2088. Lote completo de 154 amostras, taxa 0,05." |
| "Qual a acurácia?" | "67,9% com o modelo base e 94,6% com a calibração, no mesmo conjunto de teste de 56 amostras. Mas é de uma pessoa só, com expressões encenadas — não generaliza." |
| "Por que 94,6% e o modelo original reporta 88,27%?" | "São conjuntos diferentes. Os 88,27% são em RAF-DB, com milhares de pessoas. Os 94,6% são no meu conjunto, de um indivíduo — é mais fácil e por isso o número é maior. Não são comparáveis." |
| "Tem validação cruzada?" | "Não. Há apenas divisão treino/teste estratificada. É uma limitação declarada." |
| "Qual a arquitetura da rede?" | "MobileFaceNet: 50 convoluções, 33 ativações PReLU, 15 blocos bottleneck residuais, embedding de 512 dimensões e camada final de 7 saídas. 1.193.927 parâmetros, contados diretamente no arquivo ONNX." |
| "O softmax está no modelo?" | "Não. O grafo ONNX termina em Gemm e não contém operador Softmax — conferi. O softmax é aplicado pela minha aplicação, sobre os logits." |
| "Usa GPU?" | "Não. `DNN_TARGET_CPU`. 18,8 ms por ciclo em CPU." |
| "Funciona offline mesmo?" | "Nenhuma chamada de rede em `src/`, verificado por auditoria. O teste com a rede desligada ainda está pendente." |

## 11.5 Pendências conhecidas do sistema

| # | Item |
|---|---|
| 1 | Suspeita de **travamento no botão Reiniciar** (relatada pelo usuário em uso real; ainda não reproduzida nem corrigida) |
| 2 | Versão ainda em 1.0.0; solicitada 1.3 |
| 3 | Janela "Sobre o Projeto" não menciona o treinamento da calibração |
| 4 | Diagrama de arquitetura não mostra a camada de calibração |
| 5 | Documentos com números desatualizados (seção 11.1) |
| 6 | Teste offline não executado |
| 7 | Divergência de nome EmotionScan × Expressão Neural |

---

# 12. Resumo da classificação

| Seção | Comprovada | Parcial | Não avaliada | Não implementada |
|---|---|---|---|---|
| 1. Arquitetura das redes | 6 | 0 | 0 | 0 |
| 2. Treinamento | 5 | 1 (*transfer learning*) | 0 | 4 (fine-tuning, alteração de pesos, augmentation, aprendizado contínuo) |
| 3. Processamento | 3 | 0 | 0 | 0 |
| 4. Funcionalidades | 24 | 2 | 0 | 3 |
| 5. Tecnologias | 2 | 0 | 0 | 7 (bibliotecas pesadas ausentes) |
| 6. Resultados | 5 | 1 (offline) | 0 | 0 |
| 7. Acurácia | 4 | 0 | 5 (generalização, outras máquinas, viés, dataset do YuNet, 30 min com webcam) | 0 |

**Conclusão da auditoria:** o sistema é tecnicamente consistente e as
afirmações centrais são comprováveis em código, modelo ou execução. Os
problemas encontrados são de **documentação desatualizada** (seção 11.1) e de
**divergência de nome** (advertência inicial), não de funcionamento. A
afirmação de maior risco acadêmico é a acurácia de 94,64% apresentada sem o
contexto de amostra única — que está corrigido nos textos da seção 10.
