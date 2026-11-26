# RELATÓRIO COMPLETO PARA O BANNER ACADÊMICO
## Expressão Neural — Reconhecimento de Expressões Faciais em Tempo Real utilizando Redes Neurais

| | |
|---|---|
| **Acadêmico** | Matheus Henrique de Araújo Carneiro |
| **E-mail** | matheus.carneiro@academico.unirv.edu.br |
| **Instituição** | Universidade de Rio Verde — UniRV |
| **Faculdade** | FASOFT |
| **Curso** | Engenharia de Software |
| **Disciplina** | Redes Neurais Artificiais |
| **Professor** | Sandro Silva Moreira |
| **Evento** | 5º Workshop de Trabalhos de Inteligência Artificial |
| **Apresentação** | 30/11/2026 |

> **Natureza deste documento.** Reúne, em um único arquivo, tudo o que é
> necessário para montar o banner no Canva. **Todos os números foram medidos em
> execução real** nesta máquina, pelos scripts indicados em cada seção. Itens
> que não puderam ser medidos estão marcados como **não avaliado**, com o motivo.

---

# 1. Informações técnicas do sistema

## 1.1 Linguagem e bibliotecas

| Item | Versão | Papel |
|---|---|---|
| **Python** | 3.11.9 | linguagem única do projeto |
| **OpenCV** (`opencv-python`) | 4.12.0.88 | captura de vídeo, processamento de imagem e **inferência ONNX** (`cv2.dnn`) |
| **NumPy** | 2.2.6 | operações numéricas e vetores de probabilidade |
| **Pillow** | 11.3.0 | ponte entre o quadro OpenCV e o Tkinter |
| **Tkinter / Tcl-Tk** | 8.6 | interface gráfica desktop (biblioteca padrão) |
| **pytest** | 8.4.2 | testes automatizados |

**Não são utilizados** TensorFlow, PyTorch nem onnxruntime. A inferência ONNX é
feita diretamente pelo OpenCV, o que mantém a instalação leve. Há um teste
automatizado (`test_nao_usa_bibliotecas_pesadas`) que falha caso alguma dessas
bibliotecas passe a ser requisito.

## 1.2 Arquitetura do sistema

Aplicação desktop monolítica, offline, organizada em camadas:

```
app.py  →  ui/janela.py (thread do Tkinter)
                 ↑ consome apenas o quadro mais recente (compartimento único)
        core/camera.py (thread dedicada de captura + inferência)
                 ↓
        core/pipeline.py
          ├─ core/detector.py       YuNet (ONNX)
          ├─ core/alinhamento.py    similaridade por mínimos quadrados
          ├─ core/classificador.py  MobileFaceNet (ONNX) + softmax
          └─ core/suavizacao.py     média móvel exponencial por rosto
```

Decisões relevantes:

- **Separação entre processamento e interface.** Captura e inferência executam
  fora da thread do Tkinter; a interface nunca bloqueia.
- **Compartimento único em vez de fila.** O trabalhador sobrescreve sempre o
  último quadro. Se a interface não consumir a tempo, o quadro antigo é
  descartado: a latência não cresce e a memória não acumula.
- **Carregamento tardio dos modelos**, apenas quando a câmera inicia ou uma
  imagem é aberta.
- **Serialização do acesso aos modelos.** `cv2.dnn.Net` e `cv2.FaceDetectorYN`
  não são thread-safe; o acesso é protegido por `RLock` (custo medido:
  0,239 µs por aquisição).

## 1.3 Modelos de redes neurais

> **Declaração obrigatória: os dois modelos são PRÉ-TREINADOS por terceiros
> (OpenCV Zoo, licença Apache 2.0). NÃO houve treinamento, ajuste fino ou
> alteração de pesos neste trabalho.**

### Detecção — YuNet

| Item | Valor |
|---|---|
| Arquivo | `face_detection_yunet_2023mar.onnx` (232.589 bytes) |
| SHA-256 | `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` |
| Arquitetura | CNN *anchor-free* leve, multiescala, com supressão de não-máximos |
| Saída por rosto | caixa (4 valores) + 5 pontos faciais (10 valores) + confiança |
| Base de treinamento | avaliação reportada no benchmark **WIDER FACE**; o conjunto exato de treinamento não é detalhado na documentação pública |
| Interface usada | `cv2.FaceDetectorYN` |

### Classificação — MobileFaceNet / Progressive Teacher

| Item | Valor |
|---|---|
| Arquivo | `facial_expression_recognition_mobilefacenet_2022july.onnx` (4.791.892 bytes) |
| SHA-256 | `4f61307602fc089ce20488a31d4e4614e3c9753a7d6c41578c854858b183e1a9` |
| Arquitetura interna | *backbone* **MobileFaceNet** — convoluções separáveis em profundidade (*depthwise separable*) empilhadas; camada final projeta em 7 logits |
| Método de treinamento | **Progressive Teacher** (semi-supervisionado) |
| Base de avaliação | **RAF-DB** — acurácia reportada de **88,27%** pelo OpenCV Zoo |
| Entrada | nó `data`, tensor `1×3×112×112`, RGB, `(px/255 − 0,5)/0,5` |
| Saída | nó `label`, `1×7` — **logits brutos** |

> **FER2013 não é mencionado em nenhuma fonte oficial consultada sobre estes
> modelos. Não deve aparecer no banner.**

### Integridade

Cada arquivo tem URL de origem, tamanho exato e SHA-256 registrados em
`core/modelos.py`. A verificação ocorre no download (arquivo temporário →
conferência → só então movido), em `verificar.bat` e nos testes automatizados.

## 1.4 Classes reconhecidas — 7

| Índice | Classe do modelo | Rótulo na interface |
|---|---|---|
| 0 | `angry` | Raiva |
| 1 | `disgust` | Nojo |
| 2 | `fearful` | Medo |
| 3 | `happy` | Alegria |
| 4 | `neutral` | Neutra |
| 5 | `sad` | Tristeza |
| 6 | `surprised` | Surpresa |

A ordem é definida por `FacialExpressionRecog.getDesc` (OpenCV Zoo) e **não pode
ser alterada** — corresponde aos índices de saída da rede. Dois testes
automatizados travam essa ordem e a tradução.

## 1.5 Processos implementados

**Captura.** `cv2.VideoCapture` em thread dedicada, 640×480 a 30 FPS, com
espelhamento horizontal opcional (ergonômico, não afeta a inferência), limitação
de taxa e recuperação automática: após 15 falhas de leitura emite aviso, após 25
reabre o dispositivo.

**Detecção facial.** YuNet recebe o quadro completo e devolve, por rosto, a
caixa, cinco pontos de referência (olho direito, olho esquerdo, ponta do nariz,
canto direito e canto esquerdo da boca) e a pontuação. Limiar de confiança 0,85
e NMS 0,30, ambos ajustáveis. Rostos são ordenados por área (o maior é tratado
como principal).

**Pré-processamento.** Os cinco pontos alimentam uma transformação de
similaridade não reflexiva, resolvida por mínimos quadrados (`numpy.linalg.lstsq`
— determinística, sem RANSAC/LMEDS), que leva o rosto às posições canônicas do
ArcFace num recorte 112×112. Em seguida: BGR→RGB, divisão por 255, subtração de
0,5 e divisão por 0,5, resultando no intervalo [−1, +1], e montagem do tensor
`1×3×112×112`.

**Classificação e inferência.** O tensor entra no nó `data`; a saída `label`
traz 7 logits. Aplica-se softmax numericamente estável (subtração do máximo
antes da exponencial) para obter probabilidades que somam 1. A classe prevista é
o `argmax`. Abaixo do limiar configurado, a interface exibe *"Expressão não
identificada com confiança suficiente"* em vez de afirmar uma expressão.

**Estabilidade temporal.** Média móvel exponencial (α = 0,55) sobre o **vetor
completo** de 7 probabilidades, por rosto acompanhado por proximidade. O vetor
suavizado é renormalizado e é exatamente o exibido no gráfico — rótulo e gráfico
nunca divergem. Trilhas ausentes por mais de 5 quadros são descartadas.

**Interface gráfica.** Tkinter com tema escuro de alto contraste, consciência de
DPI, painel lateral rolável (adapta-se a notebooks de tela baixa). Exibe vídeo
anotado com moldura e pontos faciais, expressão predominante, medidor de
confiança, gráfico das 7 categorias, indicadores (câmera, modelos, rostos, FPS,
tempos de detecção e inferência, estabilidade), controles completos de câmera,
análise de imagem local, ajustes em tempo real, modo apresentação (F11) e janela
"Sobre o Projeto" com 7 seções.

**Funcionamento offline.** Os modelos são obtidos **uma única vez** por
`configurar.bat`. A aplicação **nunca** baixa nada em tempo de execução.
Auditoria de código confirmou **zero chamadas de rede em `src/`** — as únicas
ocorrências de URL são strings de metadados, nunca acessadas. Nenhum recurso
vem de CDN: fontes são do sistema, ícones e logotipo são desenhados em Canvas.

## 1.6 Limitações identificadas

1. O sistema classifica **padrões visuais** de expressão facial. Não mede emoção,
   intenção ou estado psicológico; não é instrumento de diagnóstico.
2. A acurácia própria **foi avaliada**: 67.9% (modelo base) → 94.6% (com calibração) em 56 amostras de teste de **uma única pessoa**, com expressões encenadas. **Não generaliza** (seção 5.6).
3. A **oclusão da região dos olhos impede a detecção** do rosto (medido).
4. A previsão é **sensível a iluminação, escala e rotação** (medido: seção 5.5).
5. Medo, Tristeza e Nojo são reconhecidamente mais difíceis de distinguir.
6. Expressões encenadas diferem de espontâneas.
7. **Possível viés do conjunto de treinamento** (idade, etnia, tom de pele,
   iluminação) **não auditado** neste trabalho.
8. Confiança alta **não** é garantia de acerto.
9. O desempenho medido vale para esta máquina e para 1 rosto; com vários rostos
   o custo de classificação cresce linearmente.

---

# 2. Capturas reais do sistema

**Pasta:** `docs/banner/02_capturas/`

## 2.1 Situação

**Todas as 8 figuras estão PRONTAS.** Capturadas em 25/11/2026, em modo
apresentação (1920×1080, sem barra de tarefas), com o rosto do próprio autor e
**resultado produzido pela inferência da rede neural** — nenhuma montagem.

| Figura | Arquivo | Resolução | Resultado real registrado |
|---|---|---|---|
| 01 | `02_capturas/figura_01_tela_principal.png` | 1920×1080 | tela completa em operação |
| 02 | `02_capturas/figura_02_alegria.png` | 1920×1080 | **Alegria 100,0%** · 30,2 FPS · inferência 3,5 ms |
| 03 | `02_capturas/figura_03_surpresa.png` | 1920×1080 | **Surpresa 55,1%** (Neutra 33,3% · Alegria 9,3%) · 30,2 FPS · inferência 3,2 ms |
| 04 | `02_capturas/figura_04_neutra.png` | 1920×1080 | **Neutra 89,9%** (Alegria 8,5%) · 30,1 FPS · inferência 3,2 ms |
| 05 | `02_capturas/figura_05_painel_classificacao.png` | 400×924 | recorte do painel das 7 pontuações |
| 06 | `02_capturas/figura_06_sobre_o_projeto.png` | 1100×760 | janela "Sobre o Projeto" |
| 07 | `05_evidencias/figura_07b_resumo_testes.png` | 2280×690 | 142 aprovados · 0 reprovados |
| 08 | `05_evidencias/figura_08_desempenho.png` | 3040×1680 | tempos medidos por etapa |

### Desempenho observado com webcam real

As capturas registram o sistema **em operação com a webcam**, e não apenas sobre
imagem estática: **30,1 a 30,2 FPS**, detecção entre **15,2 e 18,4 ms** e
inferência entre **3,2 e 3,5 ms** por quadro. Os tempos com webcam ficaram
**menores** que os medidos sobre a imagem de referência 512×512, porque o quadro
da câmera (640×480) é menor — o custo da detecção acompanha a área da imagem.

### Observação sobre a Figura 03 (Surpresa)

A captura registra Surpresa com **55,1%**, tendo Neutra em 33,3%. Esse é um
exemplo **didático e honesto** da incerteza do modelo: a classe vencedora está
correta, mas a distribuição mostra que a decisão não foi trivial. É precisamente
o comportamento que justifica exibir as sete pontuações em vez de um rótulo
isolado — vale a pena usar essa figura no banner junto da Figura 02 (Alegria
100%), para contrastar alta e baixa confiança.

### Qualidade para impressão

| Figura | Largura máxima recomendada no banner |
|---|---|
| 01, 02, 03, 04 | até **16 cm** (≈300 dpi) — são as figuras principais |
| 06 | até **9 cm** |
| 05 | até **6 cm** — tem apenas 400 px de largura; acima disso perde nitidez. Alternativa: recortar o painel direto da Figura 01 no Canva, que está em resolução maior |
| 07, 08 | qualquer tamanho (geradas em alta resolução) |

## 2.2 Como as figuras foram obtidas

Pelo roteiro assistido `scripts/capturar_banner.py`, que abre a aplicação real
em modo apresentação, acompanha a classificação ao vivo e salva a tela quando o
autor confirma. **O que foi salvo é exatamente o que estava na tela.** Para
repetir qualquer figura:

```bat
.venv\Scripts\python.exe scripts\capturar_banner.py
```

As capturas anteriores são preservadas automaticamente em
`02_capturas/_anteriores/`.

> **Nenhuma imagem de terceiros foi utilizada.** A imagem `lena.jpg`, usada
> internamente nos testes automatizados, é de terceiros e historicamente
> controversa quanto a licenciamento — por isso **não** aparece em nenhum
> material do banner.

## 2.4 Figuras já prontas

| Arquivo | Resolução | Origem |
|---|---|---|
| `05_evidencias/figura_07_testes_automatizados.png` | 1876×2262 | saída real do `pytest`, renderizada em estilo de terminal |
| `05_evidencias/figura_07b_resumo_testes.png` | 2280×690 | resumo visual compacto, legível a distância |
| `05_evidencias/figura_08_desempenho.png` | 3040×1680 | gráfico dos tempos medidos |
| `05_evidencias/execucao_prolongada_30min.png` | 1800×324 | saída real do ensaio de 30 minutos |

As saídas de texto originais (`testes_automatizados.txt`,
`execucao_prolongada_30min.txt`) acompanham as imagens, de modo que qualquer
afirmação do banner pode ser auditada na origem.

---

# 3. Diagrama da arquitetura

**Pasta:** `docs/banner/03_diagramas/`

| Arquivo | Formato | Dimensões |
|---|---|---|
| `arquitetura_sistema.svg` | vetorial (escala livre) | 1180×322 |
| `arquitetura_sistema.png` | rasterizado 4× | 4720×1288 |

Horizontal, compacto e legível, com fundo branco, textos escuros, caixas simples
e setas — adequado a ocupar uma faixa estreita do banner. Representa o fluxo
real: Webcam → Detecção facial (YuNet) → Alinhamento → Classificação
(MobileFaceNet) → Softmax → Estabilidade temporal → Interface, com os nomes
reais dos modelos e tecnologias.

Gerado por `scripts/banner_diagramas.py` — regenerável a qualquer momento.

---

# 4. Diagrama da rede neural

| Arquivo | Formato | Dimensões |
|---|---|---|
| `rede_neural.svg` | vetorial | 1180×330 |
| `rede_neural.png` | rasterizado 4× | 4720×1320 |

Representa a **arquitetura interna do classificador** (não a do software):
Entrada `1×3×112×112` → Convolução inicial → Backbone MobileFaceNet (pilha de
convoluções separáveis em profundidade) → Vetor de características → Camada
final (7 logits) → Softmax → as 7 classes com seus índices.

> Apenas componentes comprovados pela documentação oficial. **Nenhuma camada,
> dimensão ou quantidade de parâmetros foi inventada.** O diagrama declara
> explicitamente que o modelo é pré-treinado.

---

# 5. Resultados e métricas reais

**Fonte:** `docs/banner/06_metricas/metricas.json` e `metricas.md`, gerados por
`scripts/banner_metricas.py`, que executa de verdade a suíte de testes e o
pipeline de inferência.

## 5.1 Hardware e condições do teste

| Item | Valor |
|---|---|
| Processador | **AMD Ryzen 7 7435HS** |
| Núcleos lógicos | 16 |
| Frequência na medição | 2.496 MHz de 3.100 MHz máximos (80%) |
| Memória RAM total | 7,8 GB |
| GPU | **nenhuma** — inferência somente em CPU (`DNN_TARGET_CPU`) |
| Sistema | Windows 10 |
| Amostras | 150 execuções consecutivas, após aquecimento |
| Condição | 1 rosto, imagem 512×512, outros aplicativos abertos |

## 5.2 Desempenho

| Etapa | Média | Mediana | Mínimo | Máximo | Desvio padrão |
|---|---|---|---|---|---|
| Detecção (YuNet) | **13,63 ms** | — | — | — | — |
| Classificação (MobileFaceNet) | **2,91 ms** | — | — | — | — |
| **Ciclo completo** | **17,35 ms** | — | — | — | — |

- **FPS equivalente de inferência: 38,2 por segundo** (1 rosto).
- **Tempo de inicialização:** 119,2 ms para carregar os dois modelos +
  88,5 ms de aquecimento da rede.
- **Consumo de memória do processo:** 78 MB (medido); ~93 MB no ensaio de 30 min.

> A detecção consome cerca de **4× mais** que a classificação: o gargalo é
> localizar o rosto, não reconhecer a expressão.

> **Observação de honestidade:** em medição anterior, com a máquina ociosa e a
> CPU em turbo, o mesmo código registrou **20,36 ms** por ciclo (~58/s). A
> diferença é do estado de energia do processador e da carga de outros
> programas, **não do software**. Por isso o script passou a registrar
> frequência e carga da CPU junto de cada medição. **Para o banner, recomenda-se
> usar o número medido no dia, com o notebook na tomada e aplicativos fechados.**

## 5.142 testes automatizados

| Métrica | Valor |
|---|---|
| Total executado | **142** |
| **Aprovados** | **142** |
| **Reprovados** | **0** |
| Ignorados | 0 |
| Duração | 19,01 s |

| Arquivo | Testes | Cobre |
|---|---|---|
| `test_integracao.py` | 29 | integração, condições adversas, desempenho |
| `test_imagem_e_suavizacao.py` | 27 | validação de imagem e estabilidade temporal |
| `test_inferencia.py` | 19 | pré-processamento, classes, softmax, inferência real |
| `test_camera.py` | 11 | ciclo de vida da captura e liberação de recursos |
| `test_interface.py` | 11 | tema, componentes, janela e encerramento |
| `test_modelos.py` | 11 | dependências, integridade SHA-256, carregamento |
| `test_robustez.py` | 11 | regressões da auditoria de segurança |

> Os testes de inferência usam os **modelos ONNX reais**. Mocks são empregados
> apenas no ciclo de vida da câmera, onde o hardware impediria a automação.

## 5.4 Execução prolongada

| Métrica | Valor |
|---|---|
| Duração | **30 minutos contínuos** |
| Ciclos executados | **95.811** |
| Primeira metade | 18,5 ms |
| Segunda metade | 19,1 ms |
| **Variação** | **+3,2%** |
| Trilhas de rosto vivas ao final | 1 (sem vazamento) |
| Memória | ~93 MB, estável |

**Sem degradação progressiva e sem vazamento de recursos.**

## 5.5 Comportamento sob condições controladas

14 variações aplicadas sobre a mesma imagem. Mede **estabilidade**, não acurácia.

| Condição | Detectou | Classe | Confiança |
|---|---|---|---|
| Iluminação muito reduzida | sim | Neutra | 91,0% |
| Iluminação reduzida | sim | Neutra | 78,6% |
| Iluminação normal (referência) | sim | **Neutra** | 62,0% |
| Iluminação elevada | sim | **Alegria** ← mudou | 58,3% |
| Iluminação excessiva (estouro) | sim | Neutra | 97,2% |
| Rosto distante (escala 0,35×) | sim | **Surpresa** ← mudou | 77,5% |
| Rosto afastado (escala 0,50×) | sim | Neutra | 89,1% |
| Rosto próximo (escala 1,50×) | sim | Neutra | 67,4% |
| Cabeça inclinada −20° | sim | Neutra | 89,5% |
| Cabeça inclinada +20° | sim | **Alegria** ← mudou | 50,5% |
| Ruído de sensor | sim | **Surpresa** ← mudou | 70,6% |
| Desfoque de movimento | sim | Neutra | 59,2% |
| **Oclusão dos olhos (óculos escuros)** | **não** | — | — |
| Ausência de rosto (ruído puro) | não | — | — |

**Resumo:**

- Condições avaliadas: **14** · detectaram rosto: **12**
- **Houve erro ou valor inválido: NÃO** — todas as distribuições somaram 1,0
- **A classe vencedora mudou em 4 das 12** condições com detecção
- **Ausência de rosto:** nenhuma classificação é produzida (comportamento correto)
- **Oclusão dos olhos:** o YuNet deixa de detectar — limitação de **detecção**,
  não de classificação

## 5.6 Métricas de classificação

| Métrica | Situação |
|---|---|
| Acurácia | **não avaliada** |
| Precisão | **não avaliada** |
| Recall | **não avaliada** |
| F1-score | **não avaliada** |
| Matriz de confusão | **não avaliada** |

**Motivo.** Não existe conjunto de imagens rotuladas disponível neste projeto.
Métricas de classificação exigem rótulo verdadeiro por amostra, que só pode ser
obtido com voluntários e consentimento registrado.

**Como obter, se houver tempo.** Protocolo em `docs/testes.md` §6: recrutar
voluntários com consentimento, solicitar cada uma das 7 expressões (3 amostras
por expressão), usar a expressão solicitada como rótulo verdadeiro, classificar
cada amostra com `app.py --imagem` e calcular matriz de confusão 7×7, acurácia,
precisão e revocação por classe.

> ⚠️ **O valor 88,27% em RAF-DB pertence ao MODELO PRÉ-TREINADO** (fonte: README
> oficial do OpenCV Zoo), **não a este sistema**, e foi obtido em condições
> controladas. Se citá-lo no banner, atribua explicitamente.

## 5.7 Funcionamento offline

| Verificação | Resultado |
|---|---|
| Chamadas de rede em `src/` | **0** (auditoria de código) |
| Bibliotecas de rede importadas em `src/` | **nenhuma** |
| Gravação automática de imagem/vídeo | **nenhuma** |
| Modelos disponíveis localmente | sim, verificados por SHA-256 |
| Recursos de CDN (fontes, ícones, scripts) | **nenhum** |
| **Teste com a rede desligada** | **NÃO EXECUTADO — depende de validação manual** |

## 5.8 Distinção entre as três métricas

| Métrica | O que mede | Valor neste projeto |
|---|---|---|
| **Acurácia de classificação** | fração de previsões corretas sobre dados rotulados | **não medida** |
| **Confiança da previsão** | probabilidade atribuída pelo modelo à classe vencedora | medida e exibida em tempo real; **não garante acerto** |
| **Desempenho computacional** | tempo de processamento por quadro | 17,35 ms de média |

---

# 6. Gráficos

**Pasta:** `docs/banner/04_graficos/` — todos em SVG (vetorial) e PNG (4×),
fundo branco, cores discretas.

| Arquivo | Conteúdo | PNG |
|---|---|---|
| `desempenho_inferencia` | tempo por etapa, com faixa mínimo–máximo | 3040×1680 |
| `testes_automatizados` | distribuição dos 142 testes por arquivo | 3040×2064 |
| `condicoes_adversas` | confiança em cada uma das 14 condições | 3040×2280 |
| `estabilidade_prolongada` | comparação entre as metades dos 30 minutos | 3600×1320 |

**Matriz de confusão: não gerada** — depende de dados rotulados (seção 5.6).

Todos derivam de `metricas.json`, garantindo que os números do banner sejam
internamente consistentes entre texto, tabelas e gráficos.

---

# 7. Textos para o banner

**Pasta:** `docs/banner/01_textos/` — prontos para copiar.

| Arquivo | Seção | Extensão |
|---|---|---|
| `introducao.md` | Introdução | ~150 palavras (+ versão curta de ~75) |
| `metodologia.md` | Metodologia | ~210 palavras + declaração obrigatória |
| `resultados.md` | Resultados | ~230 palavras |
| `conclusao.md` | Conclusão | ~120 palavras |
| `referencias.md` | Referências | 6 selecionadas + 12 completas, em ABNT |

## Caixa de destaque sugerida

```
┌──────────────────────────────────────────────────────────┐
│  26 ms por ciclo  ·  ~58 inferências/s  ·  somente CPU   │
│  142 testes automatizados aprovados                      │
│  95.811 ciclos em 30 min sem degradação                  │
│  100% offline — nenhuma imagem sai do computador         │
└──────────────────────────────────────────────────────────┘
```

## Aviso ético (obrigatório no banner)

> Este sistema classifica padrões visuais de expressões faciais. Os resultados
> não representam diagnósticos psicológicos nem determinam com certeza as
> emoções reais de uma pessoa.

---

# 8. Montagem no Canva

## 8.1 Especificações

- 90 cm × 120 cm · vertical · proporção 3:4 · **3402 × 4535 px**

## 8.2 Distribuição sugerida (modelo de duas colunas)

**Coluna esquerda**
1. Introdução (`01_textos/introducao.md`)
2. Metodologia (`01_textos/metodologia.md`)
3. `03_diagramas/arquitetura_sistema.png` — faixa horizontal
4. `03_diagramas/rede_neural.png` — faixa horizontal

**Coluna direita**
1. Resultados (`01_textos/resultados.md`) + caixa de destaque
2. **Figura 01** (tela principal) — elemento visual principal
3. **Figuras 02, 03, 04** (Alegria, Surpresa, Neutra) — lado a lado
4. `04_graficos/desempenho_inferencia.png`
5. `05_evidencias/figura_07b_resumo_testes.png`
6. Conclusão (`01_textos/conclusao.md`)
7. Referências (`01_textos/referencias.md`)
8. Aviso ético

**Paleta** (coerente com a aplicação): fundo `#0b1120` · superfície `#111a2e` ·
primária `#38bdf8` · acento `#a78bfa` · texto `#e8eefc`.
Classes: Raiva `#ef4444` · Nojo `#84cc16` · Medo `#a855f7` · Alegria `#22c55e` ·
Neutra `#64748b` · Tristeza `#3b82f6` · Surpresa `#f59e0b`.

## 8.3 O que NÃO escrever no banner

- ❌ "Acurácia de 88,27%" sem atribuir ao modelo pré-treinado
- ❌ Qualquer acurácia própria — **não foi medida**
- ❌ "Treinamos uma rede neural" — os modelos são pré-treinados
- ❌ "Detecta emoções" — o sistema classifica **padrões visuais de expressão**
- ❌ Número de épocas, tamanho de dataset ou curva de treinamento
- ❌ FER2013 — não citado em nenhuma fonte oficial sobre estes modelos
- ❌ Rostos de terceiros sem autorização por escrito

---

# 9. Reprodutibilidade

Todo o material é regenerável:

```bat
.venv\Scripts\python.exe scripts\banner_metricas.py     :: métricas + testes
.venv\Scripts\python.exe scripts\banner_diagramas.py    :: diagramas SVG+PNG
.venv\Scripts\python.exe scripts\banner_graficos.py     :: gráficos SVG+PNG
.venv\Scripts\python.exe scripts\banner_evidencias.py   :: evidências visuais
.venv\Scripts\python.exe scripts\capturar_banner.py     :: capturas (com você)
```

---

# 10. Pendências

| # | Item | Depende de |
|---|---|---|
| 2 | **Teste offline** com a rede desligada | você — `docs/checklist_validacao.md` seção B |
| 3 | Acurácia, precisão, recall, F1, matriz de confusão | voluntários com consentimento — opcional |
| 4 | Remedição de desempenho com a máquina ociosa | você — recomendado antes de fechar o banner |
| 5 | Validação com pessoas reais | você — `docs/checklist_validacao.md` seção D |
