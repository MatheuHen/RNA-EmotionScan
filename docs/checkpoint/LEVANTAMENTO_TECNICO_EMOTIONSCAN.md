# Levantamento Técnico — EmotionScan

**Atividade:** Checkpoint de Desenvolvimento — Descritivo Técnico e Arquitetura da Solução
**Disciplina:** Redes Neurais Artificiais (RNA) — UniRV
**Professor:** Sandro Silva Moreira
**Documento de coleta — NÃO é o entregável final.** Elaborado para conferência antes da produção do diagrama (Draw.io) e do PDF final.

Gerado em: 2026-11-26, a partir de leitura direta do código-fonte, métricas medidas, testes executados agora e histórico Git real do repositório. Nenhum número foi copiado de resumos anteriores sem reverificação.

---

## 1. Identificação do projeto

| Campo | Valor | Status |
|---|---|---|
| Título oficial | EmotionScan: Reconhecimento de Expressões Faciais utilizando Redes Neurais | Comprovado (`README.md`, `src/expressao_neural/config.py:22`) |
| Nome interno da aplicação | `EmotionScan` (`config.APP_NOME`) — renomeado de "Expressão Neural" (pacote Python ainda chamado `expressao_neural`, só o nome interno do pacote) | Comprovado |
| Versão atual | 1.3 (`config.APP_VERSAO`) | Comprovado |
| Trilha tecnológica | Visão computacional / redes neurais pré-treinadas com camada de calibração supervisionada própria, aplicação desktop em Python | Comprovado |
| Acadêmico | Matheus Henrique de Araújo Carneiro | Comprovado (`config.py:25`, `README.md`) |
| Curso | Engenharia de Software — UniRV | Comprovado |
| Responsabilidades do acadêmico | Projeto, implementação integral da aplicação (captura, pipeline, interface), coleta das 210 amostras próprias e treinamento da camada de calibração, documentação e evidências | Comprovado pelo histórico de commits (autor único, 34 commits) |
| Repositório remoto | **https://github.com/MatheuHen/RNA-EmotionScan.git** | Comprovado — `git remote -v` confirma origin configurado; `git push -u origin main` executado com sucesso nesta sessão |
| Branch principal | `main` | Comprovado |
| Situação do versionamento | 34 commits, working tree limpo (`git status` sem pendências), tudo publicado no remoto | Comprovado |

---

## 2. Descritivo técnico da solução

### 2.1 Problema e público-alvo

- **Problema abordado:** demonstrar, de forma interativa e tangível, como duas redes neurais convolucionais pré-treinadas podem ser combinadas (detecção facial + classificação de expressões) e como uma camada de calibração supervisionada simples pode ajustar a saída de um modelo congelado a um contexto específico, sem retrabalhar a rede.
- **Finalidade prática:** peça de demonstração para o workshop acadêmico, permitindo que o visitante veja o próprio rosto sendo processado em tempo real.
- **Público-alvo:** público do 5º Workshop de Trabalhos de IA da FASOFT/UniRV — professores, colegas e visitantes do evento, com conhecimento técnico variável.
- **Interação no workshop:** o visitante se posiciona diante da webcam; a interface exibe a imagem anotada (caixa facial, 5 pontos de referência), o rótulo da expressão predominante, a confiança e o gráfico das 7 probabilidades, atualizados continuamente.
- **Limite explícito (respeitado na documentação):** o sistema **não** faz diagnóstico psicológico nem afirma reconhecer a emoção real da pessoa — classifica padrões visuais de expressão facial. Essa frase está registrada em `config.AVISO_PRIVACIDADE` (`config.py:32-36`) e é exibida na interface (aba "Sobre").

### 2.2 Funcionamento e fluxo de uso

1. Usuário executa `iniciar.bat` (ou `python app.py`) — não exige rede.
2. A janela principal (`ui/janela.py`, classe `JanelaPrincipal`) abre; nenhuma câmera é acionada automaticamente até o botão "Iniciar câmera" ser pressionado.
3. Ao iniciar a câmera, `core/camera.py` (`TrabalhadorCamera`) abre o dispositivo via `cv2.VideoCapture` (tenta `CAP_DSHOW`, depois `CAP_MSMF`, depois `CAP_ANY`).
4. Uma thread de **captura** lê quadros continuamente e os publica de imediato na interface.
5. Uma thread separada de **análise** consome o quadro mais recente (nunca uma fila — sempre o último) e executa o pipeline de inferência.
6. O pipeline (`core/pipeline.py`, `PipelineExpressao.processar`) executa: detecção facial (YuNet) → alinhamento 112×112 (`core/alinhamento.py`) → classificação (MobileFaceNet) → softmax → suavização temporal (`core/suavizacao.py`).
7. O resultado mais recente é combinado ao quadro publicado e desenhado na interface (`ui/desenho.py`): caixa delimitadora, rótulo, confiança, gráfico de barras das 7 classes.
8. Entradas aceitas além da webcam: **imagem estática** (`core/imagem_estatica.py`), para quando a câmera falhar ou como contingência.
9. Configurações disponíveis na interface: limiar de confiança, limiar de detecção, suavização (ativa/alfa), espelhamento, múltiplos rostos, índice de câmera, passo de inferência (`inferir_a_cada`).
10. Funcionalidades adicionais implementadas e confirmadas em código: pausar/retomar câmera, trocar câmera, reiniciar câmera, modo tela cheia (apresentação), janela "Sobre o projeto", ícone próprio na barra de tarefas (`ui/icone.py`).

### 2.3 Stack tecnológica

| Tecnologia | Versão | Função | Fase |
|---|---|---|---|
| Python | 3.11.9 (testado; compatível 3.9–3.12 conforme `README.md`) | Linguagem da aplicação | Desenvolvimento/execução |
| OpenCV (`opencv-python`) | 4.12.0.88 | Captura de vídeo, `cv2.dnn` (motor de inferência ONNX), `cv2.FaceDetectorYN` | Execução |
| NumPy | 2.2.6 | Operações vetoriais (softmax, suavização, matrizes de calibração) | Execução |
| Pillow | 11.3.0 | Conversão de quadro para exibição no Tkinter | Execução |
| Tkinter | biblioteca padrão do Python (sem versão própria) | Interface gráfica desktop | Execução |
| pytest | 8.4.2 | Suíte automatizada | Testes |
| ONNX (formato) | — | Formato de arquivo dos dois modelos pré-treinados | Execução (não há biblioteca `onnx` nem `onnxruntime` instalada — a leitura é feita pelo `cv2.dnn`) |

Confirmado em `requirements.txt`: **nenhum** framework de deep learning (TensorFlow, PyTorch, onnxruntime) é dependência de execução. O comentário no próprio arquivo registra essa decisão explicitamente.

Treinamento da camada de calibração: usa apenas NumPy (gradiente descendente implementado manualmente — não há framework de treino como scikit-learn, TensorFlow ou PyTorch envolvido).

### 2.4 Modelos e arquiteturas neurais

Duas redes neurais convolucionais pré-treinadas, **congeladas** (pesos nunca alterados):

| | Detector — YuNet | Classificador — MobileFaceNet / Progressive Teacher |
|---|---|---|
| Arquivo | `models/face_detection_yunet_2023mar.onnx` | `models/facial_expression_recognition_mobilefacenet_2022july.onnx` |
| Tamanho | 232.589 bytes | 4.791.892 bytes |
| SHA-256 | verificado em `configurar.bat`/`test_modelos.py` | idem |
| Origem | OpenCV Zoo (Hugging Face: `opencv/face_detection_yunet`) | OpenCV Zoo (Hugging Face: `opencv/facial_expression_recognition`) |
| Licença | Apache License 2.0 | Apache License 2.0 |
| Função | Detecção facial + 5 pontos de referência (olhos, nariz, cantos da boca) | Classificação de 7 expressões a partir do rosto alinhado |
| Entrada | Imagem BGR, tamanho variável | Recorte alinhado 112×112 (`config.FER_TAMANHO_ENTRADA`), normalizado com média/desvio 0,5/0,5 |
| Saída | Caixas + pontos + confiança, pós-NMS | 7 **logits brutos** (`config.FER_SAIDA_E_LOGIT = True`), nome de saída `"label"` |
| Categorias | — (não classifica expressão) | `angry, disgust, fearful, happy, neutral, sad, surprised` (ordem fixa, fonte: `opencv_zoo/.../facial_fer_model.py`) |
| Pré-treinado | Sim | Sim |
| Fine-tuning | Não | Não |

**Componente treinado pelo próprio autor — NÃO é uma rede neural nova:**

- **Camada de calibração**: uma transformação linear sobre os 7 logits do classificador congelado — matriz `W` 7×7 e vetor de bias `b` com 7 elementos, **56 parâmetros treináveis** no total.
- Inicializada como identidade (`W = I`, `b = 0`), de modo que sem treino ela reproduz exatamente a saída do modelo base.
- Implementada em `src/expressao_neural/core/calibracao.py`, carregada de `models/calibracao_pessoal.npz`.
- **Desligada por padrão** (`Ajustes.usar_calibracao: bool = False`, `config.py:129`) — o sistema entrega a saída do modelo pré-treinado a menos que o usuário ative a calibração conscientemente na interface.

**Softmax:** aplicado em `core/classificador.py` (função `softmax`, linha 26), *depois* dos logits brutos (do modelo base ou, se ativa, depois da camada de calibração) — ver `inferir_recorte`, linha 129: `probs = softmax(logits) if config.FER_SAIDA_E_LOGIT else logits`. Nenhum dos dois modelos ONNX contém um nó Softmax no próprio grafo; a operação é feita pela aplicação.

**Estabilidade temporal:** `core/suavizacao.py` (`SuavizadorTemporal`) aplica média exponencial sobre as probabilidades por rosto rastreado (`suavizacao_alfa = 0.55` por padrão), evitando oscilação quadro a quadro. É suavização estatística, não é rede neural nem parte do treinamento.

### 2.5 Bases de dados e treinamento

**Dataset de origem dos modelos pré-treinados (YuNet / MobileFaceNet):** não está documentado no repositório nem em `metricas.json`. A única fonte de origem registrada é o OpenCV Zoo (licença e arquitetura), sem menção ao dataset de treinamento original. **Não deve ser citado FER2013** sem confirmação em fonte oficial — essa confirmação não foi encontrada. O número **88,27%** registrado em `metricas.json` (`metricas_de_classificacao.acuracia_do_modelo_original`) refere-se ao desempenho do modelo pré-treinado no dataset **RAF-DB**, conforme o README oficial do OpenCV Zoo — não é uma métrica deste sistema.

**Treinamento próprio (camada de calibração)** — fonte: `docs/banner/06_metricas/treinamento_calibracao.json`, gerado em 2026-11-26T08:32:21:

| Item | Valor da auditoria anterior | Valor confirmado agora | Situação |
|---|---|---|---|
| Total de amostras | 210 | **210** (30 por classe × 7) | Confirmado, sem alteração |
| Classes | 7 | **7** | Confirmado |
| Amostras de treino | 154 | **154** | Confirmado |
| Amostras de teste | 56 | **56** | Confirmado |
| Divisão | — | estratificada por classe, semente fixa (20261130) | Confirmado |
| Épocas | 400 | **400** | Confirmado |
| Parâmetros treináveis | 56 | **56** (W 7×7 + b 7) | Confirmado |
| Acurácia antes da calibração | 67,86% | **67,86%** (38/56 acertos) | Confirmado |
| Acurácia após a calibração | 94,64% | **94,64%** (53/56 acertos) | Confirmado |

Nenhuma divergência encontrada nesses números — a auditoria anterior está correta e atualizada.

**Métricas complementares do treinamento (não estavam na auditoria anterior, disponíveis agora):**
- Precisão macro: 0,9459 · Recall macro: 0,9464 · F1 macro: 0,9452
- Ganho: +26,78 pontos percentuais
- Taxa de aprendizado: 0,05 · regularização L2 em direção à identidade: 0,001 · otimizador: gradiente descendente em lote completo
- Perda inicial 2,032 → perda final 0,209

**Limitação documentada e que deve ser preservada em qualquer texto novo:** o conjunto de teste tem 56 amostras de **uma única pessoa** (o próprio autor), com expressões **encenadas**. A acurácia de 94,64% mede ajuste a esse contexto específico e **não comprova generalização** para o público do workshop. Essa frase já está em `metricas.json.metricas_de_classificacao.limitacao` e deve ser repetida sempre que o número for citado.

**Relação treino → inferência em tempo real:** o treinamento é **totalmente offline** — roda uma vez, gera `models/calibracao_pessoal.npz`, e esse arquivo é lido (não recalculado) a cada execução da aplicação, apenas se `usar_calibracao = True`.

---

## 3. Levantamento para o diagrama de arquitetura (4 camadas)

### Camada 1 — Interface e entrada

- Interface gráfica desktop: **Tkinter** (`ui/janela.py`, classe `JanelaPrincipal`).
- Entrada de vídeo: **webcam** via `cv2.VideoCapture` (`core/camera.py`).
- Entrada alternativa: **imagem estática** (upload/seleção de arquivo), `core/imagem_estatica.py`, para contingência.
- Controles expostos na interface: limiar de confiança, limiar de detecção/NMS, suavização (on/off e alfa), espelhamento, múltiplos rostos, seleção/troca de câmera, passo de inferência, pausar/retomar/reiniciar, modo tela cheia.
- Formato de dados que entra no processamento: quadro de vídeo BGR (NumPy array `uint8`), resolução padrão 640×480.

### Camada 2 — Orquestração e processamento

- **Captura** (`core/camera.py`, classe `TrabalhadorCamera`): thread dedicada que lê a câmera e publica o quadro mais recente num slot único (nunca uma fila, para não acumular memória).
- **Análise** (mesma classe, thread separada `_laco_de_analise`): consome o quadro mais recente e dispara o pipeline de inferência, sem bloquear a captura.
- **Pipeline** (`core/pipeline.py`, classe `PipelineExpressao`): orquestra detecção → alinhamento → classificação → suavização, protegido por `RLock` porque os modelos OpenCV não são thread-safe.
- **Pré-processamento/alinhamento**: `core/alinhamento.py`, usa os 5 pontos do detector e os pontos de referência ArcFace (`config.PONTOS_PADRAO_5`) para recortar e normalizar o rosto em 112×112.
- **Controle de quadros**: `Ajustes.inferir_a_cada` (padrão 2) decide a cada quantos quadros capturados a inferência é disparada.
- **Concorrência**: duas threads (captura e análise), sincronizadas por `threading.Lock`/`Event`, nenhuma delas roda na thread do Tkinter.
- **Tratamento de falhas**: contagem de falhas de leitura consecutivas, reconexão automática da câmera após 25 falhas, mensagens de erro expostas à interface sem detalhes internos.
- **Estabilidade temporal**: `core/suavizacao.py`, média exponencial por rosto rastreado.

### Camada 3 — Inferência e modelos

- **YuNet** (`core/detector.py`, classe `DetectorFacial`) — detecção + 5 pontos.
- **MobileFaceNet/Progressive Teacher** (`core/classificador.py`, classe `ClassificadorExpressao`) — 7 logits.
- **Softmax** — aplicado em `classificador.py` imediatamente após obter os logits (seja do modelo base, seja já ajustados pela calibração).
- **Camada de calibração** (`core/calibracao.py`, classe `Calibracao`) — opcional, aplicada **entre** os logits brutos e o softmax, carregando `models/calibracao_pessoal.npz`.
- **Ordem exata da inferência**: quadro → YuNet (detecção) → alinhamento 112×112 → MobileFaceNet (logits) → [calibração, se ativa] → softmax → suavização temporal.
- **Treinamento offline** (não faz parte da execução normal): script de treino separado lê as 210 amostras, calcula `W`/`b` por gradiente descendente, grava `calibracao_pessoal.npz`. Esse fluxo não roda durante a apresentação nem é acionado pela interface.

### Camada 4 — Saída e ação

- Classificação final (rótulo em português, `config.CLASSES_PT`).
- Pontuação de confiança (0–1, exibida em %).
- Gráfico de barras das 7 probabilidades (`ui/componentes.py`, `GraficoProbabilidades`).
- Caixa delimitadora e pontos faciais desenhados sobre o quadro (`ui/desenho.py`).
- Indicadores de desempenho na interface (FPS, tempos de detecção/classificação).
- Indicador de estabilidade da trilha (quantos quadros consecutivos sustentam a mesma predição).
- Mensagem "Indefinida" quando a confiança fica abaixo do limiar **e** `sempre_exibir_previsao = False` (padrão atual é `True`: a predição é sempre mostrada, sinalizando baixa confiança em vez de escondê-la).

### Tabela de conexões

| Origem | Destino | Dados transmitidos | Finalidade |
|---|---|---|---|
| Webcam (hardware) | `TrabalhadorCamera` (thread de captura) | Quadro BGR bruto | Entrada de vídeo |
| Thread de captura | Thread de análise | Quadro mais recente (ponteiro único) | Desacoplar leitura de inferência |
| Thread de captura | Interface (Tkinter) | Quadro + último resultado disponível | Exibição em tempo real |
| Thread de análise | `PipelineExpressao.processar` | Quadro a classificar | Disparo da inferência |
| `PipelineExpressao` | `DetectorFacial` (YuNet) | Quadro completo | Detecção facial |
| `DetectorFacial` | `core/alinhamento.py` | Caixa + 5 pontos | Recorte e normalização |
| `core/alinhamento.py` | `ClassificadorExpressao` (MobileFaceNet) | Recorte 112×112 normalizado | Classificação da expressão |
| `ClassificadorExpressao` | `Calibracao` (opcional) | 7 logits brutos | Ajuste pessoal, se ativado |
| `Calibracao`/logits | função `softmax` | 7 valores (calibrados ou não) | Converter em probabilidades |
| `softmax` | `SuavizadorTemporal` | Vetor de 7 probabilidades | Estabilização temporal por rosto |
| `SuavizadorTemporal` | `ResultadoQuadro` | Vetor suavizado + metadados | Empacotar resultado |
| `ResultadoQuadro` | Interface (painéis, gráfico, desenho) | Rótulo, confiança, probabilidades, caixa | Exibição ao usuário |
| Amostras próprias (webcam, offline) | Script de treino da calibração | 210 imagens rotuladas | Gerar `W`/`b` |
| Script de treino | `models/calibracao_pessoal.npz` | Matriz 7×7 + vetor 7 | Persistir calibração treinada |
| `calibracao_pessoal.npz` | `Calibracao.carregar()` | Matriz/vetor | Carregar no início da execução, se ativada |

**Execução local:** todos os componentes acima executam localmente, em CPU, sem GPU.
**Chamadas externas em execução normal:** nenhuma (zero ocorrências de `urllib`/`requests`/`socket`/`http` em `src/`, confirmado por auditoria de código e por teste real com a rede desligada).
**Banco de dados:** não existe.
**Armazenamento local:** arquivos `.onnx` (modelos), `.npz` (calibração), `config.local.json` (preferências do usuário, não versionado).
**Dependência de internet:** apenas na instalação (`configurar.bat`, baixa os modelos uma vez). Nenhuma depois disso.
**Arquivos carregados na inicialização:** os dois `.onnx` em `models/`; `calibracao_pessoal.npz` somente se `usar_calibracao = True`; `config.local.json` se existir.
**Componentes usados somente no treinamento:** script de treino da calibração (não listado nos módulos de execução), as 210 imagens próprias (fora do repositório versionado, por política de privacidade).
**Componentes usados na execução normal:** todos os listados nas Camadas 1–4 acima, exceto o script de treino.

### Fluxo sequencial (texto)

```
Webcam → TrabalhadorCamera (captura) → [publica quadro] → Interface (exibição imediata)
                                      ↘
                                        TrabalhadorCamera (análise) → PipelineExpressao
                                                                        → DetectorFacial (YuNet)
                                                                        → alinhamento 112×112
                                                                        → ClassificadorExpressao (MobileFaceNet)
                                                                        → [Calibracao, opcional]
                                                                        → softmax
                                                                        → SuavizadorTemporal
                                                                        → ResultadoQuadro
                                                                        → Interface (painéis, gráfico, caixa)
```

Fluxo de treinamento (offline, separado):
```
210 amostras próprias → extração de logits (modelo congelado) → gradiente descendente (400 épocas)
   → W (7×7) + b (7) → models/calibracao_pessoal.npz
```

---

## 4. Status atual e checklist de maturidade

### 4.1 Implementado e validado

| Item | Evidência |
|---|---|
| Interface gráfica completa | `src/expressao_neural/ui/janela.py`; `tests/test_interface.py` (11 testes) |
| Captura de webcam em duas threads | `src/expressao_neural/core/camera.py`; `tests/test_captura_desacoplada.py` |
| Detecção facial (YuNet) | `src/expressao_neural/core/detector.py`; `tests/test_inferencia.py` |
| Classificação de expressões (MobileFaceNet) | `src/expressao_neural/core/classificador.py`; `tests/test_inferencia.py` |
| Camada de calibração treinada | `models/calibracao_pessoal.npz`; `docs/banner/06_metricas/treinamento_calibracao.json`; `tests/test_calibracao.py` (16 testes) |
| Exibição de previsões, confiança e gráfico | `src/expressao_neural/ui/componentes.py` |
| Indicadores de desempenho na interface | `ui/janela.py` |
| Suíte automatizada | **159 testes aprovados, 1 skipped** (execução real agora, `python -m pytest -q`, 157,55 s) |
| Teste offline com rede desligada | `evidencias/teste_offline.md` — APROVADO, 26/11/2026 16:17:51, 5 alvos de rede confirmados inacessíveis |
| Execução prolongada | `evidencias/execucao_prolongada.log`; `treinamento 30 min`, 95.811 ciclos, degradação de 3,2% |
| Versionamento remoto | `https://github.com/MatheuHen/RNA-EmotionScan.git`, branch `main`, 34 commits, push confirmado |

### 4.2 O que falta para a demonstração interativa

Pendências reais, sem invenção:

- Validação com **pessoas diferentes do autor** não foi feita — a acurácia de 94,64% vale só para o conjunto próprio.
- Teste em **projetor/resolução externa** não foi executado nesta sessão.
- **Ensaio com visitantes reais** (fora do autor) ainda não ocorreu.
- `metricas.json` (`docs/banner/06_metricas/metricas.json`) está com números de desempenho **gerados antes** da correção de threads (14:43, enquanto a correção foi às 15:04) — precisa ser regenerado para refletir a arquitetura final antes de qualquer citação de FPS/ciclo no documento acadêmico. Ver divergência na seção 7.

### 4.3 Gargalos técnicos já resolvidos

| Problema | Impacto | Solução aplicada | Situação |
|---|---|---|---|
| FPS da webcam caindo de ~30 para ~15 | Vídeo visivelmente menos fluido na demonstração | Thread de captura separada da thread de inferência (`core/camera.py`); `cap.read()` bloqueia numa grade fixa de tempo, e inferência na mesma thread fazia perder o próximo ponto da grade | Resolvido — medido 30,2 FPS reais após a correção (commit `8bbe402`) |
| Botão "Reiniciar" deixava o sistema sem imagem permanentemente | Travava a demonstração, exigia reiniciar o processo | `_modo` não era resetado antes de reabrir a câmera; corrigido em `commit b937784` | Resolvido, com teste de regressão (`tests/test_reiniciar_camera.py`, 5 testes) |
| Backup de capturas do banner apagava arquivos anteriores | Perda de imagens já capturadas para o banner acadêmico | Troca de `rename` (mover, destrutivo) por `shutil.copy2` (copiar) em `scripts/capturar_banner.py` | Resolvido (commit `b051420`) |
| Ícone da aplicação na barra de tarefas era o ícone genérico do Python | Identidade visual fraca na apresentação | `AppUserModelID` declarado via `ctypes.windll.shell32`, ícone `.ico` próprio gerado (`ui/icone.py`) | Resolvido (commit `f319d50`) |

---

## 5. Resultados técnicos atualizados

| Métrica | Auditoria anterior | Valor confirmado agora | Fonte | Situação |
|---|---|---|---|---|
| Testes automatizados aprovados | 137 | **159 passed, 1 skipped** | Execução real `pytest -q`, incorporada em `metricas.json` após regeneração | **Mudou** — suíte cresceu; fonte do banner agora sincronizada |
| Detecção (YuNet) | 16,90 ms | 16,97 ms (benchmark isolado, `metricas.json`, 150 amostras) | `docs/banner/06_metricas/metricas.json` | Confirmado, mesma ordem de grandeza |
| Classificação (MobileFaceNet) | 3,70 ms | 3,81 ms | idem | Confirmado, mesma ordem de grandeza |
| Ciclo completo (benchmark isolado) | 21,45 ms | 21,86 ms | idem | Confirmado, mesma ordem de grandeza |
| Inferências/s em benchmark | 46,6 | 45,8 | idem | Confirmado |
| FPS real da webcam (em uso, app completo) | ~30 (não distinguido de benchmark) | **30,2 FPS medidos** com a arquitetura de duas threads (antes da correção: 14,8–25 FPS) | `docs/resultados.md` | Confirmado, agora corretamente separado de benchmark |
| Hardware | — | AMD Ryzen 7 7435HS, 16 núcleos lógicos, 7,8 GB RAM, sem GPU (CPU apenas) | `metricas.json.hardware_de_teste` | Comprovado |
| Treinamento próprio | 67,86% → 94,64% | **67,86% → 94,64%** (confirmado, sem mudança) | `treinamento_calibracao.json` | Confirmado |
| Estabilidade (execução prolongada) | — | 30 min, 95.811 ciclos, degradação 3,2%, aprovado | `evidencias/execucao_prolongada.log`, `metricas.json` | Comprovado |
| Funcionamento offline | não mencionado como testado | **Testado com rede fisicamente desligada, aprovado** (26/11/2026 16:17:51) | `evidencias/teste_offline.md`, refletido em `metricas.json.funcionamento_offline` | Comprovado — divergência da seção 6 corrigida: o campo passou a ler o resultado real do teste em vez de um texto fixo |

**Importante — diferenciação benchmark vs. FPS real:** os valores de "ciclo completo"/"inferências por segundo" em `metricas.json` (21,86 ms / 45,8 por segundo) vêm de um **benchmark isolado**, processando quadros em sequência sem a disputa de CPU entre threads de captura e renderização. O **FPS real da webcam em uso**, medido com a aplicação completa rodando, é 30,2 — número diferente e que depende da arquitetura de duas threads implementada nesta sessão. Qualquer documento acadêmico deve apresentar os dois números separadamente, nomeando a fonte de cada um.

---

## 6. Divergências encontradas — status: **corrigidas**

As três divergências identificadas na primeira versão deste levantamento foram resolvidas depois da coleta inicial:

1. ~~`metricas.json` desatualizado em relação à arquitetura pós-threads~~ — **corrigido**: `scripts/banner_metricas.py` foi executado de novo; `testes_automatizados.total` agora é **159** (igual à execução real de `pytest`).
2. ~~`metricas.json.funcionamento_offline.teste_com_rede_desligada` fixo em "NÃO EXECUTADO"~~ — **corrigido**: o script passou a ler `evidencias/teste_offline.md` dinamicamente (`ler_teste_offline()`, `scripts/banner_metricas.py`). O campo agora registra `"resultado": "APROVADO"` e `"executado_em": "26/11/2026 às 16:17:51"`.
3. ~~Contagem de testes oscilando entre 154 / 158 / 159~~ — **corrigido**: toda a documentação foi ressincronizada com `scripts/sincronizar_numeros.py` (35 ajustes aplicados em 6 arquivos) e `scripts/verificar_consistencia.py` confirma **0 divergências** contra a fonte única de verdade.
4. Nenhuma divergência encontrada entre código e documentação quanto a: arquitetura dos modelos, parâmetros da calibração, dataset de treinamento próprio, licenças, ausência de chamadas de rede.

---

## 7. Resumo de classificação das informações

- **Comprovadas** (verificadas agora, com evidência de arquivo/execução): identificação do projeto, repositório remoto, stack tecnológica, arquitetura dos dois modelos pré-treinados, natureza e parâmetros da camada de calibração, posição do softmax, números de treinamento (210/7/154/56/400/56/67,86%/94,64%), suíte de testes (159 passed, 1 skipped), teste offline aprovado, execução prolongada aprovada, gargalos técnicos resolvidos (FPS, botão Reiniciar, backup destrutivo, ícone).
- **Pendentes de validação**: apenas de ação humana — validação com pessoas além do autor; teste em projetor/resolução externa. A pendência técnica de regeneração de `metricas.json` foi resolvida (ver seção 6).
- **Não implementadas / não aplicáveis a este projeto**: banco de dados, chamadas externas em tempo de execução, GPU, fine-tuning dos modelos pré-treinados, dataset de origem desses modelos (não documentado pela fonte oficial).

---

## Entrega desta coleta

- **Arquivo gerado:** `C:\ProjetosIA\Projetos\ExpressaoNeural\docs\checkpoint\LEVANTAMENTO_TECNICO_EMOTIONSCAN.md`
- **Repositório:** https://github.com/MatheuHen/RNA-EmotionScan.git (branch `main`, confirmado configurado e sincronizado)
- **Percentual estimado de conclusão do levantamento:** **100%** das 4 seções exigidas pelo professor, com dados verificados em código/execução real. As divergências técnicas encontradas (métricas desatualizadas) foram corrigidas na sequência. O que resta é exclusivamente ação humana — ensaio com terceiros e teste em projetor — já listado nas pendências de apresentação, não na coleta de informação.
