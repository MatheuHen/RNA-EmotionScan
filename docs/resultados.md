# Relatório de resultados — Expressão Neural

> Todos os números deste documento foram **medidos em execução real** nesta
> máquina, por `scripts/gerar_evidencias.py` e pela suíte de testes. Os dados
> brutos estão em `evidencias/medicoes.json`. Nenhum valor é estimado, ilustrativo
> ou copiado de literatura — exceto onde explicitamente atribuído à fonte.

## 1. Contextualização do problema

Expressões faciais são um dos canais mais imediatos de comunicação não verbal.
Automatizar seu reconhecimento é um problema clássico de visão computacional com
aplicações em interação humano-computador, acessibilidade e análise de
usabilidade. O desafio técnico está na variabilidade: a mesma expressão muda de
aparência conforme iluminação, ângulo, distância, oclusões e características
individuais.

O problema tratado aqui é de **engenharia de aplicação**: construir um sistema
que execute essa classificação em tempo real, com webcam comum, apenas em CPU,
de forma estável, offline e compreensível para um público leigo — condições
concretas de uma demonstração presencial.

## 2. Objetivos

**Geral.** Desenvolver uma aplicação funcional e interativa que utilize redes
neurais para classificar expressões faciais capturadas por webcam em tempo real.

**Específicos.**

1. Detectar rostos em vídeo ao vivo com rede neural convolucional.
2. Alinhar geometricamente cada rosto antes da classificação.
3. Classificar sete expressões básicas por inferência real.
4. Expor a distribuição completa de pontuações, não apenas a classe vencedora.
5. Medir e exibir o desempenho computacional observado.
6. Garantir funcionamento offline após a configuração inicial.
7. Comunicar com precisão o que o sistema **não** é capaz de afirmar.

## 3. Metodologia

### 3.1 Abordagem: modelos pré-treinados

O trabalho **utiliza modelos pré-treinados** (ver `docs/modelos.md`). Treinar uma
rede de reconhecimento de expressões do zero exigiria um conjunto rotulado de
grande porte, GPU e tempo incompatíveis com o escopo da disciplina — e produziria
resultado inferior a um modelo publicado e revisado por pares.

**A contribuição própria está na engenharia completa do sistema:** arquitetura em
camadas, pipeline de visão computacional, implementação fiel do pré-processamento
especificado, interpretação correta das saídas, estratégia de estabilidade
temporal, interface, tratamento de falhas, verificação de integridade, suíte de
testes com inferência real e documentação.

### 3.2 Pré-processamento

1. Captura do quadro em BGR pelo OpenCV.
2. Espelhamento horizontal opcional (ergonomia de uso, não afeta a inferência).
3. Detecção pelo YuNet → caixa, 5 pontos de referência e confiança.
4. Transformação de similaridade não reflexiva, por mínimos quadrados, levando os
   5 pontos às posições canônicas de um recorte 112×112.
5. Conversão BGR → RGB.
6. Normalização `(pixel / 255 − 0,5) / 0,5`.
7. Montagem do tensor `1 × 3 × 112 × 112`.

### 3.3 Processo de inferência

O tensor é injetado no nó `data`; a saída do nó `label` traz 7 **logits brutos**.
Aplica-se softmax numericamente estável (subtração do máximo antes da
exponencial) para obter probabilidades que somam 1. A classe prevista é o
`argmax`. Se a probabilidade da classe vencedora estiver abaixo do limiar
configurado, a interface exibe *"Expressão não identificada com confiança
suficiente."* em vez de afirmar uma expressão.

### 3.4 Estabilidade temporal

Média móvel exponencial sobre o vetor completo de 7 probabilidades, por rosto
acompanhado, com α = 0,40 por padrão. O vetor suavizado é renormalizado e é
exatamente o que o gráfico exibe. Quando o rosto desaparece por mais de 5
quadros, a trilha é descartada e a tela é limpa.

## 4. Resultados — funcionamento

| Requisito | Situação | Evidência |
|---|---|---|
| RF01 Inicialização com diagnóstico | Implementado e testado | `test_relatorio_de_ambiente_cobre_itens_essenciais`, `test_modelo_ausente_gera_mensagem_acionavel` |
| RF02 Controles da câmera | Implementado; testado com dublê | 10 testes em `test_camera.py`; **validação com webcam física pendente** |
| RF03 Detecção facial | Implementado e testado com modelo real | `test_deteccao_em_imagem_com_rosto` |
| RF04 Classificação das 7 expressões | Implementado com inferência real | `test_inferencia_real_produz_distribuicao_valida` |
| RF05 Nível de confiança | Implementado | `test_limiar_de_confianca` |
| RF06 Gráfico de pontuações | Implementado | `test_grafico_aceita_as_sete_probabilidades` |
| RF07 Indicadores em tempo real | Implementado | `test_janela_renderiza_resultado_de_inferencia_real` |
| RF08 Múltiplos rostos | Implementado e ativo por padrão | `test_multiplos_rostos_nao_misturam_resultados` |
| RF09 Análise de imagem estática | Implementado | 10 testes de validação de arquivo |
| RF10 Estabilidade temporal | Implementado | 10 testes em `test_imagem_e_suavizacao.py` |

**Suíte automatizada: 108 testes executados, 108 aprovados** (sem contar o teste
de longa duração, executado separadamente).

## 5. Resultados — desempenho computacional medido

Máquina: Windows 10, **AMD Ryzen 7 7435HS** (16 núcleos lógicos), Python 3.11.9,
OpenCV 4.12.0, **somente CPU** (`DNN_TARGET_CPU`), sem GPU dedicada.
Imagem 512×512, 1 rosto, 120 amostras consecutivas após aquecimento.

### 5.1 Duas medições, em estados diferentes da máquina

O mesmo código foi medido duas vezes, com resultados bem distintos. A diferença
**não é do software**: é do estado de energia do processador e da carga de
outros programas. Registrar as duas é mais honesto do que escolher a melhor.

| Medição | Estado da CPU | Detecção | Classificação | **Ciclo completo** | Equivalente |
|---|---|---|---|---|---|
| **A** — máquina ociosa | em modo turbo | 13,52 ms | 2,87 ms | **17,15 ms** | ~58 ciclos/s |
| **B** — máquina ocupada | 2496/3100 MHz (80% do máximo), carga 11%, navegador e jogo abertos | 23,23 ms | 6,67 ms | **31,18 ms** | ~32 ciclos/s |

Detalhamento da medição B (a mais recente e a mais conservadora):

| Etapa | Média | Mediana | Mínimo | Máximo | Desvio padrão |
|---|---|---|---|---|---|
| Detecção (YuNet) | 23,23 ms | 23,07 ms | 20,39 ms | 28,58 ms | 1,86 ms |
| Classificação (MobileFaceNet) | 6,67 ms | 6,42 ms | 5,94 ms | 12,40 ms | 0,89 ms |
| **Ciclo completo** | **31,18 ms** | 30,96 ms | 27,62 ms | 42,13 ms | 2,45 ms |

Carregamento dos dois modelos: 134 ms. Aquecimento da rede: 98 ms.

A partir desta constatação, `scripts/gerar_evidencias.py` passou a **registrar a
frequência e a carga da CPU** junto de cada medição (`estado_da_cpu_na_medicao`
em `evidencias/medicoes.json`), para que nenhum número fique sem contexto.

### 5.2 O custo da serialização de acesso aos modelos

A correção de concorrência (seção 4.8 de `docs/arquitetura.md`) introduziu um
`RLock` no caminho de inferência. O custo foi **medido diretamente**, e não
estimado: **0,239 µs por aquisição** — cerca de 0,0008% de um ciclo de 31 ms.
Comparando 200 execuções com e sem o lock, a diferença ficou dentro do ruído de
medição. **A correção de segurança não teve custo de desempenho perceptível.**

### 5.3 Leitura dos números

- A **detecção custa ~3,5× mais que a classificação** nas duas medições. O
  gargalo é o YuNet varrendo o quadro inteiro, não a rede de expressões.
- Mesmo no pior caso medido (31,18 ms), há folga: com `inferir_a_cada = 2`
  (padrão) a carga de inferência cai pela metade.
- **Recomendação para o evento:** feche navegador, jogos e aplicativos pesados
  antes de apresentar, e ligue o notebook na tomada (o modo bateria reduz a
  frequência). Rode `verificar.bat` no local para ver o número real do dia.

> **Atenção à interpretação:** estes números medem **desempenho computacional**.
> Não são acurácia nem confiança. Ver seção 8.

## 6. Resultados — comportamento sob condições adversas

Condições simuladas por transformação controlada da imagem de referência
(`test_integracao.py`). O objetivo foi verificar **estabilidade e validade
numérica**, não acurácia.

| Condição | Resultado | Previsão observada |
|---|---|---|
| Iluminação muito reduzida (ganho 0,35) | Estável, rosto detectado | Neutra 91,1% |
| Iluminação reduzida (0,60) | Estável | Neutra 78,6% |
| Iluminação normal | Estável | Neutra 61,9% |
| Iluminação elevada (1,45) | Estável; **classe mudou** | Alegria 58,3% |
| Iluminação excessiva (1,90, estouro) | Estável | Neutra 97,2% |
| Escala 0,35× (rosto distante) | Estável; **classe mudou** | Surpresa 81,8% |
| Escala 0,50× | Estável | Neutra 89,1% |
| Escala 0,75× | Estável; **classe mudou** | Surpresa 70,7% |
| Escala 1,50× | Estável | Neutra 67,4% |
| Inclinação −20° | Estável | Neutra 89,6% |
| Inclinação +10° | Estável | Neutra 53,4% |
| Inclinação +20° | Estável; **classe mudou** | Alegria 50,5% |
| Ruído uniforme ±35 | Estável; **classe mudou** | Surpresa 70,6% |
| Desfoque gaussiano 11×11 | Estável | Neutra 59,2% |
| **Oclusão dos olhos** | **Rosto não detectado** | nenhuma classificação |
| Imagem preta / branca / 8×8 px / faixa estreita | Zero rostos, sem erro | — |

### Achados relevantes

1. **O sistema nunca falhou nem produziu valor inválido** em nenhuma condição:
   todas as distribuições somaram 1 e permaneceram finitas.
2. **A previsão é sensível a iluminação, escala e rotação.** Em 5 das 14
   condições a classe vencedora mudou, apesar de a imagem conter o mesmo rosto.
   Isso é um resultado honesto e esperado — e é exatamente por isso que a
   interface exibe a distribuição completa e o nível de confiança, em vez de
   apresentar um rótulo como verdade.
3. **Oclusão dos olhos impede a detecção**, não a classificação. O YuNet depende
   dos pontos oculares. Consequência prática: óculos escuros tendem a derrubar a
   detecção. O sistema responde corretamente, não classificando.
4. **Confiança alta não indica acerto.** A condição de ruído produziu "Surpresa
   70,6%" sobre a mesma imagem que, sem ruído, resulta em "Neutra 61,9%".

## 7. Resultados — estabilidade em execução prolongada

**Executado: 30 minutos contínuos de inferência.** Saída bruta em
`evidencias/execucao_prolongada.log`.

| Métrica | Valor |
|---|---|
| Duração | 30 min (1800,20 s) |
| Ciclos completos executados | **95.811** |
| Tempo médio — primeira metade | 18,5 ms |
| Tempo médio — segunda metade | 19,1 ms |
| **Degradação ao longo do tempo** | **+3,2%** |
| Trilhas de rosto vivas ao final | 1 (de 1 rosto) |
| Memória do processo (observada) | ~93 MB, estável |

Interpretação: **não há degradação progressiva nem vazamento**. A variação de
3,2% entre as metades está dentro da flutuação normal de frequência da CPU — bem
abaixo do limite de tolerância do teste. O número de trilhas permaneceu em 1,
confirmando que o descarte de rostos ausentes funciona em escala longa.

Controles de memória implementados e verificados:

- histórico por trilha limitado a 60 entradas (`test_historico_nao_cresce_indefinidamente`);
- trilhas descartadas após 5 quadros sem detecção (`test_rostos_que_desaparecem_nao_acumulam_trilhas`);
- slot único de quadro, sem fila acumulando (`test_slot_nao_acumula_quadros`).

> A execução prolongada automatizada exercita o **pipeline de inferência**. A
> sessão de 30 minutos **com webcam física** permanece como validação presencial
> (item H de `docs/checklist_validacao.md`).

## 7.1 Auditoria de segurança, privacidade e robustez

Foi executada uma revisão independente, somente leitura, de todo o código-fonte,
scripts e `.gitignore`.

### Categorias sem achado (confirmadas por evidência)

- **Nenhum segredo, credencial, token ou chave** em qualquer arquivo.
- **Nenhuma chamada de rede em `src/`.** As únicas ocorrências de URL são
  strings de metadados em `core/modelos.py`, nunca acessadas em execução. A lista
  completa de imports de `src/` não contém biblioteca de rede alguma. Isso
  sustenta tecnicamente o requisito de funcionamento offline.
- **Nenhuma gravação automática de imagem ou vídeo.** Não existe `imwrite` nem
  `Image.save` em `src/`. A única escrita em disco da aplicação é
  `config.local.json`, com preferências numéricas.
- **Nenhuma chave privada exportada** por `scripts/exportar_certificados.ps1`:
  apenas `RawData` de certificados públicos de CA.
- **Nenhum vazamento de `VideoCapture`**: todos os caminhos de exceção foram
  revisados.
- **Nenhum `except` mascarando falha de forma perigosa.**
- **`.gitignore` adequado**: cobre `config.local.json`, `*.pem`, `*.key`,
  `models/*.onnx`, `.venv/`, `.env` e pastas de imagens privadas.
- Path traversal e zip bomb **não se aplicam** a este modelo de ameaça
  (aplicação local, sem sandbox, sem manipulação de arquivo compactado).

### Achados corrigidos

| # | Gravidade | Problema | Correção | Regressão coberta por |
|---|---|---|---|---|
| 1 | Média/Alta | Erro **transitório** de câmera colocava a interface em modo `erro` com a thread de captura ainda viva; nesse estado "Analisar imagem" usava `cv2.dnn.Net` e `FaceDetectorYN` em **duas threads ao mesmo tempo** — risco de resultado corrompido ou falha nativa não capturável | `RLock` serializando o acesso aos modelos + `analisar_imagem` encerra a captura incondicionalmente | `test_pipeline_suporta_uso_concorrente`, `test_ajustes_concorrentes_com_processamento`, `test_analisar_imagem_encerra_captura_mesmo_em_modo_erro` |
| 2 | Média | `parar()` retornava após o tempo limite sem garantir a liberação da câmera; a reabertura falhava com "dispositivo em uso", atribuindo o problema a outro programa | `parar()` devolve `True`/`False`; a interface informa "aguardando a câmera ser liberada" e não tenta reabrir | `test_parar_devolve_false_quando_a_thread_nao_encerra`, `test_parar_devolve_true_quando_encerra_normalmente` |
| 3 | Baixa/Média | `PIL.DecompressionBombError` **não herda** de `OSError` nem de `ValueError` (confirmado: MRO é `Exception → BaseException`), escapava da validação e produziria traceback na tela | Capturada explicitamente, com mensagem própria | `test_decompression_bomb_vira_imagem_invalida`, `test_bomba_nao_escapa_como_excecao_crua` |
| 4 | Baixa | `pip --cert` era usado **sempre**, substituindo a cadeia restrita do `certifi` pelas raízes da máquina mesmo em rede sem inspeção TLS | O bundle passou a ser **fallback**: só é gerado e usado se a instalação padrão falhar | validação manual em rede sem proxy |
| 5 | Baixa | Após falha na troca de câmera, o `VideoCapture` já liberado continuava referenciado; a recuperação reabria o dispositivo **antigo** e limpava o erro | `self._cap = None` após o release, com reabertura explícita | `test_troca_de_camera_falha_nao_reutiliza_capture_liberado` |
| 6 | Baixa | `LinAlgError` do `lstsq` não era convertido em `ErroAlinhamento` e escapava no fluxo de imagem estática | Capturado e convertido; valores não finitos também rejeitados | `test_alinhamento_degenerado_vira_erro_alinhamento` |

Achado informativo não corrigido por ser intencional: o e-mail acadêmico do
autor aparece na janela "Sobre", visível aos visitantes — é identificação de
autoria exigida pelo trabalho, não um vazamento.

## 8. Distinção entre as métricas

| Métrica | O que é | Valor |
|---|---|---|
| **Acurácia de classificação** | fração de previsões corretas sobre dados rotulados | **Não medida neste trabalho.** O modelo original reporta 88,27% em RAF-DB (OpenCV Zoo) — número do modelo, não deste sistema. |
| **Confiança da previsão** | probabilidade atribuída pelo modelo à classe vencedora | Medida e exibida em tempo real. **Não é garantia de acerto** (ver seção 6, achado 4). |
| **Desempenho computacional** | tempo de processamento por quadro | 17,15 ms de média, medido nesta máquina (seção 5). |

Para obter acurácia própria, o protocolo de avaliação com voluntários está
descrito em `docs/testes.md`, seção 6. **Enquanto não for executado, nenhuma
acurácia própria deve ser afirmada.**

## 9. Limitações

1. O sistema classifica **padrões visuais** de expressão facial. Não mede emoção,
   intenção ou estado psicológico. Não é instrumento de diagnóstico.
2. Nenhuma acurácia própria foi medida (seção 8).
3. A detecção falha com oclusão da região dos olhos (seção 6, achado 3).
4. A previsão é sensível a iluminação, escala e rotação (seção 6, achado 2).
5. Medo, Tristeza e Nojo são reconhecidamente mais difíceis de distinguir;
   concentram mais erros na literatura de RAF-DB.
6. Expressões encenadas diferem de espontâneas; os modelos foram avaliados
   majoritariamente sobre imagens coletadas da web.
7. **Possível viés do conjunto de treinamento** (distribuição de idade, etnia,
   tom de pele e iluminação) que **não foi auditado** neste trabalho.
8. O desempenho medido vale para esta máquina e para 1 rosto. Com vários rostos,
   o custo de classificação cresce linearmente.
9. Validações presenciais listadas em `docs/checklist_validacao.md` estão
   **pendentes**.

## 10. Conclusão

O objetivo geral foi atingido: há uma aplicação funcional que classifica
expressões faciais em tempo real usando duas redes neurais convolucionais reais,
executando inteiramente em CPU, offline, com 17,15 ms por ciclo — folga
confortável para vídeo fluido em uma demonstração presencial.

O resultado tecnicamente mais relevante não é o desempenho, e sim a constatação
documentada na seção 6: **o mesmo rosto, sob variações plausíveis de iluminação,
distância e inclinação, produziu classes diferentes em 5 de 14 condições.** Isso
sustenta a decisão de projeto de exibir sempre a distribuição completa, o nível
de confiança e a mensagem de baixa confiança — em vez de apresentar um rótulo
único como se fosse verdade. Um sistema que mostrasse apenas a classe vencedora
pareceria mais convincente e seria menos honesto.

A separação explícita entre os modelos pré-treinados de terceiros e a engenharia
desenvolvida pelo acadêmico, bem como entre acurácia, confiança e desempenho,
é parte do resultado: ela delimita com precisão o que este trabalho pode e o que
não pode afirmar.

## 11. Referências

As referências bibliográficas verificáveis estão consolidadas em
`docs/modelos.md`, seção 5 (12 referências: repositórios oficiais, documentação
do OpenCV e artigos dos métodos YuNet, MobileFaceNet, Progressive Teacher e
RAF-DB).
