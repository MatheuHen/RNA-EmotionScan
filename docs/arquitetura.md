# Arquitetura do sistema — Expressão Neural

## 1. Visão geral

O sistema é uma aplicação desktop monolítica, offline e orientada a camadas. Não
há servidor, banco de dados, serviço em nuvem ou chamada de rede em tempo de
execução. Todo o processamento ocorre no computador do usuário, em CPU.

```
                        ┌───────────────────────────────┐
                        │   Thread da interface (Tk)    │
                        │   app.py → ui/janela.py       │
                        └───────────────┬───────────────┘
                                        │ consome o quadro mais recente
                                        │ (slot único, sem fila)
                        ┌───────────────▼───────────────┐
                        │  Thread de captura            │
                        │  core/camera.py               │
                        └───────────────┬───────────────┘
                                        │ quadro BGR
                        ┌───────────────▼───────────────┐
                        │  core/pipeline.py             │
                        └───┬───────────┬───────────┬───┘
                            │           │           │
              ┌─────────────▼──┐  ┌─────▼──────┐  ┌─▼───────────────┐
              │ core/detector  │  │ core/      │  │ core/           │
              │ YuNet (ONNX)   │─▶│ alinhamento│─▶│ classificador   │
              │ caixa + 5 pts  │  │ 112×112    │  │ MobileFaceNet   │
              └────────────────┘  └────────────┘  └─────┬───────────┘
                                                        │ 7 logits
                                                 ┌──────▼──────────┐
                                                 │ softmax         │
                                                 │ core/suavizacao │
                                                 └─────────────────┘
```

## 2. Fluxograma de funcionamento

```
          ┌─────────────┐
          │   Início    │
          └──────┬──────┘
                 ▼
   ┌──────────────────────────┐
   │ RF01 Verificar ambiente  │
   │ deps + modelos + hash    │
   └──────┬───────────────────┘
          │ falha → mensagem acionável → fim
          ▼ ok
   ┌──────────────────────────┐
   │ Interface pronta         │
   │ (modelos ainda NÃO       │
   │  carregados)             │
   └──────┬───────────────────┘
          ▼
   ┌──────────────────────────┐        ┌───────────────────────────┐
   │ Usuário clica em         │        │ Usuário escolhe imagem    │
   │ "Iniciar câmera"         │        │ local (RF09)              │
   └──────┬───────────────────┘        └──────────┬────────────────┘
          ▼                                       ▼
   ┌──────────────────────────┐        ┌───────────────────────────┐
   │ Carrega modelos (1ª vez) │◀───────│ Valida extensão, tamanho, │
   │ e aquece a rede          │        │ cabeçalho e decodificação │
   └──────┬───────────────────┘        └──────────┬────────────────┘
          ▼                                       │
   ┌──────────────────────────┐                   │
   │ Abre VideoCapture        │                   │
   │ falha → mensagem + RF09  │                   │
   └──────┬───────────────────┘                   │
          ▼                                       │
   ┌──────────────────────────┐                   │
   │ Lê quadro                │◀──────────────┐   │
   └──────┬───────────────────┘               │   │
          ▼                                   │   │
   ┌──────────────────────────┐               │   │
   │ É quadro de inferência?  │── não ────────┤   │
   │ (contador % N == 0)      │               │   │
   └──────┬───────────────────┘               │   │
          ▼ sim                               │   │
   ┌──────────────────────────┐◀──────────────────┘
   │ YuNet: detectar rostos   │               │
   └──────┬───────────────────┘               │
          │ 0 rostos → limpa resultado ───────┤
          ▼ ≥1 rosto                          │
   ┌──────────────────────────┐               │
   │ Alinhar 5 pontos → 112²  │               │
   └──────┬───────────────────┘               │
          ▼                                   │
   ┌──────────────────────────┐               │
   │ Normalizar e inferir     │               │
   │ → 7 logits → softmax     │               │
   └──────┬───────────────────┘               │
          ▼                                   │
   ┌──────────────────────────┐               │
   │ Suavização temporal (EMA)│               │
   │ por trilha de rosto      │               │
   └──────┬───────────────────┘               │
          ▼                                   │
   ┌──────────────────────────┐               │
   │ Publicar no slot único   │               │
   └──────┬───────────────────┘               │
          ▼                                   │
   ┌──────────────────────────┐               │
   │ Interface desenha vídeo, │               │
   │ gráfico e indicadores    │───────────────┘
   └──────┬───────────────────┘
          ▼ usuário encerra
   ┌──────────────────────────┐
   │ Parar thread, liberar    │
   │ câmera, limpar estado    │
   └──────────────────────────┘
```

## 3. Estrutura de diretórios

| Caminho | Responsabilidade |
|---|---|
| `app.py` | Ponto de entrada; modos GUI, `--verificar` e `--imagem`. |
| `src/expressao_neural/config.py` | Constantes, classes, limiares, identidade acadêmica, `Ajustes`. |
| `src/expressao_neural/core/modelos.py` | Registro dos modelos: URL, SHA-256, licença, procedência, verificação de integridade. |
| `src/expressao_neural/core/detector.py` | `DetectorFacial` (YuNet) → lista de `Rosto`. |
| `src/expressao_neural/core/alinhamento.py` | Transformação de similaridade para o recorte canônico 112×112. |
| `src/expressao_neural/core/classificador.py` | `ClassificadorExpressao`, pré-processamento e softmax. |
| `src/expressao_neural/core/suavizacao.py` | `SuavizadorTemporal`: trilhas por rosto e média móvel exponencial. |
| `src/expressao_neural/core/pipeline.py` | Orquestra detecção → alinhamento → classificação → suavização. |
| `src/expressao_neural/core/camera.py` | `TrabalhadorCamera`: thread de captura, recuperação de falhas, slot único. |
| `src/expressao_neural/core/imagem_estatica.py` | Validação em camadas e carregamento de imagem local. |
| `src/expressao_neural/ui/tema.py` | Paleta, tipografia, estilos ttk, consciência de DPI. |
| `src/expressao_neural/ui/componentes.py` | Cartão, painel rolável, gráfico de barras, medidor, ícone, indicadores. |
| `src/expressao_neural/ui/desenho.py` | Sobreposição no vídeo (caixa, pontos, rótulo) e placas de mensagem. |
| `src/expressao_neural/ui/janela.py` | Janela principal, máquina de estados e laço de atualização. |
| `src/expressao_neural/ui/sobre.py` | Janela "Sobre o Projeto". |
| `src/expressao_neural/utils/ambiente.py` | Diagnóstico de dependências, modelos e pastas (RF01). |
| `scripts/` | Obtenção de modelos, verificação de prontidão, geração de evidências, certificados. |
| `models/` | Arquivos ONNX (não versionados). |
| `tests/` | Suíte automatizada com inferência real. |
| `docs/` | Documentação técnica e acadêmica. |
| `evidencias/` | Capturas, gráficos e medições reais. |

## 4. Decisões de arquitetura e justificativas

### 4.1 Inferência ONNX pelo OpenCV, sem TensorFlow/PyTorch/onnxruntime

`cv2.dnn.readNet` executa ambos os modelos ONNX. Como o OpenCV já é obrigatório
para captura e processamento de imagem, nenhuma biblioteca adicional de inferência
é necessária. Consequências medidas: instalação total de ~100 MB em vez de
vários gigabytes, e inicialização em menos de um segundo. O teste
`test_nao_usa_bibliotecas_pesadas` falha se alguma dessas bibliotecas passar a
ser requisito.

### 4.2 Separação entre processamento e interface

A leitura da câmera e a inferência rodam em `TrabalhadorCamera`, uma `Thread`
daemon. A thread do Tkinter apenas consome o resultado. Isso atende à exigência
de interface responsiva: nenhuma operação de I/O ou de rede neural bloqueia o
laço de eventos.

### 4.3 Slot único em vez de fila

O trabalhador publica em um único campo protegido por `Lock`, sobrescrevendo o
valor anterior. Se a interface não consumir a tempo, o quadro antigo é
descartado — a latência não cresce e a memória não acumula. Uma `Queue` com
tamanho livre permitiria acúmulo indefinido; uma fila limitada exigiria política
de descarte equivalente, com mais código. Verificado por
`test_slot_nao_acumula_quadros`.

### 4.4 Carregamento tardio dos modelos

Os modelos só são lidos quando o usuário inicia a câmera ou abre uma imagem. A
janela abre imediatamente, o que importa numa apresentação em que o sistema é
aberto e fechado muitas vezes.

### 4.5 Frequência de inferência configurável

`inferir_a_cada = N` roda a rede a cada N quadros, reaproveitando o último
resultado nos demais. O vídeo continua fluido mesmo em máquinas modestas.
Padrão: 2.

### 4.6 Suavização sobre o vetor completo

A média móvel exponencial é aplicada ao vetor de 7 probabilidades, não ao rótulo
vencedor. O vetor suavizado é renormalizado e continua somando 1, e é exatamente
o que o gráfico exibe. O rótulo é sempre o `argmax` do vetor exibido — não há
divergência entre o que o gráfico mostra e o que o rótulo afirma. O ajuste pode
ser desligado na interface para demonstrar a saída bruta.

### 4.7 Acompanhamento de rostos por proximidade

Cada rosto detectado é associado à trilha mais próxima dentro de um raio
proporcional ao tamanho do rosto. Uma trilha não pode ser reivindicada duas
vezes no mesmo quadro. Trilhas ausentes por mais de 5 quadros são descartadas,
o que impede vazamento de memória e garante o requisito RF10.

### 4.7-A Limiar de confiança: 0,35, não 0,50

O limiar separa "exibir a expressão" de "exibir *Expressão não identificada com
confiança suficiente*". Ele **não altera a previsão do modelo** — apenas decide
se o rótulo é mostrado.

O padrão inicial de 0,50 se mostrou alto na validação com webcam: expressões
**naturais** (sorriso de conversa, surpresa discreta) costumam vencer com 35% a
55% de confiança. Com 0,50, uma previsão correta do modelo aparecia como
"Indefinida", obrigando o usuário a exagerar a expressão — o que não representa
uso real nem funciona bem numa demonstração a visitantes.

O padrão passou a **0,35**, e continua ajustável na interface. Duas
consequências que devem ser comunicadas:

- expressões sutis passam a ser rotuladas, com a confiança real sempre visível
  ao lado — o usuário vê que a decisão foi apertada;
- a taxa de rótulos errados exibidos aumenta, porque menos previsões ficam
  retidas. Por isso o gráfico das sete pontuações é exibido sempre: ele mostra a
  disputa real entre as classes.

Pela mesma razão, `suavizacao_alfa` subiu de 0,40 para **0,55**: com 0,40 uma
expressão sutil levava vários quadros para se firmar no gráfico.

### 4.8 Serialização do acesso aos modelos

`cv2.dnn.Net` e `cv2.FaceDetectorYN` **não são thread-safe**: `setInput`/`forward`
e `setInputSize`/`detect` são pares não atômicos, e o OpenCV libera o GIL durante
a execução. Como a análise de imagem estática roda na thread do Tkinter enquanto
a captura roda na própria thread, `PipelineExpressao` serializa `processar`,
`aplicar_ajustes`, `reiniciar_estado` e `liberar` com um `threading.RLock`.

Essa necessidade foi identificada por auditoria: existia um caminho real em que
um erro **transitório** da câmera (ex.: "a câmera parou de entregar imagens")
colocava a interface em modo `erro` com a thread de captura **ainda viva**; nesse
estado, "Analisar imagem" não encerrava a captura e usava os mesmos objetos
nativos em paralelo — risco de resultado corrompido ou de falha nativa não
capturável. Além do `RLock`, `analisar_imagem` passou a encerrar a captura
incondicionalmente. Regressão coberta por `test_pipeline_suporta_uso_concorrente`
e `test_analisar_imagem_encerra_captura_mesmo_em_modo_erro`.

### 4.9 Parada da captura com confirmação

`TrabalhadorCamera.parar()` devolve `True` apenas quando a thread realmente
terminou — garantindo que o `finally` de `run()` liberou o `VideoCapture`. Se o
prazo expirar (típico quando `cap.read()` trava com webcam USB), devolve `False`
e a interface **não tenta reabrir o dispositivo**: exibe "Aguardando a câmera
anterior ser liberada". Sem isso, a reabertura falharia com "dispositivo em uso",
mensagem que atribuiria o problema a outro programa.

## 5. Tratamento de falhas

| Falha | Comportamento |
|---|---|
| Modelo ausente ou corrompido | Mensagem citando `configurar.bat`; aplicação não inicia a inferência. |
| Câmera inexistente ou ocupada | Mensagem citando conflito de programa e permissão do Windows; "Analisar imagem" continua disponível. |
| Câmera para de entregar quadros | Após 15 falhas, aviso; após 25, reabertura automática do dispositivo. |
| Exceção dentro da inferência | Capturada na thread; registrada no status; a captura continua viva. |
| Imagem inválida | Recusada antes da decodificação, com motivo específico. |
| Imagem com dimensões absurdas (*decompression bomb*) | Recusada pelo cabeçalho, antes de qualquer decodificação, com mensagem própria. |
| Alinhamento degenerado (pontos colineares/idênticos) | Convertido em `ErroAlinhamento`; o rosto é ignorado, a aplicação segue. |
| Captura não encerra no prazo | Interface informa que a câmera está sendo liberada e **não** tenta reabrir. |
| Quadro degenerado | Detector devolve zero rostos; nenhuma classificação é feita. |
| Fechamento da janela | Thread encerrada, `VideoCapture.release()` garantido por `finally`. |

## 6. Modelo de threads

| Thread | Criada em | Encerramento |
|---|---|---|
| Principal (Tk) | processo | `encerrar()` cancela o `after` e destrói a janela |
| `ExpressaoNeural-Captura` | `iniciar_camera()` | `parar()` sinaliza `Event` e faz `join(3s)`; `finally` libera a câmera |
| `BuscaCameras` | `procurar_cameras()` | encerra sozinha; resultado devolvido via `after(0, ...)` |

Todas são daemon: nenhuma impede o encerramento do processo.
