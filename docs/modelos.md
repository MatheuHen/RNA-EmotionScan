# Modelos neurais utilizados

> **Declaração obrigatória:** os dois modelos são **pré-treinados por terceiros** e
> redistribuídos pelo OpenCV Zoo sob licença Apache 2.0. **Nenhum treinamento,
> ajuste fino ou alteração de pesos foi realizado neste trabalho.** Toda
> informação abaixo foi conferida na documentação e no código-fonte oficiais,
> citados ao final.

## 1. Detecção facial — YuNet

| Item | Valor |
|---|---|
| Arquivo | `face_detection_yunet_2023mar.onnx` |
| Tamanho | 232.589 bytes |
| SHA-256 | `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` |
| Origem | OpenCV Zoo — espelho em Hugging Face `opencv/face_detection_yunet` |
| Licença | Apache License 2.0 |
| Interface usada | `cv2.FaceDetectorYN` (API nativa do OpenCV) |
| Papel no sistema | RF03 — detectar rostos e fornecer os 5 pontos de referência |

**Arquitetura.** Rede convolucional *anchor-free* leve. A imagem inteira é
processada por uma pilha de convoluções que produz, em múltiplas escalas, mapas
de confiança, de regressão de caixa e de pontos faciais. A supressão de
não-máximos elimina detecções redundantes.

**Saída por rosto** (15 valores): `x, y, largura, altura`, seguidos de 10 valores
com as coordenadas de 5 pontos (olho direito, olho esquerdo, ponta do nariz,
canto direito da boca, canto esquerdo da boca) e, por fim, a pontuação de
confiança da detecção.

**Dados de treinamento.** O repositório oficial reporta avaliação no benchmark
**WIDER Face**. O conjunto exato de treinamento e o protocolo não são detalhados
na documentação pública consultada; portanto, nenhum número de acurácia de
detecção é afirmado neste trabalho.

**Parâmetros em uso.** Limiar de confiança 0,85 (ajustável); limiar de NMS 0,30;
`top_k` 5000; `input_size` sincronizado com a resolução real do quadro a cada
mudança de tamanho.

## 2. Classificação de expressões — Progressive Teacher / MobileFaceNet

| Item | Valor |
|---|---|
| Arquivo | `facial_expression_recognition_mobilefacenet_2022july.onnx` |
| Tamanho | 4.791.892 bytes |
| SHA-256 | `4f61307602fc089ce20488a31d4e4614e3c9753a7d6c41578c854858b183e1a9` |
| Origem | OpenCV Zoo — espelho em Hugging Face `opencv/facial_expression_recognition` |
| Licença | Apache License 2.0 |
| Interface usada | `cv2.dnn.readNet` |
| Papel no sistema | RF04/RF05 — classificar 7 expressões |

**Arquitetura.** *Backbone* **MobileFaceNet** (convoluções separáveis em
profundidade, projetado para dispositivos móveis), treinado pelo método
semi-supervisionado **Progressive Teacher**. A camada final projeta o vetor de
características em 7 saídas, uma por expressão.

**Dados de treinamento e acurácia reportada.** O README oficial do OpenCV Zoo
reporta **88,27% de acurácia na avaliação sobre o conjunto RAF-DB**
(*Real-world Affective Faces Database*). O repositório **não menciona FER2013**
como conjunto de treinamento ou avaliação deste modelo — por isso este trabalho
não faz tal afirmação. Esse número **não é** a acurácia do sistema aqui
desenvolvido: refere-se ao modelo original, em conjunto de imagens controlado.

### 2.1 Contrato de entrada e saída (verificado no código-fonte oficial)

Fonte: `opencv_zoo/models/facial_expression_recognition/facial_fer_model.py`.

| Item | Valor |
|---|---|
| Nome do nó de entrada | `data` |
| Formato da entrada | `1 × 3 × 112 × 112` (NCHW) |
| Ordem dos canais | **RGB** (o OpenCV entrega BGR; a conversão é obrigatória) |
| Normalização | `(pixel / 255 − 0,5) / 0,5` → intervalo `[−1, +1]` |
| Nome do nó de saída | `label` |
| Formato da saída | `1 × 7` |
| Natureza da saída | **logits brutos** — não são probabilidades |

Como a saída é composta de logits, **o softmax é tecnicamente necessário** para
obter valores comparáveis que somem 1. É exatamente o que o sistema faz, em
`core/classificador.py`. Se o modelo já entregasse probabilidades, aplicar
softmax distorceria a escala — por isso o comportamento é controlado pela
constante `FER_SAIDA_E_LOGIT`, documentada no código.

### 2.2 Ordem das classes

A ordem é definida por `FacialExpressionRecog.getDesc` no código oficial e **não
pode ser alterada**, pois corresponde aos índices de saída da rede:

| Índice | Classe do modelo | Rótulo na interface |
|---|---|---|
| 0 | `angry` | Raiva |
| 1 | `disgust` | Nojo |
| 2 | `fearful` | Medo |
| 3 | `happy` | Alegria |
| 4 | `neutral` | Neutra |
| 5 | `sad` | Tristeza |
| 6 | `surprised` | Surpresa |

O teste `test_ordem_das_classes_bate_com_o_modelo_oficial` trava essa ordem, e
`test_traducao_alinhada_indice_por_indice` garante que a tradução não
desalinhe os índices.

### 2.3 Alinhamento facial

O classificador espera o rosto já **alinhado**, não um simples recorte da caixa.
O sistema implementa a mesma transformação de similaridade não reflexiva do
código de referência: os 5 pontos detectados pelo YuNet são levados, por mínimos
quadrados, às posições canônicas usadas pelo ArcFace em um recorte 112×112:

```
(38,2946 ; 51,6963)   olho direito
(73,5318 ; 51,5014)   olho esquerdo
(56,0252 ; 71,7366)   ponta do nariz
(41,5493 ; 92,3655)   canto direito da boca
(70,7299 ; 92,2041)   canto esquerdo da boca
```

A solução é por mínimos quadrados (`numpy.linalg.lstsq`), **determinística** —
ao contrário de estimadores robustos como RANSAC ou LMEDS, que introduziriam
variação entre execuções idênticas. O teste
`test_inferencia_e_deterministica` comprova que a mesma entrada produz sempre a
mesma saída.

O método de alinhamento tem efeito mensurável na saída. Durante o
desenvolvimento, a mesma imagem de referência produziu, com um estimador afim
aproximado (`cv2.estimateAffinePartial2D` com LMEDS), `neutral` a 81,3%; com a
transformação oficial por mínimos quadrados, `neutral` a 61,9%. A classe
vencedora não mudou, mas a distribuição sim — razão pela qual a implementação
reproduz a referência oficial em vez de usar um atalho.

## 3. Integridade e procedência

`core/modelos.py` guarda, para cada arquivo, a URL de origem, o tamanho exato e o
SHA-256 esperado. A verificação ocorre em três momentos:

1. **No download** (`scripts/baixar_modelos.py`): o arquivo vai para um temporário,
   tem tamanho e hash conferidos, e só então é movido para `models/`. Um download
   interceptado por proxy ou antivírus é rejeitado em vez de ser usado.
2. **Na verificação** (`verificar.bat`): confere o hash dos arquivos em disco.
3. **Nos testes** (`test_integridade_dos_arquivos_onnx`).

## 4. Licenças de terceiros

Consolidadas em `TERCEIROS.md`, na raiz do projeto.

## 5. Referências

1. OpenCV Zoo — repositório oficial de modelos.
   <https://github.com/opencv/opencv_zoo>
2. OpenCV Zoo — *Facial Expression Recognition* (README com a acurácia em RAF-DB
   e a indicação do backbone MobileFaceNet).
   <https://github.com/opencv/opencv_zoo/tree/main/models/facial_expression_recognition>
3. OpenCV Zoo — `facial_fer_model.py` (pré-processamento, nomes dos nós e ordem
   das classes).
   <https://github.com/opencv/opencv_zoo/blob/main/models/facial_expression_recognition/facial_fer_model.py>
4. OpenCV Zoo — *Face Detection YuNet*.
   <https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet>
5. Hugging Face — `opencv/facial_expression_recognition`.
   <https://huggingface.co/opencv/facial_expression_recognition>
6. Hugging Face — `opencv/face_detection_yunet`.
   <https://huggingface.co/opencv/face_detection_yunet>
7. JIANG, J.; WANG, M. et al. *Boosting Facial Expression Recognition by a
   Semi-Supervised Progressive Teacher*. IEEE Transactions on Affective
   Computing. <https://ieeexplore.ieee.org/abstract/document/9629313>
8. CHEN, S. et al. *MobileFaceNets: Efficient CNNs for Accurate Real-Time Face
   Verification on Mobile Devices*. CCBR 2018.
   <https://link.springer.com/chapter/10.1007/978-3-319-97909-0_46>
9. LI, S.; DENG, W.; DU, J. *Reliable Crowdsourcing and Deep Locality-Preserving
   Learning for Expression Recognition in the Wild* (RAF-DB).
   <http://whdeng.cn/RAF/model1.html>
10. WU, W. et al. *YuNet: A Tiny Millisecond-level Face Detector*. Machine
    Intelligence Research, 2023.
    <https://link.springer.com/article/10.1007/s11633-023-1423-y>
11. Documentação do OpenCV — `cv::FaceDetectorYN`.
    <https://docs.opencv.org/4.x/df/d20/classcv_1_1FaceDetectorYN.html>
12. Documentação do OpenCV — módulo `dnn`.
    <https://docs.opencv.org/4.x/d6/d0f/group__dnn.html>
