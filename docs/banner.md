# Material para o banner acadêmico

> Este documento **não é o banner**. Ele reúne o conteúdo e as evidências prontas
> para você montar o banner no modelo oficial do professor. Nenhum arquivo de
> banner foi criado ou alterado.

## 1. Especificação

| Item | Valor |
|---|---|
| Largura | 90 cm |
| Altura | 120 cm |
| Orientação | vertical |
| Proporção | 3:4 |
| Exportação no Canva | **3402 × 4535 px** |
| Seções obrigatórias | Título · Autoria e e-mail · Introdução · Metodologia · Resultados e conclusões · Referências |

## 2. Título

**Expressão Neural: Reconhecimento de Expressões Faciais em Tempo Real utilizando Redes Neurais**

## 3. Autoria

Matheus Henrique de Araújo Carneiro
matheus.carneiro@academico.unirv.edu.br
Engenharia de Software — UniRV
Redes Neurais Artificiais (RNA) · Prof. Sandro Silva Moreira
5º Workshop de Trabalhos de Inteligência Artificial da FASOFT — UniRV

## 4. Introdução (texto pronto, ~110 palavras)

> Expressões faciais são um dos canais mais imediatos de comunicação não verbal, e
> seu reconhecimento automático é um problema clássico de visão computacional. O
> desafio está na variabilidade: a mesma expressão muda de aparência conforme
> iluminação, ângulo, distância e características individuais.
>
> Este trabalho apresenta o **Expressão Neural**, uma aplicação desktop que
> classifica sete expressões faciais em tempo real a partir da webcam, utilizando
> duas redes neurais convolucionais pré-treinadas em formato ONNX. O sistema
> executa inteiramente em CPU, sem GPU dedicada e sem qualquer conexão com a
> internet, de modo que nenhuma imagem deixa o computador do usuário.

## 5. Metodologia (texto pronto, ~150 palavras)

> O sistema encadeia duas redes neurais. A rede **YuNet** detecta rostos no quadro
> e devolve a caixa delimitadora, cinco pontos faciais de referência e uma
> pontuação de confiança. Esses cinco pontos alimentam uma transformação de
> similaridade por mínimos quadrados que alinha o rosto em um recorte canônico de
> 112×112 pixels. O recorte é convertido para RGB e normalizado pela expressão
> (pixel ÷ 255 − 0,5) ÷ 0,5.
>
> O tensor resultante alimenta a rede **MobileFaceNet**, treinada pelo método
> semi-supervisionado Progressive Teacher, que produz sete logits — um por
> expressão. A função softmax converte os logits em probabilidades que somam 100%.
> Uma média móvel exponencial aplicada ao vetor completo de probabilidades reduz
> oscilações entre quadros consecutivos, por rosto acompanhado.
>
> **Ambos os modelos são pré-treinados por terceiros (OpenCV Zoo, licença Apache
> 2.0). A contribuição deste trabalho é a engenharia completa da aplicação:**
> arquitetura, pipeline de visão computacional, interpretação das saídas,
> estabilidade temporal, interface, tratamento de falhas e 109 testes
> automatizados com inferência real.

## 6. Resultados e conclusões (texto pronto, ~160 palavras)

> O sistema processa um ciclo completo de detecção e classificação em **17,15 ms
> em média** (desvio padrão 1,0 ms) em CPU AMD Zen 3, equivalente a **58
> inferências por segundo** — folga confortável para vídeo fluido. A detecção
> consome 13,52 ms e a classificação apenas 2,87 ms: o gargalo é localizar o
> rosto, não reconhecer a expressão. A suíte automatizada executa **108 testes**
> com os modelos reais, todos aprovados.
>
> O resultado mais relevante, porém, é outro. Submetido a 14 variações
> controladas da mesma imagem — iluminação, escala, rotação e ruído — o sistema
> **nunca falhou nem produziu valor inválido**, mas **a classe vencedora mudou em
> 5 das 14 condições**. Isso sustenta a decisão de projeto de exibir sempre a
> distribuição completa das sete pontuações e o nível de confiança, em vez de
> apresentar um rótulo único como verdade.
>
> Conclui-se que é viável construir um classificador de expressões faciais em
> tempo real, offline e em CPU comum — desde que a interface comunique
> honestamente a incerteza do modelo.

## 7. Caixa de destaque sugerida

```
┌──────────────────────────────────────────────────┐
│  17,15 ms por ciclo  ·  ~58 inferências/s  ·  CPU │
│  108 testes automatizados aprovados               │
│  100% offline  ·  nenhuma imagem sai do PC        │
└──────────────────────────────────────────────────┘
```

## 8. Aviso ético (obrigatório no banner)

> Este sistema classifica padrões visuais de expressões faciais. Os resultados
> não representam diagnósticos psicológicos nem determinam com certeza as emoções
> reais de uma pessoa.

## 9. Imagens disponíveis em `evidencias/`

| Arquivo | O que mostra | Onde usar |
|---|---|---|
| `captura_interface.png` | tela completa com resultado real | elemento visual principal |
| `deteccao_anotada.png` | caixa, 5 pontos e rótulo sobre a imagem | Metodologia |
| `rosto_alinhado_112.png` | a entrada real do classificador | Metodologia (ilustra o alinhamento) |
| `grafico_probabilidades.png` | barras das 7 categorias | Resultados |
| `desempenho.png` | distribuição dos tempos medidos | Resultados |
| `medicoes.md` / `medicoes.json` | números brutos | conferência |

Regenerar a qualquer momento:

```bat
.venv\Scripts\python.exe scripts\gerar_evidencias.py
```

> **Importante:** as capturas atuais usam a imagem de referência `lena.jpg`. Se
> quiser uma captura com o seu próprio rosto, abra o sistema, posicione-se e use
> a tecla `Print Screen` — mas **não inclua rostos de terceiros sem autorização
> por escrito**.

## 10. Diagrama de arquitetura (para redesenhar no Canva)

```
   WEBCAM
     │
     ▼
┌─────────────┐   caixa + 5 pontos   ┌──────────────┐   112×112   ┌───────────────┐
│   YuNet     │────────────────────▶ │ Alinhamento  │───────────▶ │ MobileFaceNet │
│  (detecção) │                      │ (similaridade│             │ (classificação)│
│   13,5 ms   │                      │  por mínimos │             │    2,9 ms     │
└─────────────┘                      │  quadrados)  │             └───────┬───────┘
                                     └──────────────┘                     │ 7 logits
                                                                          ▼
                                                                   ┌─────────────┐
                                                                   │   softmax   │
                                                                   └──────┬──────┘
                                                                          ▼
                                                                   ┌─────────────┐
                                                                   │ suavização  │
                                                                   │  temporal   │
                                                                   └──────┬──────┘
                                                                          ▼
                                                                    INTERFACE
                                                              gráfico · confiança
                                                                  · indicadores
```

Paleta do projeto, para manter coerência visual com a aplicação:

| Uso | Cor |
|---|---|
| Fundo | `#0b1120` |
| Superfície / cartões | `#111a2e` |
| Primária (ciano) | `#38bdf8` |
| Acento (violeta) | `#a78bfa` |
| Texto | `#e8eefc` |
| Texto secundário | `#9fb0d0` |

Cores das classes: Raiva `#ef4444` · Nojo `#84cc16` · Medo `#a855f7` ·
Alegria `#22c55e` · Neutra `#64748b` · Tristeza `#3b82f6` · Surpresa `#f59e0b`

## 11. Referências sugeridas para o banner (5 de 12)

Lista completa em `docs/modelos.md`, seção 5.

1. JIANG, J.; WANG, M. et al. *Boosting Facial Expression Recognition by a
   Semi-Supervised Progressive Teacher*. IEEE Transactions on Affective
   Computing, 2021.
2. CHEN, S. et al. *MobileFaceNets: Efficient CNNs for Accurate Real-Time Face
   Verification on Mobile Devices*. CCBR, 2018.
3. WU, W. et al. *YuNet: A Tiny Millisecond-level Face Detector*. Machine
   Intelligence Research, 2023.
4. LI, S.; DENG, W.; DU, J. *Reliable Crowdsourcing and Deep
   Locality-Preserving Learning for Expression Recognition in the Wild*
   (RAF-DB). CVPR, 2017.
5. OPENCV. *OpenCV Zoo*. Disponível em: https://github.com/opencv/opencv_zoo.

## 12. O que NÃO escrever no banner

Para preservar a integridade acadêmica do trabalho:

- ❌ "Acurácia de 88,27%" — esse número é **do modelo original em RAF-DB**, não
  deste sistema. Se citar, atribua explicitamente: *"o modelo pré-treinado
  utilizado reporta 88,27% em RAF-DB"*.
- ❌ Qualquer acurácia própria — **não foi medida**. Protocolo para medi-la em
  `docs/testes.md`, seção 6.
- ❌ "Treinamos uma rede neural" — os modelos são pré-treinados.
- ❌ "Detecta emoções" — o sistema classifica **padrões visuais de expressão**.
- ❌ Número de épocas, tamanho de dataset ou curva de treinamento — não houve
  treinamento.
- ❌ Rostos de terceiros sem autorização por escrito.
