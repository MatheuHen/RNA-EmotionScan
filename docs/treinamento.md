# Treinamento da camada de calibração

> **Leia esta seção antes de afirmar qualquer coisa sobre treinamento.** Ela
> define com precisão o que foi e o que não foi treinado neste trabalho.

## 1. O que é, exatamente

A rede convolucional de expressões (MobileFaceNet / Progressive Teacher) é
**pré-treinada por terceiros e permanece congelada** — nenhum peso dela é
alterado. Sobre os 7 logits que ela produz, aplica-se uma camada linear própria:

```
z' = W · z + b          W: 7×7,  b: 7   →   56 parâmetros treináveis
p  = softmax(z')
```

`W` e `b` são **aprendidos por descida de gradiente** sobre um conjunto rotulado
coletado pelo próprio autor, com função de perda de entropia cruzada.

### O que isto É

- Aprendizado supervisionado de uma camada de saída própria.
- Com dataset próprio, divisão treino/teste estratificada, épocas, curva de
  perda e métricas próprias.

### O que isto NÃO É

- ❌ Treinamento da rede convolucional.
- ❌ Ajuste fino (*fine-tuning*) do backbone.
- ❌ Um modelo novo de reconhecimento facial.

**Como declarar no banner e na apresentação:**

> "Os modelos de detecção e classificação são pré-treinados (OpenCV Zoo). Sobre
> as saídas do classificador foi treinada uma camada de calibração supervisionada
> de 56 parâmetros, com um conjunto de N amostras rotuladas coletadas pelo autor,
> atingindo X% de acurácia no conjunto de teste."

Isso é verdadeiro e verificável. Dizer "treinei uma rede neural para reconhecer
expressões" **não** seria.

## 2. Por que inicializar na identidade

`W` começa como matriz identidade e `b` como zero. Nesse ponto a camada é a
função identidade e o sistema **reproduz exatamente o modelo base**. A
regularização L2 penaliza o *afastamento da identidade*, não o tamanho de `W`.

Consequência: o treino só se afasta do modelo base quando os dados justificam, e
o ganho é mensurável — basta comparar antes e depois no mesmo conjunto de teste.
Com poucas amostras, isso evita que a camada "aprenda" ruído.

## 3. Como executar

### Passo 1 — coletar amostras (precisa da webcam)

```bat
.venv\Scripts\python.exe scripts\coletar_amostras.py --por-classe 30
```

Abre uma janela com a câmera. Para cada expressão: posicione-se, clique em
**COLETAR ESTA EXPRESSÃO** (ou pressione ESPAÇO) e mantenha a expressão
enquanto as amostras são salvas (uma a cada 0,25 s).

**Varie levemente** a pose, a distância e a intensidade durante a coleta —
variedade no conjunto de treino melhora a generalização.

As imagens vão para `dados/amostras/<classe>/` como recortes 112×112 já
alinhados, exatamente a entrada que o classificador recebe. **Não são
versionadas** (contêm rosto humano).

Recomendação: **no mínimo 20 por classe**; 30 a 50 é melhor.

### Passo 2 — treinar

```bat
.venv\Scripts\python.exe scripts\treinar_calibracao.py
```

Opções: `--epocas 600`, `--taxa 0.05`, `--regularizacao 1e-3`, `--teste 0.25`.

Saídas:

| Arquivo | Conteúdo |
|---|---|
| `models/calibracao_pessoal.npz` | pesos treinados (não versionado) |
| `docs/banner/06_metricas/treinamento_calibracao.json` | relatório completo |
| `docs/banner/06_metricas/treinamento_calibracao.md` | mesmo relatório em tabela |
| `docs/banner/04_graficos/matriz_confusao.png` | matriz com calibração |
| `docs/banner/04_graficos/matriz_confusao_base.png` | matriz do modelo base |
| `docs/banner/04_graficos/curva_de_perda.png` | perda por época |

### Passo 3 — ativar na aplicação (opcional)

A calibração vem **desligada por padrão**: o sistema entrega a saída do modelo
pré-treinado a menos que você opte conscientemente por usá-la.

Para ativar, edite `config.local.json` na raiz:

```json
{ "usar_calibracao": true }
```

## 4. Se o treinamento piorar o resultado

Acontece, e o script avisa. Causas comuns:

- **poucas amostras** — colete mais;
- **classes desbalanceadas** — equilibre a quantidade por expressão;
- **amostras muito parecidas** — varie pose, distância e intensidade;
- **sobreajuste** — aumente `--regularizacao` (ex.: `1e-2`).

Um resultado pior **também é resultado** e deve ser relatado como tal. Não
ative a calibração se ela não melhorou o teste.

## 5. Limitações obrigatórias de declarar

- O conjunto vem de **uma única pessoa**. A acurácia medida vale para esse
  contexto e **não generaliza** para público diverso.
- As expressões são **encenadas sob solicitação**, não espontâneas.
- O conjunto de teste é pequeno; a acurácia tem **incerteza alta**.
- A rede convolucional não foi treinada: o ganho possível é de **calibração**,
  não de capacidade de representação.
- Os **88,27% em RAF-DB** continuam sendo do modelo pré-treinado e **não devem
  ser confundidos** com a acurácia medida aqui.

## 6. Verificação automatizada

`tests/test_calibracao.py` cobre 16 casos, incluindo a prova de que o
treinamento **reduz a perda** e **melhora a acurácia** sobre um viés
sistemático controlado, que a inicialização é a identidade, e que a persistência
dos pesos é íntegra.
