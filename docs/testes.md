# Documentação dos testes — EmotionScan

## 1. Como executar

```bat
testar.bat              :: suíte padrão (ignora os testes de longa duração)
testar.bat --lentos     :: inclui a execução prolongada
```

Execução prolongada com duração personalizada:

```bat
set EXPRESSAO_NEURAL_DURACAO_MIN=30
.venv\Scripts\python.exe -m pytest tests/test_integracao.py::test_execucao_prolongada -m "" -s
```

## 2. Natureza dos testes

**Os testes de inferência usam os modelos ONNX reais, não dublês.** Mocks são
usados apenas onde o hardware impediria a automação (dublê de `VideoCapture` em
`tests/test_camera.py`), e nesse caso o objetivo é exercitar o ciclo de vida da
captura — não simular resultados de rede neural.

Se os arquivos ONNX estiverem ausentes, os testes dependentes são **ignorados
com motivo explícito** (`skip`), nunca aprovados em falso.

## 3. Cobertura por requisito

| Arquivo | Alvo | Testes |
|---|---|---|
| `test_modelos.py` | Dependências, integridade, carregamento, RF01 | 10 |
| `test_inferencia.py` | Pré-processamento, classes, softmax, alinhamento, inferência real, limiares, RF03–RF05, RF08 | 24 |
| `test_camera.py` | RF02: indisponibilidade, pausa, reinício, recuperação, troca, liberação, controle de fila | 10 |
| `test_imagem_e_suavizacao.py` | RF09 (validação de imagem) e RF10 (estabilidade temporal) | 26 |
| `test_interface.py` | Tema, componentes, desenho, janela, encerramento, integração com inferência real | 11 |
| `test_integracao.py` | Integração ponta a ponta, condições do workshop, desempenho, memória, execução prolongada | 28 |

### 3.1 Mapeamento detalhado

| Exigência do escopo | Teste responsável |
|---|---|
| Importação das dependências | `test_dependencias_importam` |
| Carregamento dos modelos reais | `test_detector_carrega_de_verdade`, `test_classificador_carrega_e_aquece` |
| Integridade dos arquivos ONNX | `test_integridade_dos_arquivos_onnx` (SHA-256) |
| Pré-processamento | `test_blob_de_entrada_respeita_a_especificacao` |
| Dimensões das imagens | `test_preprocessar_redimensiona_entrada_fora_de_112`, `test_alinhamento_gera_112x112` |
| Mapeamento das 7 categorias | `test_ordem_das_classes_bate_com_o_modelo_oficial`, `test_traducao_alinhada_indice_por_indice` |
| Interpretação das saídas | `test_inferencia_real_produz_distribuicao_valida`, `test_saidas_diferem_entre_rostos_diferentes` |
| Normalização das probabilidades | `test_softmax_normaliza_e_preserva_argmax`, `test_softmax_estavel_com_valores_extremos` |
| Limites de confiança | `test_limiar_de_confianca` |
| Ausência de rosto | `test_ausencia_de_rosto_nao_classifica`, `test_entradas_degeneradas_nao_derrubam` |
| Presença de rosto | `test_deteccao_em_imagem_com_rosto` |
| Imagem inválida | 10 testes em `test_imagem_e_suavizacao.py` |
| Modelo ausente | `test_modelo_ausente_gera_mensagem_acionavel` |
| Câmera indisponível | `test_camera_indisponivel_gera_mensagem_acionavel`, `test_trabalhador_reporta_erro_sem_derrubar_a_thread` |
| Inicialização | `test_janela_abre_atualiza_e_encerra_liberando_recursos` |
| Encerramento | mesmo teste (verifica `_pipeline`, `_trabalhador` e `_foto` nulos) |
| Reinicialização da câmera | `test_recuperacao_apos_falhas_de_leitura`, `test_troca_de_camera_reinicia_o_estado_temporal` |
| Liberação de recursos | `test_captura_publica_quadros_e_libera_a_camera`, `test_parar_duas_vezes_e_seguro` |
| Integração entre componentes | `test_integracao_pipeline_completo`, `test_integracao_com_captura_simulada`, `test_janela_renderiza_resultado_de_inferencia_real` |
| Execução prolongada | `test_execucao_prolongada` (marcador `lento`) |

## 4. Condições do workshop — o que foi testado automaticamente

Os testes abaixo aplicam transformações controladas sobre a imagem de referência
e verificam que o sistema **permanece estável e numericamente válido**. Eles
**não medem acurácia sobre pessoas reais**.

| Condição | Como é simulada | Teste |
|---|---|---|
| Iluminação reduzida / excessiva | `convertScaleAbs` com 5 níveis de ganho e deslocamento | `test_condicoes_de_iluminacao` |
| Diferentes distâncias | reescala da imagem em 5 fatores (0,35× a 1,5×) | `test_diferentes_distancias_da_camera` |
| Movimentação do rosto | rotação de −20° a +20° | `test_rosto_inclinado` |
| Óculos / oclusão | faixa escura sobre a região dos olhos | `test_oclusao_parcial_tipo_oculos` |
| Ruído e desfoque | ruído uniforme e `GaussianBlur` | `test_ruido_e_desfoque_nao_quebram_o_pipeline` |
| Múltiplos rostos | imagem composta por dois rostos | `test_multiplos_rostos_nao_misturam_resultados` |
| Ausência de rosto | ruído, preto, branco, faixa estreita, 8×8 px | `test_entradas_degeneradas_nao_derrubam` |

### 4.1 Achado relevante

No teste de oclusão, com os olhos cobertos por uma faixa escura, **o YuNet deixa
de detectar o rosto** e, corretamente, nenhuma classificação é produzida. Essa é
uma limitação de detecção, não de classificação, e está registrada em
`docs/resultados.md` e na aba "Limitações" da janela "Sobre o Projeto".

## 5. O que NÃO foi testado automaticamente

Estes itens **dependem de hardware e de pessoas** e não foram executados. Não há
afirmação de aprovação sobre eles.

- Captura com webcam física real.
- Expressões reproduzidas por pessoas.
- Público diverso (idade, etnia, tom de pele, óculos reais, barba, maquiagem).
- Iluminação física do ambiente do evento.
- Sessão real de 30 minutos **com câmera ligada** (a execução prolongada
  automatizada roda o pipeline de inferência, não a captura física).
- Execução com a rede desligada (ver `docs/checklist_validacao.md`).

O checklist para essas validações presenciais está em
`docs/checklist_validacao.md`.

## 6. Protocolo de avaliação proposto (métricas próprias)

Caso se queira obter acurácia própria — não a do modelo original — o protocolo
abaixo é viável e compatível com o escopo acadêmico:

1. **Coleta.** Recrutar voluntários com consentimento informado e registrado.
   Para cada voluntário, solicitar a reprodução de cada uma das 7 expressões,
   capturando 3 amostras por expressão.
2. **Rotulagem.** O rótulo verdadeiro é a expressão **solicitada** — registrar
   que se trata de expressão encenada, não espontânea. Descartar amostras em que
   o voluntário declare não ter conseguido reproduzir a expressão.
3. **Execução.** Classificar cada amostra com `app.py --imagem`, registrando a
   classe prevista e a distribuição completa.
4. **Métricas.** Calcular matriz de confusão 7×7, acurácia global, precisão e
   revocação por classe, e a acurácia considerando apenas as previsões acima do
   limiar de confiança.
5. **Relato.** Informar sempre: número de voluntários, número de amostras,
   condição de iluminação, e o fato de as expressões serem encenadas.

> Enquanto esse protocolo não for executado, **nenhuma acurácia própria deve ser
> afirmada**. O número 88,27% pertence ao modelo original em RAF-DB e deve ser
> sempre atribuído como tal.

## 7. Distinção entre as três métricas

Confundi-las é o erro mais comum neste tipo de trabalho:

| Métrica | O que mede | Valor neste projeto |
|---|---|---|
| **Acurácia de classificação** | fração de previsões corretas sobre dados rotulados | Não medida neste trabalho. O modelo original reporta 88,27% em RAF-DB. |
| **Confiança da previsão** | probabilidade que o modelo atribui à classe vencedora | Medida e exibida em tempo real. Confiança alta **não** garante acerto. |
| **Desempenho computacional** | tempo de processamento por quadro | Medido nesta máquina: ver `evidencias/medicoes.md`. |
