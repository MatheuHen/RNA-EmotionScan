# Progresso do desenvolvimento — EmotionScan

Última atualização: 25/11/2026.

## Situação geral

Sistema **funcional e demonstrável**. Todas as 12 etapas de implementação foram
percorridas. O que resta é exclusivamente **validação presencial com hardware e
pessoas reais**, que não pode ser automatizada.

## Etapas concluídas

| # | Etapa | Situação | Evidência |
|---|---|---|---|
| 1 | Diagnóstico e configuração do ambiente | Concluída | Python 3.11.9, venv próprio, dependências fixadas |
| 2 | Arquitetura e repositório | Concluída | `docs/arquitetura.md`, Git `main` |
| 3 | Obtenção e validação dos modelos | Concluída | SHA-256 conferido; contrato extraído do código oficial |
| 4 | Visão computacional | Concluída | YuNet + alinhamento por mínimos quadrados |
| 5 | Classificação facial | Concluída | Inferência real; softmax sobre logits |
| 6 | Interface gráfica | Concluída | `evidencias/captura_interface.png` |
| 7 | Integração | Concluída | `test_integracao_pipeline_completo` |
| 8 | Validações e tratamento de falhas | Concluída | Tabela em `docs/arquitetura.md` §5 |
| 9 | Testes automatizados | Concluída | 142 aprovados, 0 falhas |
| 10 | Desempenho e offline | Concluída (offline pendente de teste manual) | `evidencias/medicoes.md` |
| 11 | Documentação acadêmica | Concluída | 6 documentos em `docs/` |
| 12 | Revisão final | Concluída | Revisão de segurança independente executada |

## Decisões técnicas registradas

1. **Python 3.11.9, não 3.14.** O 3.14 instalado na máquina não possui wheels
   compatíveis para a versão fixada do OpenCV.
2. **Inferência por `cv2.dnn`, não `onnxruntime`.** O OpenCV já é dependência
   obrigatória; evita ~300 MB e reduz o tempo de inicialização.
3. **Alinhamento por mínimos quadrados, não `estimateAffinePartial2D`.** Medido:
   o estimador robusto dava `neutral` 81,3% e a transformação oficial dá 61,9%
   na mesma imagem. A referência oficial foi mantida por fidelidade.
4. **Slot único em vez de fila de quadros.** Impede acúmulo de latência e de
   memória em execução prolongada.
5. **Suavização sobre o vetor completo de probabilidades**, não sobre o rótulo —
   o gráfico e o rótulo nunca divergem.
6. **Sem emoji na interface.** Tk 8.6 no Windows não renderiza emoji colorido;
   glifos quebrados pareceriam componente decorativo sem função.

## Problemas encontrados e corrigidos durante o desenvolvimento

| Problema | Como foi detectado | Correção |
|---|---|---|
| `pip` falhando com `CERTIFICATE_VERIFY_FAILED` | instalação das dependências | `scripts/exportar_certificados.ps1` + `pip --cert` em `configurar.bat` |
| Testes sobrescreviam `config.local.json` real | limiar errado apareceu na captura de evidência | variável `EXPRESSAO_NEURAL_SEM_PERSISTENCIA` |
| Interface borrada e captura cortada | inspeção visual da evidência | `tema.ativar_consciencia_dpi()` |
| Emojis como caixas vazias | inspeção visual da evidência | `IconeExpressao` e marcadores desenhados em Canvas |
| Painel lateral cortado em tela baixa | inspeção visual da evidência | `componentes.PainelRolavel` |
| `abrir_sobre` não devolvia a janela | teste de interface falhou | retorno adicionado |
| `-m ""` quebrava o pytest no PowerShell/cmd | execução do teste prolongado | `testar.bat` reescrito com `-m "lento or not lento"` |

## Resultados medidos

- Ciclo completo: **19,77 ms** de média (desvio padrão 1,50 ms) em CPU AMD Zen 3.
- Detecção 15,98 ms · Classificação 3,47 ms · ~51 ciclos/s.
- Carregamento dos modelos: 66,8 ms.
- Suíte automatizada: **142 testes aprovados**.

Dados brutos em `evidencias/medicoes.json`.

## Pendências reais

Todas dependem de ação presencial — ver `docs/checklist_validacao.md`:

1. Teste **offline** com rede desligada (seção B).
2. Teste com **webcam física** (seção C).
3. Teste com **pessoas reais** reproduzindo as 7 expressões (seção D).
4. Sessão de **30 minutos com a câmera ligada** (seção H).
5. Teste em **segundo notebook / outra resolução** (seção G).
6. ~~Acurácia própria~~ — **executada**: 210 amostras, 67,9% → 94,6% no
   conjunto de teste (`docs/treinamento.md`).

Nenhuma dessas foi executada. Nenhum resultado foi afirmado sobre elas.
