# Progresso do desenvolvimento — EmotionScan

Última atualização: 26/11/2026. Versão 1.3.

## Situação

**Sistema concluído e validado.** Todas as pendências que dependiam de
validação presencial foram executadas, inclusive o teste offline.

## Números verificados

| Item | Valor | Fonte |
|---|---|---|
| Testes automatizados | **154 aprovados** | `testar.bat` |
| Ciclo completo | **21,86 ms** | `06_metricas/metricas.json` |
| Detecção / classificação | 15,44 ms / 3,39 ms | idem |
| Inferências por segundo | 51 | benchmark |
| FPS da webcam | **30,2** | medido com câmera real |
| Execução prolongada | 95.811 ciclos em 30 min, +3,2% | `evidencias/execucao_prolongada.log` |
| Acurácia própria | **67,9% → 94,6%** | `06_metricas/treinamento_calibracao.json` |
| Teste offline | **APROVADO** | `evidencias/teste_offline.md` |

## Decisões técnicas registradas

1. **Python 3.11.9**, não 3.14 — o 3.14 não tem wheels para a versão fixada do
   OpenCV.
2. **Inferência por `cv2.dnn`**, não `onnxruntime` — o OpenCV já era
   dependência obrigatória; evita vários GB.
3. **Alinhamento por mínimos quadrados**, não estimador robusto — determinístico.
   Medido: 81,3% com LMEDS contra 61,9% com a transformação oficial, na mesma
   imagem.
4. **Captura e inferência em threads separadas** — `cap.read()` bloqueia 33 ms
   esperando o quadro; inferir na mesma thread estourava a cadência da câmera e
   derrubava a taxa pela metade. Medido: 15 → 30 FPS.
5. **`inferir_a_cada = 2`** — decidido por medição, não por cálculo.
6. **Limiar 0,35 e exibição de baixa confiança** — expressões naturais vencem
   com 35–55%; escondê-las dava impressão de sistema parado.
7. **Suavização sobre o vetor completo**, não sobre o rótulo — gráfico e rótulo
   nunca divergem.
8. **Calibração desligada por padrão** — a saída exibida é a do modelo
   pré-treinado, a menos que o usuário opte conscientemente.

## Defeitos encontrados e corrigidos

| Defeito | Como apareceu | Correção |
|---|---|---|
| `pip` com `CERTIFICATE_VERIFY_FAILED` | instalação | bundle de CAs do Windows como *fallback* |
| Testes sobrescreviam preferências reais | limiar errado numa captura | `EXPRESSAO_NEURAL_SEM_PERSISTENCIA` |
| Interface borrada, captura cortada | inspeção visual | consciência de DPI |
| Emojis como caixas vazias | inspeção visual | marcadores desenhados em Canvas |
| Painel cortado em tela baixa | inspeção visual | painel rolável |
| Barra de tarefas nas capturas | inspeção visual | captura em modo apresentação |
| Rodapé contradizendo o painel | captura do banner | status reflete o quadro atual |
| **Reiniciar deixava o sistema sem imagem** | uso real | `_modo` coerente + `join` fora da thread do Tk |
| 6 achados de auditoria de segurança | revisão independente | `tests/test_robustez.py` |
| **15 FPS com câmera de 30 FPS** | capturas do banner | duas threads |
| **Backup destrutivo apagou as capturas** | conferência final | copiar em vez de mover |

## Pendências

Nenhuma técnica. Resta apenas o ensaio da apresentação — ver `TODO.md`.
