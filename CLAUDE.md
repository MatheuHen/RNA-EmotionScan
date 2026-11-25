# Expressão Neural — instruções do projeto

Projeto acadêmico isolado. **Este arquivo é a fonte de verdade do projeto** e
prevalece sobre qualquer configuração global.

## Identidade

Aplicação desktop Python/Tkinter que classifica 7 expressões faciais em tempo
real pela webcam, usando dois modelos ONNX pré-treinados executados por
`cv2.dnn`. 100% offline, somente CPU.

Acadêmico: Matheus Henrique de Araújo Carneiro — Engenharia de Software, UniRV.
Disciplina: Redes Neurais Artificiais. Apresentação: 30/11/2026.

## Stack e comandos

| Ação | Comando |
|---|---|
| Interpretador | `.venv\Scripts\python.exe` (Python 3.11.9) |
| Testes | `testar.bat` ou `.venv\Scripts\python.exe -m pytest -m "not lento"` |
| Testes completos | `testar.bat --lentos` |
| Verificação | `verificar.bat` |
| Executar | `iniciar.bat` ou `.venv\Scripts\python.exe app.py` |
| Evidências | `.venv\Scripts\python.exe scripts\gerar_evidencias.py` |

## Regras inegociáveis deste projeto

1. **Nunca simular inferência.** Nada de valor fixo, aleatório ou heurística
   apresentada como rede neural. Se algo não puder ser inferido, não exibir.
2. **Nunca afirmar acurácia própria.** Os 88,27% são do modelo original em
   RAF-DB. Acurácia deste sistema **não foi medida** — ver `docs/testes.md` §6.
3. **Nunca declarar que os modelos foram treinados aqui.** São pré-treinados
   (OpenCV Zoo, Apache 2.0).
4. **Nunca adicionar chamada de rede em `src/`.** Downloads só em
   `scripts/baixar_modelos.py`, executado uma vez por `configurar.bat`.
   O modo offline depende disso.
5. **Nunca adicionar TensorFlow, PyTorch ou onnxruntime.** O teste
   `test_nao_usa_bibliotecas_pesadas` falha se isso acontecer.
6. **Nunca alterar a ordem das 7 classes** em `config.CLASSES_MODELO`: ela
   corresponde aos índices de saída da rede.
7. **Nunca gravar imagem de pessoa automaticamente** nem versionar fotos de
   terceiros.
8. **Inferência jamais na thread do Tkinter.** Todo processamento fica em
   `TrabalhadorCamera`.
9. **Nunca enfraquecer teste para obter verde.** Se um teste falhar, corrigir a
   causa.
10. Scripts `.bat` devem funcionar de qualquer diretório (`%~dp0`).

## Arquitetura resumida

```
camera.py (thread) → pipeline.py → detector.py (YuNet)
                                 → alinhamento.py (112×112)
                                 → classificador.py (softmax)
                                 → suavizacao.py (EMA)
                   → slot único → ui/janela.py (thread Tk)
```

Detalhes em `docs/arquitetura.md`.

## Particularidades do ambiente

- A rede local faz **inspeção de TLS**: o `pip` falha com
  `CERTIFICATE_VERIFY_FAILED`. `configurar.bat` resolve exportando as CAs do
  Windows via `scripts/exportar_certificados.ps1` e usando `pip --cert`.
- O `.pem` gerado **não é versionado**.
- Tk 8.6 no Windows não renderiza emoji colorido — usar marcadores desenhados em
  Canvas, não caracteres emoji.

## Documentação

`docs/arquitetura.md` · `docs/modelos.md` · `docs/testes.md` ·
`docs/resultados.md` · `docs/checklist_validacao.md` · `docs/banner.md`
