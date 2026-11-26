# TODO — EmotionScan

Apresentação: **30/11/2026**.

## ✅ Concluído e verificado

| Item | Evidência |
|---|---|
| Sistema funcional com as 7 expressões | `docs/banner/02_capturas/` |
| Treinamento da camada de calibração | `docs/treinamento.md` · 67,9% → 94,6% |
| Teste offline com a rede desligada | `evidencias/teste_offline.md` · APROVADO |
| Execução prolongada (30 min) | `evidencias/execucao_prolongada.log` |
| Suíte automatizada | **154 testes aprovados** |
| Material do banner | `docs/banner/EmotionScan_MaterialBanner.zip` |
| Auditoria técnica | `docs/banner/AUDITORIA_TECNICA_FINAL_EMOTIONSCAN.md` |

## 🔴 Véspera da apresentação

- [ ] Notebook **na tomada** (em bateria a CPU reduz a frequência e o FPS cai).
- [ ] Desativar suspensão automática de tela e disco.
- [ ] Desativar notificações do Windows (Assistente de Foco).
- [ ] Fechar Teams, Zoom, Meet e navegador — qualquer coisa que use a câmera
      ou consuma CPU.
- [ ] Rodar `verificar.bat` **no local**, com a iluminação real do evento.
- [ ] Atalho de `iniciar.bat` na área de trabalho.
- [ ] 3 a 5 imagens de contingência numa pasta conhecida, caso a webcam falhe.

## 🟡 Ensaio recomendado

- [ ] Testar com outras pessoas — ver seção D de `docs/checklist_validacao.md`.
      Alegria, Surpresa e Neutra são confiáveis; Medo, Tristeza e Nojo erram
      mais. Isso é esperado e está documentado como limitação.
- [ ] Deixar rodando 30 min **com a câmera ligada** (seção H).
- [ ] Testar em outra resolução de tela, se for usar projetor.

## 🟢 Banner

Material pronto em `docs/banner/`. Antes de imprimir, releia a seção 8.3 do
`RELATORIO_COMPLETO_BANNER.md` — a lista do que **não** escrever.

- [ ] Montar no modelo oficial — 90×120 cm, 3402×4535 px no Canva.

## ⚪ Nunca

- ❌ Citar **94,6%** sem o contexto "56 amostras de teste do próprio autor,
      expressões encenadas".
- ❌ Dizer "treinei uma rede neural" — treinei uma **camada de calibração de
      56 parâmetros** sobre modelos congelados.
- ❌ Citar **FER2013** — não aparece em nenhuma fonte oficial destes modelos.
- ❌ Atribuir os **88,27% em RAF-DB** a este sistema.
- ❌ Publicar em repositório remoto sem autorização.
- ❌ Versionar fotos de terceiros.
