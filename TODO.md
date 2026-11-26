# TODO — EmotionScan

Apresentação: **30/11/2026**.

## 🔴 Obrigatório antes da apresentação (só você pode fazer)

Guia completo em [`docs/checklist_validacao.md`](docs/checklist_validacao.md).

- [ ] **Teste offline** — desligar Wi-Fi e cabo, executar `verificar.bat` e
      `iniciar.bat`, confirmar classificação em tempo real sem rede.
      *(Única pendência que invalida o requisito de funcionamento offline se falhar.)*
- [ ] **Teste com webcam física** — iniciar, pausar, retomar, reiniciar,
      encerrar; confirmar que a luz da câmera apaga ao encerrar.
- [ ] **Teste com pessoas** — reproduzir as 7 expressões e anotar previsão e
      confiança na tabela da seção D do checklist.
- [ ] **Sessão de 30 minutos com a câmera ligada** — anotar FPS e memória no
      início e no fim.
- [ ] **Teste com desconexão da webcam durante a execução** — confirmar mensagem
      clara e ausência de travamento.
- [ ] **Teste em outra resolução de tela** (segundo notebook ou escala do Windows).
- [ ] Preparar 3 a 5 **imagens de contingência** numa pasta conhecida.

## 🟡 Véspera do evento

- [ ] Desativar suspensão automática de tela e disco.
- [ ] Desativar notificações do Windows (Assistente de Foco).
- [ ] Fechar Teams, Zoom, Meet e qualquer programa que use a câmera.
- [ ] Executar `verificar.bat` uma última vez no local do evento.
- [ ] Testar com a **iluminação real** do local.
- [ ] Criar atalho de `iniciar.bat` na área de trabalho.
- [ ] Levar carregador e confirmar tomada.

## 🟢 Banner (modelo oficial do professor)

Conteúdo pronto em [`docs/banner.md`](docs/banner.md): textos de Introdução,
Metodologia e Resultados, caixa de destaque, paleta, diagrama e referências.

- [ ] Montar no modelo oficial — 90×120 cm, vertical, 3402×4535 px no Canva.
- [ ] Usar as imagens de `evidencias/`.
- [ ] Revisar a seção 12 de `docs/banner.md` (**o que NÃO escrever**) antes de
      finalizar.

## 🔵 Opcional — só se houver tempo e sem risco ao escopo

- [ ] **Acurácia própria**: executar o protocolo de `docs/testes.md` §6 com
      voluntários e consentimento registrado. Renderia uma matriz de confusão
      própria para o banner — mas exige tempo e pessoas.
- [ ] **Modo desafio de expressões** (item 7 do escopo original). Só avaliar
      depois que todas as validações acima estiverem fechadas; **não implementar
      se comprometer o prazo**. O reconhecimento continua sendo o objetivo
      principal.
- [ ] Capturar uma tela da interface com o seu próprio rosto para o banner
      (substituindo a imagem de referência atual).

## ⚪ Não fazer

- ❌ Publicar em repositório remoto sem autorização explícita.
- ❌ Versionar fotos de terceiros.
- ❌ Afirmar acurácia própria antes de medi-la.
- ❌ Adicionar TensorFlow, PyTorch ou onnxruntime.
- ❌ Adicionar qualquer chamada de rede dentro de `src/`.
