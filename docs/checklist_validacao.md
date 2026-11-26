# Checklist de validação presencial

Os itens abaixo **não podem ser automatizados** porque dependem de webcam física,
de pessoas reais e do ambiente do evento. Nada aqui foi executado pelo
desenvolvimento automatizado — todos estão **pendentes de validação por você**.

Marque cada item e anote o resultado. Reserve cerca de 40 minutos.

---

## A. Preparação (fazer uma vez, com internet)

- [ ] `configurar.bat` concluiu com a mensagem "CONFIGURACAO CONCLUIDA".
- [ ] `verificar.bat` terminou em "PRONTO PARA APRESENTAR" ou "PRONTO COM RESSALVAS".
- [ ] `testar.bat` terminou com todos os testes aprovados.

## B. Teste offline (o mais importante para o evento)

**Existe um script que faz isso sozinho e grava a evidência:**

```bat
offline.bat
```

Ele se recusa a rodar com a internet ligada — validar modo offline com rede
acessível não prova nada. Ao terminar, grava `evidencias/teste_offline.md`.

### Passo a passo

- [ ] **1. Desligue a rede.** Clique no ícone de rede na bandeja (canto
      inferior direito), desligue o Wi-Fi e desconecte o cabo, se houver.
- [ ] **2. Execute `offline.bat`.** Ele confirma que a rede está fora e então
      verifica, nesta ordem:
      integridade dos modelos (SHA-256) · carregamento · inferência real ·
      distribuição somando 1,0 · ausência de rosto · suíte de testes ·
      abertura da interface.
- [ ] **3. Confira o resultado.** `APROVADO` significa que o sistema funciona
      sem internet. A evidência fica em `evidencias/teste_offline.md`.

### Verificações manuais complementares

O script não consegue fazer estas — exigem você na frente da câmera:

- [ ] **Ainda sem rede**, abra `iniciar.bat` e clique em "Iniciar câmera".
      Confirme classificação em tempo real.
- [ ] Abra "Sobre o projeto" e navegue pelas 8 abas — nada faltando.
- [ ] Use "Analisar imagem" com uma foto local.
- [ ] Teste o botão "Reiniciar" — a câmera deve voltar.

- [ ] **4. RELIGUE a rede** ao terminar.

Resultado: ________________________________________________

Data e hora do teste: _____________________________________

## C. Webcam

- [ ] "Iniciar câmera" abre a imagem em até ~3 segundos.
- [ ] "Pausar" congela a imagem; o botão vira "Retomar".
- [ ] "Retomar" volta a atualizar.
- [ ] "Reiniciar" reabre a câmera sem travar a janela.
- [ ] "Encerrar" apaga a imagem e **a luz da webcam apaga**.
- [ ] "Iniciar câmera" novamente após o encerramento funciona.
- [ ] "Procurar" lista as câmeras do notebook.
- [ ] Com duas câmeras: trocar pelo seletor funciona.
- [ ] **Desconectar a webcam USB durante a execução** → mensagem de erro clara, janela não trava.
- [ ] Reconectar e usar "Reiniciar" → volta a funcionar.
- [ ] Abrir o sistema com outro programa usando a câmera (Teams/Meet) → mensagem clara, sem travamento.
- [ ] Fechar a janela pelo X → o processo encerra e a câmera é liberada.

Resultado: ________________________________________________

## D. Detecção e classificação com pessoas

Para cada item, anote a expressão prevista e a confiança.

- [ ] Rosto a ~50 cm da câmera → detectado: ______________
- [ ] Rosto a ~1,5 m → detectado? ______________
- [ ] Rosto a ~3 m → detectado? ______________
- [ ] Virar o rosto ~30° para o lado → ______________
- [ ] Inclinar a cabeça → ______________
- [ ] Mover o rosto rapidamente → há travamento? ______________
- [ ] Sair totalmente do quadro → o painel limpa e mostra "Nenhum rosto"? ______________
- [ ] Voltar ao quadro → volta a classificar? ______________

### Expressões (reproduzidas voluntariamente)

| Expressão solicitada | Previsão | Confiança | Acertou? |
|---|---|---|---|
| Alegria (sorriso aberto) | | | |
| Surpresa (olhos e boca abertos) | | | |
| Neutra | | | |
| Raiva (sobrancelhas franzidas) | | | |
| Tristeza | | | |
| Medo | | | |
| Nojo | | | |

> Esperado conforme a literatura: Alegria, Surpresa e Neutra tendem a acertar com
> mais frequência; Medo, Tristeza e Nojo são as mais difíceis. Registre o
> resultado real, inclusive os erros — eles são parte legítima do trabalho.

### Variações de pessoa

- [ ] Pessoa com óculos de grau → detecta? ______________
- [ ] Pessoa com óculos escuros → detecta? ______________ *(esperado: não detectar — ver `docs/resultados.md`)*
- [ ] Pessoa com barba → ______________
- [ ] Pessoa com franja cobrindo a testa → ______________
- [ ] Tons de pele diferentes → desempenho comparável? ______________
- [ ] Criança / pessoa idosa → ______________

### Iluminação (no local do evento, se possível)

- [ ] Luz do ambiente do workshop → ______________
- [ ] Luz fraca (parte das lâmpadas apagadas) → ______________
- [ ] Contraluz (janela atrás da pessoa) → ______________
- [ ] Luz direta no rosto → ______________

## E. Múltiplos rostos

- [ ] Duas pessoas no quadro → duas caixas, dois rótulos independentes.
- [ ] Três a cinco pessoas → desempenho permanece aceitável? FPS: ______
- [ ] Se houver queda de desempenho: desmarcar "Classificar múltiplos rostos" e confirmar melhora.

## F. Imagem estática (contingência, RF09)

- [ ] "Analisar imagem" abre o seletor de arquivos.
- [ ] Imagem JPG com rosto → classifica corretamente.
- [ ] Imagem PNG com rosto → funciona.
- [ ] Imagem sem rosto → mensagem "nenhum rosto detectado".
- [ ] Arquivo `.txt` renomeado para `.jpg` → recusado com mensagem clara.
- [ ] Arquivo muito grande (> 25 MB) → recusado com mensagem clara.

## G. Interface e apresentação

- [ ] F11 entra no modo apresentação; Esc sai.
- [ ] No modo apresentação, todos os painéis continuam legíveis.
- [ ] Barra de espaço pausa/retoma.
- [ ] F1 abre "Sobre o projeto".
- [ ] Ctrl+O abre o seletor de imagem.
- [ ] Mover o controle "Limiar de confiança" muda o comportamento visivelmente.
- [ ] Desligar "Estabilidade temporal" torna o resultado mais oscilante (comportamento esperado).
- [ ] Desligar "Espelhar imagem" inverte a imagem.
- [ ] Mudar "Inferir a cada N quadros" para 4 aumenta o FPS.
- [ ] Nenhum botão sem efeito; nenhum indicador congelado.
- [ ] Redimensionar a janela até o tamanho mínimo: o painel lateral ganha rolagem e nada fica inacessível.
- [ ] Em um segundo notebook / resolução diferente, o layout continua utilizável.

## H. Estabilidade prolongada (ensaio do evento)

- [ ] Deixar o sistema rodando **com a câmera ligada por 30 minutos seguidos**.
- [ ] Ao final: a interface continua responsiva? ______________
- [ ] FPS no início: ______ FPS após 30 min: ______
- [ ] Uso de memória no Gerenciador de Tarefas — início: ______ MB, fim: ______ MB.
- [ ] Nenhuma mensagem de erro acumulada no rodapé.
- [ ] Encerrar normalmente ao final.

> A execução prolongada automatizada (`testar.bat --lentos`) exercita o pipeline
> de inferência, mas **não** a captura física. Este item H é a validação real.

## I. Véspera da apresentação

- [ ] Notebook com carregador e tomada confirmada no local.
- [ ] Desativar suspensão automática de tela e de disco.
- [ ] Desativar notificações do Windows (Assistente de Foco).
- [ ] Fechar Teams, Zoom, Meet e qualquer app que use a câmera.
- [ ] Ter 3 a 5 imagens de contingência prontas numa pasta conhecida.
- [ ] Executar `verificar.bat` uma última vez.
- [ ] Testar o sistema no local, com a iluminação real do evento.
- [ ] Deixar `iniciar.bat` com atalho na área de trabalho.

---

## Registro de problemas encontrados

| # | O que aconteceu | Como reproduzir | Gravidade |
|---|---|---|---|
| 1 | | | |
| 2 | | | |
| 3 | | | |
