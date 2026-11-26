# 02_capturas — PENDENTE (depende de você)

Esta pasta está vazia **de propósito**. As figuras 01 a 06 exigem um rosto
humano real diante da webcam, e não podem ser produzidas automaticamente.

**Não usar rosto de terceiros.** A imagem usada internamente nos testes
(`lena.jpg`) é de terceiros e historicamente controversa quanto a
licenciamento — não deve ir para um banner acadêmico impresso.

## Como gerar as seis figuras

```bat
cd C:\ProjetosIA\Projetos\ExpressaoNeural
.venv\Scripts\python.exe scripts\capturar_banner.py
```

A aplicação real abre em 1920x1080 com uma janela "Roteiro de captura" que
mostra, em tempo real, o que a rede está classificando:

- `PRONTO  Alegria 87%` em verde  -> está no alvo, pode capturar
- `ajuste  Neutra 61%`  em âmbar  -> ainda não está no alvo

Teclas: ENTER captura · P pula · ESC encerra.

| Figura | Arquivo gerado | O que fazer |
|---|---|---|
| 01 | figura_01_tela_principal.png | Rosto enquadrado e centralizado, câmera ativa. |
| 02 | figura_02_alegria.png | Sorria abertamente, mostrando os dentes. |
| 03 | figura_03_surpresa.png | Arregale os olhos, abra a boca, sobrancelhas levantadas. |
| 04 | figura_04_neutra.png | Rosto totalmente relaxado, sem sorrir nem franzir. |
| 05 | figura_05_painel_classificacao.png | Mantenha o rosto; salva recorte do painel lateral. |
| 06 | figura_06_sobre_o_projeto.png | Automático — a janela abre sozinha. |

## Qualidade para impressão

- Iluminação frontal; evite janela ou luz forte atrás de você.
- Fundo limpo, sem pessoas ao fundo.
- Rosto ocupando cerca de 1/3 da altura da área de vídeo.
- Notebook **na tomada** (em bateria a CPU reduz a frequência e o FPS cai).
- Feche navegador e aplicativos pesados antes de capturar.
- Pode rodar o script quantas vezes quiser até ficar bom.

## Depois de capturar

Refaça o ZIP para incluir as imagens:

```bat
powershell -NoProfile -Command "Compress-Archive -Path docs\banner\01_textos,docs\banner\02_capturas,docs\banner\03_diagramas,docs\banner\04_graficos,docs\banner\05_evidencias,docs\banner\06_metricas,docs\banner\RELATORIO_COMPLETO_BANNER.md -DestinationPath docs\banner\EmotionScan_MaterialBanner.zip -Force"
```
