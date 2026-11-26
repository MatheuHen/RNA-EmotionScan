# Resultados

> Texto para a coluna direita do banner. ~230 palavras.
> Todos os números foram medidos em execução real — ver `06_metricas/metricas.md`.

## Funcionalidades implementadas e verificadas

O sistema detecta rostos em tempo real, classifica as sete expressões básicas
(Raiva, Nojo, Medo, Alegria, Neutra, Tristeza e Surpresa) e exibe
simultaneamente o vídeo anotado, o gráfico das sete pontuações, o nível de
confiança e indicadores técnicos (FPS, tempo de detecção e de inferência,
quantidade de rostos). Inclui controle completo da câmera, classificação de
múltiplos rostos, análise de imagem local como contingência e modo de
apresentação em tela cheia.

## Desempenho (CPU, sem GPU dedicada)

Em AMD Ryzen 7 7435HS, com 150 amostras consecutivas sobre um rosto, o
ciclo completo de detecção e classificação levou em média **19,77 ms**,
equivalente a cerca de **51 inferências por segundo**. A detecção consome
aproximadamente quatro quintos desse tempo: o gargalo é localizar o rosto, não
reconhecer a expressão. Os modelos carregam em cerca de 120 ms.

## Verificação automatizada

A suíte executa **154 testes, todos aprovados**, utilizando os modelos ONNX
reais — mocks são empregados apenas no ciclo de vida da câmera. Em ensaio de
**30 minutos contínuos**, o sistema completou **95.811 ciclos** com variação de
apenas **+3,2%** entre a primeira e a segunda metade e memória estável em
aproximadamente 93 MB, sem degradação nem vazamento.

## Resultado mais relevante

Submetido a 14 variações controladas da mesma imagem — iluminação, escala,
rotação, ruído e oclusão — o sistema **nunca falhou nem produziu valor
inválido**: todas as distribuições somaram 1,0. Entretanto, **a classe vencedora
mudou em 4 das 12 condições em que houve detecção**, e a oclusão da região dos
olhos impediu a detecção do rosto. Essa constatação sustenta a decisão de
projeto de exibir sempre a distribuição completa das sete pontuações e o nível
de confiança, em vez de apresentar um rótulo único como verdade.

## Limitações

O sistema classifica **padrões visuais** de expressão facial; não mede emoção
nem constitui instrumento de diagnóstico. A acurácia própria de **94,6%** vale
para **56 amostras de teste de uma única pessoa**, com expressões encenadas, e
**não generaliza** para público diverso. O valor de 88,27% em RAF-DB pertence ao
modelo pré-treinado, não a este sistema. Confiança alta não é garantia de acerto.
