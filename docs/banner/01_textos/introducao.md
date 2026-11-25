# Introdução

> Texto para a coluna esquerda do banner. ~150 palavras.

As expressões faciais constituem um dos canais mais imediatos de comunicação não
verbal humana, e seu reconhecimento automático é um problema consolidado da
visão computacional. O desafio central está na variabilidade: uma mesma
expressão altera sua aparência conforme iluminação, ângulo, distância, oclusões
e características individuais do rosto.

Redes neurais convolucionais tornaram-se a abordagem predominante para esse
problema por aprenderem, diretamente dos dados, as representações visuais
relevantes — dispensando a engenharia manual de atributos.

Este trabalho apresenta o **Expressão Neural**, uma aplicação desktop que
classifica sete expressões faciais em tempo real a partir da webcam, encadeando
duas redes neurais convolucionais pré-treinadas em formato ONNX. O sistema
executa integralmente em CPU, sem GPU dedicada e sem qualquer conexão com a
internet, de modo que nenhuma imagem deixa o computador do usuário.

O objetivo é demonstrar que é viável construir um classificador de expressões
faciais funcional, estável e eticamente transparente em hardware comum.

---

## Versão curta (~75 palavras, se o espaço for reduzido)

O reconhecimento automático de expressões faciais é um problema clássico da
visão computacional, dificultado pela variabilidade de iluminação, ângulo e
distância. Este trabalho apresenta o **Expressão Neural**, aplicação desktop que
classifica sete expressões faciais em tempo real pela webcam, encadeando duas
redes neurais convolucionais pré-treinadas em formato ONNX. O sistema executa
integralmente em CPU, sem GPU e sem internet: nenhuma imagem deixa o computador
do usuário.
