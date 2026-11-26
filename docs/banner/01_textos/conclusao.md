# Conclusão

> Pode ser incorporada ao final da seção de Resultados, conforme o modelo
> oficial. ~120 palavras.

O objetivo geral foi alcançado: o **EmotionScan** é uma aplicação funcional
que classifica expressões faciais em tempo real por meio de duas redes neurais
convolucionais reais, executando integralmente em CPU e sem conexão com a
internet, com desempenho de 20,36 ms por ciclo — folga confortável para vídeo
fluido em demonstração presencial. A verificação automatizada de 142 testes e o
ensaio de 30 minutos contínuos comprovam estabilidade e ausência de vazamento de
recursos.

A contribuição central, contudo, é metodológica. A constatação de que o mesmo
rosto produz classes distintas sob variações plausíveis de iluminação e
enquadramento levou a um projeto de interface que comunica a incerteza em vez de
ocultá-la. Delimitar com precisão o que o sistema pode e o que não pode afirmar
— separando modelos pré-treinados de terceiros da engenharia desenvolvida, e
acurácia de confiança e de desempenho — é parte do resultado deste trabalho.
