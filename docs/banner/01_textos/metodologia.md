# Metodologia

> Texto para a coluna esquerda do banner, acompanhando o diagrama técnico.
> ~210 palavras.

O sistema encadeia duas redes neurais convolucionais, ambas executadas em
formato ONNX pelo módulo `cv2.dnn` do OpenCV, somente em CPU.

**Detecção facial.** A rede **YuNet** processa cada quadro e devolve, para cada
rosto, a caixa delimitadora, cinco pontos faciais de referência (olhos, ponta do
nariz e cantos da boca) e uma pontuação de confiança. Trata-se de uma
arquitetura *anchor-free* leve, que opera em múltiplas escalas e aplica
supressão de não-máximos para eliminar detecções redundantes.

**Alinhamento.** Os cinco pontos detectados alimentam uma transformação de
similaridade não reflexiva, resolvida por mínimos quadrados, que leva o rosto às
posições canônicas de um recorte de 112×112 pixels. Essa etapa é determinística
e reproduz a especificação oficial do modelo.

**Pré-processamento.** O recorte é convertido de BGR para RGB e normalizado pela
expressão (pixel ÷ 255 − 0,5) ÷ 0,5, resultando no intervalo [−1, +1], e
organizado no tensor 1×3×112×112 exigido pelo nó de entrada `data`.

**Classificação.** A rede **MobileFaceNet**, treinada pelo método
semi-supervisionado *Progressive Teacher*, utiliza convoluções separáveis em
profundidade para extrair um vetor de características do rosto alinhado. A
camada final projeta esse vetor em sete logits, convertidos pela função softmax
em probabilidades que somam 100%.

**Estabilidade temporal.** Uma média móvel exponencial aplicada ao vetor completo
de probabilidades, por rosto acompanhado, reduz oscilações entre quadros
consecutivos sem alterar a classe vencedora.

**Arquitetura de software.** A captura e a inferência executam em thread
dedicada, publicando apenas o quadro mais recente em um compartimento único; a
interface Tkinter consome esse resultado sem jamais bloquear. Essa separação
mantém a aplicação responsiva e impede o acúmulo de quadros em memória.

---

## Declaração obrigatória

Os dois modelos são **pré-treinados por terceiros** (OpenCV Zoo, licença Apache
2.0) e permanecem **congelados**: nenhum peso foi alterado e **não houve
ajuste fino (*fine-tuning*)**. Houve treinamento próprio de uma **camada de
calibração supervisionada de 56 parâmetros** sobre as saídas do classificador,
com 210 amostras rotuladas coletadas pelo autor (154 treino / 56 teste, 400
épocas), elevando a acurácia no conjunto de teste de 67,9% para 94,6%. A contribuição do acadêmico é a engenharia completa da aplicação:
arquitetura, pipeline de visão computacional, implementação fiel do
pré-processamento, interpretação das saídas, estabilidade temporal, interface,
tratamento de falhas, verificação de integridade dos modelos, suíte de testes
automatizados com inferência real e documentação técnica.
