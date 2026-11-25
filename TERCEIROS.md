# Licenças e referências de terceiros

Este projeto utiliza software e modelos de terceiros. Nenhum deles foi
modificado. As licenças abaixo foram conferidas nas fontes oficiais.

## Modelos neurais (não versionados neste repositório)

Os arquivos ONNX são obtidos por `configurar.bat` a partir das fontes oficiais e
verificados por tamanho e SHA-256 antes do uso.

### YuNet — `face_detection_yunet_2023mar.onnx`

- Projeto: OpenCV Zoo
- Licença: **Apache License 2.0**
- Fonte: https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
- Espelho: https://huggingface.co/opencv/face_detection_yunet
- SHA-256: `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`

### Progressive Teacher / MobileFaceNet — `facial_expression_recognition_mobilefacenet_2022july.onnx`

- Projeto: OpenCV Zoo
- Licença: **Apache License 2.0**
- Copyright: Shenzhen Institute of Artificial Intelligence and Robotics for Society
- Fonte: https://github.com/opencv/opencv_zoo/tree/main/models/facial_expression_recognition
- Espelho: https://huggingface.co/opencv/facial_expression_recognition
- SHA-256: `4f61307602fc089ce20488a31d4e4614e3c9753a7d6c41578c854858b183e1a9`

## Algoritmo de alinhamento facial

`src/expressao_neural/core/alinhamento.py` é um porte direto da classe
`FaceAlignment` de `opencv_zoo/models/facial_expression_recognition/facial_fer_model.py`,
sob **Apache License 2.0**, mantendo o comportamento numérico original. O porte
foi feito para garantir que o pré-processamento seja idêntico ao esperado pelo
modelo. A atribuição também consta no cabeçalho do arquivo.

## Bibliotecas Python

| Biblioteca | Versão | Licença | Fonte |
|---|---|---|---|
| opencv-python | 4.12.0.88 | Apache License 2.0 (MIT para os wrappers) | https://github.com/opencv/opencv-python |
| numpy | 2.2.6 | BSD 3-Clause | https://github.com/numpy/numpy |
| pillow | 11.3.0 | MIT-CMU | https://github.com/python-pillow/Pillow |
| pytest | 8.4.2 | MIT | https://github.com/pytest-dev/pytest |
| Tkinter / Tcl-Tk | 8.6 | Tcl/Tk License (estilo BSD) | biblioteca padrão do Python |

## Imagem de teste

`tests/fixtures/lena.jpg` provém do repositório de amostras do OpenCV
(`opencv/samples/data/lena.jpg`), usado aqui exclusivamente como fixture
determinística de teste automatizado e geração de evidências. É uma imagem
historicamente empregada em literatura de processamento de imagens.

## Texto completo das licenças

- Apache License 2.0: https://www.apache.org/licenses/LICENSE-2.0
- BSD 3-Clause: https://opensource.org/license/bsd-3-clause
- MIT: https://opensource.org/license/mit
