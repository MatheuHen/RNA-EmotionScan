# Referências

> Padrão ABNT (NBR 6023). Todas as referências foram verificadas nas fontes
> originais durante o desenvolvimento. Nenhuma foi inventada.

## Seleção recomendada para o banner (6 referências)

O espaço de um banner costuma comportar de 5 a 7 referências. Estas são as
diretamente ligadas ao que foi efetivamente implementado:

CHEN, Sheng; LIU, Yang; GAO, Xiang; HAN, Zhen. **MobileFaceNets**: efficient
CNNs for accurate real-time face verification on mobile devices. In: CHINESE
CONFERENCE ON BIOMETRIC RECOGNITION (CCBR), 13., 2018, Urumqi. *Lecture Notes
in Computer Science*, v. 10996. Cham: Springer, 2018. p. 428-438. DOI:
10.1007/978-3-319-97909-0_46. Disponível em:
https://link.springer.com/chapter/10.1007/978-3-319-97909-0_46. Acesso em: 25
nov. 2026.

JIANG, Jing; DENG, Weihong. **Boosting facial expression recognition by a
semi-supervised progressive teacher**. *IEEE Transactions on Affective
Computing*, v. 13, n. 4, p. 2026-2038, 2022. DOI: 10.1109/TAFFC.2021.3131621.
Disponível em: https://ieeexplore.ieee.org/abstract/document/9629313. Acesso em:
25 nov. 2026.

LI, Shan; DENG, Weihong; DU, JunPing. **Reliable crowdsourcing and deep
locality-preserving learning for expression recognition in the wild**. In: IEEE
CONFERENCE ON COMPUTER VISION AND PATTERN RECOGNITION (CVPR), 2017, Honolulu.
*Proceedings* [...]. Honolulu: IEEE, 2017. p. 2584-2593. DOI:
10.1109/CVPR.2017.277. Disponível em: http://whdeng.cn/RAF/model1.html. Acesso
em: 25 nov. 2026.

WU, Wei; PENG, Hanyang; YU, Shiqi. **YuNet**: a tiny millisecond-level face
detector. *Machine Intelligence Research*, v. 20, n. 5, p. 656-665, 2023. DOI:
10.1007/s11633-023-1423-y. Disponível em:
https://link.springer.com/article/10.1007/s11633-023-1423-y. Acesso em: 25 nov.
2026.

OPENCV. **OpenCV Zoo**: a zoo for models tuned for OpenCV DNN with
benchmarks. [S. l.], 2023. Disponível em: https://github.com/opencv/opencv_zoo.
Acesso em: 25 nov. 2026.

BRADSKI, Gary. **The OpenCV Library**. *Dr. Dobb's Journal of Software Tools*,
v. 25, n. 11, p. 120-125, 2000. Disponível em: https://opencv.org. Acesso em: 25
nov. 2026.

---

## Lista completa (12 referências, para a documentação)

As seis acima, mais:

OPENCV. **Facial expression recognition**: Progressive Teacher. OpenCV Zoo.
[S. l.], 2022. Disponível em:
https://github.com/opencv/opencv_zoo/tree/main/models/facial_expression_recognition.
Acesso em: 25 nov. 2026.

OPENCV. **Face detection YuNet**. OpenCV Zoo. [S. l.], 2023. Disponível em:
https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet.
Acesso em: 25 nov. 2026.

OPENCV. **cv::FaceDetectorYN Class Reference**. OpenCV Documentation, v. 4.x.
[S. l.], 2024. Disponível em:
https://docs.opencv.org/4.x/df/d20/classcv_1_1FaceDetectorYN.html. Acesso em: 25
nov. 2026.

OPENCV. **Deep Neural Network module (dnn)**. OpenCV Documentation, v. 4.x.
[S. l.], 2024. Disponível em:
https://docs.opencv.org/4.x/d6/d0f/group__dnn.html. Acesso em: 25 nov. 2026.

HARRIS, Charles R. *et al.* **Array programming with NumPy**. *Nature*, v. 585,
n. 7825, p. 357-362, 2020. DOI: 10.1038/s41586-020-2649-2. Disponível em:
https://www.nature.com/articles/s41586-020-2649-2. Acesso em: 25 nov. 2026.

YANG, Shuo; LUO, Ping; LOY, Chen Change; TANG, Xiaoou. **WIDER FACE**: a face
detection benchmark. In: IEEE CONFERENCE ON COMPUTER VISION AND PATTERN
RECOGNITION (CVPR), 2016, Las Vegas. *Proceedings* [...]. Las Vegas: IEEE, 2016.
p. 5525-5533. DOI: 10.1109/CVPR.2016.596. Disponível em:
http://shuoyang1213.me/WIDERFACE/. Acesso em: 25 nov. 2026.

---

## Observações sobre a atribuição correta

- **RAF-DB** (LI; DENG; DU, 2017) é o conjunto em que o **modelo pré-treinado**
  de classificação reporta 88,27% de acurácia. Esse conjunto **não foi usado
  neste trabalho** para treinar ou avaliar nada; a referência existe para
  atribuir corretamente a procedência do número.
- **WIDER FACE** (YANG *et al.*, 2016) é o benchmark citado pelo OpenCV Zoo na
  avaliação do YuNet. Igualmente, não foi utilizado neste trabalho.
- **FER2013 não é citado** em nenhuma das fontes oficiais consultadas sobre os
  modelos empregados. Por isso **não deve aparecer** no banner.
