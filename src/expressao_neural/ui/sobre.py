"""Janela "Sobre o Projeto".

Texto em primeira pessoa, no tom de quem desenvolveu o sistema e esta
explicando para um visitante do workshop. Informal de proposito -- mas sem
abrir mao de nenhum fato: tudo o que esta escrito aqui e verificavel no
codigo, nos modelos ou nos relatorios em docs/.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from .. import config
from ..core import modelos
from . import icone, tema


def _objetivo() -> str:
    return (
        "A ideia aqui é simples de enunciar e trabalhosa de fazer: apontar a\n"
        "webcam para um rosto e dizer qual das sete expressões básicas aquele\n"
        "rosto está mostrando — na hora, sem travar e sem mandar nada para a\n"
        "internet.\n\n"
        "Escolhi esse problema porque ele junta várias coisas que eu queria\n"
        "aprender de verdade: visão computacional, inferência de rede neural,\n"
        "programação com threads e construção de interface. E porque dá para\n"
        "demonstrar ao vivo — o que é bem mais interessante do que mostrar um\n"
        "gráfico pronto.\n\n"
        "O que eu me propus a entregar:\n\n"
        "  • Detectar rostos em vídeo ao vivo usando rede neural.\n"
        "  • Alinhar cada rosto antes de classificar (isso importa mais do que\n"
        "    parece — explico na aba Redes neurais).\n"
        "  • Classificar as sete expressões com inferência real, nada de\n"
        "    resultado inventado ou aleatório.\n"
        "  • Mostrar as pontuações de TODAS as classes, não só a vencedora.\n"
        "  • Medir o desempenho de verdade, em vez de chutar.\n"
        "  • Funcionar 100% offline depois de instalado.\n\n"
        "E, principalmente: deixar claro o que o sistema NÃO sabe fazer. Isso\n"
        "está na aba Limitações, e vale a pena ler."
    )


def _arquitetura() -> str:
    return (
        "O sistema é dividido em camadas, cada uma com uma responsabilidade só:\n\n"
        "  1. Captura — lê a webcam numa thread separada. Isso é proposital: se\n"
        "     a leitura da câmera acontecesse na mesma thread da interface, ela\n"
        "     congelaria a cada quadro.\n\n"
        "  2. Detecção facial — a rede YuNet encontra os rostos e devolve, para\n"
        "     cada um, a caixa e cinco pontos de referência: os dois olhos, a\n"
        "     ponta do nariz e os dois cantos da boca.\n\n"
        "  3. Alinhamento — esses cinco pontos são levados para posições fixas\n"
        "     num recorte de 112×112. É o que corrige a inclinação da cabeça e a\n"
        "     distância da câmera.\n\n"
        "  4. Classificação — a rede MobileFaceNet olha o recorte alinhado e\n"
        "     produz sete valores, um por expressão.\n\n"
        "  5. Estabilidade temporal — uma média móvel suaviza o resultado entre\n"
        "     quadros, senão o rótulo fica piscando entre duas classes.\n\n"
        "  6. Interface — pega só o quadro mais recente e desenha.\n\n"
        "Duas decisões de projeto que eu defendo:\n\n"
        "A interface nunca espera pelo processamento. A thread de captura\n"
        "publica o resultado num único espaço, sobrescrevendo o anterior. Se a\n"
        "tela não consumir a tempo, o quadro velho é descartado. Com uma fila, a\n"
        "latência iria crescendo e a memória junto.\n\n"
        "Nenhum número na tela é estimado. FPS, tempo de detecção e tempo de\n"
        "inferência são medidos a cada quadro, não calculados por aproximação."
    )


def _redes() -> str:
    return (
        "São duas redes neurais convolucionais, uma depois da outra. Os números\n"
        "abaixo eu contei diretamente dentro dos arquivos .onnx, abrindo o\n"
        "protobuf — não são estimativas.\n\n"
        "──────────────────────────────────────────────────────────────────\n"
        "DETECÇÃO — YuNet\n"
        "──────────────────────────────────────────────────────────────────\n\n"
        "  53.121 parâmetros · 106 operações no grafo\n"
        "  53 convoluções · 15 ReLU · 6 Sigmoid · 4 MaxPool · 2 Resize\n\n"
        "É um detector anchor-free: em vez de testar caixas pré-definidas, ele\n"
        "prevê direto, em três escalas diferentes (strides 8, 16 e 32). São 12\n"
        "saídas no total — classificação, objetividade, caixa e pontos faciais,\n"
        "uma de cada por escala. Depois a supressão de não-máximos descarta as\n"
        "detecções repetidas do mesmo rosto.\n\n"
        "──────────────────────────────────────────────────────────────────\n"
        "CLASSIFICAÇÃO — MobileFaceNet (método Progressive Teacher)\n"
        "──────────────────────────────────────────────────────────────────\n\n"
        "  1.193.927 parâmetros · 97 operações no grafo\n"
        "  50 convoluções · 33 PReLU · 12 somas residuais · 1 camada densa\n\n"
        "O backbone é MobileFaceNet: convoluções separáveis em profundidade,\n"
        "feitas para rodar em celular. São 15 blocos bottleneck com conexão\n"
        "residual, que terminam num vetor de 512 dimensões descrevendo o rosto.\n"
        "A camada final transforma esses 512 números em 7.\n\n"
        "Reparei numa coisa auditando o modelo: a ativação é PReLU, não ReLU. A\n"
        "diferença é que o PReLU aprende a inclinação da parte negativa em vez\n"
        "de zerar tudo.\n\n"
        "──────────────────────────────────────────────────────────────────\n"
        "O DETALHE QUE MAIS ME CUSTOU ACERTAR\n"
        "──────────────────────────────────────────────────────────────────\n\n"
        "A saída da rede NÃO é probabilidade. São sete logits brutos — números\n"
        "que podem ser negativos e não somam nada em particular.\n\n"
        "Como sei disso com certeza? Procurei um operador Softmax dentro dos\n"
        "dois grafos ONNX e não existe nenhum. O classificador termina numa\n"
        "camada densa e para por aí. Então é o meu código que aplica o softmax,\n"
        "e é ele que transforma os logits nas porcentagens que você vê.\n\n"
        "Se a saída já fosse probabilidade e eu aplicasse softmax mesmo assim, o\n"
        "resultado ficaria achatado e errado. Por isso fui conferir em vez de\n"
        "supor.\n\n"
        "O pré-processamento também é exigente e tem que bater exatamente com o\n"
        "que o modelo espera:\n\n"
        "  • recorte alinhado de 112×112, canais em RGB (o OpenCV entrega BGR);\n"
        "  • normalização (pixel ÷ 255 − 0,5) ÷ 0,5, que joga tudo em [−1, +1];\n"
        "  • tensor 1×3×112×112 no nó chamado 'data';\n"
        "  • saída no nó 'label'.\n\n"
        "Sobre o alinhamento: testei com um estimador aproximado e com a\n"
        "transformação oficial por mínimos quadrados. Na mesma imagem, um deu\n"
        "81,3% de confiança e o outro 61,9%. A classe vencedora foi a mesma,\n"
        "mas a distribuição mudou bastante — então fiquei com a oficial."
    )


def _treinamento() -> str:
    return (
        "Essa é a parte em que eu preciso ser bem preciso, porque é fácil dizer\n"
        "mais do que se fez.\n\n"
        "──────────────────────────────────────────────────────────────────\n"
        "O QUE EU NÃO TREINEI\n"
        "──────────────────────────────────────────────────────────────────\n\n"
        "As duas redes convolucionais são PRÉ-TREINADAS, baixadas do OpenCV\n"
        "Zoo, e ficam congeladas. Não treinei nenhuma delas, não fiz fine-tuning\n"
        "e não mexi em um único peso. Dá para verificar: o SHA-256 dos dois\n"
        "arquivos .onnx continua idêntico ao do download.\n\n"
        "O classificador que uso reporta 88,27% de acurácia no conjunto RAF-DB.\n"
        "Esse número é DO MODELO ORIGINAL, não deste sistema.\n\n"
        "──────────────────────────────────────────────────────────────────\n"
        "O QUE EU TREINEI\n"
        "──────────────────────────────────────────────────────────────────\n\n"
        "Treinei uma camada de calibração em cima das saídas da rede congelada:\n\n"
        "    z' = W · z + b        W: 7×7, b: 7  →  56 parâmetros\n\n"
        "É pouca coisa — 56 números — mas é treinamento de verdade: descida de\n"
        "gradiente, entropia cruzada, conjunto de treino e de teste separados.\n\n"
        "Para ter os dados, coletei amostras minhas pela webcam:\n\n"
        "  • 210 imagens, 30 de cada uma das 7 expressões\n"
        "  • divididas em 154 para treino e 56 para teste\n"
        "  • 400 épocas, taxa de aprendizado 0,05, lote completo\n"
        "  • perda caiu de 2,0322 para 0,2088\n"
        "  • treino levou 27 milissegundos (são só 56 parâmetros)\n\n"
        "Resultado no conjunto de teste, que o treino nunca viu:\n\n"
        "                       modelo base    com calibração\n"
        "    acurácia              67,9%           94,6%\n"
        "    F1 macro              0,578           0,945\n\n"
        "Um detalhe de que gostei: inicializei W como matriz identidade e b como\n"
        "zero. Na época zero a camada é a função identidade, ou seja, o sistema\n"
        "reproduz exatamente o modelo base. A regularização penaliza o\n"
        "AFASTAMENTO da identidade. Então o treino só se afasta do modelo base\n"
        "se os dados pagarem por isso — e o ganho fica medido contra a própria\n"
        "linha de base, não contra um número de fora.\n\n"
        "──────────────────────────────────────────────────────────────────\n"
        "COMO LER ESSES 94,6%\n"
        "──────────────────────────────────────────────────────────────────\n\n"
        "Com cuidado. São 56 amostras de teste, de UMA pessoa só, com expressões\n"
        "que eu fiz de propósito na frente da câmera. Não generaliza para o\n"
        "público do workshop, e se você testar o sistema agora vai ver erro.\n\n"
        "O número que eu acho mais honesto de citar é o ganho: +26,8 pontos\n"
        "percentuais, medido no mesmo conjunto, com a mesma metodologia.\n\n"
        "A calibração vem DESLIGADA por padrão. O que você está vendo na tela é\n"
        "a saída do modelo pré-treinado, sem ajuste nenhum."
    )


def _tecnologias() -> str:
    import sys
    import tkinter

    import cv2 as cv
    import numpy as np
    import PIL

    return (
        "Tudo em Python, com o mínimo de dependência possível:\n\n"
        f"  Python {sys.version.split()[0]}\n"
        f"  OpenCV {cv.__version__} — webcam, processamento de imagem e\n"
        "                   inferência ONNX (sim, o próprio OpenCV roda o modelo)\n"
        f"  NumPy {np.__version__} — softmax, vetores de probabilidade e o\n"
        "                   treinamento da calibração\n"
        f"  Pillow {PIL.__version__} — ponte entre o quadro do OpenCV e o Tkinter\n"
        f"  Tkinter {''} — esta interface (vem junto com o Python)\n"
        "  pytest — os testes automatizados\n\n"
        "Não uso TensorFlow nem PyTorch. Essa foi uma decisão consciente: o\n"
        "OpenCV já era dependência obrigatória para capturar vídeo, e ele\n"
        "executa ONNX direto pelo cv2.dnn. Resultado prático: a instalação\n"
        "inteira dá uns 100 MB em vez de vários gigabytes, e o programa abre em\n"
        "menos de um segundo — o que faz diferença quando você abre e fecha o\n"
        "sistema trinta vezes durante um workshop.\n\n"
        "Tem até um teste automatizado que falha de propósito se alguém\n"
        "acrescentar TensorFlow, PyTorch ou onnxruntime ao projeto.\n\n"
        "Os modelos vêm do OpenCV Zoo, sob licença Apache 2.0. Cada arquivo tem\n"
        "URL de origem, tamanho e SHA-256 registrados no código, e a integridade\n"
        "é conferida em três momentos: no download, na verificação de ambiente e\n"
        "nos testes."
    )


def _modelos_detalhe() -> str:
    partes = [
        "Ficha técnica dos dois modelos. Os dois são pré-treinados por\n"
        "terceiros — ver a aba Treinamento para a distinção exata entre o que é\n"
        "deles e o que é meu.\n"
    ]
    for info in modelos.TODOS:
        partes.append(
            f"▸ {info.nome}\n"
            f"   Papel ........ {info.papel}\n"
            f"   Arquivo ...... {info.arquivo}\n"
            f"   Origem ....... {info.origem}\n"
            f"   Licença ...... {info.licenca}\n"
            f"   SHA-256 ...... {info.sha256}\n"
            f"   Arquitetura .. {info.arquitetura}\n"
            f"   Treinamento .. {info.treinamento}\n"
        )
    partes.append(
        "Sobre os dados de treinamento originais: o classificador reporta\n"
        "avaliação em RAF-DB e o detector em WIDER FACE. O FER2013, que é o\n"
        "dataset que as pessoas costumam supor nesse tipo de trabalho, NÃO\n"
        "aparece em nenhuma fonte oficial destes dois modelos — então eu não\n"
        "cito."
    )
    return "\n".join(partes)


def _limitacoes() -> str:
    return (
        "Essa é a aba que eu mais queria que fosse lida.\n\n"
        "1. O sistema classifica PADRÃO VISUAL de expressão facial. Ele não mede\n"
        "   emoção. Uma pessoa pode estar sorrindo e triste ao mesmo tempo, e o\n"
        "   sistema vai dizer 'Alegria'. Não serve para diagnóstico de nada.\n\n"
        "2. A acurácia de 94,6% que aparece na aba Treinamento vale para 56\n"
        "   amostras de teste, de uma pessoa só, com expressões encenadas. Não\n"
        "   generaliza. Com você, agora, vai errar mais.\n\n"
        "3. Óculos escuros derrubam a detecção. Testei: cobrindo a região dos\n"
        "   olhos, o YuNet simplesmente não encontra o rosto, e aí nem chega a\n"
        "   classificar. É limitação do detector, não do classificador.\n\n"
        "4. A previsão é sensível a iluminação, distância e inclinação. Rodei 14\n"
        "   variações controladas da mesma imagem e, nas 12 em que houve\n"
        "   detecção, a classe vencedora mudou em 4. Mesma pessoa, mesma foto,\n"
        "   só mudando luz e enquadramento.\n\n"
        "   É exatamente por isso que a tela mostra as sete pontuações e o nível\n"
        "   de confiança em vez de só um rótulo. Um sistema que mostrasse apenas\n"
        "   a classe vencedora pareceria mais convincente e seria menos honesto.\n\n"
        "5. Medo, Tristeza e Nojo são reconhecidamente as mais difíceis na\n"
        "   literatura. Alegria, Surpresa e Neutra são as mais confiáveis.\n\n"
        "6. Confiança alta NÃO é garantia de acerto. Num dos testes, adicionar\n"
        "   ruído na imagem fez o sistema dizer 'Surpresa 70,6%' sobre o mesmo\n"
        "   rosto que, sem ruído, dava 'Neutra 61,9%'.\n\n"
        "7. Expressão encenada é diferente de espontânea, e os modelos foram\n"
        "   avaliados principalmente em imagens coletadas da internet.\n\n"
        "8. Pode haver viés do conjunto de treinamento original — idade, etnia,\n"
        "   tom de pele, iluminação. Eu não auditei isso, então não posso\n"
        "   afirmar que não existe.\n\n"
        "9. Os tempos medidos valem para esta máquina e para um rosto. Com\n"
        "   vários rostos o custo de classificação cresce proporcionalmente."
    )


def _privacidade() -> str:
    return (
        f"{config.AVISO_PRIVACIDADE}\n\n"
        "Na prática, o que isso significa:\n\n"
        "  • A câmera só liga quando você clica em 'Iniciar câmera'. Não há\n"
        "    nada acionando a webcam sozinho.\n\n"
        "  • Nenhuma foto e nenhum vídeo são gravados. Não existe imwrite nem\n"
        "    save de imagem em nenhum lugar do código da aplicação — a única\n"
        "    coisa que o programa escreve em disco é um arquivinho com as suas\n"
        "    preferências de limiar e suavização.\n\n"
        "  • Nenhuma imagem sai do computador. Não há uma única chamada de rede\n"
        "    no código da aplicação; as únicas URLs que existem são texto de\n"
        "    documentação, nunca acessadas. Os modelos foram baixados uma vez,\n"
        "    na instalação, por um script separado.\n\n"
        "  • O sistema não identifica ninguém. Não há reconhecimento biométrico\n"
        "    de identidade, não há cadastro e não há comparação entre rostos. Ele\n"
        "    sabe que existe um rosto ali e qual expressão ele mostra — só isso.\n\n"
        "  • Nada é armazenado. O acompanhamento temporal dos rostos vive só na\n"
        "    memória e some quando você fecha o programa.\n\n"
        "As imagens que usei para treinar a camada de calibração são minhas, e\n"
        "ficam apenas neste computador — não vão para o repositório."
    )


def abrir(pai: tk.Misc, fontes) -> tk.Toplevel:
    janela = tk.Toplevel(pai)
    janela.title(f"Sobre o Projeto — {config.APP_NOME}")
    janela.configure(bg=tema.FUNDO)
    janela.geometry("980x700")
    janela.minsize(640, 480)
    janela.transient(pai)
    icone.aplicar(janela)

    # ---------------------------------------------------------------- cabecalho
    topo = tk.Frame(janela, bg=tema.SUPERFICIE)
    topo.pack(fill="x")
    interno = tk.Frame(topo, bg=tema.SUPERFICIE)
    interno.pack(fill="x", padx=22, pady=16)
    ttk.Label(
        interno,
        text=f"{config.APP_NOME} v{config.APP_VERSAO}",
        style="Cartao.TLabel",
        font=fontes.destaque_medio,
    ).pack(anchor="w")
    ttk.Label(
        interno,
        text=(
            f"{config.ACADEMICO}  ·  {config.EMAIL}\n"
            f"{config.CURSO}  ·  {config.DISCIPLINA}  ·  Prof. {config.PROFESSOR}\n"
            f"{config.EVENTO}"
        ),
        style="CartaoFraco.TLabel",
        justify="left",
    ).pack(anchor="w", pady=(6, 0))

    # ------------------------------------------------------------------- abas
    abas = ttk.Notebook(janela)
    abas.pack(fill="both", expand=True, padx=16, pady=16)

    conteudo = (
        ("A ideia", _objetivo()),
        ("Como funciona", _arquitetura()),
        ("Redes neurais", _redes()),
        ("Treinamento", _treinamento()),
        ("Modelos", _modelos_detalhe()),
        ("Tecnologias", _tecnologias()),
        ("Limitações", _limitacoes()),
        ("Privacidade", _privacidade()),
    )

    for titulo, texto in conteudo:
        moldura = tk.Frame(abas, bg=tema.SUPERFICIE)
        abas.add(moldura, text=titulo)
        barra = ttk.Scrollbar(moldura, orient="vertical")
        caixa = tk.Text(
            moldura,
            wrap="word",
            bg=tema.SUPERFICIE,
            fg=tema.TEXTO,
            insertbackground=tema.PRIMARIA,
            relief="flat",
            padx=18,
            pady=16,
            font=fontes.corpo,
            yscrollcommand=barra.set,
            spacing1=2,
            spacing3=4,
        )
        barra.configure(command=caixa.yview)
        barra.pack(side="right", fill="y")
        caixa.pack(side="left", fill="both", expand=True)
        caixa.insert("1.0", texto)
        caixa.configure(state="disabled")

    rodape = tk.Frame(janela, bg=tema.FUNDO)
    rodape.pack(fill="x", padx=16, pady=(0, 14))
    ttk.Button(rodape, text="Fechar", style="Neutro.TButton", command=janela.destroy).pack(
        side="right"
    )
    ttk.Label(
        rodape,
        text="Os números todos estão documentados em docs/ — dá para conferir tudo.",
        style="Subtitulo.TLabel",
    ).pack(side="left")

    janela.bind("<Escape>", lambda _e: janela.destroy())
    janela.focus_set()
    return janela
