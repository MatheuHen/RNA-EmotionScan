"""Janela "Sobre o Projeto" com a documentacao resumida do sistema."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from .. import config
from ..core import modelos
from . import tema


def _secao_objetivo() -> str:
    return (
        "Desenvolver uma aplicação funcional e interativa que utilize redes neurais "
        "para classificar expressões faciais capturadas pela webcam em tempo real, "
        "executando integralmente no computador local.\n\n"
        "Objetivos específicos:\n"
        "  • Detectar rostos em vídeo ao vivo com uma rede neural convolucional.\n"
        "  • Alinhar geometricamente cada rosto antes da classificação.\n"
        "  • Classificar sete expressões faciais básicas por inferência real.\n"
        "  • Apresentar as pontuações de todas as classes, e não apenas a vencedora.\n"
        "  • Medir e exibir o desempenho computacional observado.\n"
        "  • Garantir funcionamento offline após a configuração inicial."
    )


def _secao_arquitetura() -> str:
    return (
        "O sistema é dividido em camadas independentes:\n\n"
        "  1. Captura (OpenCV VideoCapture) — executa em thread própria, de modo que a\n"
        "     interface nunca fica bloqueada esperando a câmera.\n"
        "  2. Detecção facial (rede YuNet) — devolve, para cada rosto, a caixa\n"
        "     delimitadora, cinco pontos faciais de referência e uma pontuação.\n"
        "  3. Alinhamento — transformação de similaridade que leva os cinco pontos\n"
        "     detectados às posições canônicas de um recorte 112×112.\n"
        "  4. Classificação (rede MobileFaceNet / Progressive Teacher) — produz sete\n"
        "     logits, convertidos em probabilidades por softmax.\n"
        "  5. Estabilidade temporal — média móvel exponencial sobre o vetor de\n"
        "     probabilidades de cada rosto acompanhado.\n"
        "  6. Interface (Tkinter) — consome apenas o quadro mais recente publicado\n"
        "     pela thread de captura, sem fila acumulando quadros.\n\n"
        "O resultado de cada etapa alimenta exclusivamente a etapa seguinte; nenhum\n"
        "valor exibido é estimado, fixo ou aleatório."
    )
def _secao_redes() -> str:
    return (
        "Ambas as etapas usam redes neurais convolucionais pré-treinadas, executadas\n"
        "em formato ONNX pelo módulo cv2.dnn (apenas CPU).\n\n"
        "Detecção — YuNet: arquitetura anchor-free leve. A imagem completa passa por\n"
        "uma pilha de convoluções que produz, em várias escalas, mapas de confiança,\n"
        "de caixa e de pontos faciais. A supressão de não-máximos (NMS) elimina\n"
        "detecções redundantes do mesmo rosto.\n\n"
        "Classificação — MobileFaceNet treinado pelo método Progressive Teacher:\n"
        "convoluções separáveis em profundidade extraem um vetor de características\n"
        "do recorte facial alinhado; a camada final projeta esse vetor em sete\n"
        "logits, um por expressão. O softmax converte os logits em uma distribuição\n"
        "de probabilidades que soma 100%.\n\n"
        "Pré-processamento exigido pelo classificador (especificação oficial):\n"
        "  • recorte alinhado de 112×112 pixels, canais em ordem RGB;\n"
        "  • normalização (pixel ÷ 255 − 0,5) ÷ 0,5, resultando no intervalo [−1, 1];\n"
        "  • tensor de entrada 1×3×112×112 no nó de nome 'data';\n"
        "  • saída no nó 'label', com sete valores por rosto."
    )


def _secao_modelos() -> str:
    partes = [
        "Os dois modelos são PRÉ-TREINADOS e redistribuídos pelo OpenCV Zoo. "
        "Não houve treinamento, ajuste fino ou alteração de pesos neste trabalho: "
        "a contribuição do acadêmico é a aplicação completa (arquitetura de "
        "software, visão computacional, pré-processamento, interpretação de "
        "saídas, estabilidade temporal, interface, testes e documentação).\n"
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
    return "\n".join(partes)


def _secao_tecnologias() -> str:
    import sys

    import cv2 as cv
    import numpy as np
    import PIL

    return (
        f"Python {sys.version.split()[0]}\n"
        f"OpenCV {cv.__version__} — captura de vídeo e inferência ONNX (cv2.dnn)\n"
        f"NumPy {np.__version__} — operações numéricas e vetores de probabilidade\n"
        f"Pillow {PIL.__version__} — ponte entre o quadro OpenCV e o Tkinter\n"
        f"Tkinter — interface gráfica desktop (biblioteca padrão)\n"
        f"Pytest — testes automatizados\n\n"
        "TensorFlow e PyTorch NÃO são utilizados: a inferência ONNX é feita\n"
        "diretamente pelo OpenCV, o que mantém a instalação leve e o tempo de\n"
        "inicialização curto."
    )


def _secao_limitacoes() -> str:
    return (
        "As limitações abaixo são inerentes ao problema e aos modelos utilizados, e\n"
        "devem ser consideradas na leitura de qualquer resultado:\n\n"
        "  • O sistema classifica PADRÕES VISUAIS de expressão facial. Ele não mede\n"
        "    emoção, intenção ou estado psicológico, e não é instrumento de\n"
        "    diagnóstico.\n"
        "  • A acurácia publicada (88,27% em RAF-DB) refere-se ao conjunto de\n"
        "    avaliação do modelo original, em condições controladas. O desempenho\n"
        "    com webcam, iluminação variável e público diverso tende a ser inferior.\n"
        "  • Classes como Medo, Tristeza e Nojo são reconhecidamente mais difíceis de\n"
        "    distinguir do que Alegria e Surpresa, e concentram mais erros.\n"
        "  • Rostos muito inclinados, parcialmente ocluídos (máscara, mão, cabelo),\n"
        "    muito distantes ou sob contraluz podem não ser detectados.\n"
        "  • Expressões encenadas diferem de expressões espontâneas; os modelos foram\n"
        "    avaliados majoritariamente em imagens coletadas da web.\n"
        "  • A confiança exibida é a probabilidade atribuída pelo modelo à classe\n"
        "    vencedora. Confiança alta não é garantia de acerto.\n"
        "  • Há viés possível de conjunto de treinamento (distribuição de idade,\n"
        "    etnia e iluminação) que não foi auditado neste trabalho."
    )


def _secao_privacidade() -> str:
    return (
        f"{config.AVISO_PRIVACIDADE}\n\n"
        "Garantias de privacidade implementadas:\n"
        "  • A câmera só é acionada por ação explícita do usuário.\n"
        "  • Nenhuma foto ou vídeo é gravado automaticamente em disco.\n"
        "  • Nenhuma imagem sai do computador: não há chamada de rede durante a\n"
        "    execução.\n"
        "  • O sistema não identifica pessoas: não há reconhecimento biométrico de\n"
        "    identidade, cadastro ou comparação de rostos.\n"
        "  • Nenhum dado pessoal é armazenado; o estado temporal é mantido apenas em\n"
        "    memória e descartado ao encerrar."
    )


def abrir(pai: tk.Misc, fontes) -> tk.Toplevel:
    janela = tk.Toplevel(pai)
    janela.title(f"Sobre o Projeto — {config.APP_NOME}")
    janela.configure(bg=tema.FUNDO)
    janela.geometry("920x680")
    janela.minsize(640, 480)
    janela.transient(pai)

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
        ("Objetivo", _secao_objetivo()),
        ("Arquitetura", _secao_arquitetura()),
        ("Redes neurais", _secao_redes()),
        ("Modelos", _secao_modelos()),
        ("Tecnologias", _secao_tecnologias()),
        ("Limitações", _secao_limitacoes()),
        ("Privacidade", _secao_privacidade()),
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
        text="Documentação completa em docs/ (arquitetura, modelos, testes, resultados).",
        style="Subtitulo.TLabel",
    ).pack(side="left")

    janela.bind("<Escape>", lambda _e: janela.destroy())
    janela.focus_set()
    return janela
