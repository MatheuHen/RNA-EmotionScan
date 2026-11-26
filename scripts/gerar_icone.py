"""Gera o icone da aplicacao a partir da marca desenhada na interface.

A marca e o mesmo grafo de neuronios que aparece no cabecalho da janela
(`ui/janela.py:_desenhar_marca`): tres nos de entrada em violeta, dois ocultos
e um de saida em ciano, sobre o fundo escuro do tema.

Produz um .ico multi-resolucao (16 a 256 px). O Windows escolhe o tamanho
conforme o contexto -- barra de tarefas, Alt+Tab, Explorer -- e um .ico com
apenas um tamanho fica serrilhado nos demais.

Uso:
    .venv\\Scripts\\python.exe scripts\\gerar_icone.py
"""
from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from PIL import Image, ImageDraw  # noqa: E402

from expressao_neural.ui import tema  # noqa: E402

DESTINO = RAIZ / "assets"
TAMANHOS = (16, 24, 32, 48, 64, 128, 256)

# Mesma topologia de ui/janela.py:_desenhar_marca, em coordenadas 0..1 para
# escalar sem perder proporcao.
NOS = [
    (0.20, 0.24), (0.20, 0.50), (0.20, 0.76),   # camada de entrada
    (0.52, 0.36), (0.52, 0.64),                 # camada oculta
    (0.83, 0.50),                               # saida
]
LIGACOES = [(0, 3), (1, 3), (1, 4), (2, 4), (3, 5), (4, 5)]

# Abaixo de 40 px os seis nos viram um borrao: nao ha pixel suficiente para
# separa-los. Esta variante mantem a leitura "rede neural" com tres nos
# maiores. E a mesma tecnica que sistemas operacionais usam em icones
# pequenos -- simplificar a forma em vez de so reduzir.
NOS_PEQUENO = [
    (0.26, 0.28), (0.26, 0.72),   # entrada
    (0.74, 0.50),                 # saida
]
LIGACOES_PEQUENO = [(0, 2), (1, 2)]
LIMITE_SIMPLIFICACAO = 40  # inclui 32 px: tamanho tipico da barra de tarefas


def desenhar(lado: int) -> Image.Image:
    """Desenha a marca num quadrado de `lado` pixels, com antisserrilhado."""
    # Supersampling 4x: o Pillow nao antisserrilha formas, entao desenhamos
    # grande e reduzimos com LANCZOS.
    s = 4
    L = lado * s
    img = Image.new("RGBA", (L, L), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    # Fundo arredondado com a cor da superfície do tema
    raio = int(L * 0.22)
    d.rounded_rectangle([0, 0, L - 1, L - 1], radius=raio, fill=tema.FUNDO)
    # Borda sutil, para o ícone não sumir sobre fundo escuro
    d.rounded_rectangle(
        [0, 0, L - 1, L - 1], radius=raio, outline=tema.BORDA_CLARA,
        width=max(1, int(L * 0.015)),
    )

    pequeno = lado < LIMITE_SIMPLIFICACAO
    nos = NOS_PEQUENO if pequeno else NOS
    ligacoes = LIGACOES_PEQUENO if pequeno else LIGACOES
    # Índice a partir do qual o nó é de saída (ciano).
    primeiro_saida = 2 if pequeno else 3
    raio_rel = 0.155 if pequeno else 0.085
    linha_rel = 0.050 if pequeno else 0.022

    pos = [(x * L, y * L) for x, y in nos]
    for a, b in ligacoes:
        d.line([pos[a], pos[b]], fill=tema.BORDA_CLARA,
               width=max(1, int(L * linha_rel)))

    raio_no = L * raio_rel
    for i, (x, y) in enumerate(pos):
        cor = tema.PRIMARIA if i >= primeiro_saida else tema.ACENTO
        d.ellipse([x - raio_no, y - raio_no, x + raio_no, y + raio_no], fill=cor)

    return img.resize((lado, lado), Image.LANCZOS)


def main() -> int:
    DESTINO.mkdir(parents=True, exist_ok=True)
    print("=" * 64)
    print(" EmotionScan — gerando o ícone da aplicação")
    print("=" * 64)

    imagens = [desenhar(n) for n in TAMANHOS]

    ico = DESTINO / "icone.ico"
    imagens[-1].save(ico, format="ICO", sizes=[(n, n) for n in TAMANHOS])
    print(f"\n   + {ico.relative_to(RAIZ)}  "
          f"({', '.join(f'{n}×{n}' for n in TAMANHOS)})")

    # PNG de 256 px: usado como fallback via iconphoto() e útil para o banner.
    png = DESTINO / "icone_256.png"
    imagens[-1].save(png, format="PNG")
    print(f"   + {png.relative_to(RAIZ)}")

    # Amostra lado a lado, para conferir legibilidade nos tamanhos pequenos
    largura = sum(TAMANHOS) + 20 * len(TAMANHOS)
    amostra = Image.new("RGB", (largura, 300), "#ffffff")
    x = 10
    for n, im in zip(TAMANHOS, imagens):
        amostra.paste(im, (x, (300 - n) // 2), im)
        x += n + 20
    prova = DESTINO / "icone_amostra.png"
    amostra.save(prova)
    print(f"   + {prova.relative_to(RAIZ)}  (conferência visual)")

    print("\n" + "=" * 64)
    print(" Ícone gerado.")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
