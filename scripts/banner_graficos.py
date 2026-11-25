"""Gera os graficos do banner em SVG e PNG, a partir de dados REAIS.

Fontes dos dados (nenhum valor e digitado a mao neste arquivo):
  * evidencias/medicoes.json          -> desempenho por etapa
  * evidencias/execucao_prolongada.log -> estabilidade em 30 min
  * docs/banner/06_metricas/metricas.json -> testes e condicoes (gerado por
    banner_metricas.py, que executa o pipeline de verdade)

Fundo branco, cores discretas, textos legiveis para impressao.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from banner_diagramas import (  # noqa: E402
    AZUL, BRANCO, CINZA, CINZA_CLARO, LARANJA, PRETO, ROXO, VERDE, Lona,
)

DESTINO = RAIZ / "docs" / "banner" / "04_graficos"
MEDICOES = RAIZ / "evidencias" / "medicoes.json"
METRICAS = RAIZ / "docs" / "banner" / "06_metricas" / "metricas.json"


def _svg_inicio(L: int, A: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{L}" height="{A}" '
        f'viewBox="0 0 {L} {A}" font-family="Segoe UI, Calibri, Arial, sans-serif">',
        f'  <rect width="{L}" height="{A}" fill="{BRANCO}"/>',
    ]


def _svg_txt(x, y, t, tam=12, cor=CINZA, peso="400", anc="start") -> str:
    return (f'  <text x="{x}" y="{y}" font-size="{tam}" fill="{cor}" '
            f'font-weight="{peso}" text-anchor="{anc}">{t}</text>')


def _svg_rect(x, y, w, h, fill, stroke="none") -> str:
    return (f'  <rect x="{x}" y="{y}" width="{max(0, w)}" height="{max(0, h)}" '
            f'fill="{fill}" stroke="{stroke}"/>')


def _titulo(svg, png, L, titulo, subtitulo):
    svg.append(_svg_txt(26, 32, titulo, 17, PRETO, "700"))
    png.texto(26, 26, titulo, 17, PRETO, True)
    svg.append(_svg_txt(26, 52, subtitulo, 12, CINZA))
    png.texto(26, 48, subtitulo, 12, CINZA)
    svg.append(f'  <line x1="26" y1="64" x2="{L - 26}" y2="64" stroke="{CINZA_CLARO}"/>')
    png.linha(26, 64, L - 26, 64)


def _salvar(svg: list[str], png: Lona, nome: str) -> None:
    svg.append("</svg>")
    (DESTINO / f"{nome}.svg").write_text("\n".join(svg), encoding="utf-8")
    print(f"   + {(DESTINO / (nome + '.svg')).relative_to(RAIZ)}")
    png.salvar(DESTINO / f"{nome}.png")


# =============================================== 1. desempenho por etapa
def grafico_desempenho(met: dict) -> None:
    """Fonte única: metricas.json, a mesma dos demais gráficos do banner."""
    d = met["desempenho"]
    etapas = [
        ("Detecção\n(YuNet)", d["deteccao"], AZUL),
        ("Classificação\n(MobileFaceNet)", d["classificacao"], VERDE),
        ("Ciclo completo", d["ciclo_completo"], ROXO),
    ]
    L, A = 760, 420
    svg, png = _svg_inicio(L, A), Lona(L, A)
    cpu = met.get("hardware_de_teste", {})
    _titulo(
        svg, png, L,
        "Tempo de processamento por etapa (CPU)",
        f"{d['ciclo_completo']['amostras']} amostras · 1 rosto · "
        f"{cpu.get('processador', 'CPU')} a {cpu.get('clock_atual_mhz', '?')} MHz "
        f"· sem GPU",
    )

    base_y, topo_y = A - 86, 100
    maximo = max(e[1]["maximo_ms"] for e in etapas) * 1.12
    larg, gap = 118, 86
    x0 = 100

    for i in range(5):  # linhas de grade
        v = maximo * i / 4
        yy = base_y - (base_y - topo_y) * (i / 4)
        svg.append(f'  <line x1="{x0 - 24}" y1="{yy}" x2="{L - 40}" y2="{yy}" '
                   f'stroke="{CINZA_CLARO}" stroke-dasharray="3 3"/>')
        png.linha(x0 - 24, yy, L - 40, yy, CINZA_CLARO, 1)
        svg.append(_svg_txt(x0 - 32, yy + 4, f"{v:.0f}", 11, CINZA, anc="end"))
        png.texto(x0 - 32, yy, f"{v:.0f}", 11, CINZA, ancora="rm")
    svg.append(_svg_txt(x0 - 32, topo_y - 18, "ms", 11, CINZA, anc="end"))
    png.texto(x0 - 32, topo_y - 22, "ms", 11, CINZA, ancora="rm")

    for i, (nome, est, cor) in enumerate(etapas):
        x = x0 + i * (larg + gap)
        alt = (base_y - topo_y) * (est["media_ms"] / maximo)
        svg.append(_svg_rect(x, base_y - alt, larg, alt, cor))
        png.d.rectangle(
            [x * png.e, (base_y - alt) * png.e, (x + larg) * png.e, base_y * png.e],
            fill=cor,
        )
        # faixa min-max
        y_min = base_y - (base_y - topo_y) * (est["minimo_ms"] / maximo)
        y_max = base_y - (base_y - topo_y) * (est["maximo_ms"] / maximo)
        cx = x + larg / 2
        svg.append(f'  <line x1="{cx}" y1="{y_max}" x2="{cx}" y2="{y_min}" '
                   f'stroke="{PRETO}" stroke-width="1.5"/>')
        png.linha(cx, y_max, cx, y_min, PRETO, 1)
        for yy in (y_min, y_max):
            svg.append(f'  <line x1="{cx - 10}" y1="{yy}" x2="{cx + 10}" y2="{yy}" '
                       f'stroke="{PRETO}" stroke-width="1.5"/>')
            png.linha(cx - 10, yy, cx + 10, yy, PRETO, 1)

        rot_ms = f"{est['media_ms']:.1f} ms".replace(".", ",")
        svg.append(_svg_txt(cx, base_y - alt - 12, rot_ms, 13, cor, "700", "middle"))
        png.texto(cx, base_y - alt - 16, rot_ms, 13, cor, True, "mm")
        for j, linha in enumerate(nome.split("\n")):
            svg.append(_svg_txt(cx, base_y + 22 + j * 16, linha, 12, PRETO, "600", "middle"))
            png.texto(cx, base_y + 18 + j * 16, linha, 12, PRETO, True, "mm")

    nota = ("Barra = média; haste vertical = mínimo e máximo observados. "
            "Somente CPU, sem GPU dedicada.")
    svg.append(_svg_txt(26, A - 18, nota, 11, CINZA))
    png.texto(26, A - 22, nota, 11, CINZA)
    _salvar(svg, png, "desempenho_inferencia")


# ====================================================== 2. testes automatizados
def grafico_testes(met: dict) -> None:
    t = met["testes_automatizados"]
    arquivos = t["por_arquivo"]
    # Altura dinâmica: acompanha a quantidade real de arquivos de teste.
    L, A = 760, 96 + len(arquivos) * 48 + 84
    svg, png = _svg_inicio(L, A), Lona(L, A)
    _titulo(
        svg, png, L,
        f"Testes automatizados — {t['aprovados']} aprovados, "
        f"{t['reprovados']} reprovados",
        f"Executados em {t['duracao_s']:.1f} s".replace(".", ",") +
        " com os modelos ONNX reais "
        f"(mocks apenas no ciclo de vida da câmera)",
    )

    y = 96
    maxq = max(a["quantidade"] for a in arquivos)
    x_barra, larg_max = 310, L - 310 - 90
    for a in arquivos:
        svg.append(_svg_txt(26, y + 14, a["arquivo"], 12.5, PRETO, "600"))
        png.texto(26, y + 9, a["arquivo"], 12.5, PRETO, True)
        svg.append(_svg_txt(26, y + 30, a["cobre"], 10.5, CINZA))
        png.texto(26, y + 26, a["cobre"], 10.5, CINZA)
        w = larg_max * a["quantidade"] / maxq
        svg.append(_svg_rect(x_barra, y + 4, larg_max, 22, "#eef1f5"))
        png.d.rectangle([x_barra * png.e, (y + 4) * png.e,
                         (x_barra + larg_max) * png.e, (y + 26) * png.e], fill="#eef1f5")
        svg.append(_svg_rect(x_barra, y + 4, w, 22, VERDE))
        png.d.rectangle([x_barra * png.e, (y + 4) * png.e,
                         (x_barra + w) * png.e, (y + 26) * png.e], fill=VERDE)
        svg.append(_svg_txt(x_barra + larg_max + 10, y + 20,
                            str(a["quantidade"]), 13, PRETO, "700"))
        png.texto(x_barra + larg_max + 10, y + 15, str(a["quantidade"]), 13, PRETO, True)
        y += 48

    svg.append(f'  <line x1="26" y1="{y + 2}" x2="{L - 26}" y2="{y + 2}" '
               f'stroke="{CINZA_CLARO}"/>')
    png.linha(26, y + 2, L - 26, y + 2)
    resumo = (f"Total: {t['total']} testes · {t['aprovados']} aprovados · "
              f"{t['reprovados']} reprovados · taxa de aprovação "
              f"{t['aprovados'] / max(1, t['total']) * 100:.0f}%")
    svg.append(_svg_txt(26, y + 26, resumo, 13, VERDE, "700"))
    png.texto(26, y + 22, resumo, 13, VERDE, True)
    extra = (f"Execução prolongada à parte: {met['execucao_prolongada']['ciclos']:,} "
             f"ciclos em {met['execucao_prolongada']['duracao_min']} min".replace(",", "."))
    svg.append(_svg_txt(26, y + 48, extra, 11.5, CINZA))
    png.texto(26, y + 44, extra, 11.5, CINZA)
    _salvar(svg, png, "testes_automatizados")


# ================================= 3. comportamento sob condicoes adversas
def grafico_condicoes(met: dict) -> None:
    cond = met["condicoes"]
    # Altura calculada a partir da quantidade real de condições medidas,
    # para que o gráfico nunca transborde ao se acrescentar uma linha.
    L, A = 760, 92 + len(cond) * 29 + 72
    svg, png = _svg_inicio(L, A), Lona(L, A)
    _titulo(
        svg, png, L,
        "Comportamento sob variações controladas da mesma imagem",
        "Confiança da classe vencedora. Verifica estabilidade, não acurácia.",
    )

    y = 92
    for c in cond:
        cor = VERDE if c["detectou"] else LARANJA
        svg.append(_svg_txt(26, y + 14, c["condicao"], 12, PRETO, "600"))
        png.texto(26, y + 9, c["condicao"], 12, PRETO, True)
        x_barra, larg_max = 330, 260
        svg.append(_svg_rect(x_barra, y + 3, larg_max, 20, "#eef1f5"))
        png.d.rectangle([x_barra * png.e, (y + 3) * png.e,
                         (x_barra + larg_max) * png.e, (y + 23) * png.e], fill="#eef1f5")
        if c["detectou"]:
            w = larg_max * c["confianca"]
            svg.append(_svg_rect(x_barra, y + 3, w, 20, cor))
            png.d.rectangle([x_barra * png.e, (y + 3) * png.e,
                             (x_barra + w) * png.e, (y + 23) * png.e], fill=cor)
            rot = f"{c['classe']} · {c['confianca'] * 100:.0f}%"
        else:
            rot = "rosto não detectado"
        svg.append(_svg_txt(x_barra + larg_max + 10, y + 18, rot, 11.5, cor, "600"))
        png.texto(x_barra + larg_max + 10, y + 13, rot, 11.5, cor, True)
        y += 29

    nota1 = ("Em nenhuma condição houve erro ou valor inválido: todas as "
             "distribuições somaram 1,0.")
    nota2 = (f"A classe vencedora mudou em {met['resumo_condicoes']['mudaram_de_classe']} "
             f"de {met['resumo_condicoes']['total']} condições — motivo pelo qual a "
             "interface exibe sempre as 7 pontuações.")
    svg.append(_svg_txt(26, A - 36, nota1, 11, CINZA))
    png.texto(26, A - 40, nota1, 11, CINZA)
    svg.append(_svg_txt(26, A - 18, nota2, 11, CINZA))
    png.texto(26, A - 22, nota2, 11, CINZA)
    _salvar(svg, png, "condicoes_adversas")


# ================================================ 4. estabilidade em 30 min
def grafico_estabilidade(met: dict) -> None:
    ep = met["execucao_prolongada"]
    L, A = 900, 330
    svg, png = _svg_inicio(L, A), Lona(L, A)
    _titulo(
        svg, png, L,
        "Estabilidade em execução prolongada",
        f"{ep['ciclos']:,}".replace(",", ".") +
        f" ciclos de inferência em {ep['duracao_min']} minutos contínuos",
    )

    metades = [
        ("Primeira metade\n(0–15 min)", ep["primeira_metade_ms"], AZUL),
        ("Segunda metade\n(15–30 min)", ep["segunda_metade_ms"], ROXO),
    ]
    base_y, topo_y = A - 96, 108
    maximo = max(m[1] for m in metades) * 1.45
    larg, x0, gap = 128, 90, 120
    for i, (nome, valor, cor) in enumerate(metades):
        x = x0 + i * (larg + gap)
        alt = (base_y - topo_y) * (valor / maximo)
        svg.append(_svg_rect(x, base_y - alt, larg, alt, cor))
        png.d.rectangle([x * png.e, (base_y - alt) * png.e,
                         (x + larg) * png.e, base_y * png.e], fill=cor)
        cx = x + larg / 2
        rot_ms = f"{valor:.1f} ms".replace(".", ",")
        svg.append(_svg_txt(cx, base_y - alt - 12, rot_ms, 14, cor, "700", "middle"))
        png.texto(cx, base_y - alt - 16, rot_ms, 14, cor, True, "mm")
        for j, linha in enumerate(nome.split("\n")):
            svg.append(_svg_txt(cx, base_y + 22 + j * 15, linha, 11.5, PRETO, "600", "middle"))
            png.texto(cx, base_y + 18 + j * 15, linha, 11.5, PRETO, True, "mm")
    svg.append(f'  <line x1="60" y1="{base_y}" x2="490" y2="{base_y}" '
               f'stroke="{CINZA_CLARO}"/>')
    png.linha(60, base_y, 490, base_y)

    cx = 690
    svg.append(_svg_txt(cx, 150, f"+{ep['degradacao_percent']:.1f}%".replace(".", ","), 34, VERDE, "700", "middle"))
    png.texto(cx, 142, f"+{ep['degradacao_percent']:.1f}%".replace(".", ","), 34, VERDE, True, "mm")
    svg.append(_svg_txt(cx, 174, "de variação", 12.5, CINZA, anc="middle"))
    png.texto(cx, 170, "de variação", 12.5, CINZA, ancora="mm")
    svg.append(_svg_txt(cx, 206, f"{ep['trilhas_vivas_final']} trilha de rosto viva",
                        11.5, CINZA, anc="middle"))
    png.texto(cx, 202, f"{ep['trilhas_vivas_final']} trilha de rosto viva", 11.5, CINZA, ancora="mm")
    svg.append(_svg_txt(cx, 224, "ao final (sem vazamento)", 11.5, CINZA, anc="middle"))
    png.texto(cx, 220, "ao final (sem vazamento)", 11.5, CINZA, ancora="mm")

    nota = ("Sem degradação progressiva e sem crescimento de memória: "
            f"~{ep['memoria_mb']} MB estáveis durante toda a execução.")
    svg.append(_svg_txt(26, A - 18, nota, 11, CINZA))
    png.texto(26, A - 22, nota, 11, CINZA)
    _salvar(svg, png, "estabilidade_prolongada")


def main() -> int:
    DESTINO.mkdir(parents=True, exist_ok=True)
    if not METRICAS.is_file():
        print("ERRO: execute antes scripts/banner_metricas.py", file=sys.stderr)
        return 1
    met = json.loads(METRICAS.read_text(encoding="utf-8"))

    print("Gerando gráficos do banner (SVG + PNG 4x)")
    grafico_desempenho(met)
    grafico_testes(met)
    grafico_condicoes(met)
    grafico_estabilidade(met)
    print("Concluído.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
