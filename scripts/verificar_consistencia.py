"""Confere se os numeros citados na documentacao batem com os medidos.

Motivacao concreta: durante a auditoria apareceram TRES valores diferentes de
ciclo circulando pelo repositorio (17,15 / 18,78 / 26,19 ms) e contagens de
teste de tres epocas distintas (108, 119, 137). Para quem le, documento
desatualizado e indistinguivel de documento errado.

Este script compara a documentacao contra a fonte unica de verdade
(docs/banner/06_metricas/metricas.json) e devolve codigo 1 se houver
divergencia.

A comparacao e NUMERICA, com tolerancia: a medicao flutua naturalmente entre
execucoes (frequencia da CPU, carga da maquina). Uma lista fixa de valores
"proibidos" ficaria desatualizada e o verificador cegaria.

Uso:
    .venv\\Scripts\\python.exe scripts\\verificar_consistencia.py
    .venv\\Scripts\\python.exe scripts\\verificar_consistencia.py --tolerancia 0.2
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]

if hasattr(sys.stdout, "reconfigure"):   # console cp1252 nao suporta ×, ≤
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

METRICAS = RAIZ / "docs" / "banner" / "06_metricas" / "metricas.json"
TREINO = RAIZ / "docs" / "banner" / "06_metricas" / "treinamento_calibracao.json"

# A auditoria registra numeros historicos de proposito, para documentar o que
# mudou. Incluí-la geraria falso positivo em toda execucao.
IGNORAR = {"AUDITORIA_TECNICA_FINAL_EMOTIONSCAN.md"}


def _vir(valor: float, casas: int = 2) -> str:
    return f"{valor:.{casas}f}".replace(".", ",")


def documentos() -> list[Path]:
    alvos = set((RAIZ / "docs").glob("*.md")) | set(RAIZ.glob("*.md"))
    alvos |= set((RAIZ / "docs" / "banner").rglob("*.md"))
    return sorted(p for p in alvos if p.name not in IGNORAR)


def _linha(texto: str, posicao: int) -> int:
    return texto[:posicao].count("\n") + 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Confere a documentação contra as métricas medidas."
    )
    ap.add_argument(
        "--tolerancia", type=float, default=0.12,
        help="divergência aceita antes de acusar (padrão: 0,12 = 12%%)",
    )
    args = ap.parse_args(argv)
    tol = max(0.01, args.tolerancia)

    if not METRICAS.is_file():
        print(f"ERRO: {METRICAS} não existe. Execute scripts/banner_metricas.py.",
              file=sys.stderr)
        return 2

    d = json.loads(METRICAS.read_text(encoding="utf-8"))
    de, t = d["desempenho"], d["testes_automatizados"]
    cic = de["ciclo_completo"]["media_ms"]
    det = de["deteccao"]["media_ms"]
    clf = de["classificacao"]["media_ms"]
    fps = de["fps_equivalente_1_rosto"]
    aprov = float(t["aprovados"])

    print("=" * 68)
    print(" Consistência da documentação x métricas medidas")
    print("=" * 68)
    print(f"\nFonte: {METRICAS.relative_to(RAIZ)}")
    print(f"  ciclo {_vir(cic)} ms | detecção {_vir(det)} ms | "
          f"classificação {_vir(clf)} ms | {fps:.0f}/s | {aprov:.0f} testes")
    print(f"  tolerância: {tol:.0%}\n")

    problemas: list[str] = []

    # ------------------------------------------------ 1. coerencia interna
    print("1. Coerência interna das métricas")
    if det + clf > cic + 0.5:
        msg = (f"detecção + classificação ({_vir(det + clf)} ms) excede o "
               f"ciclo ({_vir(cic)} ms)")
        problemas.append(msg)
        print(f"   [FALHA] {msg}")
    else:
        print(f"   [ OK ] detecção + classificação = {_vir(det + clf)} ms, "
              f"dentro do ciclo de {_vir(cic)} ms")

    esperado = 1000.0 / cic
    if abs(esperado - fps) > 1.0:
        msg = (f"taxa declarada ({fps:.1f}/s) não corresponde a "
               f"1000/{_vir(cic)} = {esperado:.1f}/s")
        problemas.append(msg)
        print(f"   [FALHA] {msg}")
    else:
        print(f"   [ OK ] {fps:.1f}/s corresponde a 1000 / {_vir(cic)} ms")

    # ------------------------------------- 2. numeros divergentes nos docs
    print("\n2. Números citados na documentação")

    def fora(valor: float, referencia: float) -> bool:
        return abs(valor - referencia) > referencia * tol

    # (regex com UM grupo numérico, referência, rótulo)
    numericos: list[tuple[str, float, str]] = [
        (r"ciclo[^.\n|]{0,70}?(\d+,\d+)\s*ms", cic, "ciclo"),
        (r"(\d+,\d+)\s*ms\s*(?:em\s+)?(?:de\s+)?m[ée]dia", cic, "ciclo médio"),
        (r"~?\s*(\d+)(?:,\d+)?\s*(?:inferências|ciclos)\s*(?:por segundo|/s)",
         fps, "taxa de inferência"),
        # Exige duas casas e forma inequívoca: "10 testes em test_camera.py" é
        # contagem por arquivo numa tabela, legítima, e não deve ser acusada.
        (r"\b(\d{2,})\s+testes\s+automatizados", aprov, "contagem de testes"),
        (r"\*\*(\d{2,})\s+testes", aprov, "contagem de testes"),
        (r"\b(\d{2,})\s+testes,\s*todos aprovados", aprov, "contagem de testes"),
    ]
    # Afirmações literais que não dependem de medição
    literais: list[tuple[str, str]] = [
        (r"α\s*=\s*0,40", "alfa da suavização (atual: 0,55)"),
        (r"limiar de confiança[^.\n]{0,24}0,50", "limiar de confiança (atual: 0,35)"),
        (r"`inferir_a_cada = 2`", "passo de inferência (atual: 1)"),
        (r"com `inferir_a_cada = 2`", "passo de inferência (atual: 1)"),
    ]

    achados = 0
    for doc in documentos():
        texto = doc.read_text(encoding="utf-8")
        for padrao, ref, rotulo in numericos:
            for m in re.finditer(padrao, texto, re.IGNORECASE):
                try:
                    valor = float(m.group(1).replace(",", "."))
                except (ValueError, IndexError):
                    continue
                if fora(valor, ref):
                    print(f"   [FALHA] {doc.relative_to(RAIZ)}:{_linha(texto, m.start())}"
                          f" - {m.group(0).strip()!r} | {rotulo} "
                          f"(medido: {_vir(ref)})")
                    problemas.append(f"{doc.name} {rotulo}")
                    achados += 1
        for padrao, descricao in literais:
            for m in re.finditer(padrao, texto):
                print(f"   [FALHA] {doc.relative_to(RAIZ)}:{_linha(texto, m.start())}"
                      f" - {m.group(0)!r} | {descricao}")
                problemas.append(f"{doc.name} {descricao}")
                achados += 1
    if not achados:
        print(f"   [ OK ] nenhum número divergente além da tolerância")

    # --------------------------------- 3. afirmacoes sobre o treinamento
    print("\n3. Afirmações sobre treinamento")
    if TREINO.is_file():
        frases = [
            "Nenhum treinamento foi realizado neste trabalho",
            "Nenhum treinamento, ajuste fino ou alteração de pesos foi realizado",
            "A acurácia própria não foi avaliada",
            "acurácia deste sistema **não foi medida**",
        ]
        encontrou = False
        for doc in documentos():
            texto = doc.read_text(encoding="utf-8")
            for frase in frases:
                pos = texto.find(frase)
                if pos >= 0:
                    print(f"   [FALHA] {doc.relative_to(RAIZ)}:{_linha(texto, pos)}"
                          f" - nega treinamento, mas a calibração foi treinada")
                    problemas.append(f"{doc.name} nega treinamento")
                    encontrou = True
        if not encontrou:
            print("   [ OK ] nenhuma negação incorreta de treinamento")
    else:
        print("   [ -- ] calibração ainda não treinada; verificação pulada")

    print("\n" + "=" * 68)
    if problemas:
        print(f" RESULTADO: {len(problemas)} divergência(s)")
        print(" Regenere as métricas e sincronize os documentos antes de imprimir.")
        print("=" * 68)
        return 1
    print(" RESULTADO: documentação consistente com as métricas medidas")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
