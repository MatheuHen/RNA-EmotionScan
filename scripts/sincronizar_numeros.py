"""Reescreve os numeros citados na documentacao a partir das metricas medidas.

Existe porque sincronizar a mao nao escala: cada execucao de
`scripts/banner_metricas.py` produz valores levemente diferentes (frequencia da
CPU, carga da maquina), e a documentacao volta a divergir. Durante a auditoria
chegaram a circular TRES valores de ciclo e contagens de teste de tres epocas.

Fluxo recomendado antes de imprimir o banner:

    1. .venv\\Scripts\\python.exe scripts\\banner_metricas.py     (mede)
    2. .venv\\Scripts\\python.exe scripts\\sincronizar_numeros.py (propaga)
    3. .venv\\Scripts\\python.exe scripts\\verificar_consistencia.py (confere)

Use --simular para ver o que mudaria sem gravar nada.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

METRICAS = RAIZ / "docs" / "banner" / "06_metricas" / "metricas.json"
# A auditoria cita numeros historicos de proposito.
IGNORAR = {"AUDITORIA_TECNICA_FINAL_EMOTIONSCAN.md"}
# Arquivos gerados por script: sincronizar aqui seria sobrescrito depois.
GERADOS = {"metricas.md", "treinamento_calibracao.md"}


def _vir(valor: float, casas: int = 2) -> str:
    return f"{valor:.{casas}f}".replace(".", ",")


def documentos() -> list[Path]:
    alvos = set((RAIZ / "docs").glob("*.md")) | set(RAIZ.glob("*.md"))
    alvos |= set((RAIZ / "docs" / "banner").rglob("*.md"))
    return sorted(
        p for p in alvos if p.name not in IGNORAR and p.name not in GERADOS
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Sincroniza os números da documentação com as métricas medidas."
    )
    ap.add_argument("--simular", action="store_true",
                    help="mostra o que mudaria, sem gravar")
    args = ap.parse_args(argv)

    if not METRICAS.is_file():
        print(f"ERRO: {METRICAS} não existe. Execute scripts/banner_metricas.py.",
              file=sys.stderr)
        return 2

    d = json.loads(METRICAS.read_text(encoding="utf-8"))
    de, t = d["desempenho"], d["testes_automatizados"]
    cic = _vir(de["ciclo_completo"]["media_ms"])
    det = _vir(de["deteccao"]["media_ms"])
    clf = _vir(de["classificacao"]["media_ms"])
    dp = _vir(de["ciclo_completo"]["desvio_padrao_ms"])
    n = de["ciclo_completo"]["amostras"]
    fps = f"{de['fps_equivalente_1_rosto']:.0f}"
    aprov = t["aprovados"]

    print("=" * 68)
    print(" Sincronizando a documentação com as métricas medidas")
    print("=" * 68)
    print(f"\n  ciclo {cic} ms | detecção {det} ms | classificação {clf} ms")
    print(f"  {fps} inferências/s | {aprov} testes | {n} amostras\n")

    # (regex, funcao que monta a substituicao a partir do match)
    regras: list[tuple[str, object]] = [
        # ciclo em destaque: **NN,NN ms**
        (r"\*\*(\d+,\d+)\s*ms\*\*(?=\s*(?:em m[ée]dia|\||$))",
         lambda m: f"**{cic} ms**"),
        (r"(?<=em m[ée]dia )\*\*(\d+,\d+)\s*ms\*\*", lambda m: f"**{cic} ms**"),
        # "ciclo completo ... NN,NN ms"
        (r"(ciclo completo[^.\n|]{0,70}?)(\d+,\d+)(\s*ms)",
         lambda m: f"{m.group(1)}{cic}{m.group(3)}"),
        (r"(ciclo de )(\d+,\d+)(\s*ms)", lambda m: f"{m.group(1)}{cic}{m.group(3)}"),
        (r"(\d+,\d+)(\s*ms de m[ée]dia)", lambda m: f"{cic}{m.group(2)}"),
        # taxa
        (r"~?\s*\d+(?:,\d+)?\s*(inferências|ciclos)\s*(por segundo|/s)",
         lambda m: f"{fps} {m.group(1)} {m.group(2)}"
                   if m.group(2) == "por segundo" else f"~{fps} {m.group(1)}/s"),
        # contagem de testes: só formas inequívocas, para não pegar
        # contagens por arquivo numa tabela
        (r"\b\d+\s+testes\s+automatizados", lambda m: f"{aprov} testes automatizados"),
        (r"\*\*\d+\s+testes\*\*", lambda m: f"**{aprov} testes**"),
        (r"\*\*\d+\s+testes,\s*todos aprovados\*\*",
         lambda m: f"**{aprov} testes, todos aprovados**"),
        (r"\b\d+\s+testes,\s*todos aprovados",
         lambda m: f"{aprov} testes, todos aprovados"),
        (r"\b\d+\s+aprovados\s*·\s*\d+\s+reprovados",
         lambda m: f"{aprov} aprovados · 0 reprovados"),
        (r"Suíte:\s*\d+\s+aprovados", lambda m: f"Suíte: {aprov} aprovados"),
        # amostras e desvio
        (r"\d+\s+amostras consecutivas", lambda m: f"{n} amostras consecutivas"),
        (r"\(desvio padrão \d+,\d+ ms\)", lambda m: f"(desvio padrão {dp} ms)"),
        # componentes
        (r"(Detecção \(YuNet\)[^|\n]*\|\s*\*{0,2})(\d+,\d+)(\s*ms)",
         lambda m: f"{m.group(1)}{det}{m.group(3)}"),
        (r"(Classificação \(MobileFaceNet\)[^|\n]*\|\s*\*{0,2})(\d+,\d+)(\s*ms)",
         lambda m: f"{m.group(1)}{clf}{m.group(3)}"),
        (r"A detecção\s*\n?consome (\d+,\d+) ms e a classificação apenas (\d+,\d+) ms",
         lambda m: f"A detecção consome {det} ms e a classificação apenas {clf} ms"),
    ]

    total = 0
    for doc in documentos():
        s0 = s = doc.read_text(encoding="utf-8")
        mudancas = 0
        for padrao, repo in regras:
            s, k = re.subn(padrao, repo, s, flags=re.IGNORECASE)
            mudancas += k
        if s != s0:
            if not args.simular:
                doc.write_text(s, encoding="utf-8")
            print(f"  {'(simulado) ' if args.simular else ''}"
                  f"{doc.relative_to(RAIZ)}: {mudancas} ajuste(s)")
            total += mudancas

    print(f"\n{total} ajuste(s) " + ("simulado(s)" if args.simular else "aplicado(s)"))
    if not args.simular:
        print("\nConfira com: .venv\\Scripts\\python.exe scripts\\verificar_consistencia.py")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
