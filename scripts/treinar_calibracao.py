"""Treina a camada de calibracao sobre as amostras coletadas.

Fluxo real de treinamento supervisionado:
  1. le as amostras rotuladas de dados/amostras/<classe>/
  2. extrai os 7 logits de cada uma com o modelo pre-treinado (congelado)
  3. divide em treino e teste, de forma estratificada
  4. treina W (7x7) e b (7) por descida de gradiente com entropia cruzada
  5. avalia ANTES e DEPOIS no MESMO conjunto de teste
  6. grava os pesos, o relatorio e a matriz de confusao

Uso:
    .venv\\Scripts\\python.exe scripts\\treinar_calibracao.py
    .venv\\Scripts\\python.exe scripts\\treinar_calibracao.py --epocas 600 --teste 0.3
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

import cv2 as cv  # noqa: E402
import numpy as np  # noqa: E402

from expressao_neural import config  # noqa: E402
from expressao_neural.core import calibracao as mod_cal  # noqa: E402
from expressao_neural.core.classificador import ClassificadorExpressao  # noqa: E402
from expressao_neural.core.detector import ErroModelo  # noqa: E402

AMOSTRAS = RAIZ / "dados" / "amostras"
SAIDA = RAIZ / "docs" / "banner" / "06_metricas"
GRAFICOS = RAIZ / "docs" / "banner" / "04_graficos"
PT_PARA_INDICE = {nome: i for i, nome in enumerate(config.CLASSES_PT)}


def carregar_amostras() -> tuple[np.ndarray, np.ndarray, dict]:
    """Extrai os logits de cada amostra rotulada com o modelo congelado."""
    if not AMOSTRAS.is_dir():
        print(f"ERRO: {AMOSTRAS} não existe.\n"
              f"Execute antes: .venv\\Scripts\\python.exe scripts\\coletar_amostras.py",
              file=sys.stderr)
        raise SystemExit(2)

    clf = ClassificadorExpressao()
    clf.aquecer()

    logits: list[np.ndarray] = []
    rotulos: list[int] = []
    contagem: dict[str, int] = {}

    for pasta in sorted(AMOSTRAS.iterdir()):
        if not pasta.is_dir():
            continue
        indice = PT_PARA_INDICE.get(pasta.name)
        if indice is None:
            print(f"   ! pasta ignorada (classe desconhecida): {pasta.name}")
            continue
        arquivos = sorted(pasta.glob("*.png"))
        validos = 0
        for arq in arquivos:
            dados = np.fromfile(str(arq), dtype=np.uint8)
            img = cv.imdecode(dados, cv.IMREAD_COLOR)
            if img is None or img.ndim != 3:
                continue
            blob = clf.preprocessar(img)
            clf._net.setInput(blob, config.FER_NOME_ENTRADA)
            z = np.asarray(
                clf._net.forward(config.FER_NOME_SAIDA), dtype=np.float64
            ).reshape(-1)
            if z.size != mod_cal.N_CLASSES:
                continue
            logits.append(z)
            rotulos.append(indice)
            validos += 1
        contagem[pasta.name] = validos
        print(f"   {pasta.name:10s} {validos:4d} amostras")

    if not logits:
        print("ERRO: nenhuma amostra válida encontrada.", file=sys.stderr)
        raise SystemExit(2)
    return np.vstack(logits), np.array(rotulos, dtype=np.int64), contagem


def dividir(X: np.ndarray, y: np.ndarray, fracao_teste: float, semente: int):
    """Divisao estratificada: cada classe contribui para treino e teste."""
    rng = np.random.default_rng(semente)
    idx_treino: list[int] = []
    idx_teste: list[int] = []
    for classe in np.unique(y):
        indices = np.where(y == classe)[0]
        rng.shuffle(indices)
        n_teste = max(1, int(round(len(indices) * fracao_teste)))
        if len(indices) <= 2:  # classe com pouquíssimas amostras: tudo em treino
            idx_treino.extend(indices.tolist())
            continue
        idx_teste.extend(indices[:n_teste].tolist())
        idx_treino.extend(indices[n_teste:].tolist())
    return (np.array(idx_treino, dtype=np.int64),
            np.array(idx_teste, dtype=np.int64))


def grafico_matriz(mc: list[list[int]], titulo: str, destino: Path) -> None:
    """Matriz de confusao em PNG, fundo branco, para o banner."""
    sys.path.insert(0, str(RAIZ / "scripts"))
    from banner_diagramas import AZUL, BRANCO, CINZA, CINZA_CLARO, PRETO, Lona

    n = mod_cal.N_CLASSES
    cel, m_esq, m_topo = 62, 150, 112
    L = m_esq + cel * n + 40
    A = m_topo + cel * n + 70
    png = Lona(L, A)
    maximo = max(1, max(max(linha) for linha in mc))

    png.texto(26, 26, titulo, 17, PRETO, True)
    png.texto(26, 50, "Linhas: classe verdadeira · Colunas: classe prevista",
              12, CINZA)
    png.linha(26, 66, L - 26, 66, CINZA_CLARO)

    for j in range(n):
        png.texto(m_esq + cel * j + cel / 2, m_topo - 16,
                  config.CLASSES_PT[j][:4], 11, PRETO, True, "mm")
    for i in range(n):
        png.texto(m_esq - 12, m_topo + cel * i + cel / 2,
                  config.CLASSES_PT[i], 11.5, PRETO, True, "rm")
        for j in range(n):
            v = mc[i][j]
            x, y = m_esq + cel * j, m_topo + cel * i
            if v:
                intensidade = v / maximo
                if i == j:
                    cor = (int(232 - 180 * intensidade), int(243 - 60 * intensidade),
                           int(236 - 160 * intensidade))
                else:
                    cor = (int(253 - 40 * intensidade), int(232 - 110 * intensidade),
                           int(232 - 110 * intensidade))
            else:
                cor = (250, 251, 252)
            png.d.rectangle(
                [x * png.e, y * png.e, (x + cel) * png.e, (y + cel) * png.e],
                fill=cor, outline=CINZA_CLARO, width=png.e,
            )
            if v:
                png.texto(x + cel / 2, y + cel / 2, str(v), 15,
                          PRETO if i == j else "#9c2a2a", True, "mm")

    total = sum(sum(linha) for linha in mc)
    acertos = sum(mc[i][i] for i in range(n))
    png.texto(26, A - 34,
              f"Acurácia: {acertos}/{total} = "
              f"{acertos / max(1, total) * 100:.1f}%".replace(".", ","),
              13, AZUL, True)
    png.texto(26, A - 14,
              "Diagonal = acertos. Fora da diagonal = confusões entre classes.",
              11, CINZA)
    png.salvar(destino)


def grafico_perda(historico: list[float], destino: Path) -> None:
    sys.path.insert(0, str(RAIZ / "scripts"))
    from banner_diagramas import AZUL, CINZA, CINZA_CLARO, PRETO, Lona

    L, A = 760, 360
    png = Lona(L, A)
    png.texto(26, 26, "Curva de perda do treinamento", 17, PRETO, True)
    png.texto(26, 50,
              f"Entropia cruzada no conjunto de treino · {len(historico)} épocas",
              12, CINZA)
    png.linha(26, 66, L - 26, 66, CINZA_CLARO)

    base_y, topo_y, x0 = A - 70, 96, 80
    x1 = L - 40
    maximo = max(historico) if historico else 1.0
    minimo = min(historico) if historico else 0.0
    faixa = max(1e-9, maximo - minimo)

    for k in range(5):
        v = maximo - faixa * k / 4
        yy = topo_y + (base_y - topo_y) * k / 4
        png.linha(x0, yy, x1, yy, CINZA_CLARO)
        png.texto(x0 - 10, yy, f"{v:.2f}".replace(".", ","), 10.5, CINZA, ancora="rm")

    pontos = []
    for i, v in enumerate(historico):
        x = x0 + (x1 - x0) * (i / max(1, len(historico) - 1))
        y = base_y - (base_y - topo_y) * ((v - minimo) / faixa)
        pontos.append((x, y))
    for (ax, ay), (bx, by) in zip(pontos, pontos[1:]):
        png.linha(ax, ay, bx, by, AZUL, 2)

    png.texto(x0, A - 40, "época 1", 10.5, CINZA)
    png.texto(x1, A - 40, f"época {len(historico)}", 10.5, CINZA, ancora="rm")
    png.texto(26, A - 16,
              f"Perda inicial {historico[0]:.4f} → final {historico[-1]:.4f}"
              .replace(".", ","),
              12, PRETO, True)
    png.salvar(destino)


def main() -> int:  # noqa: C901
    ap = argparse.ArgumentParser(
        description="Treina a camada de calibração sobre amostras próprias.")
    ap.add_argument("--epocas", type=int, default=400)
    ap.add_argument("--taxa", type=float, default=0.05)
    ap.add_argument("--regularizacao", type=float, default=1e-3)
    ap.add_argument("--teste", type=float, default=0.25,
                    help="fração separada para teste (padrão: 0,25)")
    ap.add_argument("--semente", type=int, default=20261130)
    args = ap.parse_args()

    print("=" * 72)
    print(" Expressão Neural — treinamento da camada de calibração")
    print("=" * 72)

    print("\n1. Extraindo logits das amostras rotuladas (modelo base congelado)")
    try:
        X, y, contagem = carregar_amostras()
    except ErroModelo as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        return 1

    classes_presentes = sorted(set(y.tolist()))
    print(f"\n   Total: {X.shape[0]} amostras · {len(classes_presentes)} de "
          f"{mod_cal.N_CLASSES} classes representadas")
    if X.shape[0] < 20:
        print("\nERRO: amostras insuficientes para um treinamento honesto "
              "(mínimo 20).", file=sys.stderr)
        return 2

    print("\n2. Dividindo em treino e teste (estratificado)")
    i_tr, i_te = dividir(X, y, args.teste, args.semente)
    X_tr, y_tr = X[i_tr], y[i_tr]
    X_te, y_te = (X[i_te], y[i_te]) if len(i_te) else (X_tr, y_tr)
    if not len(i_te):
        print("   ! poucas amostras: o teste reaproveita o treino (resultado otimista)")
    print(f"   treino: {len(i_tr)} amostras · teste: {len(i_te)} amostras")

    print("\n3. Avaliando o MODELO BASE no conjunto de teste (linha de base)")
    prev_antes = np.argmax(X_te, axis=1)
    m_antes = mod_cal.metricas(y_te, prev_antes)
    print(f"   acurácia: {m_antes['acuracia'] * 100:.1f}%  ·  "
          f"F1 macro: {m_antes['macro_f1']:.3f}")

    print(f"\n4. Treinando W (7×7) e b (7) — {args.epocas} épocas")
    cal, historico = mod_cal.treinar(
        X_tr, y_tr, epocas=args.epocas, taxa=args.taxa,
        regularizacao=args.regularizacao, semente=args.semente,
    )
    print(f"   perda: {historico[0]:.4f} → {historico[-1]:.4f}")

    print("\n5. Avaliando a CAMADA TREINADA no MESMO conjunto de teste")
    Z_te = X_te @ cal.W.T + cal.b
    prev_depois = np.argmax(Z_te, axis=1)
    m_depois = mod_cal.metricas(y_te, prev_depois)
    print(f"   acurácia: {m_depois['acuracia'] * 100:.1f}%  ·  "
          f"F1 macro: {m_depois['macro_f1']:.3f}")

    delta = m_depois["acuracia"] - m_antes["acuracia"]
    print(f"\n   variação de acurácia: {delta * 100:+.1f} pontos percentuais")
    if delta < 0:
        print("   ! A calibração PIOROU o resultado no teste. Isso acontece com")
        print("     poucas amostras. Colete mais ou aumente --regularizacao.")
        print("     Os pesos serão gravados mesmo assim, mas a aplicação só os usa")
        print("     se você ativar explicitamente (ver docs/treinamento.md).")

    cal.metadados.update({
        "acuracia_teste": m_depois["acuracia"],
        "acuracia_teste_modelo_base": m_antes["acuracia"],
        "f1_macro_teste": m_depois["macro_f1"],
        "f1_macro_teste_modelo_base": m_antes["macro_f1"],
        "amostras_teste": int(len(i_te)),
        "amostras_por_classe": contagem,
        "classes_representadas": [config.CLASSES_MODELO[i] for i in classes_presentes],
        "melhorou": bool(delta > 0),
    })
    caminho = cal.salvar()
    print(f"\n6. Pesos gravados em {caminho.relative_to(RAIZ)}")

    # ----------------------------------------------------------- relatorios
    SAIDA.mkdir(parents=True, exist_ok=True)
    GRAFICOS.mkdir(parents=True, exist_ok=True)
    relatorio = {
        "gerado_em": datetime.now().isoformat(timespec="seconds"),
        "natureza_do_treinamento": (
            "Aprendizado supervisionado de uma camada linear de saída (W 7×7 e b 7, "
            "56 parâmetros) sobre os logits do modelo pré-treinado, que permanece "
            "CONGELADO. Não houve treinamento nem ajuste fino da rede convolucional."
        ),
        "dados": {
            "origem": "amostras coletadas pelo próprio autor via webcam",
            "rotulo_verdadeiro": "a expressão SOLICITADA (expressões encenadas)",
            "total_amostras": int(X.shape[0]),
            "amostras_por_classe": contagem,
            "amostras_treino": int(len(i_tr)),
            "amostras_teste": int(len(i_te)),
            "divisao": "estratificada por classe",
            "semente": args.semente,
        },
        "hiperparametros": {
            "epocas": args.epocas,
            "taxa_aprendizado": args.taxa,
            "regularizacao_l2_ate_identidade": args.regularizacao,
            "funcao_de_perda": "entropia cruzada",
            "otimizador": "descida de gradiente em lote completo",
            "inicializacao": "W = identidade, b = 0 (reproduz o modelo base)",
        },
        "perda": {
            "inicial": historico[0],
            "final": historico[-1],
            "historico": [round(v, 6) for v in historico],
        },
        "resultados": {
            "modelo_base": m_antes,
            "com_calibracao": m_depois,
            "variacao_acuracia_pp": round(delta * 100, 2),
        },
    }
    (SAIDA / "treinamento_calibracao.json").write_text(
        json.dumps(relatorio, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"   + {(SAIDA / 'treinamento_calibracao.json').relative_to(RAIZ)}")

    _markdown(relatorio, SAIDA / "treinamento_calibracao.md")
    print(f"   + {(SAIDA / 'treinamento_calibracao.md').relative_to(RAIZ)}")

    print("\n7. Gráficos")
    grafico_matriz(m_depois["matriz_confusao"],
                   "Matriz de confusão — com calibração própria",
                   GRAFICOS / "matriz_confusao.png")
    grafico_matriz(m_antes["matriz_confusao"],
                   "Matriz de confusão — modelo base (sem calibração)",
                   GRAFICOS / "matriz_confusao_base.png")
    grafico_perda(historico, GRAFICOS / "curva_de_perda.png")

    print("\n" + "=" * 72)
    print(" Treinamento concluído.")
    print(f" Acurácia no teste: {m_antes['acuracia'] * 100:.1f}% (base) → "
          f"{m_depois['acuracia'] * 100:.1f}% (calibrada)")
    print("=" * 72)
    return 0


def _markdown(r: dict, destino: Path) -> None:
    d, h, res = r["dados"], r["hiperparametros"], r["resultados"]
    L = [
        "# Treinamento da camada de calibração",
        "",
        f"Executado em {r['gerado_em']} por `scripts/treinar_calibracao.py`.",
        "",
        "## O que foi treinado",
        "",
        r["natureza_do_treinamento"],
        "",
        "| Item | Valor |",
        "|---|---|",
        "| Parâmetros treináveis | **56** (W 7×7 + b 7) |",
        f"| Função de perda | {h['funcao_de_perda']} |",
        f"| Otimizador | {h['otimizador']} |",
        f"| Épocas | **{h['epocas']}** |",
        f"| Taxa de aprendizado | {h['taxa_aprendizado']} |",
        f"| Regularização L2 (até a identidade) | {h['regularizacao_l2_ate_identidade']} |",
        f"| Inicialização | {h['inicializacao']} |",
        f"| Perda inicial → final | {r['perda']['inicial']:.4f} → {r['perda']['final']:.4f} |",
        "",
        "## Conjunto de dados",
        "",
        "| Item | Valor |",
        "|---|---|",
        f"| Origem | {d['origem']} |",
        f"| Rótulo verdadeiro | {d['rotulo_verdadeiro']} |",
        f"| Total de amostras | **{d['total_amostras']}** |",
        f"| Treino | {d['amostras_treino']} |",
        f"| Teste | {d['amostras_teste']} |",
        f"| Divisão | {d['divisao']} |",
        "",
        "Amostras por classe:",
        "",
        "| Classe | Amostras |",
        "|---|---|",
    ]
    for classe, n in d["amostras_por_classe"].items():
        L.append(f"| {classe} | {n} |")

    L += [
        "",
        "## Resultados no conjunto de teste",
        "",
        "| Métrica | Modelo base | Com calibração |",
        "|---|---|---|",
        f"| **Acurácia** | {res['modelo_base']['acuracia'] * 100:.1f}% | "
        f"**{res['com_calibracao']['acuracia'] * 100:.1f}%** |",
        f"| Precisão (macro) | {res['modelo_base']['macro_precisao']:.3f} | "
        f"{res['com_calibracao']['macro_precisao']:.3f} |",
        f"| Revocação (macro) | {res['modelo_base']['macro_revocacao']:.3f} | "
        f"{res['com_calibracao']['macro_revocacao']:.3f} |",
        f"| F1-score (macro) | {res['modelo_base']['macro_f1']:.3f} | "
        f"{res['com_calibracao']['macro_f1']:.3f} |",
        "",
        f"Variação de acurácia: **{res['variacao_acuracia_pp']:+.2f} pontos percentuais**.",
        "",
        "### Por classe (com calibração)",
        "",
        "| Classe | Amostras | Precisão | Revocação | F1 |",
        "|---|---|---|---|---|",
    ]
    for c in res["com_calibracao"]["por_classe"]:
        L.append(f"| {c['classe']} | {c['amostras']} | {c['precisao']:.3f} | "
                 f"{c['revocacao']:.3f} | {c['f1']:.3f} |")

    L += [
        "",
        "### Matriz de confusão (com calibração)",
        "",
        "Linhas: classe verdadeira · Colunas: classe prevista",
        "",
        "| | " + " | ".join(c[:4] for c in config.CLASSES_PT) + " |",
        "|---" * (mod_cal.N_CLASSES + 1) + "|",
    ]
    for i, linha in enumerate(res["com_calibracao"]["matriz_confusao"]):
        L.append(f"| **{config.CLASSES_PT[i]}** | " + " | ".join(str(v) for v in linha) + " |")

    L += [
        "",
        "## Limitações deste treinamento",
        "",
        "Declaradas para que o resultado não seja superinterpretado:",
        "",
        f"- O conjunto tem **{d['total_amostras']} amostras de uma única pessoa**. "
        "A acurácia medida vale para esse contexto e **não generaliza** para "
        "público diverso.",
        "- As expressões são **encenadas sob solicitação**, não espontâneas.",
        "- O conjunto de teste é pequeno; a acurácia tem incerteza alta.",
        "- A rede convolucional **não foi treinada**: apenas a camada de saída "
        "de 56 parâmetros. O ganho possível é de **calibração**, não de "
        "capacidade de representação.",
        "- Os 88,27% em RAF-DB continuam sendo do **modelo pré-treinado**, e não "
        "devem ser confundidos com a acurácia medida aqui.",
        "",
    ]
    destino.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
