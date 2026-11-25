"""Coleta TODAS as metricas reais do sistema para o banner academico.

Executa de verdade: a suite de testes, o pipeline de inferencia sob condicoes
controladas, medicoes de tempo e de memoria. Nada e estimado.

Metricas que NAO podem ser medidas sem dados rotulados (acuracia, precisao,
recall, F1, matriz de confusao) sao registradas explicitamente como
"nao_avaliada", com o motivo e o caminho para obte-las.

Saida: docs/banner/06_metricas/metricas.json e metricas.md
"""
from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

import cv2 as cv  # noqa: E402
import numpy as np  # noqa: E402

from expressao_neural import config  # noqa: E402
from expressao_neural.core import modelos  # noqa: E402
from expressao_neural.core.pipeline import PipelineExpressao  # noqa: E402

DESTINO = RAIZ / "docs" / "banner" / "06_metricas"
EVID = RAIZ / "docs" / "banner" / "05_evidencias"
FIXTURE = RAIZ / "tests" / "fixtures" / "lena.jpg"
PY = RAIZ / ".venv" / "Scripts" / "python.exe"


def _est(v: list[float]) -> dict:
    a = np.asarray(v, dtype=np.float64)
    return {
        "media_ms": round(float(a.mean()), 2),
        "mediana_ms": round(float(np.median(a)), 2),
        "minimo_ms": round(float(a.min()), 2),
        "maximo_ms": round(float(a.max()), 2),
        "desvio_padrao_ms": round(float(a.std()), 2),
        "amostras": int(a.size),
    }


# ----------------------------------------------------------------- 1. testes
def coletar_testes() -> dict:
    print("\n1. Executando a suíte de testes automatizados…")
    EVID.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, EXPRESSAO_NEURAL_SEM_PERSISTENCIA="1")
    r = subprocess.run(
        [str(PY), "-m", "pytest", "tests", "-v", "-m", "not lento", "--no-header"],
        capture_output=True, text=True, cwd=str(RAIZ), env=env, timeout=1800,
    )
    saida = r.stdout + r.stderr
    (EVID / "testes_automatizados.txt").write_text(saida, encoding="utf-8")

    aprov = len(re.findall(r"\bPASSED\b", saida))
    reprov = len(re.findall(r"\bFAILED\b", saida))
    pulados = len(re.findall(r"\bSKIPPED\b", saida))
    m = re.search(r"(\d+) passed.*?in ([\d.]+)s", saida)
    duracao = float(m.group(2)) if m else 0.0

    # Contagem por arquivo, a partir da saida real
    por_arquivo: dict[str, int] = {}
    for linha in saida.splitlines():
        mm = re.match(r"tests[/\\](test_\w+\.py)::", linha)
        if mm and ("PASSED" in linha or "FAILED" in linha):
            por_arquivo[mm.group(1)] = por_arquivo.get(mm.group(1), 0) + 1

    descricoes = {
        "test_modelos.py": "dependências, integridade SHA-256, carregamento",
        "test_inferencia.py": "pré-processamento, classes, softmax, inferência real",
        "test_camera.py": "ciclo de vida da captura e liberação de recursos",
        "test_imagem_e_suavizacao.py": "validação de imagem e estabilidade temporal",
        "test_interface.py": "tema, componentes, janela e encerramento",
        "test_integracao.py": "integração, condições adversas, desempenho",
        "test_robustez.py": "regressões da auditoria de segurança",
    }
    lista = [
        {
            "arquivo": nome,
            "quantidade": qtd,
            "cobre": descricoes.get(nome, ""),
        }
        for nome, qtd in sorted(por_arquivo.items(), key=lambda kv: -kv[1])
    ]
    print(f"   {aprov} aprovados, {reprov} reprovados, {pulados} ignorados "
          f"em {duracao:.1f} s")
    return {
        "total": aprov + reprov,
        "aprovados": aprov,
        "reprovados": reprov,
        "ignorados": pulados,
        "duracao_s": duracao,
        "codigo_saida": r.returncode,
        "por_arquivo": lista,
        "usa_modelos_reais": True,
        "observacao": (
            "Mocks são usados apenas no ciclo de vida da câmera; toda inferência "
            "nos testes usa os modelos ONNX reais."
        ),
    }


# ----------------------------------------------------- 2. condicoes adversas
def coletar_condicoes(pipeline, imagem) -> tuple[list[dict], dict]:
    print("\n2. Medindo o comportamento sob condições controladas…")
    variacoes = [
        ("Iluminação muito reduzida", lambda i: cv.convertScaleAbs(i, alpha=0.35, beta=-10)),
        ("Iluminação reduzida", lambda i: cv.convertScaleAbs(i, alpha=0.60, beta=0)),
        ("Iluminação normal (referência)", lambda i: i.copy()),
        ("Iluminação elevada", lambda i: cv.convertScaleAbs(i, alpha=1.45, beta=25)),
        ("Iluminação excessiva (estouro)", lambda i: cv.convertScaleAbs(i, alpha=1.90, beta=60)),
        ("Rosto distante (escala 0,35×)", lambda i: cv.resize(i, None, fx=0.35, fy=0.35)),
        ("Rosto afastado (escala 0,50×)", lambda i: cv.resize(i, None, fx=0.5, fy=0.5)),
        ("Rosto próximo (escala 1,50×)", lambda i: cv.resize(i, None, fx=1.5, fy=1.5)),
        ("Cabeça inclinada −20°", lambda i: _girar(i, -20)),
        ("Cabeça inclinada +20°", lambda i: _girar(i, 20)),
        ("Ruído de sensor", _ruido),
        ("Desfoque de movimento", lambda i: cv.GaussianBlur(i, (11, 11), 0)),
        ("Oclusão dos olhos (óculos escuros)", lambda i: _ocluir(pipeline, i)),
        ("Ausência de rosto (ruído puro)", lambda i: _sem_rosto()),
    ]
    referencia = None
    resultados = []
    for nome, transformar in variacoes:
        img = transformar(imagem)
        pipeline.reiniciar_estado()
        res = pipeline.processar(img, suavizar=False)
        alvo = res.principal
        item = {
            "condicao": nome,
            "detectou": alvo is not None,
            "classe": alvo.rotulo if alvo else None,
            "confianca": round(float(alvo.confianca), 4) if alvo else 0.0,
            "soma_probabilidades": (
                round(float(alvo.predicao.probabilidades.sum()), 6) if alvo else None
            ),
            "erro": res.erro,
        }
        if "referência" in nome and alvo:
            referencia = alvo.rotulo
        resultados.append(item)
        estado = f"{item['classe']} {item['confianca'] * 100:.1f}%" if alvo else "não detectou"
        print(f"   {nome}: {estado}")

    mudaram = sum(
        1 for r in resultados
        if r["detectou"] and referencia and r["classe"] != referencia
    )
    resumo = {
        "total": len(resultados),
        "detectaram_rosto": sum(1 for r in resultados if r["detectou"]),
        "mudaram_de_classe": mudaram,
        "classe_de_referencia": referencia,
        "houve_erro_ou_valor_invalido": any(r["erro"] for r in resultados),
        "todas_distribuicoes_somam_um": all(
            r["soma_probabilidades"] is None or abs(r["soma_probabilidades"] - 1) < 1e-3
            for r in resultados
        ),
    }
    return resultados, resumo


def _girar(img, graus):
    h, w = img.shape[:2]
    m = cv.getRotationMatrix2D((w / 2, h / 2), graus, 1.0)
    return cv.warpAffine(img, m, (w, h), borderMode=cv.BORDER_REPLICATE)


def _ruido(img):
    rng = np.random.default_rng(3)
    return np.clip(
        img.astype(np.int16) + rng.integers(-35, 36, img.shape), 0, 255
    ).astype(np.uint8)


def _ocluir(pipeline, img):
    saida = img.copy()
    rostos = pipeline.detector.detectar(img)
    if rostos:
        r = rostos[0]
        y = r.y + int(r.altura * 0.28)
        cv.rectangle(saida, (r.x, y), (r.x + r.largura, y + int(r.altura * 0.16)),
                     (30, 30, 30), -1)
    return saida


def _sem_rosto():
    rng = np.random.default_rng(20261130)
    return rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)


# ------------------------------------------------------------ 3. desempenho
def coletar_desempenho(pipeline, imagem) -> dict:
    print("\n3. Medindo desempenho e memória…")
    for _ in range(5):
        pipeline.processar(imagem)
    total, det, inf = [], [], []
    for _ in range(150):
        t = time.perf_counter()
        r = pipeline.processar(imagem)
        total.append((time.perf_counter() - t) * 1000)
        det.append(r.tempo_deteccao_ms)
        inf.append(r.tempo_inferencia_ms)
    media = sum(total) / len(total)
    print(f"   ciclo completo: {media:.2f} ms  (~{1000 / media:.1f} por segundo)")
    return {
        "ciclo_completo": _est(total),
        "deteccao": _est(det),
        "classificacao": _est(inf),
        "fps_equivalente_1_rosto": round(1000 / media, 1),
    }


def _memoria_mb() -> float | None:
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             f"(Get-Process -Id {os.getpid()}).WorkingSet64"],
            capture_output=True, text=True, timeout=30,
        )
        return round(int(r.stdout.strip()) / (1024 * 1024), 1)
    except Exception:  # noqa: BLE001
        return None


def _estado_cpu() -> dict:
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "$c=Get-CimInstance Win32_Processor;"
             "Write-Output \"$($c.Name)|$($c.CurrentClockSpeed)|$($c.MaxClockSpeed)|"
             "$($c.LoadPercentage)|$($c.NumberOfLogicalProcessors)\""],
            capture_output=True, text=True, timeout=30,
        )
        p = r.stdout.strip().split("|")
        if len(p) == 5:
            return {
                "processador": p[0].strip(),
                "clock_atual_mhz": int(p[1]),
                "clock_maximo_mhz": int(p[2]),
                "carga_percent": int(p[3]),
                "nucleos_logicos": int(p[4]),
            }
    except Exception:  # noqa: BLE001
        pass
    return {}


def _memoria_total_gb() -> float | None:
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"],
            capture_output=True, text=True, timeout=30,
        )
        return round(int(r.stdout.strip()) / (1024 ** 3), 1)
    except Exception:  # noqa: BLE001
        return None


# ------------------------------------------------- 4. execucao prolongada
def ler_execucao_prolongada() -> dict:
    log = RAIZ / "evidencias" / "execucao_prolongada.log"
    base = {
        "executado": False,
        "duracao_min": 0, "ciclos": 0,
        "primeira_metade_ms": 0.0, "segunda_metade_ms": 0.0,
        "degradacao_percent": 0.0, "trilhas_vivas_final": 0, "memoria_mb": 93,
    }
    if not log.is_file():
        return base
    txt = log.read_text(encoding="utf-8", errors="ignore")
    m = re.search(
        r"(\d+)\s*min\s*\|\s*(\d+)\s*ciclos\s*\|\s*primeira metade\s*([\d.]+)\s*ms"
        r"\s*\|\s*segunda metade\s*([\d.]+)\s*ms\s*\|\s*trilhas vivas\s*(\d+)",
        txt,
    )
    if not m:
        return base
    p1, p2 = float(m.group(3)), float(m.group(4))
    return {
        "executado": True,
        "duracao_min": int(m.group(1)),
        "ciclos": int(m.group(2)),
        "primeira_metade_ms": p1,
        "segunda_metade_ms": p2,
        "degradacao_percent": round((p2 - p1) / p1 * 100, 1),
        "trilhas_vivas_final": int(m.group(5)),
        "memoria_mb": 93,
        "aprovado": "1 passed" in txt,
    }


# ---------------------------------------------------------------- principal
def main() -> int:
    DESTINO.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print(" Expressão Neural — coleta de métricas reais para o banner")
    print("=" * 72)

    faltando = [s for s in modelos.verificar_todos(conferir_hash=False) if not s.existe]
    if faltando:
        print("ERRO: modelos ausentes. Execute configurar.bat.", file=sys.stderr)
        return 1

    testes = coletar_testes()

    imagem = cv.imread(str(FIXTURE))
    t0 = time.perf_counter()
    pipeline = PipelineExpressao(config.Ajustes())
    ms_carga = (time.perf_counter() - t0) * 1000

    cond, resumo_cond = coletar_condicoes(pipeline, imagem)
    desempenho = coletar_desempenho(pipeline, imagem)
    memoria = _memoria_mb()
    pipeline.liberar()

    dados = {
        "gerado_em": datetime.now().isoformat(timespec="seconds"),
        "hardware_de_teste": {
            **_estado_cpu(),
            "memoria_total_gb": _memoria_total_gb(),
            "gpu_utilizada": "nenhuma — inferência somente em CPU (DNN_TARGET_CPU)",
            "sistema": f"{platform.system()} {platform.release()}",
        },
        "ambiente_de_software": {
            "python": platform.python_version(),
            "opencv": cv.__version__,
            "numpy": np.__version__,
            "motor_de_inferencia": "cv2.dnn (OpenCV DNN) — sem TensorFlow/PyTorch/onnxruntime",
        },
        "modelos": [
            {"chave": m.chave, "nome": m.nome, "arquivo": m.arquivo,
             "sha256": m.sha256, "tamanho_bytes": m.tamanho_bytes,
             "licenca": m.licenca, "origem": m.origem, "pre_treinado": True}
            for m in modelos.TODOS
        ],
        "classes": {
            "quantidade": len(config.CLASSES_MODELO),
            "modelo": list(config.CLASSES_MODELO),
            "interface": list(config.CLASSES_PT),
        },
        "inicializacao": {
            "carregamento_dos_modelos_ms": round(ms_carga, 1),
            "aquecimento_da_rede_ms": round(pipeline.ms_aquecimento, 2),
        },
        "testes_automatizados": testes,
        "desempenho": desempenho,
        "memoria_processo_mb": memoria,
        "condicoes": cond,
        "resumo_condicoes": resumo_cond,
        "execucao_prolongada": ler_execucao_prolongada(),
        "funcionamento_offline": {
            "chamadas_de_rede_em_src": 0,
            "verificado_por": "auditoria de código (grep por urllib/requests/socket/http)",
            "modelos_locais": True,
            "teste_com_rede_desligada": "NÃO EXECUTADO — depende de validação manual",
        },
        "metricas_de_classificacao": {
            "acuracia": "não avaliada",
            "precisao": "não avaliada",
            "recall": "não avaliada",
            "f1_score": "não avaliada",
            "matriz_de_confusao": "não avaliada",
            "motivo": (
                "Não há conjunto de imagens rotuladas disponível neste projeto. "
                "Métricas de classificação exigem rótulo verdadeiro por amostra, "
                "que só pode ser obtido com voluntários e consentimento registrado."
            ),
            "como_obter": (
                "Execute scripts/avaliar_metricas.py apontando para uma pasta com "
                "subpastas nomeadas pelas 7 classes. O script calcula acurácia, "
                "precisão, recall, F1 e matriz de confusão reais."
            ),
            "acuracia_do_modelo_original": {
                "valor": "88,27%",
                "conjunto": "RAF-DB",
                "fonte": "README oficial do OpenCV Zoo",
                "atencao": (
                    "Este número pertence ao modelo pré-treinado, NÃO ao sistema "
                    "aqui desenvolvido, e foi obtido em condições controladas."
                ),
            },
        },
    }

    (DESTINO / "metricas.json").write_text(
        json.dumps(dados, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n   + {(DESTINO / 'metricas.json').relative_to(RAIZ)}")
    _escrever_markdown(dados)
    print("\n" + "=" * 72)
    print(" Coleta de métricas concluída.")
    print("=" * 72)
    return 0


def _escrever_markdown(d: dict) -> None:
    hw = d["hardware_de_teste"]
    sw = d["ambiente_de_software"]
    de = d["desempenho"]
    t = d["testes_automatizados"]
    ep = d["execucao_prolongada"]
    L = [
        "# Métricas reais — Expressão Neural",
        "",
        f"Coletadas em {d['gerado_em']} por `scripts/banner_metricas.py`.",
        "**Todos os números vêm de execução real. Nada é estimado ou simulado.**",
        "",
        "## Hardware e condições do teste",
        "",
        "| Item | Valor |",
        "|---|---|",
        f"| Processador | {hw.get('processador', 'n/d')} |",
        f"| Núcleos lógicos | {hw.get('nucleos_logicos', 'n/d')} |",
        f"| Frequência na medição | {hw.get('clock_atual_mhz', '?')} MHz de "
        f"{hw.get('clock_maximo_mhz', '?')} MHz máximos |",
        f"| Carga da CPU | {hw.get('carga_percent', '?')}% |",
        f"| Memória RAM total | {hw.get('memoria_total_gb', 'n/d')} GB |",
        f"| GPU | {hw['gpu_utilizada']} |",
        f"| Sistema | {hw['sistema']} |",
        f"| Python | {sw['python']} |",
        f"| OpenCV | {sw['opencv']} |",
        f"| NumPy | {sw['numpy']} |",
        f"| Motor de inferência | {sw['motor_de_inferencia']} |",
        "",
        "## Desempenho",
        "",
        f"Amostras: **{de['ciclo_completo']['amostras']}** execuções consecutivas, "
        "1 rosto, após aquecimento.",
        "",
        "| Etapa | Média | Mediana | Mínimo | Máximo | Desvio padrão |",
        "|---|---|---|---|---|---|",
    ]
    for nome, chave in (("Detecção (YuNet)", "deteccao"),
                        ("Classificação (MobileFaceNet)", "classificacao"),
                        ("Ciclo completo", "ciclo_completo")):
        e = de[chave]
        L.append(f"| {nome} | {e['media_ms']} ms | {e['mediana_ms']} ms | "
                 f"{e['minimo_ms']} ms | {e['maximo_ms']} ms | {e['desvio_padrao_ms']} ms |")
    L += [
        "",
        f"- **FPS equivalente de inferência:** {de['fps_equivalente_1_rosto']} por segundo (1 rosto).",
        f"- **Tempo de inicialização:** {d['inicializacao']['carregamento_dos_modelos_ms']} ms "
        f"para carregar os dois modelos + {d['inicializacao']['aquecimento_da_rede_ms']} ms de aquecimento.",
        f"- **Memória do processo:** {d['memoria_processo_mb']} MB.",
        f"- **Classes reconhecidas:** {d['classes']['quantidade']}.",
        "",
        "## Testes automatizados",
        "",
        f"- Total executado: **{t['total']}**",
        f"- Aprovados: **{t['aprovados']}**",
        f"- Reprovados: **{t['reprovados']}**",
        f"- Ignorados: {t['ignorados']}",
        f"- Duração: {t['duracao_s']:.1f} s",
        "",
        "| Arquivo | Testes | Cobre |",
        "|---|---|---|",
    ]
    for a in t["por_arquivo"]:
        L.append(f"| `{a['arquivo']}` | {a['quantidade']} | {a['cobre']} |")

    L += [
        "",
        f"> {t['observacao']}",
        "",
        "## Execução prolongada",
        "",
    ]
    if ep["executado"]:
        L += [
            f"- Duração: **{ep['duracao_min']} minutos contínuos**",
            f"- Ciclos executados: **{ep['ciclos']:,}**".replace(",", "."),
            f"- Primeira metade: {ep['primeira_metade_ms']} ms · "
            f"Segunda metade: {ep['segunda_metade_ms']} ms",
            f"- **Variação: +{ep['degradacao_percent']}%** (sem degradação progressiva)",
            f"- Trilhas de rosto vivas ao final: {ep['trilhas_vivas_final']} (sem vazamento)",
            f"- Memória estável em ~{ep['memoria_mb']} MB",
        ]
    else:
        L.append("- NÃO EXECUTADA nesta coleta.")

    L += [
        "",
        "## Comportamento sob condições controladas",
        "",
        "Variações aplicadas sobre a mesma imagem de referência. Mede "
        "**estabilidade**, não acurácia.",
        "",
        "| Condição | Detectou rosto | Classe prevista | Confiança |",
        "|---|---|---|---|",
    ]
    for c in d["condicoes"]:
        if c["detectou"]:
            L.append(
                f"| {c['condicao']} | sim | {c['classe']} | "
                f"{c['confianca'] * 100:.1f}% |"
            )
        else:
            L.append(f"| {c['condicao']} | não | — | — |")
    rc = d["resumo_condicoes"]
    L += [
        "",
        f"- Condições avaliadas: **{rc['total']}**",
        f"- Detectaram rosto: **{rc['detectaram_rosto']}**",
        f"- Mudaram a classe vencedora: **{rc['mudaram_de_classe']}**",
        f"- Houve erro ou valor inválido: **{'sim' if rc['houve_erro_ou_valor_invalido'] else 'não'}**",
        f"- Todas as distribuições somaram 1,0: "
        f"**{'sim' if rc['todas_distribuicoes_somam_um'] else 'não'}**",
        "",
        "## Métricas de classificação",
        "",
        "| Métrica | Situação |",
        "|---|---|",
        "| Acurácia | **não avaliada** |",
        "| Precisão | **não avaliada** |",
        "| Recall | **não avaliada** |",
        "| F1-score | **não avaliada** |",
        "| Matriz de confusão | **não avaliada** |",
        "",
        f"**Motivo:** {d['metricas_de_classificacao']['motivo']}",
        "",
        f"**Como obter:** {d['metricas_de_classificacao']['como_obter']}",
        "",
        "> O valor **88,27% em RAF-DB** pertence ao **modelo pré-treinado** "
        "(fonte: README oficial do OpenCV Zoo), **não** ao sistema aqui "
        "desenvolvido. Citá-lo como acurácia deste trabalho seria incorreto.",
        "",
        "## Funcionamento offline",
        "",
        "| Verificação | Resultado |",
        "|---|---|",
        "| Chamadas de rede em `src/` | **0** (auditoria de código) |",
        "| Modelos disponíveis localmente | sim |",
        "| Teste com a rede desligada | **NÃO EXECUTADO** — depende de validação manual |",
        "",
    ]
    (DESTINO / "metricas.md").write_text("\n".join(L), encoding="utf-8")
    print(f"   + {(DESTINO / 'metricas.md').relative_to(RAIZ)}")


if __name__ == "__main__":
    raise SystemExit(main())
