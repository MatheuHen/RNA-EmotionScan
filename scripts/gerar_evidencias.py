"""Gera evidencias reais para o relatorio e o banner academico.

Produz em evidencias/:
  * captura_interface.png       — tela da aplicacao com resultado real
  * deteccao_anotada.png        — quadro com caixa, pontos e rotulo
  * rosto_alinhado_112.png      — entrada efetiva do classificador
  * grafico_probabilidades.png  — barras das 7 categorias (render proprio)
  * desempenho.png              — distribuicao dos tempos medidos
  * medicoes.json               — numeros brutos das medicoes
  * medicoes.md                 — mesma informacao em tabela

Tudo vem de inferencia real; nada e ilustrativo ou simulado.
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

if hasattr(sys.stdout, "reconfigure"):   # console cp1252 nao suporta °, ×, ≤
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import cv2 as cv  # noqa: E402
import numpy as np  # noqa: E402

from expressao_neural import config  # noqa: E402
from expressao_neural.core import modelos  # noqa: E402
from expressao_neural.core.alinhamento import alinhar_rosto  # noqa: E402
from expressao_neural.core.pipeline import PipelineExpressao  # noqa: E402
from expressao_neural.ui import desenho, tema  # noqa: E402

DESTINO = config.DIR_EVIDENCIAS
FIXTURE = RAIZ / "tests" / "fixtures" / "lena.jpg"


def _estado_da_maquina() -> dict:
    """Frequencia e carga da CPU no momento da medicao.

    Sem isso, um mesmo codigo mede tempos muito diferentes conforme o estado de
    energia do processador e os programas abertos -- e o numero deixa de ser
    auditavel.
    """
    estado: dict[str, object] = {}
    try:
        saida = subprocess.run(
            [
                "powershell", "-NoProfile", "-Command",
                "$c=Get-CimInstance Win32_Processor;"
                "Write-Output \"$($c.Name)|$($c.CurrentClockSpeed)|"
                "$($c.MaxClockSpeed)|$($c.LoadPercentage)|"
                "$($c.NumberOfLogicalProcessors)\"",
            ],
            capture_output=True, text=True, timeout=30,
        )
        partes = saida.stdout.strip().split("|")
        if len(partes) == 5:
            estado = {
                "processador": partes[0].strip(),
                "clock_atual_mhz": int(partes[1]),
                "clock_maximo_mhz": int(partes[2]),
                "carga_cpu_percent": int(partes[3]),
                "nucleos_logicos": int(partes[4]),
            }
            estado["fracao_do_clock_maximo"] = round(
                int(partes[1]) / max(1, int(partes[2])), 3
            )
    except Exception:  # noqa: BLE001 - diagnostico opcional
        estado = {"observacao": "estado da CPU não pôde ser lido"}
    return estado


def _salvar(nome: str, imagem: np.ndarray) -> Path:
    caminho = DESTINO / nome
    ok, buffer = cv.imencode(caminho.suffix, imagem)
    if not ok:
        raise RuntimeError(f"falha ao codificar {nome}")
    caminho.write_bytes(buffer.tobytes())
    print(f"   + {caminho.relative_to(RAIZ)}")
    return caminho


def _grafico_barras(probabilidades: np.ndarray, destaque: int) -> np.ndarray:
    """Renderiza o grafico das 7 categorias em imagem (para o banner)."""
    largura, altura_linha, margem = 680, 46, 26
    altura = margem * 2 + altura_linha * len(config.CLASSES_PT) + 40
    img = np.full((altura, largura, 3), tema.hex_para_bgr(tema.SUPERFICIE), dtype=np.uint8)
    cv.putText(
        img,
        "Pontuacoes do classificador (softmax dos logits)",
        (margem, margem + 6),
        cv.FONT_HERSHEY_DUPLEX,
        0.58,
        tema.hex_para_bgr(tema.PRIMARIA),
        1,
        cv.LINE_AA,
    )
    x0 = margem + 130
    x1 = largura - margem - 90
    for i, nome in enumerate(config.CLASSES_MODELO):
        y = margem + 40 + i * altura_linha
        rotulo = config.CLASSES_PT[i].encode("ascii", "ignore").decode() or nome
        cor = tema.hex_para_bgr(config.CORES_CLASSES[i])
        ativo = i == destaque
        cv.putText(
            img, rotulo, (margem, y + 16), cv.FONT_HERSHEY_DUPLEX, 0.5,
            tema.hex_para_bgr(tema.TEXTO if ativo else tema.TEXTO_SECUNDARIO), 1, cv.LINE_AA,
        )
        cv.rectangle(img, (x0, y), (x1, y + 20), tema.hex_para_bgr(tema.GRADE), -1)
        valor = float(np.clip(probabilidades[i], 0, 1))
        fim = x0 + int((x1 - x0) * valor)
        if fim > x0:
            cv.rectangle(img, (x0, y), (fim, y + 20), cor, -1)
        if ativo:
            cv.rectangle(img, (x0 - 2, y - 2), (x1 + 2, y + 22), cor, 1)
        cv.putText(
            img, f"{valor * 100:5.1f}%", (x1 + 10, y + 16), cv.FONT_HERSHEY_DUPLEX,
            0.5, cor if ativo else tema.hex_para_bgr(tema.TEXTO_FRACO), 1, cv.LINE_AA,
        )
    return img


def _grafico_desempenho(tempos: list[float], titulo: str) -> np.ndarray:
    largura, altura = 680, 320
    img = np.full((altura, largura, 3), tema.hex_para_bgr(tema.SUPERFICIE), dtype=np.uint8)
    cv.putText(
        img, titulo, (24, 32), cv.FONT_HERSHEY_DUPLEX, 0.58,
        tema.hex_para_bgr(tema.PRIMARIA), 1, cv.LINE_AA,
    )
    base_y, topo_y = altura - 56, 60
    maximo = max(tempos) * 1.15
    passo = max(1, len(tempos) // 120)
    amostras = tempos[::passo]
    larg_barra = max(1, (largura - 70) // max(1, len(amostras)))
    for i, valor in enumerate(amostras):
        x = 40 + i * larg_barra
        h = int((base_y - topo_y) * (valor / maximo))
        cv.rectangle(
            img, (x, base_y - h), (x + max(1, larg_barra - 1), base_y),
            tema.hex_para_bgr(tema.PRIMARIA), -1,
        )
    media = sum(tempos) / len(tempos)
    y_media = base_y - int((base_y - topo_y) * (media / maximo))
    cv.line(img, (40, y_media), (largura - 30, y_media),
            tema.hex_para_bgr(tema.ALERTA), 1, cv.LINE_AA)
    cv.putText(
        img, f"media {media:.1f} ms", (largura - 190, y_media - 8),
        cv.FONT_HERSHEY_DUPLEX, 0.46, tema.hex_para_bgr(tema.ALERTA), 1, cv.LINE_AA,
    )
    cv.line(img, (40, base_y), (largura - 30, base_y),
            tema.hex_para_bgr(tema.BORDA_CLARA), 1)
    cv.putText(
        img,
        f"min {min(tempos):.1f} ms   max {max(tempos):.1f} ms   "
        f"{len(tempos)} medicoes   CPU",
        (40, altura - 24), cv.FONT_HERSHEY_DUPLEX, 0.45,
        tema.hex_para_bgr(tema.TEXTO_SECUNDARIO), 1, cv.LINE_AA,
    )
    return img


def _captura_da_interface(imagem: np.ndarray, resultado) -> bool:
    """Abre a janela real, injeta o resultado e salva a tela."""
    try:
        import tkinter as tk

        from PIL import ImageGrab

        from expressao_neural.ui.janela import JanelaPrincipal
    except Exception as exc:  # noqa: BLE001
        print(f"   ! captura da interface indisponível: {exc}")
        return False

    tema.ativar_consciencia_dpi()
    try:
        janela = JanelaPrincipal()
    except tk.TclError as exc:
        print(f"   ! sem ambiente gráfico: {exc}")
        return False

    try:
        janela.geometry("1400x860+30+30")
        janela.update()
        janela._quadro_exibido = imagem
        janela._ultimo_resultado = resultado
        janela._renderizar(imagem, resultado)
        janela._atualizar_paineis(resultado)
        janela._definir_status(
            "Evidência gerada a partir de inferência real sobre imagem local.",
            tema.SUCESSO,
        )
        janela._ind["camera"].definir("imagem local", tema.PRIMARIA)
        janela._ind["modelos"].definir("ativos", tema.SUCESSO)
        for _ in range(12):
            janela.update()
            time.sleep(0.05)

        x = janela.winfo_rootx()
        y = janela.winfo_rooty()
        caixa = (x, y, x + janela.winfo_width(), y + janela.winfo_height())
        tela = ImageGrab.grab(bbox=caixa)
        caminho = DESTINO / "captura_interface.png"
        tela.save(caminho)
        print(f"   + {caminho.relative_to(RAIZ)}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"   ! falha ao capturar a interface: {exc}")
        return False
    finally:
        janela.encerrar()


def main() -> int:
    DESTINO.mkdir(parents=True, exist_ok=True)
    if not FIXTURE.is_file():
        print(f"ERRO: imagem de referência ausente: {FIXTURE}", file=sys.stderr)
        return 1

    print("=" * 72)
    print(" Expressão Neural — geração de evidências reais")
    print("=" * 72)

    faltando = [s for s in modelos.verificar_todos(conferir_hash=False) if not s.existe]
    if faltando:
        print("ERRO: modelos ausentes. Execute configurar.bat.", file=sys.stderr)
        return 1

    imagem = cv.imread(str(FIXTURE))
    t0 = time.perf_counter()
    pipeline = PipelineExpressao(config.Ajustes())
    ms_carga = (time.perf_counter() - t0) * 1000.0

    resultado = pipeline.processar(imagem, suavizar=False)
    if resultado.quantidade_rostos == 0 or resultado.principal is None:
        print("ERRO: nenhum rosto detectado na imagem de referência.", file=sys.stderr)
        return 1
    alvo = resultado.principal

    print("\n1. Imagens")
    _salvar("deteccao_anotada.png", desenho.anotar(imagem, resultado, 0.5))
    _salvar("rosto_alinhado_112.png", alinhar_rosto(imagem, alvo.rosto.pontos))
    _salvar(
        "grafico_probabilidades.png",
        _grafico_barras(alvo.predicao.probabilidades, alvo.indice),
    )

    print("\n2. Medições de desempenho (CPU)")
    estado_cpu = _estado_da_maquina()
    if "clock_atual_mhz" in estado_cpu:
        print(
            f"   estado da CPU: {estado_cpu['clock_atual_mhz']}/"
            f"{estado_cpu['clock_maximo_mhz']} MHz "
            f"({estado_cpu['fracao_do_clock_maximo'] * 100:.0f}% do máximo), "
            f"carga {estado_cpu['carga_cpu_percent']}%"
        )
    for _ in range(5):
        pipeline.processar(imagem)
    tempos_total: list[float] = []
    tempos_det: list[float] = []
    tempos_inf: list[float] = []
    for _ in range(120):
        t = time.perf_counter()
        r = pipeline.processar(imagem)
        tempos_total.append((time.perf_counter() - t) * 1000.0)
        tempos_det.append(r.tempo_deteccao_ms)
        tempos_inf.append(r.tempo_inferencia_ms)
    media = sum(tempos_total) / len(tempos_total)
    print(f"   ciclo completo: média {media:.2f} ms  (~{1000 / media:.1f} /s)")
    _salvar(
        "desempenho.png",
        _grafico_desempenho(tempos_total, "Tempo do ciclo completo por quadro (1 rosto)"),
    )

    print("\n3. Captura da interface")
    tem_captura = _captura_da_interface(imagem, resultado)

    print("\n4. Planilha de medições")

    def _est(v: list[float]) -> dict:
        arr = np.array(v, dtype=np.float64)
        return {
            "media_ms": round(float(arr.mean()), 2),
            "mediana_ms": round(float(np.median(arr)), 2),
            "minimo_ms": round(float(arr.min()), 2),
            "maximo_ms": round(float(arr.max()), 2),
            "desvio_padrao_ms": round(float(arr.std()), 2),
            "amostras": int(arr.size),
        }

    dados = {
        "gerado_em": datetime.now().isoformat(timespec="seconds"),
        "ambiente": {
            "sistema": f"{platform.system()} {platform.release()}",
            "processador": platform.processor() or "não informado",
            "arquitetura": platform.machine(),
            "python": platform.python_version(),
            "opencv": cv.__version__,
            "numpy": np.__version__,
            "aceleracao": "CPU (cv2.dnn, DNN_TARGET_CPU)",
        },
        "estado_da_cpu_na_medicao": estado_cpu,
        "modelos": [
            {
                "chave": m.chave,
                "arquivo": m.arquivo,
                "sha256": m.sha256,
                "tamanho_bytes": m.tamanho_bytes,
                "licenca": m.licenca,
                "pre_treinado": True,
            }
            for m in modelos.TODOS
        ],
        "carregamento_modelos_ms": round(ms_carga, 1),
        "aquecimento_ms": round(pipeline.ms_aquecimento, 2),
        "imagem_referencia": {
            "arquivo": FIXTURE.name,
            "dimensoes": f"{imagem.shape[1]}x{imagem.shape[0]}",
            "rostos_detectados": resultado.quantidade_rostos,
            "confianca_deteccao": round(float(alvo.rosto.confianca), 4),
        },
        "classificacao": {
            "expressao_prevista": alvo.rotulo,
            "classe_modelo": config.CLASSES_MODELO[alvo.indice],
            "confianca": round(float(alvo.confianca), 4),
            "soma_probabilidades": round(float(alvo.predicao.probabilidades.sum()), 6),
            "distribuicao": {
                config.CLASSES_PT[i]: round(float(v), 6)
                for i, v in enumerate(alvo.predicao.probabilidades)
            },
        },
        "desempenho": {
            "ciclo_completo": _est(tempos_total),
            "deteccao": _est(tempos_det),
            "classificacao": _est(tempos_inf),
            "quadros_por_segundo_equivalente": round(1000.0 / media, 1),
        },
        "captura_interface_gerada": tem_captura,
    }
    (DESTINO / "medicoes.json").write_text(
        json.dumps(dados, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"   + {(DESTINO / 'medicoes.json').relative_to(RAIZ)}")

    linhas = [
        "# Medições reais — Expressão Neural",
        "",
        f"Geradas em {dados['gerado_em']} por `scripts/gerar_evidencias.py`.",
        "Todos os números vêm de execução real nesta máquina; nenhum valor é estimado.",
        "",
        "## Ambiente de medição",
        "",
        "| Item | Valor |",
        "|---|---|",
    ]
    for k, v in dados["ambiente"].items():
        linhas.append(f"| {k.replace('_', ' ').capitalize()} | {v} |")
    for k, v in estado_cpu.items():
        linhas.append(f"| {k.replace('_', ' ').capitalize()} | {v} |")
    linhas += [
        "",
        "## Desempenho (1 rosto, imagem "
        f"{dados['imagem_referencia']['dimensoes']}, {len(tempos_total)} amostras)",
        "",
        "| Etapa | Média | Mediana | Mínimo | Máximo | Desvio padrão |",
        "|---|---|---|---|---|---|",
    ]
    for nome, chave in (
        ("Detecção (YuNet)", "deteccao"),
        ("Classificação (MobileFaceNet)", "classificacao"),
        ("Ciclo completo", "ciclo_completo"),
    ):
        e = dados["desempenho"][chave]
        linhas.append(
            f"| {nome} | {e['media_ms']} ms | {e['mediana_ms']} ms | "
            f"{e['minimo_ms']} ms | {e['maximo_ms']} ms | {e['desvio_padrao_ms']} ms |"
        )
    linhas += [
        "",
        f"Equivalente a **{dados['desempenho']['quadros_por_segundo_equivalente']} "
        "ciclos de inferência por segundo** em CPU, sem GPU dedicada.",
        f"Carregamento dos modelos: {dados['carregamento_modelos_ms']} ms "
        f"(aquecimento {dados['aquecimento_ms']} ms).",
        "",
        "## Classificação da imagem de referência",
        "",
        f"Expressão prevista: **{dados['classificacao']['expressao_prevista']}** "
        f"(classe do modelo: `{dados['classificacao']['classe_modelo']}`), "
        f"confiança {dados['classificacao']['confianca'] * 100:.1f}%.",
        f"Soma das probabilidades: {dados['classificacao']['soma_probabilidades']}.",
        "",
        "| Categoria | Probabilidade |",
        "|---|---|",
    ]
    for nome, valor in dados["classificacao"]["distribuicao"].items():
        linhas.append(f"| {nome} | {valor * 100:.2f}% |")
    linhas += [
        "",
        "> Estes números descrevem **desempenho computacional** e **confiança da "
        "previsão**, não acurácia do sistema sobre pessoas reais. Acurácia própria "
        "exige o protocolo de avaliação descrito em `docs/testes.md`.",
        "",
    ]
    (DESTINO / "medicoes.md").write_text("\n".join(linhas), encoding="utf-8")
    print(f"   + {(DESTINO / 'medicoes.md').relative_to(RAIZ)}")

    pipeline.liberar()
    print("\n" + "=" * 72)
    print(" Evidências geradas com sucesso.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
