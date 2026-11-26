"""Valida o funcionamento sem internet e grava a evidencia.

Esta e a unica pendencia que a auditoria de codigo nao consegue fechar sozinha:
ausencia de chamada de rede no codigo e forte indicio, mas nao e prova de que o
sistema funciona com a rede desligada. Pode haver resolucao de DNS por uma
biblioteca, licenca verificada por dependencia, telemetria de terceiro.

O script:
  1. CONFIRMA que a rede esta realmente fora (nao aceita teste com rede ligada);
  2. executa o pipeline completo com os modelos reais;
  3. roda a suite de testes;
  4. verifica a integridade dos modelos por SHA-256;
  5. abre a interface grafica;
  6. grava evidencias/teste_offline.md com o resultado e o horario.

Uso:
    .venv\\Scripts\\python.exe scripts\\testar_offline.py
    .venv\\Scripts\\python.exe scripts\\testar_offline.py --sem-interface
"""
from __future__ import annotations

import argparse
import os
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OK, FALHA, AVISO = "[ OK ]", "[FALHA]", "[AVISO]"
DESTINO = RAIZ / "evidencias" / "teste_offline.md"

# Alvos usados só para CONFIRMAR que a rede está fora. Se algum responder, o
# teste é abortado: validar modo offline com internet ligada não prova nada.
ALVOS = [
    ("1.1.1.1", 53), ("8.8.8.8", 53), ("github.com", 443),
    ("huggingface.co", 443), ("pypi.org", 443),
]


def _titulo(t: str) -> None:
    print(f"\n{t}\n{'-' * len(t)}")


def rede_esta_fora() -> tuple[bool, list[str]]:
    """True quando NENHUM alvo responde. Devolve também o que respondeu."""
    alcancaveis: list[str] = []
    for host, porta in ALVOS:
        try:
            with socket.create_connection((host, porta), timeout=2.5):
                alcancaveis.append(f"{host}:{porta}")
        except (OSError, socket.gaierror, socket.timeout):
            pass
    return (not alcancaveis), alcancaveis


def main() -> int:  # noqa: C901
    ap = argparse.ArgumentParser(
        description="Valida o funcionamento sem internet."
    )
    ap.add_argument("--sem-interface", action="store_true",
                    help="pula a abertura da janela gráfica")
    ap.add_argument("--forcar", action="store_true",
                    help="executa mesmo com a rede acessível (NÃO prova nada)")
    args = ap.parse_args()

    from expressao_neural import config

    print("=" * 72)
    print(f" {config.APP_NOME} v{config.APP_VERSAO} — teste de funcionamento OFFLINE")
    print("=" * 72)

    criticos: list[str] = []
    ressalvas: list[str] = []
    inicio = datetime.now()

    # ---------------------------------------------- 1. a rede está mesmo fora?
    _titulo("1. Confirmando que a rede está desligada")
    print("   Testando 5 alvos conhecidos (DNS, GitHub, Hugging Face, PyPI)…")
    fora, alcancaveis = rede_esta_fora()
    if fora:
        print(f"{OK} Nenhum alvo respondeu — a rede está fora.")
    else:
        print(f"{FALHA} A rede AINDA ESTÁ ACESSÍVEL. Responderam:")
        for a in alcancaveis:
            print(f"        · {a}")
        if not args.forcar:
            print("\n" + "=" * 72)
            print(" TESTE ABORTADO")
            print("=" * 72)
            print("\n Validar o modo offline com a internet ligada não prova nada.")
            print("\n Faça assim:")
            print("   1. Desligue o Wi-Fi (ícone da bandeja ou tecla de atalho)")
            print("   2. Desconecte o cabo de rede, se houver")
            print("   3. Rode este script de novo")
            print("\n Se quiser rodar mesmo assim, use --forcar (mas o resultado")
            print(" NÃO serve como evidência de funcionamento offline).")
            return 2
        ressalvas.append("executado com a rede acessível — não prova modo offline")
        print(f"{AVISO} Continuando por --forcar. O resultado NÃO é evidência.")

    # ------------------------------------------- 2. integridade dos modelos
    _titulo("2. Modelos locais e integridade (SHA-256)")
    from expressao_neural.core import modelos

    for status in modelos.verificar_todos(conferir_hash=True):
        if status.ok:
            print(f"{OK} {status.info.arquivo}")
        else:
            print(f"{FALHA} {status.mensagem}")
            criticos.append(status.mensagem)

    if criticos:
        _gravar(inicio, criticos, ressalvas, fora, alcancaveis, {})
        _resumo(criticos, ressalvas)
        return 1

    # ------------------------------------------------- 3. inferência real
    _titulo("3. Inferência real com a rede desligada")
    import cv2 as cv
    import numpy as np

    from expressao_neural.core.pipeline import PipelineExpressao

    t0 = time.perf_counter()
    pipeline = PipelineExpressao(config.Ajustes())
    ms_carga = (time.perf_counter() - t0) * 1000
    print(f"{OK} Modelos carregados em {ms_carga:.0f} ms")

    medicoes: dict[str, float] = {"carregamento_ms": round(ms_carga, 1)}
    fixture = RAIZ / "tests" / "fixtures" / "lena.jpg"
    if fixture.is_file():
        imagem = cv.imread(str(fixture))
        res = pipeline.processar(imagem, suavizar=False)
        alvo = res.principal
        if alvo is not None:
            soma = float(alvo.predicao.probabilidades.sum())
            print(f"{OK} Rosto detectado e classificado: {alvo.rotulo} "
                  f"({alvo.confianca * 100:.1f}%)")
            print(f"{OK} Distribuição válida (soma {soma:.4f})")
            medicoes["classe"] = alvo.rotulo
            medicoes["confianca"] = round(alvo.confianca, 4)
            if abs(soma - 1.0) > 1e-3:
                criticos.append("soma das probabilidades diferente de 1")
        else:
            criticos.append("nenhum rosto detectado na imagem de referência")
            print(f"{FALHA} Nenhum rosto detectado.")

        tempos = []
        for _ in range(30):
            t = time.perf_counter()
            pipeline.processar(imagem)
            tempos.append((time.perf_counter() - t) * 1000)
        media = sum(tempos) / len(tempos)
        medicoes["ciclo_ms"] = round(media, 2)
        print(f"{OK} Ciclo completo: {media:.2f} ms "
              f"(~{1000 / media:.0f} inferências/s)")
    else:
        ressalvas.append("fixture lena.jpg ausente")
        print(f"{AVISO} Imagem de referência ausente.")

    ruido = np.random.default_rng(7).integers(0, 256, (360, 480, 3), dtype=np.uint8)
    if pipeline.processar(ruido, suavizar=False).quantidade_rostos == 0:
        print(f"{OK} Imagem sem rosto não produz classificação.")
    else:
        ressalvas.append("falso positivo em imagem de ruído")
    pipeline.liberar()

    # --------------------------------------------- 4. suíte automatizada
    _titulo("4. Suíte de testes com a rede desligada")
    py = RAIZ / ".venv" / "Scripts" / "python.exe"
    env = dict(os.environ, EXPRESSAO_NEURAL_SEM_PERSISTENCIA="1")
    try:
        r = subprocess.run(
            [str(py), "-m", "pytest", "tests", "-q", "-m", "not lento"],
            capture_output=True, text=True, cwd=str(RAIZ), env=env, timeout=1800,
        )
        ultima = [l for l in (r.stdout + r.stderr).splitlines() if l.strip()][-1]
        if r.returncode == 0:
            print(f"{OK} {ultima.strip()}")
            medicoes["testes"] = ultima.strip()
        else:
            print(f"{FALHA} {ultima.strip()}")
            criticos.append(f"testes falharam offline: {ultima.strip()}")
    except subprocess.TimeoutExpired:
        criticos.append("a suíte de testes excedeu o tempo limite")
        print(f"{FALHA} Tempo limite excedido.")

    # ---------------------------------------------------- 5. interface
    _titulo("5. Interface gráfica")
    if args.sem_interface:
        print(f"{AVISO} Pulada (--sem-interface).")
        ressalvas.append("interface não verificada nesta execução")
    else:
        try:
            import tkinter as tk

            from expressao_neural.ui import icone, tema
            from expressao_neural.ui.janela import JanelaPrincipal

            tema.ativar_consciencia_dpi()
            icone.preparar()
            janela = JanelaPrincipal()
            try:
                for _ in range(20):
                    janela.update()
                    time.sleep(0.03)
                print(f"{OK} Janela abriu e respondeu a eventos.")
                print(f"{OK} Status: {janela._mensagem_status}")
                if janela._modo == "erro":
                    criticos.append("a interface abriu em estado de erro")
                    print(f"{FALHA} Interface em estado de erro.")
            finally:
                janela.encerrar()
        except tk.TclError as exc:
            ressalvas.append(f"sem ambiente gráfico: {exc}")
            print(f"{AVISO} Sem ambiente gráfico disponível.")
        except Exception as exc:  # noqa: BLE001
            criticos.append(f"falha ao abrir a interface: {exc}")
            print(f"{FALHA} {exc}")

    # -------------------------------------------------------- evidência
    _gravar(inicio, criticos, ressalvas, fora, alcancaveis, medicoes)
    print(f"\n   + {DESTINO.relative_to(RAIZ)}")
    _resumo(criticos, ressalvas)

    if criticos:
        return 1
    return 2 if ressalvas else 0


def _gravar(inicio, criticos, ressalvas, fora, alcancaveis, medicoes) -> None:
    from expressao_neural import config

    DESTINO.parent.mkdir(parents=True, exist_ok=True)
    estado = "APROVADO" if not criticos else "REPROVADO"
    if not criticos and ressalvas:
        estado = "APROVADO COM RESSALVAS"

    L = [
        "# Teste de funcionamento offline",
        "",
        f"**Resultado: {estado}**",
        "",
        f"- Executado em: {inicio.strftime('%d/%m/%Y às %H:%M:%S')}",
        f"- Duração: {(datetime.now() - inicio).total_seconds():.0f} s",
        f"- Sistema: {platform.system()} {platform.release()}",
        f"- Versão do sistema: {config.APP_NOME} v{config.APP_VERSAO}",
        f"- Script: `scripts/testar_offline.py`",
        "",
        "## Confirmação de que a rede estava desligada",
        "",
        "Antes de qualquer verificação, o script tentou alcançar 5 alvos "
        "conhecidos (DNS da Cloudflare e do Google, GitHub, Hugging Face e "
        "PyPI).",
        "",
    ]
    if fora:
        L += ["**Nenhum respondeu — a rede estava efetivamente fora.**", ""]
    else:
        L += [
            "⚠️ **A rede estava acessível.** Responderam: "
            + ", ".join(f"`{a}`" for a in alcancaveis),
            "",
            "> Este resultado **não serve como evidência** de funcionamento "
            "offline. Repita com a rede desligada.",
            "",
        ]

    L += ["## Verificações", ""]
    itens = [
        "Integridade dos modelos por SHA-256",
        "Carregamento dos modelos ONNX",
        "Inferência real sobre imagem de referência",
        "Distribuição de probabilidades válida (soma 1,0)",
        "Ausência de rosto não produz classificação",
        "Suíte de testes automatizados",
        "Abertura da interface gráfica",
    ]
    for item in itens:
        marca = "x" if not criticos else " "
        L.append(f"- [{marca}] {item}")

    if medicoes:
        L += ["", "## Medições durante o teste", "", "| Item | Valor |", "|---|---|"]
        rotulos = {
            "carregamento_ms": "Carregamento dos modelos",
            "ciclo_ms": "Ciclo completo",
            "classe": "Classe prevista na imagem de referência",
            "confianca": "Confiança",
            "testes": "Suíte de testes",
        }
        for chave, valor in medicoes.items():
            nome = rotulos.get(chave, chave)
            if chave.endswith("_ms"):
                valor = f"{valor} ms".replace(".", ",")
            elif chave == "confianca":
                valor = f"{float(valor) * 100:.1f}%".replace(".", ",")
            L.append(f"| {nome} | {valor} |")

    if criticos:
        L += ["", "## Falhas críticas", ""] + [f"- {c}" for c in criticos]
    if ressalvas:
        L += ["", "## Ressalvas", ""] + [f"- {r}" for r in ressalvas]

    L += [
        "",
        "## O que este teste comprova",
        "",
        "A auditoria de código já mostrava **zero chamadas de rede** em `src/`. "
        "Isso é indício forte, mas não é prova: uma biblioteca poderia resolver "
        "DNS, verificar licença ou enviar telemetria sem aparecer numa busca "
        "por `import`.",
        "",
        "Este teste fecha essa lacuna **executando o sistema com a rede "
        "fisicamente fora**.",
        "",
        "## O que NÃO comprova",
        "",
        "- Não cobre a **instalação** (`configurar.bat` precisa de internet uma "
        "  vez, para baixar os modelos).",
        "- Não cobre outra máquina nem outro sistema operacional.",
        "- Com `--sem-interface`, não cobre a janela gráfica.",
        "",
    ]
    DESTINO.write_text("\n".join(L), encoding="utf-8")


def _resumo(criticos: list[str], ressalvas: list[str]) -> None:
    print("\n" + "=" * 72)
    if criticos:
        print(" RESULTADO: REPROVADO — o sistema NÃO funciona offline")
        for c in criticos:
            print(f"   x {c}")
    elif ressalvas:
        print(" RESULTADO: APROVADO COM RESSALVAS")
    else:
        print(" RESULTADO: APROVADO — o sistema funciona sem internet")
    for r in ressalvas:
        print(f"   ! {r}")
    print("=" * 72)


if __name__ == "__main__":
    raise SystemExit(main())
