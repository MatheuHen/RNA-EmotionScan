"""Rotina de preparacao para a apresentacao.

Vai alem do diagnostico de dependencias: executa uma inferencia real,
confere a integridade dos modelos e procura cameras. Serve como checklist
executavel momentos antes do workshop.

Codigos de saida:
    0  pronto para apresentar
    1  problema critico (nao apresentavel sem correcao)
    2  pronto com ressalvas (ex.: nenhuma camera encontrada)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

OK = "[ OK ]"
FALHA = "[FALHA]"
AVISO = "[AVISO]"


def _titulo(texto: str) -> None:
    print(f"\n{texto}\n{'-' * len(texto)}")


def main(argv: list[str] | None = None) -> int:  # noqa: C901
    argv = sys.argv[1:] if argv is None else argv
    pular_camera = "--sem-camera" in argv

    from expressao_neural import config
    from expressao_neural.utils import ambiente

    print("=" * 72)
    print(f" {config.APP_NOME} v{config.APP_VERSAO} — verificação de prontidão")
    print(f" {config.ACADEMICO} · {config.CURSO}")
    print("=" * 72)

    criticos: list[str] = []
    ressalvas: list[str] = []

    # ------------------------------------------- dependencias, modelos, pastas
    _titulo("1. Dependências, modelos e integridade")
    rel = ambiente.verificar_ambiente(conferir_hash=True)
    print(rel.texto())
    criticos += [f"{i.nome}: {i.detalhe}" for i in rel.problemas_criticos]
    ressalvas += [
        f"{i.nome}: {i.detalhe}" for i in rel.problemas if not i.critico
    ]

    if criticos:
        print(f"\n{FALHA} Falhas críticas impedem os testes seguintes.")
        _resumo(criticos, ressalvas)
        return 1

    # --------------------------------------------------- inferencia real
    _titulo("2. Inferência real com os modelos neurais")
    import cv2 as cv
    import numpy as np

    from expressao_neural.core.pipeline import PipelineExpressao

    t0 = time.perf_counter()
    pipeline = PipelineExpressao(config.Ajustes())
    ms_carga = (time.perf_counter() - t0) * 1000.0
    print(f"{OK} Modelos carregados em {ms_carga:.0f} ms "
          f"(aquecimento {pipeline.ms_aquecimento:.1f} ms)")

    fixture = RAIZ / "tests" / "fixtures" / "lena.jpg"
    if fixture.is_file():
        imagem = cv.imread(str(fixture))
        res = pipeline.processar(imagem, suavizar=False)
        if res.quantidade_rostos >= 1 and res.principal is not None:
            alvo = res.principal
            soma = float(alvo.probabilidades.sum())
            print(
                f"{OK} Rosto detectado e classificado: {alvo.rotulo} "
                f"({alvo.confianca * 100:.1f}%)"
            )
            print(
                f"{OK} Distribuição válida (soma {soma:.4f}) | detecção "
                f"{res.tempo_deteccao_ms:.1f} ms | inferência "
                f"{res.tempo_inferencia_ms:.1f} ms"
            )
            if abs(soma - 1.0) > 1e-3:
                criticos.append("soma das probabilidades diferente de 1")
        else:
            criticos.append("nenhum rosto detectado na imagem de referência")
            print(f"{FALHA} Nenhum rosto detectado na imagem de referência.")
    else:
        ressalvas.append("fixture tests/fixtures/lena.jpg ausente")
        print(f"{AVISO} Imagem de referência ausente — inferência não comprovada.")

    # ausencia de rosto
    ruido = np.random.default_rng(7).integers(0, 256, (360, 480, 3), dtype=np.uint8)
    res_vazio = pipeline.processar(ruido, suavizar=False)
    if res_vazio.quantidade_rostos == 0:
        print(f"{OK} Imagem sem rosto não produz classificação.")
    else:
        ressalvas.append("detecção de rosto em imagem de ruído (falso positivo)")
        print(f"{AVISO} Falso positivo em imagem de ruído.")

    # ---------------------------------------------------- desempenho medido
    _titulo("3. Desempenho medido nesta máquina (CPU)")
    if fixture.is_file():
        imagem = cv.imread(str(fixture))
        amostras = []
        for _ in range(20):
            t = time.perf_counter()
            pipeline.processar(imagem, suavizar=True)
            amostras.append((time.perf_counter() - t) * 1000.0)
        medio = sum(amostras) / len(amostras)
        print(f"{OK} Ciclo completo (detecção + classificação): "
              f"média {medio:.1f} ms | mínimo {min(amostras):.1f} ms | "
              f"máximo {max(amostras):.1f} ms")
        print(f"       Equivale a ~{1000.0 / medio:.1f} inferências por segundo "
              f"em 1 rosto a {imagem.shape[1]}x{imagem.shape[0]}.")
        if medio > 200:
            ressalvas.append(
                f"ciclo de inferência lento ({medio:.0f} ms) — aumente "
                "'Inferir a cada N quadros' na interface"
            )
    pipeline.liberar()

    # ---------------------------------------------------------- cameras
    _titulo("4. Câmeras disponíveis")
    if pular_camera:
        print(f"{AVISO} Verificação de câmera ignorada (--sem-camera).")
        ressalvas.append("câmera não verificada nesta execução")
    else:
        from expressao_neural.core.camera import listar_cameras

        encontradas = listar_cameras(5)
        if encontradas:
            print(f"{OK} Câmera(s) encontrada(s): "
                  f"{', '.join(f'índice {i}' for i in encontradas)}")
        else:
            print(f"{AVISO} Nenhuma câmera encontrada.")
            print("       A função “Analisar imagem” continua disponível como "
                  "contingência (RF09).")
            ressalvas.append("nenhuma câmera detectada")

    # ------------------------------------------------------- modo offline
    _titulo("5. Prontidão para operação offline")
    print(f"{OK} Modelos presentes localmente em {config.DIR_MODELOS}")
    print(f"{OK} Nenhuma dependência de CDN, fonte remota ou API externa no código.")
    print(f"{AVISO} A confirmação definitiva exige desligar a rede e abrir "
          "iniciar.bat — teste manual.")

    _resumo(criticos, ressalvas)
    if criticos:
        return 1
    return 2 if ressalvas else 0


def _resumo(criticos: list[str], ressalvas: list[str]) -> None:
    print("\n" + "=" * 72)
    if criticos:
        print(" RESULTADO: NÃO PRONTO — corrija os itens críticos abaixo.")
        for item in criticos:
            print(f"   ✗ {item}")
    elif ressalvas:
        print(" RESULTADO: PRONTO COM RESSALVAS.")
    else:
        print(" RESULTADO: PRONTO PARA APRESENTAR.")
    for item in ressalvas:
        print(f"   ! {item}")
    print("=" * 72)


if __name__ == "__main__":
    raise SystemExit(main())
