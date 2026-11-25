"""Expressao Neural -- ponto de entrada da aplicacao.

Uso:
    python app.py                 abre a interface grafica
    python app.py --verificar     apenas diagnostica o ambiente (sem janela)
    python app.py --imagem X.jpg  classifica uma imagem no terminal (sem janela)

Nenhum modelo e baixado aqui: o funcionamento offline depende de configurar.bat
ter sido executado uma vez.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
SRC = RAIZ / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _verificar(conferir_hash: bool) -> int:
    from expressao_neural import config
    from expressao_neural.utils import ambiente

    print(f"{config.APP_NOME} v{config.APP_VERSAO} — verificação de ambiente\n")
    rel = ambiente.verificar_ambiente(conferir_hash=conferir_hash)
    print(rel.texto())
    if rel.ok:
        print("\nAmbiente pronto. Execute iniciar.bat para abrir a aplicação.")
        return 0
    print("\nAmbiente INCOMPLETO. Execute configurar.bat (requer internet nesta etapa).")
    return 1


def _classificar_imagem(caminho: str) -> int:
    from expressao_neural import config
    from expressao_neural.core import imagem_estatica
    from expressao_neural.core.detector import ErroModelo
    from expressao_neural.core.pipeline import PipelineExpressao

    try:
        imagem = imagem_estatica.carregar(caminho)
    except imagem_estatica.ImagemInvalida as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        return 2

    try:
        pipeline = PipelineExpressao(config.Ajustes())
    except ErroModelo as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        return 3

    resultado = pipeline.processar(
        imagem_estatica.redimensionar_para_analise(imagem), suavizar=False
    )
    if resultado.erro:
        print(f"ERRO: {resultado.erro}", file=sys.stderr)
        return 4
    if resultado.quantidade_rostos == 0:
        print("Nenhum rosto detectado — nenhuma classificação realizada.")
        return 0

    print(
        f"{resultado.quantidade_rostos} rosto(s) | detecção "
        f"{resultado.tempo_deteccao_ms:.1f} ms | inferência "
        f"{resultado.tempo_inferencia_ms:.1f} ms\n"
    )
    for i, r in enumerate(resultado.rostos, 1):
        print(f"Rosto {i} — caixa {r.rosto.caixa} (detecção {r.rosto.confianca:.3f})")
        print(f"  Expressão: {r.rotulo}  |  confiança {r.confianca * 100:.1f}%")
        for nome, valor in zip(config.CLASSES_PT, r.probabilidades):
            marca = "<<" if nome == r.rotulo else "  "
            print(f"    {nome:<9} {valor * 100:6.2f}% {marca}")
        print()
    pipeline.liberar()
    print(config.AVISO_PRIVACIDADE)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="expressao-neural",
        description="Reconhecimento de expressões faciais em tempo real com redes neurais.",
    )
    parser.add_argument(
        "--verificar", action="store_true", help="diagnostica o ambiente e sai"
    )
    parser.add_argument(
        "--sem-hash",
        action="store_true",
        help="na verificação, não confere o SHA-256 dos modelos (mais rápido)",
    )
    parser.add_argument(
        "--imagem", metavar="ARQUIVO", help="classifica uma imagem local e sai"
    )
    args = parser.parse_args(argv)

    if args.verificar:
        return _verificar(conferir_hash=not args.sem_hash)
    if args.imagem:
        return _classificar_imagem(args.imagem)

    try:
        from expressao_neural.ui.janela import executar
    except ImportError as exc:
        print(
            "ERRO: dependências ausentes para abrir a interface "
            f"({exc}).\nExecute configurar.bat para instalar o ambiente.",
            file=sys.stderr,
        )
        return 5
    return executar()


if __name__ == "__main__":
    raise SystemExit(main())
