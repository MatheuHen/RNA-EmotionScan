"""Obtem os modelos ONNX de fontes verificadas (etapa unica, com internet).

Executado por configurar.bat. A aplicacao NUNCA chama este script: o modo
offline exige que os modelos ja estejam em disco antes da apresentacao.

Verificacoes aplicadas a cada arquivo:
  * download para arquivo temporario (nunca sobre o destino final);
  * conferencia de tamanho exato;
  * conferencia de SHA-256 exato;
  * so entao o arquivo e movido para models/.

Ambientes com inspecao TLS corporativa podem recusar o certificado do Python.
Nesse caso o script tenta automaticamente o curl.exe do Windows, que usa o
repositorio de certificados do sistema.
"""
from __future__ import annotations

import os
import shutil
import ssl
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from expressao_neural.core import modelos  # noqa: E402

PEM_WINDOWS = RAIZ / "scripts" / "windows-ca-bundle.pem"
TEMPO_LIMITE = 180


def _barra(baixado: int, total: int, largura: int = 34) -> str:
    if total <= 0:
        return f"{baixado / 1024:.0f} KB"
    fracao = min(1.0, baixado / total)
    cheio = int(fracao * largura)
    return (
        f"[{'=' * cheio}{'.' * (largura - cheio)}] "
        f"{fracao * 100:5.1f}%  {baixado / 1024:7.0f}/{total / 1024:.0f} KB"
    )


def _contexto_ssl() -> ssl.SSLContext | None:
    if PEM_WINDOWS.is_file():
        try:
            ctx = ssl.create_default_context(cafile=str(PEM_WINDOWS))
            return ctx
        except ssl.SSLError:
            pass
    return None


def _baixar_urllib(url: str, destino: Path) -> None:
    requisicao = urllib.request.Request(
        url, headers={"User-Agent": "ExpressaoNeural/1.0 (academico)"}
    )
    ctx = _contexto_ssl()
    with urllib.request.urlopen(requisicao, timeout=TEMPO_LIMITE, context=ctx) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        baixado = 0
        with open(destino, "wb") as saida:
            while True:
                pedaco = resp.read(65536)
                if not pedaco:
                    break
                saida.write(pedaco)
                baixado += len(pedaco)
                print(f"\r      {_barra(baixado, total)}", end="", flush=True)
    print()


def _baixar_curl(url: str, destino: Path) -> None:
    curl = shutil.which("curl") or shutil.which("curl.exe")
    if not curl:
        raise RuntimeError("curl não disponível para a tentativa alternativa")
    print("      tentando novamente com curl (certificados do Windows)…")
    resultado = subprocess.run(
        [curl, "-fsSL", "--max-time", str(TEMPO_LIMITE), "-o", str(destino), url],
        capture_output=True,
        text=True,
    )
    if resultado.returncode != 0:
        raise RuntimeError(
            f"curl falhou (código {resultado.returncode}): "
            f"{resultado.stderr.strip() or 'sem detalhe'}"
        )


def obter(info: modelos.ModeloInfo, forcar: bool = False) -> bool:
    destino = info.caminho
    destino.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n>> {info.nome}")
    print(f"   arquivo : {info.arquivo}")
    print(f"   licença : {info.licenca}")
    print(f"   origem  : {info.origem}")

    if destino.is_file() and not forcar:
        status = modelos.verificar(info, conferir_hash=True)
        if status.ok:
            print("   situação: já presente e com integridade verificada. OK")
            return True
        print(f"   situação: arquivo existente inválido ({status.mensagem}). Baixando novamente.")

    temporario = Path(
        tempfile.mkstemp(prefix=f"{info.chave}_", suffix=".part", dir=str(destino.parent))[1]
    )
    try:
        print(f"   baixando de {info.url}")
        try:
            _baixar_urllib(info.url, temporario)
        except (urllib.error.URLError, ssl.SSLError, TimeoutError, OSError) as exc:
            print(f"\n      falha na primeira tentativa: {exc}")
            _baixar_curl(info.url, temporario)

        tamanho = temporario.stat().st_size
        if tamanho != info.tamanho_bytes:
            raise RuntimeError(
                f"tamanho inesperado: {tamanho} bytes, esperado {info.tamanho_bytes}. "
                "O download pode ter sido interceptado por proxy ou antivírus."
            )
        encontrado = modelos.sha256_do_arquivo(temporario)
        if encontrado != info.sha256:
            raise RuntimeError(
                f"SHA-256 divergente.\n        esperado: {info.sha256}"
                f"\n        obtido  : {encontrado}"
            )

        os.replace(temporario, destino)
        print(f"   integridade: SHA-256 confere ({info.sha256[:16]}…). OK")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"   ERRO: {exc}", file=sys.stderr)
        return False
    finally:
        if temporario.exists():
            try:
                temporario.unlink()
            except OSError:
                pass


def main(argv: list[str] | None = None) -> int:
    forcar = "--forcar" in (argv if argv is not None else sys.argv[1:])
    print("=" * 72)
    print(" Expressão Neural — obtenção dos modelos neurais (ONNX)")
    print(" Esta é a ÚNICA etapa que precisa de internet.")
    print("=" * 72)

    falhas = [info for info in modelos.TODOS if not obter(info, forcar)]

    print("\n" + "=" * 72)
    if falhas:
        print(" RESULTADO: FALHA")
        for info in falhas:
            print(f"   - {info.arquivo} não ficou disponível.")
        print(
            "\n Verifique a conexão e eventual proxy/antivírus com inspeção de TLS.\n"
            " Alternativa manual: baixe os arquivos pelas URLs acima e salve-os em\n"
            f" {modelos.DIR_MODELOS}\n"
            " Depois execute verificar.bat para conferir a integridade."
        )
        return 1
    print(" RESULTADO: todos os modelos estão presentes e verificados.")
    print(" A partir daqui o sistema funciona sem internet.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
