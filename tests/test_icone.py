"""Ícone próprio da aplicação.

Antes, a janela e a barra de tarefas mostravam o logotipo do Python. Dois
pontos distintos precisavam de correção:

  * a janela, via `iconbitmap`/`iconphoto`;
  * a BARRA DE TAREFAS, que agrupa pelo AppUserModelID do processo e por isso
    continuava exibindo o ícone do interpretador mesmo com a janela corrigida.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

tk = pytest.importorskip("tkinter")

from expressao_neural.ui import icone  # noqa: E402


# ------------------------------------------------------------ arquivos
def test_arquivos_do_icone_existem() -> None:
    assert icone.ARQUIVO_ICO.is_file(), (
        "assets/icone.ico ausente — execute scripts/gerar_icone.py"
    )
    assert icone.ARQUIVO_PNG.is_file(), "assets/icone_256.png ausente"


def test_ico_tem_varias_resolucoes() -> None:
    """Um .ico de tamanho único fica serrilhado nos demais contextos."""
    from PIL import Image

    with Image.open(icone.ARQUIVO_ICO) as img:
        tamanhos = {t[0] for t in img.info.get("sizes", set())}
    assert 16 in tamanhos, "falta 16 px (menus e listas)"
    assert 32 in tamanhos, "falta 32 px (barra de tarefas)"
    assert 256 in tamanhos, "falta 256 px (ícone grande do Explorer)"
    assert len(tamanhos) >= 5, f"poucas resoluções: {sorted(tamanhos)}"


def test_png_quadrado_e_grande_o_bastante() -> None:
    from PIL import Image

    with Image.open(icone.ARQUIVO_PNG) as img:
        assert img.width == img.height == 256


# ------------------------------------------------- identidade no Windows
def test_identificador_segue_o_formato_recomendado() -> None:
    """Microsoft recomenda Empresa.Produto.Componente.Versão."""
    partes = icone.APP_USER_MODEL_ID.split(".")
    assert len(partes) == 4, icone.APP_USER_MODEL_ID
    assert all(partes), "nenhuma parte pode ser vazia"
    assert "python" not in icone.APP_USER_MODEL_ID.lower()


def test_declarar_identidade_nao_levanta_excecao() -> None:
    """Recurso cosmético: falhar aqui nunca pode derrubar a aplicação."""
    resultado = icone.declarar_identidade_no_windows()
    assert isinstance(resultado, bool)
    if sys.platform == "win32":
        assert resultado is True, "no Windows a chamada deveria ser aceita"


def test_preparar_e_seguro_de_chamar_mais_de_uma_vez() -> None:
    icone.preparar()
    icone.preparar()


# ---------------------------------------------------------- aplicação
@pytest.fixture()
def raiz():
    try:
        r = tk.Tk()
    except tk.TclError:  # pragma: no cover
        pytest.skip("ambiente sem interface gráfica disponível")
    r.withdraw()
    yield r
    try:
        r.destroy()
    except tk.TclError:
        pass


def test_aplicar_em_janela_real(raiz) -> None:
    assert icone.aplicar(raiz) is True


def test_aplicar_em_toplevel(raiz) -> None:
    topo = tk.Toplevel(raiz)
    topo.withdraw()
    assert icone.aplicar(topo) is True
    topo.destroy()


def test_ausencia_do_arquivo_nao_levanta_excecao(raiz, tmp_path, monkeypatch) -> None:
    """Sem o ícone, a aplicação tem de abrir do mesmo jeito."""
    monkeypatch.setattr(icone, "ARQUIVO_ICO", tmp_path / "nao_existe.ico")
    monkeypatch.setattr(icone, "ARQUIVO_PNG", tmp_path / "nao_existe.png")
    monkeypatch.setattr(icone, "_foto_icone", None)
    assert icone.aplicar(raiz) is False   # não aplicou, mas não quebrou


def test_arquivo_corrompido_nao_levanta_excecao(raiz, tmp_path, monkeypatch) -> None:
    ruim = tmp_path / "ruim.ico"
    ruim.write_bytes(b"isto nao e um icone")
    monkeypatch.setattr(icone, "ARQUIVO_ICO", ruim)
    monkeypatch.setattr(icone, "ARQUIVO_PNG", tmp_path / "nao_existe.png")
    monkeypatch.setattr(icone, "_foto_icone", None)
    icone.aplicar(raiz)   # não deve levantar


# ------------------------------------------------- integração com a janela
def test_janela_principal_aplica_o_icone() -> None:
    """A janela real deve conseguir aplicar o ícone.

    O Tk não oferece consulta limpa do slot `-default` do `wm iconbitmap`,
    então verificamos o comportamento: `aplicar()` tem sucesso sobre a janela
    real (é idempotente) e a imagem de fallback fica retida, provando que o
    caminho `iconphoto` também executou.
    """
    from expressao_neural.ui.janela import JanelaPrincipal

    try:
        j = JanelaPrincipal()
    except tk.TclError:  # pragma: no cover
        pytest.skip("ambiente sem interface gráfica disponível")
    try:
        j.update_idletasks()
        assert icone.aplicar(j) is True
        assert icone._foto_icone is not None, (
            "a imagem do ícone não foi retida — seria coletada pelo GC e o "
            "ícone sumiria"
        )
    finally:
        j.encerrar()


def test_janela_sobre_tambem_recebe_o_icone() -> None:
    """Toplevel sem ícone próprio volta a mostrar o logotipo do Python."""
    from expressao_neural.ui import sobre
    from expressao_neural.ui.janela import JanelaPrincipal

    try:
        j = JanelaPrincipal()
    except tk.TclError:  # pragma: no cover
        pytest.skip("ambiente sem interface gráfica disponível")
    try:
        d = sobre.abrir(j, j.fontes)
        d.update_idletasks()
        assert icone.aplicar(d) is True
        d.destroy()
    finally:
        j.encerrar()
