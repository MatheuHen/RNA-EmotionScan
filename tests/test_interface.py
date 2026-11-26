"""Testes da interface: inicializacao, componentes, encerramento e recursos.

Estes testes exigem um ambiente grafico. Em maquinas sem display eles sao
automaticamente ignorados (skip), nunca falham por esse motivo.
"""
from __future__ import annotations

import numpy as np
import pytest

tk = pytest.importorskip("tkinter")


@pytest.fixture()
def raiz():
    try:
        r = tk.Tk()
    except tk.TclError:  # pragma: no cover - ambiente sem display
        pytest.skip("ambiente sem interface gráfica disponível")
    r.withdraw()
    yield r
    try:
        r.destroy()
    except tk.TclError:
        pass


# ------------------------------------------------------------------- tema
def test_tema_aplica_e_resolve_fontes(raiz) -> None:
    from expressao_neural.ui import tema

    fontes = tema.aplicar(raiz)
    assert fontes.familia
    assert isinstance(fontes.titulo, tuple) and len(fontes.titulo) >= 2


def test_conversao_de_cor_para_bgr() -> None:
    from expressao_neural.ui import tema

    assert tema.hex_para_bgr("#ff0000") == (0, 0, 255)
    assert tema.hex_para_bgr("112233") == (0x33, 0x22, 0x11)


def test_cor_de_confianca_reflete_o_limiar() -> None:
    from expressao_neural.ui import tema

    assert tema.cor_confianca(0.9, 0.5) == tema.SUCESSO
    assert tema.cor_confianca(0.35, 0.5) == tema.ALERTA
    assert tema.cor_confianca(0.05, 0.5) == tema.ERRO


# -------------------------------------------------------------- componentes
def test_grafico_aceita_as_sete_probabilidades(raiz) -> None:
    from expressao_neural.ui import componentes, tema

    fontes = tema.aplicar(raiz)
    grafico = componentes.GraficoProbabilidades(raiz, fontes)
    grafico.pack()
    raiz.update_idletasks()

    vetor = np.array([0.01, 0.02, 0.01, 0.80, 0.10, 0.03, 0.03], dtype=np.float32)
    grafico.atualizar(vetor, 3)
    grafico.atualizar(None, -1)  # limpeza ao perder o rosto
    grafico.atualizar(np.zeros(3, dtype=np.float32), 0)  # tamanho errado: ignorado
    assert len(grafico._itens) == 7


def test_medidor_e_selo_atualizam(raiz) -> None:
    from expressao_neural.ui import componentes, tema

    fontes = tema.aplicar(raiz)
    medidor = componentes.MedidorConfianca(raiz, fontes)
    medidor.pack()
    medidor.atualizar(0.73, 0.5)
    medidor.atualizar(1.5, 0.5)  # valor fora da faixa e limitado
    selo = componentes.Selo(raiz, "LOCAL", tema.SUCESSO, fontes.micro)
    selo.atualizar("OFFLINE", tema.ALERTA)
    assert selo.cget("text") == "OFFLINE"


def test_cartao_expoe_corpo_para_os_filhos(raiz) -> None:
    from expressao_neural.ui import componentes, tema

    fontes = tema.aplicar(raiz)
    cartao = componentes.Cartao(raiz, "Título", fontes)
    cartao.pack()
    tk.Label(cartao.corpo, text="conteúdo").pack()
    raiz.update_idletasks()
    assert cartao.corpo.winfo_exists()


# ----------------------------------------------------------------- desenho
def test_anotacao_nao_altera_o_quadro_original() -> None:
    from expressao_neural.core.pipeline import ResultadoQuadro
    from expressao_neural.ui import desenho

    quadro = np.zeros((120, 160, 3), dtype=np.uint8)
    saida = desenho.anotar(quadro, ResultadoQuadro(), 0.5)
    assert saida is not quadro
    assert np.array_equal(quadro, np.zeros((120, 160, 3), dtype=np.uint8))


def test_ajuste_de_area_preserva_proporcao() -> None:
    from expressao_neural.ui import desenho

    quadro = np.zeros((480, 640, 3), dtype=np.uint8)
    saida = desenho.ajustar_para_area(quadro, 800, 600)
    assert saida.shape == (600, 800, 3)


def test_placa_de_mensagem_gera_imagem_valida() -> None:
    from expressao_neural.ui import desenho

    placa = desenho.placa_mensagem(640, 360, ["EMOTIONSCAN", "Aguardando"])
    assert placa.shape == (360, 640, 3)
    assert placa.dtype == np.uint8


# ------------------------------------------------------- janela completa
@pytest.mark.usefixtures("pipeline")
def test_janela_abre_atualiza_e_encerra_liberando_recursos() -> None:
    """Inicializacao e encerramento sem camera: nenhum recurso deve vazar."""
    from expressao_neural.ui.janela import JanelaPrincipal

    try:
        janela = JanelaPrincipal()
    except tk.TclError:  # pragma: no cover
        pytest.skip("ambiente sem interface gráfica disponível")

    try:
        janela.update()
        janela.update_idletasks()
        # A camera nao foi iniciada: nada de trabalhador nem de pipeline carregado.
        assert janela._trabalhador is None
        assert janela._modo in ("parado", "erro")

        # Ajustes refletem na configuracao sem exigir reinicio.
        janela._var_limiar.set(0.70)
        janela._ao_mudar_limiar()
        assert janela.ajustes.limiar_confianca == pytest.approx(0.70)

        janela._var_suavizar.set(False)
        janela._ao_mudar_ajuste()
        assert janela.ajustes.suavizacao_ativa is False

        # Comandos de camera com camera desligada nao devem lancar excecao.
        janela.alternar_pausa()
        janela.parar_camera()
        janela._limpar_resultados()
        janela.update()

        # Modo apresentacao liga e desliga.
        janela.alternar_tela_cheia()
        janela.update_idletasks()
        assert janela._tela_cheia is True
        janela._sair_tela_cheia()
        assert janela._tela_cheia is False

        # Janela Sobre abre e fecha.
        dialogo = janela.abrir_sobre()
        janela.update_idletasks()
        assert dialogo.winfo_exists()
        dialogo.destroy()
    finally:
        janela.encerrar()

    assert janela._trabalhador is None
    assert janela._pipeline is None
    assert janela._foto is None


def test_janela_renderiza_resultado_de_inferencia_real(imagem_rosto, pipeline) -> None:
    """Integracao: pipeline real alimentando os paineis da interface."""
    from expressao_neural.ui.janela import JanelaPrincipal

    try:
        janela = JanelaPrincipal()
    except tk.TclError:  # pragma: no cover
        pytest.skip("ambiente sem interface gráfica disponível")
    try:
        resultado = pipeline.processar(imagem_rosto, suavizar=False)
        assert resultado.quantidade_rostos >= 1
        janela.update_idletasks()
        janela._renderizar(imagem_rosto, resultado)
        janela._atualizar_paineis(resultado)
        janela.update()

        alvo = resultado.principal
        assert alvo is not None
        assert janela._lbl_expressao.cget("text") in (
            alvo.rotulo, "Indefinida",
        )
        assert "confiança" in janela._lbl_confianca.cget("text")
        assert janela._ind["rostos"]._valor.cget("text") == str(
            resultado.quantidade_rostos
        )

        # Sem rosto: a tela e limpa (RF10).
        from expressao_neural.core.pipeline import ResultadoQuadro

        janela._atualizar_paineis(ResultadoQuadro())
        janela.update_idletasks()
        assert janela._lbl_expressao.cget("text") == "Nenhum rosto"
    finally:
        janela.encerrar()
