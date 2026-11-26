"""Regressões da revisão de segurança e robustez.

Cada teste corresponde a um achado concreto da auditoria. Todos falham no
código anterior às correções.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from expressao_neural.config import Ajustes
from expressao_neural.core import camera as mod_camera
from expressao_neural.core import imagem_estatica as ie
from expressao_neural.core.alinhamento import ErroAlinhamento, matriz_alinhamento

from .conftest import modelos_disponiveis


# ======================================================= achado 3: bomba de descompressão
def test_decompression_bomb_vira_imagem_invalida(tmp_path: Path, monkeypatch) -> None:
    """DecompressionBombError não herda de OSError/ValueError e escapava."""
    alvo = tmp_path / "bomba.png"
    # Imagem real pequena; o limite do Pillow é reduzido para dispará-la.
    Image.new("RGB", (900, 900), (10, 20, 30)).save(alvo)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1000)

    with pytest.raises(ie.ImagemInvalida, match="decompression bomb"):
        ie.carregar(alvo)


def test_bomba_nao_escapa_como_excecao_crua(tmp_path: Path, monkeypatch) -> None:
    """O chamador só trata ImagemInvalida: nada mais pode escapar daqui."""
    alvo = tmp_path / "bomba2.png"
    Image.new("RGB", (800, 800), (0, 0, 0)).save(alvo)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 500)
    try:
        ie.carregar(alvo)
    except ie.ImagemInvalida:
        pass
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"exceção não tratada escapou de carregar(): {exc!r}")


# ============================================= achado 6: lstsq / valores não finitos
@pytest.mark.parametrize(
    "pontos",
    [
        np.zeros((5, 2), dtype=np.float32),              # todos idênticos
        np.array([[i, i] for i in range(5)], np.float32),  # colineares
        np.full((5, 2), 1e30, dtype=np.float32),          # escala extrema
    ],
)
def test_alinhamento_degenerado_vira_erro_alinhamento(pontos) -> None:
    """Qualquer degeneração deve virar ErroAlinhamento, nunca LinAlgError."""
    try:
        matriz_alinhamento(pontos)
    except ErroAlinhamento:
        pass
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"exceção inesperada no alinhamento: {exc!r}")


# ====================================== achado 2: parar() informa se a câmera ficou presa
class _CameraLenta:
    """Dublê cujo read() demora mais que o timeout de parar()."""

    def __init__(self, atraso: float = 4.0) -> None:
        self._atraso = atraso
        self.liberada = False

    def isOpened(self):  # noqa: N802
        return True

    def read(self):
        time.sleep(self._atraso)
        return True, np.zeros((48, 64, 3), dtype=np.uint8)

    def set(self, *_a):
        return True

    def release(self):
        self.liberada = True


class _PipelineInerte:
    def processar(self, _quadro, suavizar=None):
        from expressao_neural.core.pipeline import ResultadoQuadro

        return ResultadoQuadro()

    def reiniciar_estado(self) -> None:
        pass


def test_parar_devolve_false_quando_a_thread_nao_encerra(monkeypatch) -> None:
    monkeypatch.setattr(
        mod_camera.cv, "VideoCapture", lambda *_a, **_k: _CameraLenta(4.0)
    )
    t = mod_camera.TrabalhadorCamera(_PipelineInerte(), Ajustes())
    t.start()
    try:
        time.sleep(0.3)
        assert t.parar(timeout=0.5) is False, (
            "parar() deve informar que a câmera ainda está presa"
        )
        assert t.is_alive()
    finally:
        t.parar(timeout=6.0)


def test_parar_devolve_true_quando_encerra_normalmente(monkeypatch) -> None:
    class _Rapida(_CameraLenta):
        def read(self):
            return True, np.zeros((48, 64, 3), dtype=np.uint8)

    monkeypatch.setattr(mod_camera.cv, "VideoCapture", lambda *_a, **_k: _Rapida())
    t = mod_camera.TrabalhadorCamera(_PipelineInerte(), Ajustes(fps_alvo=60))
    t.start()
    assert t.aguardar_pronto(5.0)
    assert t.parar(timeout=3.0) is True
    assert not t.is_alive()


# ============================ achado 5: capture liberado não é reutilizado após troca falha
def test_troca_de_camera_falha_nao_reutiliza_capture_liberado(monkeypatch) -> None:
    criadas: list[_CameraLenta] = []
    permitir = {"abrir": True}

    class _Simples(_CameraLenta):
        def read(self):
            return True, np.zeros((48, 64, 3), dtype=np.uint8)

    def fabricar(indice, *_a, **_k):
        if not permitir["abrir"]:
            cam = _Simples()
            cam._aberta = False
            criadas.append(cam)
            # Simula dispositivo que não abre.
            cam.isOpened = lambda: False  # type: ignore[method-assign]
            return cam
        cam = _Simples()
        criadas.append(cam)
        return cam

    monkeypatch.setattr(mod_camera.cv, "VideoCapture", fabricar)
    t = mod_camera.TrabalhadorCamera(_PipelineInerte(), Ajustes(fps_alvo=60))
    t.start()
    try:
        assert t.aguardar_pronto(5.0)
        primeira = criadas[0]
        permitir["abrir"] = False
        t.trocar_camera(1)
        # Espera a troca falhar e o erro ser registrado.
        prazo = time.time() + 5.0
        while t.erro is None and time.time() < prazo:
            time.sleep(0.05)
        assert t.erro is not None
        assert primeira.liberada, "o capture anterior deve ter sido liberado"
        assert t._cap is None or t._cap is not primeira, (
            "não pode continuar usando o capture já liberado"
        )
    finally:
        t.parar(timeout=5.0)


# ============================ achado 1: acesso concorrente aos modelos é serializado
@modelos_disponiveis
def test_pipeline_suporta_uso_concorrente(pipeline, imagem_rosto) -> None:
    """Duas threads processando tamanhos diferentes não podem corromper nada.

    Sem o RLock, `setInputSize`/`detect` e `setInput`/`forward` se intercalam
    entre threads e o resultado é indeterminado (ou crash nativo).
    """
    import cv2 as cv

    pequena = cv.resize(imagem_rosto, (320, 320))
    erros: list[BaseException] = []
    resultados: list[int] = []

    def trabalhar(img: np.ndarray, repeticoes: int) -> None:
        try:
            for _ in range(repeticoes):
                res = pipeline.processar(img, suavizar=False)
                if res.erro:
                    erros.append(RuntimeError(res.erro))
                resultados.append(res.quantidade_rostos)
        except BaseException as exc:  # noqa: BLE001
            erros.append(exc)

    threads = [
        threading.Thread(target=trabalhar, args=(imagem_rosto, 30)),
        threading.Thread(target=trabalhar, args=(pequena, 30)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(90)
        assert not t.is_alive(), "thread travou — possível deadlock"

    assert not erros, f"erros durante uso concorrente: {erros[:3]}"
    assert len(resultados) == 60


@modelos_disponiveis
def test_ajustes_concorrentes_com_processamento(pipeline, imagem_rosto) -> None:
    """Mudar limiares enquanto a inferência roda não pode quebrar nada."""
    erros: list[BaseException] = []
    parar = threading.Event()

    def processar() -> None:
        try:
            for _ in range(60):  # limite rígido: o teste nunca roda indefinidamente
                if parar.is_set():
                    break
                pipeline.processar(imagem_rosto, suavizar=True)
        except BaseException as exc:  # noqa: BLE001
            erros.append(exc)

    def ajustar() -> None:
        try:
            for i in range(40):
                pipeline.aplicar_ajustes(
                    Ajustes(
                        limiar_deteccao=0.70 + (i % 3) * 0.05,
                        multiplos_rostos=bool(i % 2),
                        suavizacao_alfa=0.2 + (i % 4) * 0.1,
                    )
                )
        except BaseException as exc:  # noqa: BLE001
            erros.append(exc)

    t1 = threading.Thread(target=processar)
    t2 = threading.Thread(target=ajustar)
    t1.start()
    t2.start()
    t2.join(60)
    parar.set()
    t1.join(60)

    assert not t1.is_alive() and not t2.is_alive()
    assert not erros, f"erros com ajustes concorrentes: {erros[:3]}"


# =============== achado 1 (lado da UI): analisar imagem sempre encerra a captura
def test_analisar_imagem_encerra_captura_mesmo_em_modo_erro(
    monkeypatch, tmp_path, caminho_rosto
) -> None:
    """Em modo 'erro' a thread continua viva: ela precisa ser encerrada antes."""
    tk = pytest.importorskip("tkinter")
    from expressao_neural.ui.janela import JanelaPrincipal

    class _Falha(_CameraLenta):
        def read(self):
            return False, None  # erro transitório: a thread segue tentando

    monkeypatch.setattr(mod_camera.cv, "VideoCapture", lambda *_a, **_k: _Falha())
    monkeypatch.setattr(
        "tkinter.filedialog.askopenfilename", lambda **_k: str(caminho_rosto)
    )
    monkeypatch.setattr("tkinter.messagebox.showerror", lambda *_a, **_k: None)

    try:
        janela = JanelaPrincipal()
    except tk.TclError:  # pragma: no cover
        pytest.skip("ambiente sem interface gráfica disponível")

    try:
        if not janela._garantir_pipeline():
            pytest.skip("modelos indisponíveis")
        janela.iniciar_camera()
        janela.update()
        time.sleep(0.6)
        janela._modo = "erro"  # estado reproduzido pela auditoria
        assert janela._trabalhador is not None
        assert janela._trabalhador.is_alive()

        janela.analisar_imagem()
        janela.update()

        assert janela._trabalhador is None, (
            "a captura precisa ser encerrada antes de usar os modelos na thread do Tk"
        )
        assert janela._modo == "imagem"
    finally:
        janela.encerrar()


# ================== rodape nao pode contradizer o painel lateral
@modelos_disponiveis
def test_status_volta_ao_normal_depois_de_multiplos_rostos(
    imagem_rosto, imagem_multiplos_rostos
) -> None:
    """Regressão: uma detecção falsa de 2 rostos congelava o rodapé.

    O texto "2 rostos — principal: Neutra; outros: Alegria" permanecia visível
    mesmo quando o quadro seguinte tinha 1 rosto classificado de outra forma,
    contradizendo o painel lateral na captura do banner.
    """
    tk = pytest.importorskip("tkinter")
    from expressao_neural.core.pipeline import PipelineExpressao
    from expressao_neural.ui.janela import JanelaPrincipal

    try:
        janela = JanelaPrincipal()
    except tk.TclError:  # pragma: no cover
        pytest.skip("ambiente sem interface gráfica disponível")

    try:
        p = PipelineExpressao(Ajustes(multiplos_rostos=True, max_rostos=5))
        janela._modo = "ativo"  # simula câmera ligada, sem abrir dispositivo

        multi = p.processar(imagem_multiplos_rostos, suavizar=False)
        if multi.quantidade_rostos < 2:
            pytest.skip("fixture não produziu dois rostos")
        janela._atualizar_paineis(multi)
        janela.update_idletasks()
        assert "rostos" in janela._mensagem_status

        # Quadro seguinte com UM rosto: o rodapé precisa acompanhar.
        p.reiniciar_estado()
        um = p.processar(imagem_rosto, suavizar=False)
        assert um.quantidade_rostos == 1
        janela._atualizar_paineis(um)
        janela.update_idletasks()
        assert "2 rostos" not in janela._mensagem_status
        assert "classificação em tempo real" in janela._mensagem_status

        # E sem nenhum rosto, idem.
        from expressao_neural.core.pipeline import ResultadoQuadro

        janela._atualizar_paineis(ResultadoQuadro())
        janela.update_idletasks()
        assert "nenhum rosto" in janela._mensagem_status.lower()
        p.liberar()
    finally:
        janela.encerrar()


def test_status_de_captura_nao_sobrescreve_modo_imagem() -> None:
    """Em análise de imagem estática o rodapé tem mensagem própria."""
    tk = pytest.importorskip("tkinter")
    from expressao_neural.core.pipeline import ResultadoQuadro
    from expressao_neural.ui.janela import JanelaPrincipal
    from expressao_neural.ui import tema

    try:
        janela = JanelaPrincipal()
    except tk.TclError:  # pragma: no cover
        pytest.skip("ambiente sem interface gráfica disponível")
    try:
        janela._modo = "imagem"
        janela._definir_status("foto.jpg: 1 rosto classificado.", tema.SUCESSO)
        janela._status_da_captura(ResultadoQuadro())
        assert janela._mensagem_status == "foto.jpg: 1 rosto classificado."
    finally:
        janela.encerrar()
