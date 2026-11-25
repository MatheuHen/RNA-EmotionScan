"""Validacao de imagem estatica (RF09) e estabilidade temporal (RF10)."""
from __future__ import annotations

from pathlib import Path

import cv2 as cv
import numpy as np
import pytest

from expressao_neural import config
from expressao_neural.core import imagem_estatica as ie
from expressao_neural.core.suavizacao import SuavizadorTemporal


# =========================================================== imagem estatica
def test_carrega_imagem_valida(caminho_rosto: Path) -> None:
    img = ie.carregar(caminho_rosto)
    assert img.ndim == 3 and img.shape[2] == 3


def test_rejeita_arquivo_inexistente(tmp_path: Path) -> None:
    with pytest.raises(ie.ImagemInvalida, match="não encontrado"):
        ie.carregar(tmp_path / "nao_existe.jpg")


def test_rejeita_diretorio(tmp_path: Path) -> None:
    with pytest.raises(ie.ImagemInvalida, match="não é um arquivo"):
        ie.carregar(tmp_path)


def test_rejeita_extensao_nao_suportada(tmp_path: Path) -> None:
    alvo = tmp_path / "dados.txt"
    alvo.write_text("conteúdo qualquer", encoding="utf-8")
    with pytest.raises(ie.ImagemInvalida, match="Extensão não suportada"):
        ie.carregar(alvo)


def test_rejeita_arquivo_vazio(tmp_path: Path) -> None:
    alvo = tmp_path / "vazio.png"
    alvo.write_bytes(b"")
    with pytest.raises(ie.ImagemInvalida, match="vazio"):
        ie.carregar(alvo)


def test_rejeita_arquivo_corrompido_com_extensao_de_imagem(tmp_path: Path) -> None:
    """Extensao mentirosa: o conteudo nao e imagem."""
    alvo = tmp_path / "falso.png"
    alvo.write_bytes(b"isto nao e uma imagem" * 50)
    with pytest.raises(ie.ImagemInvalida, match="não é uma imagem válida"):
        ie.carregar(alvo)


def test_rejeita_jpeg_truncado(tmp_path: Path, caminho_rosto: Path) -> None:
    dados = caminho_rosto.read_bytes()
    alvo = tmp_path / "truncado.jpg"
    alvo.write_bytes(dados[: len(dados) // 3])
    with pytest.raises(ie.ImagemInvalida):
        ie.carregar(alvo)


def test_rejeita_imagem_pequena_demais(tmp_path: Path) -> None:
    alvo = tmp_path / "minuscula.png"
    cv.imwrite(str(alvo), np.zeros((10, 10, 3), dtype=np.uint8))
    with pytest.raises(ie.ImagemInvalida, match="muito pequena"):
        ie.carregar(alvo)


def test_rejeita_arquivo_acima_do_limite(tmp_path: Path, monkeypatch) -> None:
    alvo = tmp_path / "grande.png"
    cv.imwrite(str(alvo), np.zeros((64, 64, 3), dtype=np.uint8))
    monkeypatch.setattr(ie, "TAMANHO_MAX_IMAGEM_BYTES", 10)
    with pytest.raises(ie.ImagemInvalida, match="muito grande"):
        ie.carregar(alvo)


def test_rejeita_lado_acima_do_limite(tmp_path: Path, monkeypatch) -> None:
    alvo = tmp_path / "larga.png"
    cv.imwrite(str(alvo), np.zeros((64, 300, 3), dtype=np.uint8))
    monkeypatch.setattr(ie, "LADO_MAX_IMAGEM", 100)
    with pytest.raises(ie.ImagemInvalida, match="pixels por lado"):
        ie.carregar(alvo)


@pytest.mark.parametrize("extensao", [".png", ".bmp", ".jpg"])
def test_aceita_formatos_declarados(tmp_path: Path, extensao: str) -> None:
    alvo = tmp_path / f"amostra{extensao}"
    cv.imwrite(str(alvo), np.full((80, 80, 3), 120, dtype=np.uint8))
    assert ie.carregar(alvo).shape[:2] == (80, 80)


def test_redimensionamento_preserva_proporcao() -> None:
    grande = np.zeros((900, 1800, 3), dtype=np.uint8)
    saida = ie.redimensionar_para_analise(grande, lado_max=600)
    assert max(saida.shape[:2]) == 600
    assert saida.shape[0] / saida.shape[1] == pytest.approx(0.5, abs=0.01)


def test_redimensionamento_nao_amplia() -> None:
    pequena = np.zeros((100, 120, 3), dtype=np.uint8)
    assert ie.redimensionar_para_analise(pequena, lado_max=600).shape == pequena.shape


# ============================================================ suavizacao
def _pico(indice: int, valor: float = 0.9) -> np.ndarray:
    v = np.full(7, (1.0 - valor) / 6.0, dtype=np.float32)
    v[indice] = valor
    return v


def test_suavizacao_converge_para_a_classe_dominante() -> None:
    s = SuavizadorTemporal(alfa=0.4)
    centro, escala = (100.0, 100.0), 80.0
    tid = None
    vetor = None
    for _ in range(20):
        tid, vetor = s.suavizar(centro, escala, _pico(3))
        s.esquecer_ausentes({tid})
    assert int(np.argmax(vetor)) == 3
    assert vetor[3] > 0.85
    assert s.quantidade_trilhas == 1


def test_resultado_suavizado_continua_sendo_distribuicao_valida() -> None:
    s = SuavizadorTemporal(alfa=0.3)
    rng = np.random.default_rng(1)
    for _ in range(30):
        bruto = rng.random(7).astype(np.float32)
        bruto /= bruto.sum()
        _, vetor = s.suavizar((50.0, 50.0), 60.0, bruto)
        assert pytest.approx(1.0, abs=1e-5) == float(vetor.sum())
        assert (vetor >= 0).all()


def test_suavizacao_amortece_oscilacao_sem_travar() -> None:
    """Alternar bruscamente entre duas classes nao deve produzir salto total."""
    s = SuavizadorTemporal(alfa=0.3)
    centro, escala = (10.0, 10.0), 50.0
    s.suavizar(centro, escala, _pico(0))
    _, depois = s.suavizar(centro, escala, _pico(6))
    assert depois[6] < 0.9, "a mudança deve ser amortecida"
    assert depois[6] > _pico(0)[6], "mas deve caminhar na direção do novo valor"
    for _ in range(25):
        _, depois = s.suavizar(centro, escala, _pico(6))
    assert int(np.argmax(depois)) == 6, "a suavização não pode travar a classe antiga"


def test_alfa_um_equivale_a_sem_suavizacao() -> None:
    s = SuavizadorTemporal(alfa=1.0)
    s.suavizar((0.0, 0.0), 40.0, _pico(1))
    _, vetor = s.suavizar((0.0, 0.0), 40.0, _pico(5))
    assert np.allclose(vetor, _pico(5), atol=1e-5)


def test_rostos_distantes_recebem_trilhas_distintas() -> None:
    s = SuavizadorTemporal(alfa=0.5)
    a, _ = s.suavizar((50.0, 50.0), 60.0, _pico(3))
    b, _ = s.suavizar((500.0, 300.0), 60.0, _pico(0), usados={a})
    assert a != b
    assert s.quantidade_trilhas == 2


def test_mesmo_rosto_com_leve_movimento_mantem_a_trilha() -> None:
    s = SuavizadorTemporal(alfa=0.5)
    a, _ = s.suavizar((100.0, 100.0), 80.0, _pico(3))
    b, _ = s.suavizar((112.0, 108.0), 80.0, _pico(3))
    assert a == b


def test_trilha_e_descartada_quando_o_rosto_desaparece() -> None:
    """RF10: nada do resultado antigo sobrevive a ausencia do rosto."""
    s = SuavizadorTemporal(alfa=0.5, tolerancia_quadros=3)
    tid, _ = s.suavizar((100.0, 100.0), 80.0, _pico(3))
    for _ in range(3):
        s.esquecer_ausentes(set())
    assert s.quantidade_trilhas == 0
    assert s.estabilidade(tid) == 0.0


def test_limpar_descarta_todo_o_estado() -> None:
    s = SuavizadorTemporal()
    s.suavizar((1.0, 1.0), 30.0, _pico(2))
    s.suavizar((400.0, 400.0), 30.0, _pico(4))
    s.limpar()
    assert s.quantidade_trilhas == 0


def test_estabilidade_reflete_constancia() -> None:
    s = SuavizadorTemporal(alfa=0.5)
    tid = None
    for _ in range(25):
        tid, _ = s.suavizar((20.0, 20.0), 40.0, _pico(4))
    assert s.estabilidade(tid) == pytest.approx(1.0)


def test_historico_nao_cresce_indefinidamente() -> None:
    """Controle de memoria em execucao prolongada."""
    s = SuavizadorTemporal(alfa=0.5)
    tid = None
    for _ in range(500):
        tid, _ = s.suavizar((20.0, 20.0), 40.0, _pico(4))
    assert len(s._trilhas[tid].historico) <= 60


def test_vetor_de_tamanho_errado_e_recusado() -> None:
    s = SuavizadorTemporal()
    with pytest.raises(ValueError):
        s.suavizar((0.0, 0.0), 10.0, np.zeros(5, dtype=np.float32))


def test_numero_de_classes_do_suavizador_bate_com_a_configuracao() -> None:
    s = SuavizadorTemporal()
    _, vetor = s.suavizar((0.0, 0.0), 10.0, np.full(7, 1 / 7, dtype=np.float32))
    assert vetor.size == len(config.CLASSES_MODELO)
