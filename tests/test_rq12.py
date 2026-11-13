from pathlib import Path

import pandas as pd
import pytest

from aoc_perplexity.rq12 import (
    MODELO,
    PACOTE,
    carregar_pares,
    confere_publicado,
    tabela_rq2,
    wilcoxon_com_efeito,
)

PUBLICADOS = PACOTE / "model_results" / MODELO / "pairwise_analysis" / "pairwise_comparisons"


def escrever_stats(tmp_path, arquivos):
    caminho = tmp_path / "stats.csv"
    pd.DataFrame({"File": arquivos, "Overall Perplexity": range(len(arquivos))}).to_csv(caminho, index=False)
    return caminho


def test_wilcoxon_ignora_diferencas_zero():
    resultado = wilcoxon_com_efeito([2, 3, 4, 5, 9], [1, 1, 1, 1, 9])
    assert resultado.n == 4
    assert resultado.w == 0


def test_wilcoxon_calcula_efeito_com_conta_a_mao():
    # n = 4 e W = 0: média 5, desvio sqrt(7,5), r = 5 / sqrt(7,5) / 2.
    resultado = wilcoxon_com_efeito([2, 3, 4, 5], [1, 1, 1, 1])
    assert resultado.r == pytest.approx(0.9128709)


def test_wilcoxon_recusa_pares_identicos():
    with pytest.raises(ValueError):
        wilcoxon_com_efeito([1, 2], [1, 2])


def test_carregar_pares_aceita_identificador_vazio(tmp_path):
    caminho = escrever_stats(tmp_path, ["-clean-v0", "-obf-v0", "9-clean-v1", "9-obf-v1"])
    pares = carregar_pares(caminho)
    assert sorted(pares["base"]) == ["", "9"]


def test_carregar_pares_recusa_par_incompleto(tmp_path):
    caminho = escrever_stats(tmp_path, ["9-clean-v0", "9-obf-v0", "9-clean-v1"])
    with pytest.raises(ValueError, match="sem par"):
        carregar_pares(caminho)


def test_carregar_pares_recusa_nome_invalido(tmp_path):
    caminho = escrever_stats(tmp_path, ["9-outro-v0"])
    with pytest.raises(ValueError, match="padrão"):
        carregar_pares(caminho)


@pytest.mark.parametrize(
    ("valor", "publicado", "esperado"),
    [
        (0.0004, "<0.001", True),
        (0.002, "<0.001", False),
        (0.474, "0.47", True),
        (0.063, "0.05", False),
        (0.2466, "0.247", True),
    ],
)
def test_confere_publicado(valor, publicado, esperado):
    assert confere_publicado(valor, publicado) is esperado


def test_rq2_detecta_correlacao_perfeita():
    analise = pd.DataFrame(
        {
            "obf_Max AOI Perplexity": [1, 2, 3, 4],
            "obf_mean_eeg": [10, 20, 30, 40],
            "clean_Max AOI Perplexity": [1, 2, 3, 4],
            "clean_mean_eeg": [40, 30, 20, 10],
        }
    )
    tabela = tabela_rq2(analise).set_index("variante")
    assert tabela.loc["confuso", "rho_recalculado"] == pytest.approx(1)
    assert tabela.loc["limpo", "rho_recalculado"] == pytest.approx(-1)


@pytest.mark.dados
def test_dados_publicados_tem_72_pares_e_24_bases():
    pares = carregar_pares(PUBLICADOS / "perplexity_stats.csv")
    assert len(pares) == 72
    assert pares["base"].nunique() == 24
    assert sorted(pares["versao"].unique()) == ["v0", "v1", "v2"]
