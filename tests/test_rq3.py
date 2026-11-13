from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from aoc_perplexity.rq3 import MODELO, PACOTE, bootstrap_agrupado, mascara_numerada, regioes_novas, sobreposicao

PUBLICADOS = PACOTE / "model_results" / MODELO


def snippet(manuais, detectadas, rotulos=1):
    return {
        "aoi_regions": {"NEXP": manuais},
        "aoi_regions_algorithm": {"EXP": detectadas},
        "aoi_labels": {str(i + 1): {} for i in range(rotulos)},
    }


def test_sobreposicao_conta_token_da_borda():
    resultados = {
        "1-obf-v0": snippet([[3, 5]], [[5, 6]]),
        "1-clean-v0": snippet([[3, 5]], [[6, 7], [0, 1]], rotulos=2),
    }
    listas, regioes = sobreposicao(resultados)
    assert listas[("confuso", True)] == ["1-obf-v0"]
    assert listas[("limpo", False)] == ["1-clean-v0"]
    assert regioes == {"limpo": 2, "confuso": 1}
    assert regioes_novas(resultados) == {"limpo": 2, "confuso": 0}


def test_snippet_sem_aoi_manual_nao_sobrepoe():
    listas, _ = sobreposicao({"1-obf-v0": {"aoi_regions": {}, "aoi_regions_algorithm": {"EXP": [[0, 1]]}}})
    assert listas[("confuso", False)] == ["1-obf-v0"]


def test_mascara_nao_confunde_1_com_10():
    df = pd.DataFrame({"file": ["1-obf-v0-1", "10-obf-v0-1", "-obf-v0-2"]})
    assert mascara_numerada(df, ["1-obf-v0"]).tolist() == [True, False, False]
    assert mascara_numerada(df, ["-obf-v0"]).tolist() == [False, False, True]


def bootstrap_dos_autores(df, n, semente=42):
    # Cópia simplificada de clustered_bootstrap_spearman (paper_plots/rq3_plots.ipynb).
    np.random.seed(semente)
    ids = df["snippet_id"].unique()
    rhos = []
    for _ in range(n):
        amostra = pd.concat([df[df["snippet_id"] == c] for c in np.random.choice(ids, size=len(ids), replace=True)])
        rhos.append(spearmanr(amostra["Max AOI Perplexity"], amostra["mean_eeg"]).statistic)
    return np.mean(rhos), np.percentile(rhos, 2.5), np.percentile(rhos, 97.5)


def test_bootstrap_reproduz_os_autores():
    gerador = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "snippet_id": [f"s{i // 2}" for i in range(20)],
            "Max AOI Perplexity": gerador.random(20),
            "mean_eeg": gerador.random(20),
        }
    )
    assert bootstrap_agrupado(df, n=200) == pytest.approx(bootstrap_dos_autores(df, n=200))


@pytest.mark.dados
def test_dados_publicados_tem_227_regioes():
    import json

    resultados = json.loads((PUBLICADOS / "perplexity_results_with_aois.json").read_text())
    _, regioes = sobreposicao(resultados)
    assert regioes == {"limpo": 112, "confuso": 115}
