import numpy as np
import pandas as pd
import pytest

from aoc_perplexity.eeg_extensao import (
    ANALISE,
    COLUNA,
    PUBLICADOS,
    analises_publicadas,
    concordancia,
    correlacoes,
    delta_com_ic,
    eeg_publicado,
    modelos_da_extensao,
    pares_publicados,
    tabela_correlacoes,
    tabela_snippets,
)
from aoc_perplexity.rq12 import PACOTE, wilcoxon_com_efeito

QWEN_32B = "Qwen2.5-Coder-32B"


def analise_sintetica(n_bases=6, fator=1.0):
    pair_ids = [f"{b}-v{v}" for b in range(n_bases) for v in range(3)]
    gerador = np.random.RandomState(0)
    limpo = np.exp(gerador.normal(size=len(pair_ids)))
    return pd.DataFrame(
        {
            "pair_id": pair_ids,
            "clean_file": [p.replace("-", "-clean-") for p in pair_ids],
            "obf_file": [p.replace("-", "-obf-") for p in pair_ids],
            "clean_mean_eeg": gerador.normal(size=len(pair_ids)),
            "obf_mean_eeg": gerador.normal(size=len(pair_ids)),
            f"clean_{COLUNA}": limpo,
            f"obf_{COLUNA}": limpo * fator * np.exp(gerador.normal(0.3, 0.1, size=len(pair_ids))),
        }
    )


@pytest.mark.dados
def test_reproduz_os_rho_publicados_do_qwen_32b():
    publicado = pd.read_csv(PACOTE / "paper_plots" / "RQ2" / "correlation_statistics_by_subset.csv").set_index("subset")
    analises = analises_publicadas()
    snippets = tabela_snippets(pares_publicados(analises[QWEN_32B]), eeg_publicado(analises))
    resultado = correlacoes(snippets, n=20).set_index("subconjunto")
    assert [resultado.loc[s, "n"] for s in ["todos", "confusos", "limpos"]] == [144, 72, 72]
    for nosso, deles in [("todos", "full"), ("confusos", "obf"), ("limpos", "clean")]:
        assert resultado.loc[nosso, "rho"] == pytest.approx(publicado.loc[deles, "spearman_r"], abs=1e-12)
        assert resultado.loc[nosso, "p"] == pytest.approx(publicado.loc[deles, "spearman_p"], rel=1e-9)
    # Os valores que o artigo cita (docs/metodos.md, seção 5).
    assert [round(resultado.loc[s, "rho"], 3) for s in ["todos", "confusos", "limpos"]] == [0.309, 0.474, 0.063]


@pytest.mark.dados
def test_eeg_identico_nos_seis_modelos_publicados():
    analises = analises_publicadas()
    assert len(analises) == 6
    eeg = eeg_publicado(analises)
    assert len(eeg) == 72
    for analise in analises.values():
        colunas = analise.sort_values("pair_id").reset_index(drop=True)[["clean_mean_eeg", "obf_mean_eeg"]]
        assert colunas.equals(eeg[["clean_mean_eeg", "obf_mean_eeg"]])


def test_eeg_diferente_e_recusado():
    a, b = analise_sintetica(), analise_sintetica()
    b.loc[0, "obf_mean_eeg"] += 1
    with pytest.raises(ValueError):
        eeg_publicado({"a": a, "b": b})


@pytest.mark.dados
def test_arquivo_de_analise_existe_para_os_seis():
    assert len(list(PUBLICADOS.glob(f"*/{ANALISE}"))) == 6


def test_correlacao_perfeita_e_ic_degenerado():
    analise = analise_sintetica()
    analise["obf_mean_eeg"] = np.log(analise[f"obf_{COLUNA}"])
    analise["clean_mean_eeg"] = np.log(analise[f"clean_{COLUNA}"])
    snippets = tabela_snippets(pares_publicados(analise), analise)
    resultado = correlacoes(snippets, n=30).set_index("subconjunto")
    assert resultado["rho"].tolist() == pytest.approx([1, 1, 1])
    assert resultado["ic_inferior"].tolist() == pytest.approx([1, 1, 1])
    assert resultado["n"].tolist() == [36, 18, 18]


def test_bootstrap_e_reprodutivel_e_agrupa_por_base():
    analise = analise_sintetica()
    snippets = tabela_snippets(pares_publicados(analise), analise)
    a, b = correlacoes(snippets, n=40), correlacoes(snippets, n=40)
    assert a.equals(b)
    assert (a["ic_inferior"] <= a["rho"] + 1e-9).all() or (a["ic_superior"] >= a["rho"] - 1e-9).all()
    assert (a["ic_inferior"] < a["ic_superior"]).all()


def test_snippets_sem_par_no_eeg_sao_recusados():
    analise = analise_sintetica()
    with pytest.raises(ValueError):
        tabela_snippets(pares_publicados(analise), analise.iloc[:-1])


def test_snippet_sem_aoi_sai_so_da_correlacao():
    analise = analise_sintetica()
    pares = pares_publicados(analise)
    pares.loc[0, f"{COLUNA}_confuso"] = np.nan
    snippets = tabela_snippets(pares, analise)
    assert len(snippets) == 2 * len(pares) - 1


def test_delta_e_concordancia_com_efeito_forte():
    pares = pares_publicados(analise_sintetica(n_bases=8, fator=5.0))
    delta = delta_com_ic(pares, n=200)
    assert delta["delta"] == pytest.approx(np.mean(np.log(pares[f"{COLUNA}_confuso"]) - np.log(pares[f"{COLUNA}_limpo"])))
    assert delta["ic_inferior"] > 0
    assert delta["n_pares"] == 24
    assert concordancia(delta) == {"wilcoxon_rejeita": True, "ic_exclui_zero": True, "concordam": True}


def test_discordancia_e_detectada():
    delta = {"p_wilcoxon": 0.01, "ic_inferior": -0.1, "ic_superior": 0.5}
    assert concordancia(delta) == {"wilcoxon_rejeita": True, "ic_exclui_zero": False, "concordam": False}


@pytest.mark.dados
def test_wilcoxon_do_qwen_32b_confere_com_o_artigo():
    # aoi_max do Qwen2.5-Coder-32B: W = 638 e p < 0,001 (Figura 2 do artigo).
    pares = pares_publicados(analises_publicadas()[QWEN_32B])
    w = wilcoxon_com_efeito(pares[f"{COLUNA}_confuso"], pares[f"{COLUNA}_limpo"])
    assert w.w == 638
    assert w.p < 0.001


def test_trios_de_java_sao_oito_com_tres_modelos():
    modelos = list(modelos_da_extensao())
    assert len(modelos) == 24
    assert len({trio for trio, *_ in modelos}) == 8
    assert {papel for _, papel, *_ in modelos} == {"base", "codigo", "instruct"}


def test_tabela_de_correlacoes_tem_os_30_modelos_e_negrito_quando_o_ic_exclui_zero():
    correlacao = pd.read_csv("results/extensao/2026-11-05-eeg-java/correlacoes.csv")
    texto = tabela_correlacoes(correlacao)
    linhas = [l for l in texto.splitlines() if l.count(" & ") == 3 and not l.startswith("Model")]
    assert len(linhas) == 30
    excluem = ((correlacao["ic_inferior"] > 0) | (correlacao["ic_superior"] < 0)).sum()
    assert texto.count(r"\textbf{") == excluem
    confusos = correlacao[correlacao["subconjunto"] == "confusos"]
    assert all(f"{r:.2f}" in texto for r in confusos["rho"])
