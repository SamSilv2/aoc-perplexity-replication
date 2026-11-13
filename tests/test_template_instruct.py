import numpy as np
import pandas as pd
import pytest

from aoc_perplexity.template_instruct import (
    CONSOLIDADO,
    EXTENSAO,
    TRIOS_COM_TEMPLATE,
    comparar,
    diferenca_delta_i,
    exclui_zero,
    posicao_zero,
    resumir,
)
from aoc_perplexity.trio import analisar


def pares(fator, n_bases=8, ruido=0.0):
    bases = [str(i) for i in range(n_bases) for _ in range(3)]
    versoes = [f"v{j}" for _ in range(n_bases) for j in range(3)]
    gerador = np.random.RandomState(0)
    limpo = np.linspace(2, 40, len(bases))
    df = pd.DataFrame({"base": bases, "versao": versoes})
    for coluna in ["Max AOI Perplexity", "Overall Perplexity"]:
        df[f"{coluna}_limpo"] = limpo
        df[f"{coluna}_confuso"] = limpo * fator * np.exp(gerador.normal(0, ruido, len(bases)))
    return df


def test_diferenca_zero_quando_o_template_nao_muda_nada():
    d = diferenca_delta_i(pares(3.0, ruido=0.3), pares(3.0, ruido=0.3), "Max AOI Perplexity", n=50)
    assert d == {"estimativa": 0.0, "ic_inferior": 0.0, "ic_superior": 0.0}


def test_diferenca_de_um_deslocamento_constante_e_o_deslocamento():
    d = diferenca_delta_i(pares(3.0, ruido=0.3), pares(3.0 * np.exp(-0.7), ruido=0.3), "Max AOI Perplexity", n=50)
    assert d["estimativa"] == pytest.approx(-0.7) and d["ic_inferior"] == pytest.approx(-0.7) and d["ic_superior"] == pytest.approx(-0.7)


def test_o_sorteio_e_o_de_trio_analisar():
    # Com o instruct "sem" sem efeito (confuso = limpo, ln da razão = 0), a diferença é o Δ_I do "com" e o IC tem de ser o do trio.analisar.
    com = pares(3.0, ruido=0.5)
    sem = pares(1.0)
    base, codigo = pares(2.0, ruido=0.2), pares(2.5, ruido=0.2)
    trio, _ = analisar({"base": base, "codigo": codigo, "instruct": com}, n=200)
    delta_i = trio[(trio["metrica"] == "aoi_max") & (trio["quantidade"] == "Delta_instruct")].iloc[0]
    d = diferenca_delta_i(sem, com, "Max AOI Perplexity", n=200)
    assert (d["estimativa"], d["ic_inferior"], d["ic_superior"]) == pytest.approx((delta_i["estimativa"], delta_i["ic_inferior"], delta_i["ic_superior"]), abs=1e-12)


def test_pares_diferentes_ou_validos_diferentes_dao_erro():
    outro = pares(3.0).iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError):
        diferenca_delta_i(pares(3.0), outro, "Max AOI Perplexity", n=5)
    sem_aoi = pares(3.0)
    sem_aoi.loc[0, "Max AOI Perplexity_confuso"] = np.nan
    with pytest.raises(ValueError):
        diferenca_delta_i(pares(3.0), sem_aoi, "Max AOI Perplexity", n=5)


def test_posicao_zero_por_condicao():
    resultados = {f"{b}-{c}-v1": {"results": [{"log_prob": lp}]} for b, c, lp in [(1, "clean", -1.0), (2, "clean", -3.0), (1, "obf", -5.0)]}
    r = posicao_zero(resultados)
    assert r == {"todos": (3, pytest.approx(-3.0)), "limpo": (2, pytest.approx(-2.0)), "confuso": (1, pytest.approx(-5.0))}


def test_exclui_zero():
    assert exclui_zero(0.1, 0.5) and exclui_zero(-0.5, -0.1) and not exclui_zero(-0.003, 0.58)


def comparacao_sintetica():
    def trio(delta):
        linhas = []
        for metrica in ["aoi_max", "snippet_media"]:
            for q, v in [("Delta_instruct", delta), ("D_instrucao", delta - 1), ("D", delta - 2)]:
                linhas.append({"metrica": metrica, "quantidade": q, "estimativa": v, "ic_inferior": v - 0.5, "ic_superior": v + 0.5})
        return pd.DataFrame(linhas)

    diferenca = {m: {"estimativa": -0.4, "ic_inferior": -0.6, "ic_superior": -0.2} for m in ["aoi_max", "snippet_media"]}
    return pd.concat([comparar(trio(1.0), trio(0.6), diferenca, "T1", "bergum"), comparar(trio(0.2), trio(-0.1), diferenca, "T2", "gopstein")], ignore_index=True)


def test_comparar_e_resumir_os_casos():
    c = comparacao_sintetica()
    assert len(c) == 12 and set(c["linguagem"]) == {"Java", "C"}
    assert (c["diferenca"] == -0.4).all()  # a mesma diferença nas três quantidades
    r = resumir(c)
    assert r["casos"] == 4 and r["delta_i_cai"] == 4
    assert r["maior_queda"]["queda"] == pytest.approx(0.4)
    assert r["muda_de_sinal"] == ["T2 C aoi_max", "T2 C snippet_media"]
    # T1: Δ_I 1,0 [0,5; 1,5] → 0,6 [0,1; 1,1] segue fora de zero; T2: 0,2 [−0,3; 0,7] → −0,1 [−0,6; 0,4] segue incluindo zero.
    assert r["muda_status_do_ic"] == [
        {"trio": "T1", "linguagem": "Java", "metrica": m, "quantidade": q, "sem": pytest.approx([v - 0.5, v + 0.5]), "com": pytest.approx([v - 0.9, v + 0.1])}
        for m in ["aoi_max", "snippet_media"]
        for q, v in [("D_instrucao", 0.0), ("D", -1.0)]
        if (v - 0.5 > 0 or v + 0.5 < 0) != (v - 0.9 > 0 or v + 0.1 < 0)
    ] + [
        {"trio": "T2", "linguagem": "C", "metrica": m, "quantidade": q, "sem": pytest.approx([v - 0.5, v + 0.5]), "com": pytest.approx([v - 0.8, v + 0.2])}
        for m in ["aoi_max", "snippet_media"]
        for q, v in [("D", -1.8)]
        if (v - 0.5 > 0 or v + 0.5 < 0) != (v - 0.8 > 0 or v + 0.2 < 0)
    ]


def test_valores_sem_template_sao_os_do_consolidado_3_e_os_quatro_trios_existem():
    consolidado = pd.read_csv(CONSOLIDADO)
    for (familia, conjunto), pasta in TRIOS_COM_TEMPLATE.items():
        assert (pasta / "trio.csv").is_file() and pasta.name.endswith("-template")
        com = pd.read_csv(pasta / "trio.csv")
        assert len(com) == 12
        assert len(consolidado[(consolidado["trio"] == familia) & (consolidado["conjunto"] == conjunto)]) == 12


def test_conferir_com_o_consolidado_aceita_igual_e_recusa_diferente():
    from aoc_perplexity.template_instruct import conferir_com_o_consolidado

    consolidado = pd.DataFrame(
        {"trio": ["F"], "conjunto": ["bergum"], "metrica": ["aoi_max"], "quantidade": ["Delta_base"], "estimativa": [0.5], "ic_inferior": [0.1], "ic_superior": [0.9]}
    )
    sem = consolidado[["metrica", "quantidade", "estimativa", "ic_inferior", "ic_superior"]]
    conferir_com_o_consolidado(sem, consolidado, "F", "bergum")
    with pytest.raises(ValueError, match="ic_superior"):
        conferir_com_o_consolidado(sem.assign(ic_superior=0.8), consolidado, "F", "bergum")


def test_conferir_tokenizacao_do_template_para_se_a_fronteira_funde(tmp_path):
    import json

    from test_perplexidade import TokenizadorFalso, TokenizadorQueFunde

    from aoc_perplexity.template_instruct import conferir_tokenizacao_do_template

    (tmp_path / "run.json").write_text(json.dumps({"config": {"chat_template": {"prefixo": "cb"}}}))
    assert conferir_tokenizacao_do_template(tmp_path, "bergum", TokenizadorFalso(), {"x": "acca", "y": "ab"}) == 2
    with pytest.raises(ValueError, match="tokenização conjunta"):
        conferir_tokenizacao_do_template(tmp_path, "bergum", TokenizadorQueFunde(), {"x": "acca"})
