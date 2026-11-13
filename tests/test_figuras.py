import matplotlib
import numpy as np
import pandas as pd
import pytest

from aoc_perplexity.consolidar import TRIOS
from aoc_perplexity.eeg_extensao import analises_publicadas, eeg_publicado
from aoc_perplexity.figuras import (
    COR_NEUTRA,
    CONSOLIDADO,
    EXEMPLO,
    EXTENSAO,
    FAMILIAS,
    FONTES,
    LINGUAGENS,
    PAPEIS,
    QUADRANTES,
    contagens,
    dados_delta,
    dados_eeg_pares,
    estilo,
    figura_delta,
    figura_eeg,
    fonte_serifada,
    quadrante,
    salvar,
    snippets_do_exemplo,
)
from aoc_perplexity.trio import pares_do_modelo


def snippets_sinteticos():
    gerador = np.random.RandomState(1)
    linhas = [
        {"base": str(b), "pair_id": f"{b}-v{v}", "condicao": c, "ppl": float(np.exp(gerador.normal(2, 1))), "frp": float(gerador.normal(0, 4e-6))}
        for b in range(6)
        for v in range(3)
        for c in ["confuso", "limpo"]
    ]
    return pd.DataFrame(linhas)


def test_dados_do_delta_sao_as_48_linhas_do_consolidado():
    consolidado = pd.read_csv(CONSOLIDADO)
    dados = dados_delta(consolidado)
    assert len(dados) == 48 == len(FAMILIAS) * len(PAPEIS) * len(LINGUAGENS)
    assert FAMILIAS == list(dict.fromkeys(dados["familia"]))  # ordem da tabela compacta
    bruto = consolidado[(consolidado["metrica"] == "aoi_max") & consolidado["quantidade"].str.startswith("Delta_")]
    assert len(bruto) == 48
    chaves = bruto.assign(papel=bruto["quantidade"].str.removeprefix("Delta_")).set_index(["conjunto", "trio", "papel"])
    for linha in dados.itertuples():
        esperado = chaves.loc[(linha.conjunto, linha.familia, linha.papel)]
        assert (linha.estimativa, linha.ic_inferior, linha.ic_superior) == (esperado["estimativa"], esperado["ic_inferior"], esperado["ic_superior"])


def test_pontos_da_figura_1_batem_com_o_consolidado():
    dados = dados_delta(pd.read_csv(CONSOLIDADO))
    figura = figura_delta(dados)
    encontrados = 0
    for eixo, conjunto in zip(figura.axes, LINGUAGENS):
        assert len(eixo.containers) == len(PAPEIS)
        for container, papel in zip(eixo.containers, PAPEIS):
            esperado = dados[(dados["conjunto"] == conjunto) & (dados["papel"] == papel)].set_index("familia").loc[FAMILIAS]
            assert list(container.lines[0].get_xdata()) == esperado["estimativa"].tolist()
            barras = container.lines[2][0].get_segments()  # uma barra de IC por família
            extremos = np.array([(s[0][0], s[1][0]) for s in barras])
            assert extremos == pytest.approx(esperado[["ic_inferior", "ic_superior"]].to_numpy(), abs=1e-12)
            encontrados += len(esperado)
    assert encontrados == 48


def test_figura_1_tem_a_mesma_escala_nos_dois_paineis_e_zero_marcado():
    figura = figura_delta(dados_delta(pd.read_csv(CONSOLIDADO)))
    java, c = figura.axes[:2]
    assert java.get_xlim() == c.get_xlim()
    assert [t.get_text() for t in java.get_yticklabels()] == FAMILIAS
    assert any(list(linha.get_xdata()) == [0, 0] for linha in java.lines)


def test_figura_2_conta_os_pares_por_quadrante_e_marca_o_delta():
    dados = dados_eeg_pares(snippets_sinteticos())
    assert len(dados) == 18
    n = contagens(dados)
    assert sum(n.values()) == 18 and set(n) == set(QUADRANTES)
    figura = figura_eeg(dados)
    eixo = figura.axes[0]
    textos = [t.get_text() for t in eixo.texts if t.get_text().startswith("n = ")]
    assert sorted(textos) == sorted(f"n = {n[q]}" for q in ["sup_dir", "sup_esq", "inf_esq", "inf_dir"])
    assert "Δ" in "".join(t.get_text() for t in eixo.texts).replace(r"$\Delta$", "Δ")
    deltas = [linha for linha in eixo.lines if len(set(linha.get_xdata())) == 1 and linha.get_xdata()[0] == pytest.approx(dados["ell"].mean())]
    assert len(deltas) == 1 and deltas[0].get_linestyle() != "-"


def test_quadrante_por_sinal():
    assert [quadrante(1, 1), quadrante(-1, 1), quadrante(-1, -1), quadrante(1, -1), quadrante(0, 1), quadrante(1, 0)] == ["sup_dir", "sup_esq", "inf_esq", "inf_dir", "eixo", "eixo"]


def test_par_sem_aoi_valida_sai_da_figura_2():
    snippets = snippets_sinteticos()
    snippets = snippets[~((snippets["pair_id"] == "0-v0") & (snippets["condicao"] == "confuso"))]
    dados = dados_eeg_pares(snippets)
    assert len(dados) == 17 and "0-v0" not in set(dados["par"])


def test_figura_2_sem_estatistica_e_com_cor_neutra():
    figura = figura_eeg(dados_eeg_pares(snippets_sinteticos()))
    eixo = figura.axes[0]
    assert eixo.get_legend() is None
    assert not any("rho" in t.get_text().lower() or "ρ" in t.get_text() for t in eixo.texts)
    cores = eixo.collections[0].get_facecolor()[0][:3]
    assert cores == pytest.approx(matplotlib.colors.to_rgb(COR_NEUTRA))
    assert COR_NEUTRA.upper() not in {m["cor"].upper() for m in PAPEIS.values()}


@pytest.mark.dados
@pytest.mark.tokenizador
def test_figura_2_bate_com_pares_do_modelo_e_eeg_publicado():
    snippets, modelo, _, execucao, total = snippets_do_exemplo()
    dados = dados_eeg_pares(snippets)
    assert modelo == "Qwen/Qwen2.5-7B" and total == 72 and len(dados) == 72
    pares, _ = pares_do_modelo(execucao, "bergum", "manual")
    eeg = eeg_publicado(analises_publicadas()).set_index("pair_id")
    for linha in dados.itertuples():
        base, versao = linha.par.split("-")
        original = pares[(pares["base"] == base) & (pares["versao"] == versao)].iloc[0]
        assert linha.ell == pytest.approx(np.log(original["Max AOI Perplexity_confuso"]) - np.log(original["Max AOI Perplexity_limpo"]), abs=1e-12)
        assert linha.dif_frp_uv == pytest.approx((eeg.loc[linha.par, "obf_mean_eeg"] - eeg.loc[linha.par, "clean_mean_eeg"]) * 1e6, abs=1e-9)
    n = contagens(dados)
    assert sum(n.values()) == 72
    assert n["sup_dir"] + n["inf_dir"] == (dados["ell"] > 0).sum() and n["sup_dir"] + n["sup_esq"] == (dados["dif_frp_uv"] > 0).sum()


@pytest.mark.dados
@pytest.mark.tokenizador
def test_media_de_ell_e_o_delta_base_do_trio_qwen_7b():
    snippets, *_ = snippets_do_exemplo()
    dados = dados_eeg_pares(snippets)
    trio = pd.read_csv(TRIOS[(EXEMPLO["trio"], "bergum")] / "trio.csv")
    esperado = trio[(trio["metrica"] == "aoi_max") & (trio["quantidade"] == "Delta_base")]["estimativa"].iloc[0]
    assert dados["ell"].mean() == pytest.approx(esperado, abs=1e-9)


def test_estilo_unico_com_fonte_serifada_registrada():
    fonte = fonte_serifada()
    assert fonte in FONTES
    rc = estilo(fonte)
    assert rc["font.serif"] == [fonte] and rc["mathtext.fontset"] == "stix"
    assert (rc["axes.labelsize"], rc["xtick.labelsize"], rc["legend.fontsize"]) == (8, 7, 7)
    assert rc["axes.spines.top"] is False and rc["axes.spines.right"] is False


def test_figura_1_tem_eixo_x_unico_e_painel_sem_moldura_na_legenda():
    figura = figura_delta(dados_delta(pd.read_csv(CONSOLIDADO)))
    assert "Confusion effect" in figura._supxlabel.get_text()
    assert all(eixo.get_xlabel() == "" for eixo in figura.axes)
    assert [eixo.get_title(loc="left") for eixo in figura.axes[:2]] == [rf"$\mathbf{{{l['titulo']}}}$" for l in LINGUAGENS.values()]
    assert not figura.legends[0].get_frame_on() and figura.legends[0]._ncols == 3


def test_duas_execucoes_geram_pdfs_e_pngs_identicos(tmp_path):
    dados = dados_delta(pd.read_csv(CONSOLIDADO))
    pares = dados_eeg_pares(snippets_sinteticos())
    for rodada in ["a", "b"]:
        for nome, figura in [("delta", figura_delta(dados)), ("pares", figura_eeg(pares))]:
            salvar(figura, tmp_path / f"{nome}-{rodada}.pdf")
            salvar(figura, tmp_path / f"{nome}-{rodada}.png", dpi=150)
    for nome in ["delta", "pares"]:
        for extensao in ["pdf", "png"]:
            assert (tmp_path / f"{nome}-a.{extensao}").read_bytes() == (tmp_path / f"{nome}-b.{extensao}").read_bytes()
    pdf = (tmp_path / "delta-a.pdf").read_bytes()
    assert pdf.startswith(b"%PDF") and b"CreationDate" not in pdf


def test_figura_1_em_coluna_tem_largura_acm_java_sobre_c_e_rotulos():
    figura = figura_delta(dados_delta(pd.read_csv(CONSOLIDADO)), "coluna")
    largura, altura = figura.get_size_inches()
    assert largura == pytest.approx(3.33) and altura > largura
    java, c = figura.axes[:2]
    assert java.get_position().y0 > c.get_position().y0 and java.get_xlim() == c.get_xlim()
    assert [eixo.get_title(loc="left") for eixo in (java, c)] == [rf"$\mathbf{{{l['titulo']}}}$" for l in LINGUAGENS.values()]
    assert [t.get_text() for t in java.get_yticklabels()] == FAMILIAS
    assert "Confusion effect" in figura._supxlabel.get_text() and figura._supxlabel.get_fontsize() == 8
    assert [t.get_text() for t in figura.legends[0].get_texts()] == [f"{m['rotulo']} ({m['nome']})" for m in PAPEIS.values()]


def test_figura_1_em_coluna_tem_os_mesmos_valores_e_cores_da_larga():
    dados = dados_delta(pd.read_csv(CONSOLIDADO))
    larga, coluna = figura_delta(dados), figura_delta(dados, "coluna")
    for eixo_l, eixo_c in zip(larga.axes[:2], coluna.axes[:2]):
        for cont_l, cont_c in zip(eixo_l.containers, eixo_c.containers, strict=True):
            assert list(cont_l.lines[0].get_xdata()) == list(cont_c.lines[0].get_xdata())
            assert list(cont_l.lines[0].get_ydata()) == list(cont_c.lines[0].get_ydata())
            assert cont_l.lines[0].get_color() == cont_c.lines[0].get_color()
            assert [s.tolist() for s in cont_l.lines[2][0].get_segments()] == [s.tolist() for s in cont_c.lines[2][0].get_segments()]


def test_delta_invalido_e_recusado():
    consolidado = pd.read_csv(CONSOLIDADO)
    consolidado.loc[(consolidado["metrica"] == "aoi_max") & (consolidado["quantidade"] == "Delta_base"), "valida"] = "nao"
    with pytest.raises(ValueError):
        dados_delta(consolidado)


def test_figura_1_em_grade_tem_largura_acm_quatro_paineis_e_mesma_escala():
    figura = figura_delta(dados_delta(pd.read_csv(CONSOLIDADO)), "grade")
    largura, altura = figura.get_size_inches()
    assert largura == pytest.approx(3.33) and altura < 4.6
    eixos = figura.axes[:4]
    assert len({eixo.get_xlim() for eixo in eixos}) == 1
    java_esq, java_dir, c_esq, c_dir = eixos
    assert java_esq.get_position().y0 > c_esq.get_position().y0 and java_esq.get_position().x0 < java_dir.get_position().x0
    assert [eixo.get_title(loc="left") for eixo in (java_esq, c_esq)] == [rf"$\mathbf{{{l['titulo']}}}$" for l in LINGUAGENS.values()]
    rotulos = [t.get_text() for eixo in (java_esq, java_dir) for t in eixo.get_yticklabels()]
    assert rotulos == [familia.replace(" (4 bits)", "") for familia in FAMILIAS]
    assert [t.get_text() for t in figura.legends[0].get_texts()] == [f"{m['rotulo']} ({m['nome']})" for m in PAPEIS.values()]


def test_figura_1_em_grade_tem_os_mesmos_valores_e_cores_da_coluna():
    dados = dados_delta(pd.read_csv(CONSOLIDADO))
    coluna, grade = figura_delta(dados, "coluna"), figura_delta(dados, "grade")
    metade = len(FAMILIAS) // 2
    for eixo_c, (esq, dir_) in zip(coluna.axes[:2], [grade.axes[0:2], grade.axes[2:4]]):
        for cont_c, cont_e, cont_d in zip(eixo_c.containers, esq.containers, dir_.containers, strict=True):
            assert list(cont_c.lines[0].get_xdata()) == list(cont_e.lines[0].get_xdata()) + list(cont_d.lines[0].get_xdata())
            assert cont_c.lines[0].get_color() == cont_e.lines[0].get_color() == cont_d.lines[0].get_color()
            intervalos = [s[:, 0].tolist() for s in cont_c.lines[2][0].get_segments()]
            assert intervalos[:metade] == [s[:, 0].tolist() for s in cont_e.lines[2][0].get_segments()]
            assert intervalos[metade:] == [s[:, 0].tolist() for s in cont_d.lines[2][0].get_segments()]
