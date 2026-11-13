import re

import matplotlib
import matplotlib.text
import numpy as np
import pandas as pd
import pytest

from aoc_perplexity.figuras import CONSOLIDADO, PAPEIS, salvar
from aoc_perplexity.figuras_grade import (
    CATEGORIAS,
    CORRELACOES,
    GRADES,
    COR_MEDIA,
    MEDIA_GERAL,
    X_MAX,
    carregar,
    categoria,
    categorias_dos_grupos,
    contagem,
    contagem_e_proporcao,
    contrastes_da_dupla,
    dados_do_modelo,
    delta_da_media_geral,
    delta_do_consolidado,
    estilo_grade,
    figura_grade,
    limites,
    nome_curto,
    pontos_alem_da_borda,
    rho_confusos,
    tabela_grade,
    texto_do_painel,
)


@pytest.fixture(scope="module")
def modelos():
    return carregar()


@pytest.fixture(scope="module")
def correlacoes():
    return pd.read_csv(CORRELACOES)


def snippets_sinteticos(semente, n_bases=24):
    gerador = np.random.RandomState(semente)
    linhas = [
        {"base": str(b) if b else "", "pair_id": f"{b if b else ''}-v{v}", "condicao": c, "ppl": float(np.exp(gerador.normal(3, 2))), "frp": float(gerador.normal(0, 4e-6))}
        for b in range(n_bases)
        for v in range(3)
        for c in ["confuso", "limpo"]
    ]
    return pd.DataFrame(linhas)


def modelos_sinteticos():
    chaves = {(f, p) for grade in GRADES.values() for f in grade["familias"] for p in PAPEIS}
    return {c: {"modelo": f"org/Modelo-{i}" + ("-4bit" if "4 bits" in c[0] else ""), "dados": dados_do_modelo(snippets_sinteticos(i))} for i, c in enumerate(sorted(chaves))}


def correlacoes_sinteticas(modelos):
    return pd.DataFrame([{"trio": f, "papel": p, "subconjunto": "confusos", "rho": 0.1 + 0.01 * i} for i, (f, p) in enumerate(sorted(modelos))])


@pytest.mark.dados
@pytest.mark.tokenizador
def test_24_grupos_por_2_condicoes_em_cada_um_dos_18_modelos(modelos):
    familias = {f for g in GRADES.values() for f in g["familias"]}
    assert len(modelos) == 18 == len(familias) * len(PAPEIS)
    for item in modelos.values():
        dados = item["dados"]
        grupos = dados[dados["grupo"] != MEDIA_GERAL]
        assert grupos["grupo"].nunique() == 24 and len(grupos) == 48
        assert grupos["n"].eq(3).all() and grupos.groupby("grupo")["condicao"].nunique().eq(2).all()
        assert dados[dados["grupo"] == MEDIA_GERAL]["n"].tolist() == [72, 72]


@pytest.mark.dados
@pytest.mark.tokenizador
def test_delta_da_media_geral_e_o_delta_do_consolidado(modelos):
    consolidado = pd.read_csv(CONSOLIDADO)
    for (familia, papel), item in modelos.items():
        assert delta_da_media_geral(item["dados"]) == pytest.approx(delta_do_consolidado(consolidado, familia, papel), abs=1e-9)


@pytest.mark.dados
@pytest.mark.tokenizador
def test_y_das_medias_por_grupo_e_igual_em_todos_os_modelos(modelos):
    series = [item["dados"].set_index(["grupo", "condicao"])["y"] for item in modelos.values()]
    for serie in series[1:]:
        assert serie.index.equals(series[0].index)
        assert serie.to_numpy() == pytest.approx(series[0].to_numpy(), abs=1e-9)


@pytest.mark.dados
@pytest.mark.tokenizador
def test_categoria_bate_com_os_sinais_e_a_soma_e_24(modelos):
    for item in modelos.values():
        dados = item["dados"]
        grupos = dados[dados["grupo"] != MEDIA_GERAL].pivot(index="grupo", columns="condicao", values=["x", "y"])
        dx = grupos[("x", "confuso")] - grupos[("x", "limpo")]
        dy = grupos[("y", "confuso")] - grupos[("y", "limpo")]
        por_grupo = pd.Series(categorias_dos_grupos(dados))
        assert (por_grupo[(dx > 0) & (dy > 0)] == "expected").all() and ((dx > 0) & (dy > 0)).sum() == (por_grupo == "expected").sum()
        assert (por_grupo[(dx < 0) & (dy < 0)] == "opposite").all() and ((dx < 0) & (dy < 0)).sum() == (por_grupo == "opposite").sum()
        assert (por_grupo[~(((dx > 0) & (dy > 0)) | ((dx < 0) & (dy < 0)))] == "partial").all()
        assert sum(contagem(dados).values()) == 24
        # a categoria de cada linha da tabela é a do grupo (e a média geral não tem)
        assert (dados[dados["grupo"] == MEDIA_GERAL]["categoria"] == "").all()


def test_categoria_nos_casos_de_borda():
    assert [categoria(0, 0, 1, 1), categoria(1, 1, 0, 0), categoria(0, 0, 1, -1), categoria(0, 0, -1, 1), categoria(0, 0, 0, 1), categoria(0, 0, 1, 0)] == ["expected", "opposite", "partial", "partial", "partial", "partial"]


@pytest.mark.dados
@pytest.mark.tokenizador
def test_rho_e_proporcoes_do_painel_batem_com_correlacoes_e_contagens(modelos, correlacoes):
    for nome, grade in GRADES.items():
        figura = figura_grade(modelos, correlacoes, grade["familias"], grade["tamanho"])
        painel = 0
        for linha, familia in enumerate(grade["familias"]):
            for coluna, papel in enumerate(PAPEIS):
                eixo = figura.axes[linha * 3 + coluna]
                texto = next(t.get_text() for t in eixo.texts if "expected" in t.get_text())
                linhas = texto.split("\n")
                assert len(linhas) == 3
                rho = float(re.search(r"= (-?\d+\.\d+)", linhas[0]).group(1))
                esperado = correlacoes[(correlacoes["trio"] == familia) & (correlacoes["papel"] == papel) & (correlacoes["subconjunto"] == "confusos")]["rho"].iloc[0]
                assert rho == round(esperado, 2) and rho_confusos(correlacoes, familia, papel) == esperado
                n = contagem(modelos[(familia, papel)]["dados"])
                e = re.fullmatch(r"expected (\d+)/24 \((\d+)%\)", linhas[1])
                p, o = re.fullmatch(r"partial (\d+)/24 \((\d+)%\) · opposite (\d+)/24 \((\d+)%\)", linhas[2]).group(1, 3), re.fullmatch(r"partial (\d+)/24 \((\d+)%\) · opposite (\d+)/24 \((\d+)%\)", linhas[2]).group(2, 4)
                assert int(e.group(1)) == n["expected"] and int(p[0]) == n["partial"] and int(p[1]) == n["opposite"]
                assert int(e.group(1)) + int(p[0]) + int(p[1]) == 24
                assert [int(e.group(2)), int(o[0]), int(o[1])] == [int(100 * n[c] / 24 + 0.5) for c in ["expected", "partial", "opposite"]]
                painel += 1
        assert painel == len(grade["familias"]) * 3


def test_texto_do_painel_arredonda_para_inteiro():
    dados = dados_do_modelo(snippets_sinteticos(5))
    c = contagem_e_proporcao(dados)
    assert c["expected_n"] + c["partial_n"] + c["opposite_n"] == 24
    assert all(c[f"{k}_pct"] == int(100 * c[f"{k}_n"] / 24 + 0.5) for k in CATEGORIAS)
    assert texto_do_painel(0.4567, dados).split("\n")[0] == r"$\rho_{\mathrm{conf}}$ = 0.46"


@pytest.mark.dados
@pytest.mark.tokenizador
def test_grade_contraste_usa_llama_13b_e_gemma_7b_nos_tres_papeis(modelos, correlacoes):
    assert GRADES["eeg-grade-contraste"]["familias"] == ["Llama-2 13B", "Gemma 7B"] and GRADES["eeg-grade-contraste"]["tamanho"] == (7, 4.8)
    assert {(f, p) for f in GRADES["eeg-grade-contraste"]["familias"] for p in PAPEIS} <= set(modelos)
    figura = figura_grade(modelos, correlacoes, GRADES["eeg-grade-contraste"]["familias"], GRADES["eeg-grade-contraste"]["tamanho"])
    nomes = [t.get_text() for e in figura.axes for t in e.texts if isinstance(t, matplotlib.text.Annotation) and t.arrow_patch is None]
    assert nomes == ["Llama-2-13b-hf", "CodeLlama-13b-hf", "CodeLlama-13b-Instruct-hf", "gemma-7b", "codegemma-7b", "codegemma-7b-it"]


def test_contrastes_da_dupla_no_consolidado_3():
    c = contrastes_da_dupla(pd.read_csv(CONSOLIDADO))
    assert {k: round(v["estimativa"], 2) for k, v in c.items()} == {"Llama-2 13B D_codigo": 1.14, "Llama-2 13B D_instrucao": -0.54, "Gemma 7B D_codigo": -0.5, "Gemma 7B D_instrucao": 0.78}
    assert all(v["ic_exclui_zero"] for v in c.values())


@pytest.mark.dados
@pytest.mark.tokenizador
def test_tabela_traz_o_rho_usado_e_as_linhas_da_media_geral(modelos, correlacoes):
    tabela = tabela_grade(modelos, correlacoes)
    assert len(tabela) == 18 * 50 and (tabela["grupo"] == MEDIA_GERAL).sum() == 36
    assert list(tabela.columns) == ["familia", "tamanho", "papel", "modelo", "grupo", "condicao", "x", "y", "n", "categoria", "rho_confusos", "expected_n", "partial_n", "opposite_n", "expected_pct", "partial_pct", "opposite_pct"]
    linha = tabela[(tabela["familia"] == "Qwen2.5") & (tabela["tamanho"] == "7B") & (tabela["papel"] == "base")].iloc[0]
    assert linha["rho_confusos"] == rho_confusos(correlacoes, "Qwen2.5 7B", "base")
    assert linha["expected_n"] + linha["partial_n"] + linha["opposite_n"] == 24
    assert {"Llama-2", "Gemma"} <= set(tabela["familia"])


def test_grade_tem_eixos_compartilhados_legenda_unica_e_marca_4_bits():
    modelos = modelos_sinteticos()
    correlacoes = correlacoes_sinteticas(modelos)
    figura = figura_grade(modelos, correlacoes, GRADES["eeg-grade-qwen25"]["familias"], GRADES["eeg-grade-qwen25"]["tamanho"])
    assert len(figura.axes) == 9 and tuple(figura.get_size_inches()) == (7, 6.8)
    assert len({tuple(e.get_xlim()) for e in figura.axes}) == 1 and len({tuple(e.get_ylim()) for e in figura.axes}) == 1
    assert all(e.get_xscale() == "log" for e in figura.axes)
    assert [t.get_text() for t in figura.legends[0].get_texts()] == ["Clean", "Confusing", "Expected direction", "Partial", "Opposite", "Mean over base pairs"]
    assert "geometric mean per base pair, log scale" in figura._supxlabel.get_text() and "FRP" in figura._supylabel.get_text()
    titulos = [t.get_text() for e in figura.axes for t in e.texts if "(4-bit)" in t.get_text()]
    assert len(titulos) == 3
    # rótulo da linha à esquerda e só na primeira coluna
    assert [figura.axes[i].get_ylabel() for i in (0, 3, 6)] == GRADES["eeg-grade-qwen25"]["familias"]
    assert all(figura.axes[i].get_ylabel() == "" for i in (1, 2, 4))


def test_cor_do_papel_no_titulo_e_estilo_das_setas_por_categoria():
    modelos = modelos_sinteticos()
    figura = figura_grade(modelos, correlacoes_sinteticas(modelos), GRADES["eeg-grade-llama2"]["familias"], GRADES["eeg-grade-llama2"]["tamanho"])
    for coluna, papel in enumerate(PAPEIS):
        etiqueta = next(t for t in figura.axes[coluna].texts if "mathbf" in t.get_text())
        assert matplotlib.colors.to_hex(etiqueta.get_color()).upper() == PAPEIS[papel]["cor"].upper()
    e, p, o = CATEGORIAS["expected"], CATEGORIAS["partial"], CATEGORIAS["opposite"]
    assert (e["cor"], e["lw"], e["alpha"]) == ("#4F8A9E", 1.0, 0.85)
    assert (p["cor"], p["lw"], p["alpha"]) == ("#7F7F7F", 0.9, 0.8)
    assert (o["cor"], o["lw"], o["alpha"]) == ("#B5655A", 1.0, 0.85)
    assert p["zorder"] < o["zorder"] < e["zorder"]
    # expected e opposite: mesmo formato, só a cor muda
    assert {k: v for k, v in e.items() if k not in {"cor", "zorder", "rotulo"}} == {k: v for k, v in o.items() if k not in {"cor", "zorder", "rotulo"}}
    setas = [t for t in figura.axes[0].texts if isinstance(t, matplotlib.text.Annotation) and t.arrow_patch is not None]
    assert len(setas) == 25  # 24 grupos e a média geral
    # todas as linhas são cheias
    assert all(t.arrow_patch.get_linestyle() in ("-", "solid") for t in setas)
    opostas = [t for t in setas if matplotlib.colors.to_hex(t.arrow_patch.get_edgecolor()).upper() == "#B5655A"]
    assert len(opostas) == contagem(modelos[("Llama-2 7B", "base")]["dados"])["opposite"]
    assert all(matplotlib.colors.to_hex(t.arrow_patch.get_facecolor()).upper() == "#B5655A" for t in opostas)
    media = setas[-1].arrow_patch
    assert matplotlib.colors.to_hex(media.get_edgecolor()).upper() == COR_MEDIA.upper() and media.get_linewidth() == 2.6 and media.get_path_effects()
    assert media.get_zorder() >= 8 or setas[-1].get_zorder() == 8


def test_estilo_da_grade_nao_usa_hinting():
    assert estilo_grade()["text.hinting"] == "none" and estilo_grade()["mathtext.fontset"] == "stix"


def test_x_vai_ate_1e6_e_o_que_passa_encosta_na_borda_com_marcador():
    modelos = modelos_sinteticos()
    chave = ("Llama-2 7B", "base")
    snippets = snippets_sinteticos(99)
    snippets.loc[(snippets["condicao"] == "confuso") & (snippets["base"].isin(["1", "2"])), "ppl"] = 5e7
    modelos[chave] = {**modelos[chave], "dados": dados_do_modelo(snippets)}
    assert pontos_alem_da_borda(modelos[chave]["dados"]) == 2
    figura = figura_grade(modelos, correlacoes_sinteticas(modelos), GRADES["eeg-grade-llama2"]["familias"], GRADES["eeg-grade-llama2"]["tamanho"])
    assert all(e.get_xlim()[1] == X_MAX for e in figura.axes)
    eixo = figura.axes[0]
    triangulos = [c for c in eixo.collections if len(c.get_offsets()) == 1 and c.get_offsets()[0][0] == X_MAX]
    assert len(triangulos) == 2 and all(not c.get_clip_on() for c in triangulos)
    assert pontos_alem_da_borda(modelos[("Llama-2 7B", "codigo")]["dados"]) == 0
    # nenhum dado é removido: a tabela guarda os x originais
    assert modelos[chave]["dados"].query("grupo != 'media_geral'")["x"].max() > X_MAX


def test_rotulos_de_linha_das_tres_grades():
    modelos = modelos_sinteticos()
    correlacoes = correlacoes_sinteticas(modelos)
    esperados = {
        "eeg-grade-llama2": ["Llama-2 7B", "Llama-2 13B"],
        "eeg-grade-qwen25": ["Qwen2.5 3B", "Qwen2.5 7B", "Qwen2.5 32B (4 bits)"],
        "eeg-grade-contraste": ["Llama-2 13B", "Gemma 7B"],
    }
    assert set(GRADES) == set(esperados)
    for nome, grade in GRADES.items():
        figura = figura_grade(modelos, correlacoes, grade["familias"], grade["tamanho"])
        linhas = len(grade["familias"])
        assert [figura.axes[3 * i].get_ylabel() for i in range(linhas)] == esperados[nome]
        assert all(figura.axes[3 * i + j].get_ylabel() == "" for i in range(linhas) for j in (1, 2))
    assert "Llama-2 13B" in [e.get_ylabel() for e in figura_grade(modelos, correlacoes, ["Llama-2 13B", "Gemma 7B"], (7, 4.8)).axes]


def test_nome_curto_marca_o_32b_em_4_bits():
    assert nome_curto("mlx-community/Qwen2.5-Coder-32B-4bit", "Qwen2.5 32B (4 bits)") == "Qwen2.5-Coder-32B (4-bit)"
    assert nome_curto("Qwen/Qwen2.5-7B", "Qwen2.5 7B") == "Qwen2.5-7B"


def test_limites_deixam_folga_no_alto_e_cobrem_os_pontos():
    modelos = modelos_sinteticos()
    chaves = list(modelos)[:3]
    (x_min, x_max), (y_min, y_max) = limites(modelos, chaves)
    assert x_max == X_MAX
    for chave in chaves:
        pontos = modelos[chave]["dados"].query("grupo != 'media_geral'")
        assert pontos["x"].min() > x_min and pontos["x"].max() <= x_max
        assert pontos["y"].min() > y_min and pontos["y"].max() < y_max - 0.4 * (pontos["y"].max() - pontos["y"].min())


def test_duas_execucoes_geram_pdfs_e_pngs_identicos(tmp_path):
    modelos = modelos_sinteticos()
    correlacoes = correlacoes_sinteticas(modelos)
    for rodada in ["a", "b"]:
        for nome, grade in GRADES.items():
            figura = figura_grade(modelos, correlacoes, grade["familias"], grade["tamanho"])
            salvar(figura, tmp_path / f"{nome}-{rodada}.pdf", rc=estilo_grade())
            salvar(figura, tmp_path / f"{nome}-{rodada}.png", rc=estilo_grade(), dpi=150)
    for nome in GRADES:
        for extensao in ["pdf", "png"]:
            assert (tmp_path / f"{nome}-a.{extensao}").read_bytes() == (tmp_path / f"{nome}-b.{extensao}").read_bytes()
    assert b"CreationDate" not in (tmp_path / "eeg-grade-llama2-a.pdf").read_bytes()


def test_estilo_das_setas_parciais_cor_espessura_e_alpha():
    modelos = modelos_sinteticos()
    chave = ("Llama-2 7B", "base")
    figura = figura_grade(modelos, correlacoes_sinteticas(modelos), GRADES["eeg-grade-llama2"]["familias"], GRADES["eeg-grade-llama2"]["tamanho"])
    n_parciais = contagem(modelos[chave]["dados"])["partial"]
    assert n_parciais > 0
    setas = [t for t in figura.axes[0].texts if isinstance(t, matplotlib.text.Annotation) and t.arrow_patch is not None]
    parciais = [t for t in setas if matplotlib.colors.to_hex(t.arrow_patch.get_edgecolor()).upper() == "#7F7F7F"]
    assert len(parciais) == n_parciais
    for t in parciais:
        patch = t.arrow_patch
        assert patch.get_linewidth() == 0.9 and patch.get_alpha() == 0.8
        assert matplotlib.colors.to_hex(patch.get_facecolor()).upper() == "#7F7F7F"  # cabeça cheia na mesma cor
    # partial por baixo, opposite no meio e expected por cima
    por_cor = {matplotlib.colors.to_hex(t.arrow_patch.get_edgecolor()).upper(): t.get_zorder() for t in setas[:-1]}
    assert por_cor["#7F7F7F"] < por_cor["#B5655A"] < por_cor["#4F8A9E"]
    # os pontos confusos parciais acompanham a cor nova
    pontos = [c for c in figura.axes[0].collections if len(c.get_offsets()) == 1 and len(c.get_facecolor()) and matplotlib.colors.to_hex(c.get_facecolor()[0]).upper() == "#7F7F7F"]
    assert len(pontos) == n_parciais
    # a legenda traz a mesma cor, espessura e alpha
    legenda = next(h for h in figura.legends[0].legend_handles if h.get_alpha() == 0.8)
    assert matplotlib.colors.to_hex(legenda.get_edgecolor()).upper() == "#7F7F7F" and legenda.get_linewidth() == 0.9


def test_percentual_arredonda_meio_para_cima():
    from aoc_perplexity.figuras_grade import percentual

    assert [percentual(k) for k in (0, 3, 9, 12, 24)] == [0, 13, 38, 50, 100]
    assert percentual(1, total=8) == 13
