import json
from types import SimpleNamespace

import pandas as pd

from aoc_perplexity import consolidar
from aoc_perplexity.resultados import Manifesto

QUANTIDADES = ["Delta_base", "Delta_codigo", "Delta_instruct", "D", "D_codigo", "D_instrucao"]


def trio_falso(pasta):
    pasta.mkdir(parents=True)
    linhas = [
        {"metrica": m, "quantidade": q, "estimativa": 0.5, "ic_inferior": 0.1, "ic_superior": 0.9}
        for m in ["aoi_max", "snippet_media"]
        for q in QUANTIDADES
    ]
    pd.DataFrame(linhas).to_csv(pasta / "trio.csv", index=False)


def test_saida_escolhe_a_pasta_e_nao_sobrescreve(tmp_path, monkeypatch):
    trio_falso(tmp_path / "entrada" / "t")
    monkeypatch.setattr(consolidar, "TRIOS", {("T", "bergum"): tmp_path / "entrada" / "t", ("T", "gopstein"): tmp_path / "entrada" / "t"})
    monkeypatch.setattr(consolidar, "SLUGS", {"T": "t"})
    manifesto = tmp_path / "manifesto.json"
    pasta = str(tmp_path / "entrada" / "t")
    manifesto.write_text(json.dumps({"trios": {"principais": {"T|bergum": pasta, "T|gopstein": pasta}}, "etapas": {}}))
    saida = tmp_path / "saida"
    saida.mkdir()

    consolidar.main(["--manifesto", str(manifesto), "--saida", str(saida)])
    consolidar.main(["--manifesto", str(manifesto), "--saida", str(saida)])

    pastas = sorted(p.name for p in saida.iterdir())
    assert len(pastas) == 2 and pastas[1].endswith("-consolidado-2")
    assert (saida / pastas[0] / "consolidado.csv").exists()
    assert (saida / pastas[0] / "tabela-aoi_max.tex").exists()


def test_limite_que_arredondaria_para_zero_mostra_o_valor():
    assert consolidar.numero(0.00206) == "0.002"
    assert consolidar.numero(-0.00494) == "-0.005"
    assert consolidar.numero(-0.04784) == "-0.05"
    assert consolidar.numero(0.0) == "0.00"
    assert consolidar.numero(8e-05) == "0.00008" and consolidar.numero(-3.2e-05) == "-0.00003"
    linha = SimpleNamespace(estimativa=-0.0478, ic_inferior=-0.0968, ic_superior=0.00206)
    assert consolidar.celula(linha) == "$-$0.05 [$-$0.10, 0.002]"


def test_consolidado_real_tem_os_trios_esperados_e_bate_com_cada_trio_csv(tmp_path):
    consolidar.main(["--saida", str(tmp_path)])
    tabela = pd.read_csv(next(tmp_path.iterdir()) / "consolidado.csv")

    esperados = {
        "bergum": ["Qwen2.5 3B", "Qwen2.5 7B", "Qwen2.5 32B (4 bits)", "DeepSeek 7B", "Yi 9B", "Llama-2 7B", "Llama-2 13B", "Gemma 7B"],
        "gopstein": ["Qwen2.5 3B", "Qwen2.5 7B", "Qwen2.5 32B (4 bits)", "DeepSeek 7B", "Yi 9B", "Llama-2 7B", "Llama-2 13B", "Gemma 7B"],
    }
    for conjunto, trios in esperados.items():
        assert sorted(tabela[tabela.conjunto == conjunto].trio.unique()) == sorted(trios)
    assert len(tabela.groupby(["trio", "conjunto"])) == sum(len(v) for v in esperados.values())

    chave = ["metrica", "quantidade"]
    for (trio, conjunto), parte in tabela.groupby(["trio", "conjunto"]):
        origem = pd.read_csv(consolidar.TRIOS[(trio, conjunto)] / "trio.csv")
        colunas = [c for c in origem.columns]
        pd.testing.assert_frame_equal(
            parte[colunas].sort_values(chave).reset_index(drop=True),
            origem.sort_values(chave).reset_index(drop=True),
            check_exact=False,
        )
        if conjunto == "gopstein":
            assert "aoi-manual-v2.1" in consolidar.TRIOS[(trio, conjunto)].name  # C vem da AOI v2.1; as pastas sem sufixo são a v1


def test_gemma_aparece_como_travessao_no_snippet_media_e_fica_marcado_no_csv(tmp_path):
    consolidar.main(["--saida", str(tmp_path)])
    pasta = next(tmp_path.iterdir())
    tabela = pd.read_csv(pasta / "consolidado.csv")

    invalidas = tabela[tabela.valida == "nao"]
    assert set(zip(invalidas.trio, invalidas.metrica)) == {("Gemma 7B", "snippet_media")}
    assert len(invalidas) == 12  # 6 quantidades em Java e 6 em C: marcadas, não removidas
    assert tabela[(tabela.trio == "Gemma 7B") & (tabela.metrica == "aoi_max")].valida.eq("sim").all()

    snippet = (pasta / "tabela-snippet_media.tex").read_text()
    gemma = [linha for linha in snippet.splitlines() if linha.startswith("Gemma 7B")]
    assert len(gemma) == 2 and all(linha.count("---") == 6 for linha in gemma)
    assert "---: not reported" in snippet and "decision" not in snippet and "e$-$" not in snippet
    aoi = (pasta / "tabela-aoi_max.tex").read_text()
    assert "---" not in aoi and "Gemma 7B" in aoi


def test_pares_do_aoi_max_em_c_sao_60_e_o_snippet_media_63(tmp_path):
    consolidar.main(["--saida", str(tmp_path)])
    tabela = pd.read_csv(next(tmp_path.iterdir()) / "consolidado.csv")
    c = tabela[tabela.conjunto == "gopstein"].groupby("metrica").n_pares.unique()
    assert list(c["aoi_max"]) == [60] and list(c["snippet_media"]) == [63]
    java = tabela[tabela.conjunto == "bergum"].n_pares.unique()
    assert list(java) == [72]


def linhas_da_tabela_compacta(tmp_path):
    consolidar.main(["--saida", str(tmp_path)])
    texto = (next(tmp_path.iterdir()) / "tabela-contrastes-aoi_max.tex").read_text()
    familias = [t for t, c in consolidar.TRIOS if c == "bergum"]
    return texto, [linha for linha in texto.splitlines() if linha.startswith(tuple(familias))]


def test_tabela_compacta_tem_uma_linha_por_familia_e_seis_colunas_de_numero(tmp_path):
    texto, linhas = linhas_da_tabela_compacta(tmp_path)
    assert len(linhas) == 8
    assert [linha.split(" & ")[0] for linha in linhas] == [
        "Qwen2.5 3B", "Qwen2.5 7B", "Qwen2.5 32B (4 bits)", "DeepSeek 7B", "Yi 9B", "Llama-2 7B", "Llama-2 13B", "Gemma 7B",
    ]
    assert all(len(linha.split(" & ")) == 7 for linha in linhas)
    assert r"\footnotesize" in texto


def test_tabela_compacta_nao_tem_celula_vazia_porque_toda_familia_tem_c(tmp_path):
    texto, linhas = linhas_da_tabela_compacta(tmp_path)
    assert "---" not in texto and all(len(linha.rstrip(r" \\").split(" & ")) == 7 for linha in linhas)


def test_tabela_compacta_do_relatorio_nao_marca_dependencia_da_anotacao(tmp_path):
    texto, _ = linhas_da_tabela_compacta(tmp_path)  # a dependência da anotação fica no texto, não na tabela
    assert "dagger" not in texto and "AOI v2.1" not in texto


def test_tabela_compacta_marca_com_adaga_quando_recebe_a_robustez(tmp_path):
    consolidar.main(["--saida", str(tmp_path)])
    tabela = pd.read_csv(next(tmp_path.iterdir()) / "consolidado.csv")
    robustez = pd.read_csv(Manifesto.ler().pasta("sensibilidade") / consolidar.ROBUSTEZ)
    dependentes = {(r.trio, r.quantidade) for r in robustez.itertuples() if r.classe == consolidar.DEPENDENTE and r.quantidade in consolidar.CONTRASTES}
    texto = consolidar.tabela_contrastes(tabela, robustez)
    assert texto.count(r"$^\dagger$") == len(dependentes) + 1  # uma por contraste dependente e uma na nota


def test_celulas_da_tabela_compacta_batem_com_o_consolidado_e_negrito_quando_o_ic_exclui_zero(tmp_path):
    texto, linhas = linhas_da_tabela_compacta(tmp_path)
    tabela = pd.read_csv(next(tmp_path.iterdir()) / "consolidado.csv")
    aoi = tabela[tabela.metrica == "aoi_max"].set_index(["conjunto", "trio", "quantidade"])
    for linha in linhas:
        familia, *celulas = linha.rstrip(r" \\").split(" & ")
        for conjunto, parte in [("bergum", celulas[:3]), ("gopstein", celulas[3:])]:
            for q, c in zip(consolidar.CONTRASTES, parte):
                esperado = consolidar.celula(aoi.loc[(conjunto, familia, q)])
                assert c.removesuffix(r"$^\dagger$") == esperado


def test_tabela_em_coluna_tem_java_sobre_c_com_as_mesmas_celulas_da_larga(tmp_path):
    texto_largo, linhas_largas = linhas_da_tabela_compacta(tmp_path)
    texto = (next(tmp_path.iterdir()) / "tabela-contrastes-aoi_max-coluna.tex").read_text()
    assert texto.startswith("{\\scriptsize\\setlength{\\tabcolsep}{2.5pt}\n\\begin{tabular}{lccc}\n") and "dagger" not in texto
    linhas = texto.splitlines()
    assert linhas.index(r"\multicolumn{4}{l}{\textit{Java (Bergum et al.)}} \\") < linhas.index(r"\multicolumn{4}{l}{\textit{C (Gopstein et al.)}} \\")
    dados = [l for l in linhas if l.endswith(r" \\") and " & " in l and not l.startswith("Family")]
    assert len(dados) == 16 and all(len(l.split(" & ")) == 4 for l in dados)
    largas = [l.rstrip(r" \\").split(" & ") for l in linhas_largas]
    esperado = [[f, *c[:3]] for f, *c in largas] + [[f, *c[3:]] for f, *c in largas]
    assert [l.rstrip(r" \\").split(" & ") for l in dados] == esperado
    rotulo_largo = next(l for l in texto_largo.splitlines() if l.startswith("Family"))
    assert next(l for l in linhas if l.startswith("Family")) == " & ".join(rotulo_largo.split(" & ")[:4]) + r" \\"


def test_as_tres_tabelas_usam_family_no_cabecalho(tmp_path):
    consolidar.main(["--saida", str(tmp_path)])
    pasta = next(tmp_path.iterdir())
    for nome in ["aoi_max", "snippet_media", "contrastes-aoi_max"]:
        texto = (pasta / f"tabela-{nome}.tex").read_text()
        assert "Family &" in texto and "Trio &" not in texto
