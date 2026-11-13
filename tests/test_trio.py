import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aoc_perplexity import gopstein
from aoc_perplexity.trio import analisar, filtrar_atomos, montar_pares, pares_validos, repositorio_tokenizador
from test_gopstein import TokenizadorPorCaractere, anotacao_falsa


def pares(fator_confuso):
    bases = [str(i) for i in range(6) for _ in range(3)]
    versoes = [f"v{j}" for _ in range(6) for j in range(3)]
    limpo = np.linspace(2, 20, 18)
    df = pd.DataFrame({"base": bases, "versao": versoes})
    for coluna in ["Max AOI Perplexity", "Overall Perplexity"]:
        df[f"{coluna}_limpo"] = limpo
        df[f"{coluna}_confuso"] = limpo * fator_confuso
    return df


def test_delta_e_d_com_razao_constante():
    trio, por_modelo = analisar({"base": pares(2.0), "codigo": pares(4.0), "instruct": pares(8.0)}, n=50)
    t = trio[trio["metrica"] == "aoi_max"].set_index("quantidade")
    assert t.loc["Delta_base", "estimativa"] == pytest.approx(np.log(2))
    assert t.loc["D", "estimativa"] == pytest.approx(np.log(4))
    assert t.loc["D_codigo", "estimativa"] == pytest.approx(np.log(2))
    # Razão constante: o bootstrap não varia.
    assert t.loc["D", "ic_inferior"] == pytest.approx(np.log(4))
    assert len(por_modelo) == 6


def test_recusa_pares_diferentes():
    outro = pares(2.0).iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError):
        analisar({"base": pares(2.0), "codigo": outro, "instruct": pares(2.0)}, n=5)


def test_tokenizador_do_modelo_ou_o_registrado():
    antigo = {"model": "m", "model_revision": "r1", "config": {}}
    assert repositorio_tokenizador(antigo) == ("m", "r1")
    novo = {"model": "m", "model_revision": "r1", "config": {"tokenizador": "t", "tokenizador_revisao": "r2"}}
    assert repositorio_tokenizador(novo) == ("t", "r2")


def resultados_dos_pares(n_bases, fator):
    """Pares de 6 caracteres: o confuso tem log-prob menor (pior) que o limpo por `fator` em todas as posições."""
    resultados = {}
    for b in range(1, n_bases + 1):
        for lado, f in [("clean", 0.0), ("obf", fator + 0.1 * b)]:
            codigo = f"x{b}abcd"
            resultados[f"{b}-{lado}-v0"] = {"code": codigo, "results": [{"log_prob": -1.0 - 0.3 * i - f * (i >= 1)} for i in range(len(codigo))]}
    return resultados


def test_par_excluido_fica_fora_do_aoi_max_e_dentro_do_snippet_media(tmp_path, monkeypatch):
    n_bases = 5
    codigos = {nome: r["code"] for nome, r in resultados_dos_pares(n_bases, 1.0).items()}
    csv = tmp_path / "a.csv"
    anotacao_falsa(csv, codigos, fora=["5-obf-v0", "5-clean-v0"])
    monkeypatch.setitem(gopstein.ANOTACOES, "falsa", csv)

    pares_por_papel = {}
    for papel, fator in [("base", 1.0), ("codigo", 1.5), ("instruct", 2.0)]:
        with pytest.warns(UserWarning, match="fora do aoi_max"):
            pares_por_papel[papel] = montar_pares(resultados_dos_pares(n_bases, fator), TokenizadorPorCaractere(), "gopstein", "manual", "falsa")

    pares = pares_por_papel["base"]
    quinto = pares["base"] == "5"
    assert pares.loc[quinto, "Max AOI Perplexity_confuso"].isna().all()
    assert pares.loc[~quinto, "Max AOI Perplexity_confuso"].notna().all()
    assert pares["Overall Perplexity_confuso"].notna().all()
    assert pares_validos(pares_por_papel, "Max AOI Perplexity").sum() == 4
    assert pares_validos(pares_por_papel, "Overall Perplexity").sum() == 5

    trio, por_modelo = analisar(pares_por_papel, n=20)
    # Δ do aoi_max usa só os 4 pares; Δ do snippet_media usa os 5
    base = pares_por_papel["base"]
    esperado_aoi = np.mean(np.log(base["Max AOI Perplexity_confuso"][:4]) - np.log(base["Max AOI Perplexity_limpo"][:4]))
    esperado_snippet = np.mean(np.log(base["Overall Perplexity_confuso"]) - np.log(base["Overall Perplexity_limpo"]))
    delta = trio[trio.quantidade == "Delta_base"].set_index("metrica").estimativa
    assert delta["aoi_max"] == pytest.approx(esperado_aoi)
    assert delta["snippet_media"] == pytest.approx(esperado_snippet)
    assert delta["aoi_max"] != pytest.approx(delta["snippet_media"])


def test_modelos_com_pares_validos_diferentes_dao_erro():
    a, b = pares(2.0), pares(2.0)
    b.loc[0, "Max AOI Perplexity_confuso"] = np.nan
    with pytest.raises(ValueError, match="mesmos pares válidos"):
        analisar({"base": a, "codigo": b, "instruct": a}, n=5)


RESULTADOS = Path("results/extensao")
TRIOS_C_PUBLICADOS = {
    "qwen2.5-3b": "2026-11-02-trio-qwen2.5-3b-gopstein-aoi-manual",
    "qwen2.5-32b-4bit": "2026-11-03-trio-qwen2.5-32b-4bit-gopstein-aoi-manual",
    "deepseek-7b": "2026-11-03-trio-deepseek-7b-gopstein-aoi-manual",
    "gemma-7b": "2026-11-03-trio-gemma-7b-gopstein-aoi-manual",
    "yi-9b": "2026-11-04-trio-yi-9b-gopstein-aoi-manual",
    "llama-2-7b": "2026-11-04-trio-llama-2-7b-gopstein-aoi-manual",
}


def pares_do_trio_offline(pasta, anotacao):
    """Roda o pipeline do trio sobre os JSON da main; pula o teste se o tokenizador não estiver no cache local."""
    from transformers import AutoTokenizer

    from aoc_perplexity import trio

    execucoes = json.loads((RESULTADOS / pasta / "run.json").read_text())["config"]["execucoes"]
    pares = {}
    for papel, d in execucoes.items():
        registro = json.loads((Path(d) / "run.json").read_text())
        repo, revisao = trio.repositorio_tokenizador(registro)
        tokenizador = AutoTokenizer.from_pretrained(repo, revision=revisao, local_files_only=True)
        resultados = gopstein.com_codigo(json.loads((Path(d) / "perplexidade-gopstein.json").read_text()), tokenizador)
        pares[papel] = trio.montar_pares(resultados, tokenizador, "gopstein", "manual", anotacao)
    return pares


@pytest.mark.dados
@pytest.mark.tokenizador
@pytest.mark.parametrize("nome", list(TRIOS_C_PUBLICADOS))
def test_v1_reproduz_o_trio_csv_publicado_de_gopstein(nome):
    pasta = TRIOS_C_PUBLICADOS[nome]
    pares = pares_do_trio_offline(pasta, "v1")
    trio, _ = analisar(pares)
    publicado = (RESULTADOS / pasta / "trio.csv").read_text()
    assert trio.to_csv(index=False) == publicado


@pytest.mark.dados
@pytest.mark.tokenizador
def test_d_do_yi_na_v2_1_com_60_pares_no_aoi_max():
    pastas = sorted(RESULTADOS.glob("*-trio-yi-9b-gopstein-aoi-manual-v2.1"))
    assert len(pastas) == 1
    trio = pd.read_csv(pastas[0] / "trio.csv")
    d = trio[(trio.metrica == "aoi_max") & (trio.quantidade == "D")].iloc[0]
    assert d.estimativa == pytest.approx(1.1107, abs=5e-5)
    n_pares = json.loads((pastas[0] / "run.json").read_text())["config"]["n_pares"]
    assert n_pares == {"aoi_max": 60, "snippet_media": 63}


def test_filtrar_atomos_mantem_so_os_atomos_listados_nos_tres_modelos():
    todos = {"base": pares(2.0), "codigo": pares(4.0), "instruct": pares(8.0)}
    filtrados = filtrar_atomos(todos, ["1", "4"])
    for p in filtrados.values():
        assert list(p["base"]) == ["1"] * 3 + ["4"] * 3
        assert p.index.tolist() == list(range(6))


def test_filtrar_atomos_sem_lista_devolve_os_mesmos_pares():
    todos = {"base": pares(2.0), "codigo": pares(4.0), "instruct": pares(8.0)}
    assert filtrar_atomos(todos, None) is todos


def test_filtrar_atomos_recusa_atomo_inexistente():
    with pytest.raises(ValueError):
        filtrar_atomos({"base": pares(2.0), "codigo": pares(2.0), "instruct": pares(2.0)}, ["99"])


@pytest.mark.dados
@pytest.mark.tokenizador
def test_sufixo_da_pasta_e_trio_csv_igual_ao_vigente_com_as_mesmas_execucoes(tmp_path, monkeypatch):
    import sys
    from datetime import date

    from aoc_perplexity import trio as modulo_trio

    pasta = "2026-11-02-trio-qwen2.5-3b-bergum-aoi-manual"
    execucoes = json.loads((RESULTADOS / pasta / "run.json").read_text())["config"]["execucoes"]
    argv = ["trio", "--nome", "qwen2.5-3b", "--conjunto", "bergum", "--aoi", "manual", "--sufixo", "template", "--saida", str(tmp_path)]
    monkeypatch.setattr(sys, "argv", argv + [a for papel, d in execucoes.items() for a in (f"--{papel}", d)])
    modulo_trio.main()
    destino = tmp_path / f"{date.today().isoformat()}-trio-qwen2.5-3b-bergum-aoi-manual-template"
    assert destino.is_dir()
    assert (destino / "trio.csv").read_bytes() == (RESULTADOS / pasta / "trio.csv").read_bytes()


def test_revisoes_omitidas_preservam_o_run_json():
    from aoc_perplexity.trio import conferir_revisao, revisoes_por_papel

    registro = {"model": "org/m", "model_revision": "aaa", "config": {}}
    assert repositorio_tokenizador(registro) == ("org/m", "aaa")
    assert repositorio_tokenizador(registro, "bbb") == ("org/m", "bbb")
    conferir_revisao(registro, None)
    conferir_revisao(registro, "aaa")
    with pytest.raises(ValueError, match="revisão aaa"):
        conferir_revisao(registro, "bbb")
    assert revisoes_por_papel(None, "--revision") == {}
    assert revisoes_por_papel(["base=aaa", "instruct=ccc"], "--revision") == {"base": "aaa", "instruct": "ccc"}
    for ruim in ["aaa", "outro=aaa", "base="]:
        with pytest.raises(ValueError, match="PAPEL=SHA"):
            revisoes_por_papel([ruim], "--tokenizer-revision")
