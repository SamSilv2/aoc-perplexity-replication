from pathlib import Path

import pandas as pd
import pytest

from aoc_perplexity.gopstein import ANOTACOES, GOPSTEIN, hash_snippet, ler_pares, ler_snippets, mapear_por_anotacao, spans_por_diff


def test_diff_marca_tokens_trocados():
    assert spans_por_diff([1, 2, 3, 4], [1, 9, 9, 4]) == ([[1, 2]], [[1, 2]])


def test_diff_insercao_marca_vizinho():
    a, b = spans_por_diff([1, 2, 3], [1, 7, 2, 3])
    assert a == [[1, 1]]
    assert b == [[1, 1]]


def test_diff_sem_mudanca_nao_marca_nada():
    assert spans_por_diff([1, 2], [1, 2]) == ([], [])


@pytest.mark.dados
def test_gopstein_tem_63_pares_de_21_atomos():
    pares = ler_pares()
    assert len(pares) == 63
    assert pares["base"].nunique() == 21
    assert sorted(pares["versao"].unique()) == ["v0", "v1", "v2"]
    assert (pares["confuso"] != pares["limpo"]).all()


@pytest.mark.dados
def test_atomos_agrupados_por_id():
    pares = ler_pares().set_index("id_confuso")
    assert pares.loc[[1, 3, 5], "atomo"].unique().tolist() == ["add_CONDITION_atom"]
    assert pares.loc[[25, 27, 29], "atomo"].unique().tolist() == ["replace_CONSTANTVARIABLE_atom"]
    assert "&lt;" not in "".join(pares["limpo"])


@pytest.mark.dados
def test_anotacao_encontra_todos_os_trechos():

    from aoc_perplexity.gopstein import ler_snippets, mapear_por_anotacao

    class PorCaractere:
        def __call__(self, texto, **_):
            return {"offset_mapping": [(i, i + 1) for i in range(len(texto))]}

    resultados = {a: {"code": c} for a, c in ler_snippets().items()}
    mapear_por_anotacao(resultados, PorCaractere())
    assert all(r["aoi_regions"]["ANOT"] for r in resultados.values())


def test_sem_codigo_remove_codigo_e_texto_dos_tokens():
    from aoc_perplexity.gopstein import sem_codigo

    r = {"x": {"code": "int a;", "tokens": ["int", " a", ";"], "summary": {"token_count": 3},
               "results": [{"token": "int", "position": 0, "log_prob": -1.0}]}}
    saida = sem_codigo(r)["x"]
    assert set(saida) == {"summary", "results"}
    assert saida["results"] == [{"position": 0, "log_prob": -1.0}]


class TokenizadorPorCaractere:
    """Um token por caractere, o suficiente para mapear intervalos de caracteres em tokens."""

    def __call__(self, texto, return_offsets_mapping=False, add_special_tokens=False):
        return {"offset_mapping": [(i, i + 1) for i in range(len(texto))], "input_ids": list(range(len(texto)))}


def anotacao_falsa(caminho, codigos, fora=(), hash_errado=None):
    """CSV no formato da v2.1: AOI = do 2º ao 4º caractere; `fora` lista os snippets com na_metrica_aoi = nao."""
    linhas = []
    for nome, codigo in codigos.items():
        linhas.append(
            {
                "snippet": nome,
                "inicio": "" if nome in fora and "-clean-" in nome else "1",
                "fim": "" if nome in fora and "-clean-" in nome else "4",
                "na_metrica_aoi": "nao" if nome in fora else "sim",
                "sha256_snippet": hash_errado or hash_snippet(codigo),
            }
        )
    pd.DataFrame(linhas).to_csv(caminho, index=False)


def resultados_falsos(codigos):
    return {nome: {"code": codigo, "results": [{"log_prob": -1.0 - 0.1 * i} for i in range(len(codigo))]} for nome, codigo in codigos.items()}


CODIGOS = {"1-obf-v0": "abcdef", "1-clean-v0": "abcxyz"}


def test_intervalo_de_caracteres_vira_tokens(tmp_path):
    csv = tmp_path / "a.csv"
    anotacao_falsa(csv, CODIGOS)
    r = mapear_por_anotacao(resultados_falsos(CODIGOS), TokenizadorPorCaractere(), anotacao=csv)
    assert r["1-obf-v0"]["aoi_regions"] == {"ANOT": [[1, 3]]}


def test_hash_errado_gera_erro(tmp_path):
    csv = tmp_path / "a.csv"
    anotacao_falsa(csv, CODIGOS, hash_errado="0" * 16)
    with pytest.raises(ValueError, match="hash"):
        mapear_por_anotacao(resultados_falsos(CODIGOS), TokenizadorPorCaractere(), anotacao=csv)


def test_snippet_alterado_depois_da_anotacao_gera_erro(tmp_path):
    csv = tmp_path / "a.csv"
    anotacao_falsa(csv, CODIGOS)
    alterado = resultados_falsos({**CODIGOS, "1-obf-v0": "abcdeX"})
    with pytest.raises(ValueError, match="hash"):
        mapear_por_anotacao(alterado, TokenizadorPorCaractere(), anotacao=csv)


def test_par_fora_da_metrica_sai_inteiro_com_aviso(tmp_path):
    csv = tmp_path / "a.csv"
    anotacao_falsa(csv, CODIGOS, fora=["1-clean-v0"])
    with pytest.warns(UserWarning, match="fora do aoi_max"):
        r = mapear_por_anotacao(resultados_falsos(CODIGOS), TokenizadorPorCaractere(), anotacao=csv)
    assert r["1-obf-v0"]["fora_do_aoi_max"] and r["1-clean-v0"]["fora_do_aoi_max"]
    assert r["1-obf-v0"]["aoi_regions"] == {"ANOT": []}


@pytest.mark.dados
def test_anotacoes_publicadas_batem_com_o_codigo_baixado():
    snippets = ler_snippets()
    for versao in ["v2", "v2.1"]:
        tabela = pd.read_csv(ANOTACOES[versao], keep_default_na=False, dtype=str)
        assert len(tabela) == 126
        assert all(hash_snippet(snippets[r.snippet]) == r.sha256_snippet for r in tabela.itertuples())
        assert (tabela.na_metrica_aoi == "nao").sum() == 6  # os 3 pares de código morto (átomo 17)


class TokenizadorPorCodigo:
    """Um token por caractere, com o código do caractere como ID."""

    def __call__(self, texto, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in texto]}


def test_com_codigo_confere_hash_dos_ids_quando_salvo(monkeypatch):
    from aoc_perplexity import gopstein
    from aoc_perplexity.gopstein import com_codigo, hash_ids

    monkeypatch.setattr(gopstein, "ler_snippets", lambda pasta: {"1-obf-v0": "ab"})
    tokenizador = TokenizadorPorCodigo()

    def resultado(**summary):
        return {"1-obf-v0": {"summary": summary, "results": [{"position": 0}, {"position": 1}]}}

    # Resultado antigo, sem hash: só a contagem é conferida.
    assert com_codigo(resultado(), tokenizador)["1-obf-v0"]["code"] == "ab"
    assert com_codigo(resultado(token_ids_sha256=hash_ids([97, 98])), tokenizador)["1-obf-v0"]["code"] == "ab"
    # Mesma contagem, IDs diferentes: o hash pega.
    with pytest.raises(ValueError, match="IDs de token"):
        com_codigo(resultado(token_ids_sha256=hash_ids([97, 99])), tokenizador)


def test_sem_codigo_mantem_o_hash():
    from aoc_perplexity.gopstein import sem_codigo

    saida = sem_codigo({"x": {"code": "ab", "tokens": ["a"], "summary": {"token_ids_sha256": "f"}, "results": [{"token": "a", "position": 0}]}})
    assert saida["x"]["summary"] == {"token_ids_sha256": "f"}
