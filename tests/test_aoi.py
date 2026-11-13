import math

import pytest

from aoc_perplexity.aoi import ler_aois, para_tokens, resumo


def test_ler_aois_remove_marcacoes_aninhadas():
    limpo, regioes = ler_aois("b = [aoi=WEXP]([aoi=NEXP]V1[aoi=OP]++[/aoi][/aoi] > 1)[/aoi];")
    assert limpo == "b = (V1++ > 1);"
    assert regioes["OP"] == [(7, 9)]
    assert regioes["NEXP"] == [(5, 9)]
    assert regioes["WEXP"] == [(4, 14)]


def test_ler_aois_ignora_tipo_desconhecido():
    limpo, _ = ler_aois("[aoi=XYZ]a")
    assert limpo == "[aoi=XYZ]a"


def test_para_tokens_usa_sobreposicao_sem_tocar_borda():
    offsets = [(0, 3), (3, 5), (5, 8)]
    assert para_tokens(offsets, [(3, 5)]) == [[1, 1]]
    assert para_tokens(offsets, [(4, 6)]) == [[1, 2]]
    assert para_tokens(offsets, [(8, 9)]) == []


def test_resumo_calcula_snippet_e_aoi():
    r = {"results": [{"log_prob": math.log(p)} for p in [0.5, 0.25, 0.125]], "aoi_regions": {"OP": [[1, 2]]}}
    saida = resumo(r)
    assert saida["Max Perplexity"] == pytest.approx(8)
    assert saida["Overall AOI Perplexity"] == pytest.approx(math.sqrt(4 * 8))
    assert saida["Max AOI Perplexity"] == pytest.approx(8)


def test_resumo_sem_aoi_devolve_nan():
    saida = resumo({"results": [{"log_prob": -1.0}], "aoi_regions": {}})
    assert math.isnan(saida["Max AOI Perplexity"])
