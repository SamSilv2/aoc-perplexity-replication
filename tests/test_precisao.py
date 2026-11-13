import math

import pandas as pd

from aoc_perplexity.precisao import aoi_max, resumir


def test_aoi_max_usa_o_menor_log_prob_da_aoi():
    r = {"results": [{"log_prob": -1.0}, {"log_prob": -3.0}, {"log_prob": -2.0}], "aoi_regions": {"A": [[1, 2]]}}
    assert aoi_max(r) == math.exp(3.0)


def test_aoi_max_sem_aoi_e_nan():
    assert math.isnan(aoi_max({"results": [{"log_prob": -1.0}], "aoi_regions": {}}))


def test_resumir_mede_variacao_da_aoi_em_ln():
    tabela = pd.DataFrame(
        {
            "conjunto": ["bergum", "bergum"],
            "arquivo": ["a", "b"],
            "logprob_pos0_referencia": [-10.0, -12.0],
            "logprob_pos0_nova": [-10.0, -12.0],
            "dif_media_pos_maior_0": [0.1, 0.3],
            "dif_maxima_pos_maior_0": [0.5, 0.9],
            "aoi_max_referencia": [1.0, 2.0],
            "aoi_max_nova": [math.e, 2.0],
            "aoi_inclui_pos0": [False, False],
        }
    )
    linha = resumir(tabela).iloc[0]
    assert linha["dif_media"] == 0.2 and linha["dif_maxima"] == 0.9
    assert linha["aoi_ln_maxima"] == 1.0 and linha["aoi_ln_mediana"] == 0.5
