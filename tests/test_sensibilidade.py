from types import SimpleNamespace

import pandas as pd

from aoc_perplexity.sensibilidade import classificar, conclusao


def linha(trio, anotacao, estimativa, inferior, superior, quantidade="D", metrica="aoi_max"):
    return {"trio": trio, "anotacao": anotacao, "metrica": metrica, "quantidade": quantidade, "estimativa": estimativa, "ic_inferior": inferior, "ic_superior": superior}


def test_conclusao_e_sinal_e_status_do_ic():
    assert conclusao(SimpleNamespace(estimativa=0.5, ic_inferior=-0.1, ic_superior=1.0)) == "+, IC inclui 0"
    assert conclusao(SimpleNamespace(estimativa=-0.5, ic_inferior=-0.9, ic_superior=-0.1)) == "-, IC exclui 0"


def test_robusto_so_se_as_tres_anotacoes_concordam():
    tabela = pd.DataFrame(
        [
            linha("a", "v1", -0.5, -0.9, -0.1),
            linha("a", "v2", -0.4, -0.8, -0.05),
            linha("a", "v2.1", -0.6, -1.0, -0.2),
            linha("b", "v1", 0.7, -0.4, 1.9),
            linha("b", "v2", 1.0, 0.1, 2.0),
            linha("b", "v2.1", 1.1, 0.1, 2.0),
            linha("c", "v1", 0.2, -0.3, 0.7, metrica="snippet_media"),
        ]
    )
    r = classificar(tabela).set_index("trio")
    assert r.loc["a", "classe"] == "robusto"
    assert r.loc["b", "classe"] == "dependente da anotação"
    assert "c" not in r.index  # só o aoi_max é classificado


def test_sensibilidade_cobre_as_oito_familias_com_c():
    from aoc_perplexity import consolidar, sensibilidade

    assert set(sensibilidade.TRIOS) == set(consolidar.SLUGS.values())
    assert len(sensibilidade.TRIOS) == 8
