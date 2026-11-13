import json

from aoc_perplexity import tabela_modelos
from aoc_perplexity.tabela_modelos import pares_da_reexecucao


def test_pares_da_reexecucao_nao_escreve_ao_lado_do_resultado(tmp_path, monkeypatch):
    pasta = tmp_path / "results" / "execucao"
    pasta.mkdir(parents=True)
    caminho = pasta / "perplexidade.json"
    caminho.write_text(json.dumps({"1-obf-v0": {"summary": {"overall_perplexity": 3.0}}, "1-clean-v0": {"summary": {"overall_perplexity": 2.0}}}))
    vistos = []

    def carregar(temporario):
        vistos.append(temporario)
        assert temporario.exists() and pasta not in temporario.parents
        return "pares"

    monkeypatch.setattr(tabela_modelos, "carregar_pares", carregar)
    assert pares_da_reexecucao(caminho) == "pares"
    assert sorted(p.name for p in pasta.iterdir()) == ["perplexidade.json"]
    assert not vistos[0].exists()
