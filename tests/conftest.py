"""Marcadores dos testes que precisam de algo além do código (veja `make verificar`).

- dados: pacote dos autores e snippets de Gopstein em data/external (make data);
- tokenizador: os tokenizadores das execuções, nas revisões fixadas, no cache local do Hugging Face (make tokenizadores);
- inferencia: torch (make setup).
Sem o pré-requisito, o teste marcado é pulado com o motivo. Os testes não acessam a rede.
"""

import importlib.util

import pytest

from aoc_perplexity import ambiente


def pytest_configure(config):
    config.addinivalue_line("markers", "dados: precisa de data/external (make data)")
    config.addinivalue_line("markers", "tokenizador: precisa dos tokenizadores no cache local (make tokenizadores)")
    config.addinivalue_line("markers", "inferencia: precisa de torch (make setup)")


def pytest_collection_modifyitems(config, items):
    faltam_dados = ambiente.dados_faltando()
    sem_torch = importlib.util.find_spec("torch") is None
    faltam_tokenizadores = ambiente.tokenizadores_faltando() if any("tokenizador" in i.keywords for i in items) else []
    for item in items:
        if "tokenizador" in item.keywords and faltam_tokenizadores:
            item.add_marker(pytest.mark.skip(reason=f"{len(faltam_tokenizadores)} tokenizadores fora do cache: rode make tokenizadores"))
        if "dados" in item.keywords and faltam_dados:
            item.add_marker(pytest.mark.skip(reason=f"faltam dados ({', '.join(faltam_dados)}): rode make data"))
        if "inferencia" in item.keywords and sem_torch:
            item.add_marker(pytest.mark.skip(reason="torch não instalado: rode make setup"))
