from pathlib import Path

from aoc_perplexity.resultados import Manifesto

EXTENSAO = Path("results/extensao")


def test_indice_cita_todas_as_pastas_da_extensao():
    indice = (EXTENSAO / "INDICE.md").read_text()
    pastas = sorted(p.name for p in EXTENSAO.iterdir() if p.is_dir())
    assert [p for p in pastas if f"`{p}`" not in indice] == []


def test_indice_cita_as_pastas_do_manifesto_e_nada_que_nao_existe():
    import re

    indice = (EXTENSAO / "INDICE.md").read_text()
    manifesto = Manifesto.ler()
    pastas = [p for grupo in ["principais", "template", "atomos"] for p in manifesto.trios(grupo).values()]
    pastas += [manifesto.pasta(e) for e in manifesto.dados["etapas"] if manifesto.pasta(e).parent == EXTENSAO]
    assert [p.name for p in pastas if f"`{p.name}`" not in indice] == []
    citadas = re.findall(r"^\| `([^`]+)` \|", indice, re.M)
    assert [c for c in citadas if not (EXTENSAO / c).is_dir()] == []
