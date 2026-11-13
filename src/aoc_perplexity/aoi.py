"""Mapeia as AOIs anotadas para tokens de qualquer tokenizador e resume a perplexidade.

Segue parse_aoi_regions e map_aois_to_perplexity_file (main/utils.py do pacote), com uma
diferença: as posições de caractere vêm do offset_mapping do tokenizador, e não da soma
dos tamanhos dos tokens decodificados.
"""

import math
from pathlib import Path

import pandas as pd

PACOTE = Path("data/external/llm-perplexity-AoC")
TIPOS = ["BLOCK", "STAT", "WEXP", "NEXP", "OP"]


def ler_aois(codigo, tipos=TIPOS):
    """Remove as marcações [aoi=X]...[/aoi] e devolve o código limpo e os intervalos de caracteres."""
    limpo, regioes, pilha = [], {t: [] for t in tipos}, []
    i = 0
    while i < len(codigo):
        fim_tag = codigo.find("]", i)
        if codigo.startswith("[aoi=", i) and fim_tag != -1 and codigo[i + 5 : fim_tag] in regioes:
            pilha.append((codigo[i + 5 : fim_tag], len(limpo)))
            i = fim_tag + 1
        elif codigo.startswith("[/aoi]", i):
            if pilha:
                tipo, inicio = pilha.pop()
                regioes[tipo].append((inicio, len(limpo)))
            i += 6
        else:
            limpo.append(codigo[i])
            i += 1
    return "".join(limpo), regioes


def aoi_otima(caminho=PACOTE / "code_snippets_with_aois" / "aoi_sizes.csv"):
    tabela = pd.read_csv(caminho, sep=";")
    return dict(zip(tabela.loc[tabela["OPTIMAL"], "Snippet"], tabela.loc[tabela["OPTIMAL"], "NO"]))


def para_tokens(offsets, intervalos):
    """Primeiro e último token que tocam cada intervalo de caracteres."""
    spans = []
    for inicio, fim in intervalos:
        tocados = [i for i, (a, b) in enumerate(offsets) if not (b <= inicio or a >= fim)]
        if tocados:
            spans.append([tocados[0], tocados[-1]])
    return spans


def mapear(resultados, tokenizador, pasta=PACOTE / "code_snippets_with_aois"):
    """Acrescenta `aoi_regions` (só a AOI ótima) a cada arquivo de `resultados`."""
    otima = aoi_otima(pasta / "aoi_sizes.csv")
    anotados = {p.stem: p.read_text(encoding="utf-8") for p in pasta.glob("v*/*.java")}
    for arquivo, r in resultados.items():
        limpo, regioes = ler_aois(anotados[arquivo])
        if limpo != r["code"]:
            raise ValueError(f"código anotado difere do original em {arquivo}")
        offsets = tokenizador(limpo, return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]
        tipo = otima.get(arquivo)
        r["aoi_regions"] = {tipo: para_tokens(offsets, regioes[tipo])} if tipo else {}
    return resultados


def resumo(resultado):
    """Perplexidade média (geométrica) e máxima no snippet e na AOI."""
    lp = [t["log_prob"] for t in resultado["results"]]
    aoi = [lp[i] for spans in resultado["aoi_regions"].values() for a, b in spans for i in range(a, b + 1)]
    return {
        "Overall Perplexity": math.exp(-sum(lp) / len(lp)),
        "Max Perplexity": math.exp(-min(lp)),
        "Overall AOI Perplexity": math.exp(-sum(aoi) / len(aoi)) if aoi else math.nan,
        "Max AOI Perplexity": math.exp(-min(aoi)) if aoi else math.nan,
    }
