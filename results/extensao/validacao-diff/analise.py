"""Valida a AOI por diff nos dados de Bergum: compara com a AOI manual e refaz a RQ1 (perplexidade máxima na AOI).

Rode da raiz: uv run --locked python results/extensao/validacao-diff/analise.py [PASTA_DE_SAIDA]
Grava tabela.csv em PASTA_DE_SAIDA (padrão: results/novos/validacao-diff), nunca na pasta publicada.
As revisões dos tokenizadores são as que eram a main dos repositórios em 2026-11-02, quando a tabela foi gerada.
"""

import copy
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from aoc_perplexity.aoi import mapear, resumo
from aoc_perplexity.gopstein import mapear_por_diff
from aoc_perplexity.rq12 import carregar_pares, wilcoxon_com_efeito

MODELOS = {
    "Qwen2.5-Coder-32B": ("Qwen/Qwen2.5-Coder-32B", "2e12b5f7bc878d424d222e224ed40aee564ec45f"),
    "deepseek-coder-1.3b-base": ("deepseek-ai/deepseek-coder-1.3b-base", "c919139c3a9b4070729c8b2cca4847ab29ca8d94"),
    "deepseek-coder-6.7b-base": ("deepseek-ai/deepseek-coder-6.7b-base", "ce2207a8bfef3ee92bd7dd4cc31c52cfa0046912"),
    "Artigenz-Coder-DS-6.7B": ("Artigenz/Artigenz-Coder-DS-6.7B", "a0dea4a1c6cfdef8043c8accffa803887f444f45"),
    "Yi-Coder-9B": ("01-ai/Yi-Coder-9B", "bc6e63fd26a654ee42dba737a553310bfa01dc5d"),
}
PACOTE = Path("data/external/llm-perplexity-AoC/model_results")
SAIDA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/novos/validacao-diff")
TEMP = Path(tempfile.mkdtemp()) / "stats_temporario.csv"


def tokens(spans):
    return {i for a, b in spans for i in range(a, b + 1)}


def rq1(resultados):
    pd.DataFrame([{"File": a, **resumo(r)} for a, r in resultados.items()]).to_csv(TEMP, index=False)
    pares = carregar_pares(TEMP)
    TEMP.unlink()
    return wilcoxon_com_efeito(pares["Max AOI Perplexity_confuso"], pares["Max AOI Perplexity_limpo"])


linhas = []
for modelo, (repo, revisao) in MODELOS.items():
    tokenizador = AutoTokenizer.from_pretrained(repo, revision=revisao)
    publicado = json.loads((PACOTE / modelo / "perplexity_results.json").read_text())
    manual = mapear(copy.deepcopy(publicado), tokenizador)
    diff = mapear_por_diff(copy.deepcopy(publicado), tokenizador)
    cobertura, precisao = [], []
    for arquivo in manual:
        m = tokens(s for v in manual[arquivo]["aoi_regions"].values() for s in v)
        d = tokens(diff[arquivo]["aoi_regions"]["DIFF"])
        if m:
            cobertura.append(len(m & d) / len(m))
        if d:
            precisao.append(len(m & d) / len(d))
    r_manual, r_diff = rq1(manual), rq1(diff)
    linhas.append(
        {
            "modelo": modelo,
            "cobertura_media": np.mean(cobertura),
            "precisao_media": np.mean(precisao),
            "manual_W": r_manual.w, "manual_p": r_manual.p, "manual_r": r_manual.r,
            "diff_W": r_diff.w, "diff_p": r_diff.p, "diff_r": r_diff.r,
        }
    )
tabela = pd.DataFrame(linhas)
SAIDA.mkdir(parents=True, exist_ok=True)
tabela.to_csv(SAIDA / "tabela.csv", index=False)
print(f"Tabela em {SAIDA / 'tabela.csv'}")
print(tabela.round(3).to_string(index=False))
