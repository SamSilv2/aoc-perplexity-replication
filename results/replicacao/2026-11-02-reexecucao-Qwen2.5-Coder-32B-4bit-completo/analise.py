"""Validação do Qwen2.5-Coder-32B em 4 bits contra o JSON publicado (fp16).

Rode da raiz: uv run python <este arquivo> <pasta da reexecução>
A AOI usa os intervalos de tokens publicados, válidos porque os tokens são idênticos.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, "src")
from aoc_perplexity.rq12 import carregar_pares, wilcoxon_com_efeito

BASE = Path("data/external/llm-perplexity-AoC/model_results/Qwen2.5-Coder-32B")
pasta = Path(sys.argv[1])
nosso = json.loads((pasta / "perplexidade.json").read_text())
com_aoi = json.loads((BASE / "perplexity_results_with_aois.json").read_text())
stats_pub = pd.read_csv(BASE / "pairwise_analysis/pairwise_comparisons/perplexity_stats.csv").set_index("File")


def metricas(resultados):
    linhas = []
    for arquivo, r in resultados.items():
        lp = np.array([t["log_prob"] for t in r["results"]])
        spans = [s for v in com_aoi[arquivo]["aoi_regions"].values() for s in v]
        aoi = np.concatenate([lp[a : b + 1] for a, b in spans]) if spans else np.array([np.nan])
        linhas.append(
            {
                "File": arquivo,
                "Overall Perplexity": np.exp(-lp.mean()),
                "Max Perplexity": np.exp(-lp.min()),
                "Overall AOI Perplexity": np.exp(-np.nanmean(aoi)),
                "Max AOI Perplexity": np.exp(-np.nanmin(aoi)),
            }
        )
    return pd.DataFrame(linhas)


publicado = metricas(com_aoi)
nossa = metricas(nosso)

# Validação do recorte: com as perplexidades publicadas, a AOI tem de bater com a coluna dos autores.
p = publicado.set_index("File")
erro = (p["Max AOI Perplexity"] / stats_pub.loc[p.index, "Max AOI Perplexity"] - 1).abs().max()
print(f"Recorte da AOI contra a coluna publicada: maior erro relativo {erro:.2e}")

lp_n = np.array([t["log_prob"] for r in nosso.values() for t in r["results"]])
lp_p = np.array([t["log_prob"] for a in nosso for t in com_aoi[a]["results"]])
print(f"Token a token: mediana |Δ| {np.median(np.abs(lp_n - lp_p)):.3f}, média com sinal {np.mean(lp_n - lp_p):.3f}, Spearman {spearmanr(lp_n, lp_p).statistic:.4f}")

for nome, df in [("publicado (fp16)", publicado), ("4 bits", nossa)]:
    df.to_csv(pasta / "stats_temporario.csv", index=False)
    pares = carregar_pares(pasta / "stats_temporario.csv")
    (pasta / "stats_temporario.csv").unlink()
    for coluna in ["Overall Perplexity", "Max Perplexity", "Overall AOI Perplexity", "Max AOI Perplexity"]:
        r = wilcoxon_com_efeito(pares[f"{coluna}_confuso"], pares[f"{coluna}_limpo"])
        print(f"RQ1 {nome:17} {coluna:24} n={r.n:2} W={r.w:6.0f} p={r.p:.4f} r={r.r:.3f}")
    for coluna in ["Overall Perplexity", "Max AOI Perplexity"]:
        print(f"    mediana {coluna}: limpo {pares[f'{coluna}_limpo'].median():.2f}, confuso {pares[f'{coluna}_confuso'].median():.2f}")
