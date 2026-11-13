"""Recalcula RQ1 e RQ2 do artigo a partir dos arquivos publicados."""

import argparse
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from aoc_perplexity.resultados import commit_externo, escrever_json, pasta_nova, proveniencia

PACOTE = Path("data/external/llm-perplexity-AoC")
MODELO = "Qwen2.5-Coder-32B"
SAIDA = Path("results/novos")

# Colunas dos autores: "Overall" é a PPL do artigo, a média geométrica.
METRICAS_RQ1 = {
    "snippet_media": "Overall Perplexity",
    "snippet_max": "Max Perplexity",
    "aoi_media": "Overall AOI Perplexity",
    "aoi_max": "Max AOI Perplexity",
}

# Valores da Figura 2 do artigo. p com "<" indica limite superior.
ARTIGO_RQ1 = {
    "snippet_media": {"W": 941, "p": "0.036", "r": "0.247"},
    "snippet_max": {"W": 97, "p": "0.046", "r": "0.391"},
    "aoi_media": {"W": 686, "p": "<0.001", "r": "0.415"},
    "aoi_max": {"W": 638, "p": "<0.001", "r": "0.447"},
}

# Valores da Seção 5.2 do artigo.
ARTIGO_RQ2 = {
    "confuso": {"rho": "0.47", "p": "<0.001"},
    "limpo": {"rho": "0.05", "p": "0.664"},
}


@dataclass(frozen=True)
class ResultadoWilcoxon:
    w: float
    p: float
    r: float
    n: int


def wilcoxon_com_efeito(confuso, limpo):
    """Wilcoxon pareado com r = |Z| / sqrt(N), como no notebook dos autores.

    Diferenças iguais a zero saem do teste, e N conta só as demais.
    """
    diferencas = np.asarray(confuso, dtype=float) - np.asarray(limpo, dtype=float)
    diferencas = diferencas[diferencas != 0]
    n = len(diferencas)
    if n == 0:
        raise ValueError("Todas as diferenças são zero.")

    w, p = wilcoxon(diferencas)
    media = n * (n + 1) / 4
    desvio = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    return ResultadoWilcoxon(float(w), float(p), float(abs(w - media) / desvio / np.sqrt(n)), n)


def carregar_pares(caminho):
    """Junta as linhas limpa e confusa de cada par de `perplexity_stats.csv`."""
    dados = pd.read_csv(caminho)
    partes = dados["File"].str.extract(r"^(?P<base>\d*)-(?P<condicao>clean|obf)-(?P<versao>v\d)$")
    if partes.isna().any(axis=None):
        raise ValueError("Nome de arquivo fora do padrão esperado.")
    dados = pd.concat([dados.drop(columns="File"), partes], axis=1)

    limpo = dados[dados["condicao"] == "clean"].drop(columns="condicao")
    confuso = dados[dados["condicao"] == "obf"].drop(columns="condicao")
    pares = limpo.merge(
        confuso,
        on=["base", "versao"],
        how="outer",
        suffixes=("_limpo", "_confuso"),
        validate="one_to_one",
    )
    if pares.isna().any(axis=None):
        raise ValueError("Há snippets sem par.")
    return pares


def tabela_rq1(pares):
    linhas = []
    for nome, coluna in METRICAS_RQ1.items():
        resultado = wilcoxon_com_efeito(pares[f"{coluna}_confuso"], pares[f"{coluna}_limpo"])
        artigo = ARTIGO_RQ1[nome]
        linhas.append(
            {
                "metrica": nome,
                "n": resultado.n,
                "W_artigo": artigo["W"],
                "W_recalculado": resultado.w,
                "p_artigo": artigo["p"],
                "p_recalculado": resultado.p,
                "r_artigo": artigo["r"],
                "r_recalculado": resultado.r,
                "confere": resultado.w == artigo["W"]
                and confere_publicado(resultado.p, artigo["p"])
                and confere_publicado(resultado.r, artigo["r"]),
            }
        )
    return pd.DataFrame(linhas)


def tabela_rq2(analise):
    """Correlaciona a perplexidade máxima na AOI com o FRP médio."""
    colunas = {
        "confuso": ("obf_Max AOI Perplexity", "obf_mean_eeg"),
        "limpo": ("clean_Max AOI Perplexity", "clean_mean_eeg"),
    }
    linhas = []
    for nome, (perplexidade, frp) in colunas.items():
        resultado = spearmanr(analise[perplexidade], analise[frp])
        artigo = ARTIGO_RQ2[nome]
        linhas.append(
            {
                "variante": nome,
                "n": len(analise),
                "rho_artigo": artigo["rho"],
                "rho_recalculado": float(resultado.statistic),
                "p_artigo": artigo["p"],
                "p_recalculado": float(resultado.pvalue),
                "confere": confere_publicado(resultado.statistic, artigo["rho"])
                and confere_publicado(resultado.pvalue, artigo["p"]),
            }
        )
    return pd.DataFrame(linhas)


def confere_publicado(valor, publicado):
    """Compara `valor` com o texto publicado, na mesma precisão."""
    if publicado.startswith("<"):
        return bool(valor < float(publicado[1:]))
    casas = len(publicado.split(".")[1])
    return bool(round(float(valor), casas) == round(float(publicado), casas))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pacote", type=Path, default=PACOTE)
    parser.add_argument("--saida", type=Path, default=SAIDA)
    parser.add_argument("--pasta", type=Path, default=None, help="caminho exato da pasta de saída (sem data)")
    args = parser.parse_args(argv)

    entrada = args.pacote / "model_results" / MODELO / "pairwise_analysis" / "pairwise_comparisons"
    pares = carregar_pares(entrada / "perplexity_stats.csv")
    analise = pd.read_csv(entrada / "eeg_perplexity_analysis.csv")

    rq1 = tabela_rq1(pares)
    rq2 = tabela_rq2(analise)

    # O registro vem antes da pasta de saída, que contaria como alteração.
    registro = {
        **proveniencia(),
        "config": {
            "pacote_commit": commit_externo(args.pacote),
            "fonte": str(entrada),
        },
        "seed": None,
        "model": f"{MODELO} (resultados publicados, sem reexecução)",
        "model_revision": None,
        "date": date.today().isoformat(),
    }

    final = args.pasta or args.saida / f"{date.today().isoformat()}-rq1-rq2"
    with pasta_nova(final) as destino:
        rq1.to_csv(destino / "rq1.csv", index=False)
        rq2.to_csv(destino / "rq2.csv", index=False)
        escrever_json(destino / "run.json", registro)

    print(rq1.to_string(index=False))
    print(rq2.to_string(index=False))
    print(f"Resultados em {final}")
    return final


if __name__ == "__main__":
    main()
