"""Recalcula RQ3 do artigo a partir dos arquivos publicados.

Segue main/rq3.1_perplexity_algorithm.ipynb e paper_plots/rq3_plots.ipynb do pacote.
As regiões detectadas vêm do JSON publicado; o algoritmo de detecção não é reexecutado.
"""

import argparse
import json
import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from aoc_perplexity.rq12 import MODELO, PACOTE, SAIDA, confere_publicado
from aoc_perplexity.resultados import commit_externo, escrever_json, pasta_nova, proveniencia

# Figura 4 do artigo: snippets com e sem sobreposição, e regiões detectadas.
ARTIGO_FIG4 = {
    "atomos_confusos_detectados": 46,
    "atomos_confusos_nao_detectados": 26,
    "atomos_limpos_detectados": 27,
    "atomos_limpos_nao_detectados": 45,
    "regioes_em_confusos": 115,
    "regioes_em_limpos": 112,
}

# Figura 5 e Tabela 3: rho e IC de 95%.
ARTIGO_RHO = {
    "confuso": ("0.37", "0.19", "0.53"),
    "limpo": ("0.31", "0.10", "0.50"),
    "sobrepoe_limpo": ("0.38", "0.10", "0.70"),
    "nao_sobrepoe_limpo": ("0.24", "-0.05", "0.49"),
    "sobrepoe_confuso": ("0.33", "0.12", "0.52"),
    "nao_sobrepoe_confuso": ("0.36", "-0.05", "0.71"),
}


def subconjunto(arquivo):
    return "limpo" if "clean" in arquivo.lower() else "confuso"


def sobreposicao(resultados):
    """Um snippet sobrepõe se alguma região detectada divide ao menos um token com a AOI manual."""
    listas = {(s, c): [] for s in ["limpo", "confuso"] for c in [True, False]}
    regioes = {"limpo": 0, "confuso": 0}
    for arquivo, r in resultados.items():
        manuais = [span for spans in r["aoi_regions"].values() for span in spans]
        detectadas = [span for spans in r["aoi_regions_algorithm"].values() for span in spans]
        sobrepoe = any(a <= d_fim and d_ini <= b for a, b in manuais for d_ini, d_fim in detectadas)
        listas[(subconjunto(arquivo), sobrepoe)].append(arquivo)
        regioes[subconjunto(arquivo)] += len(r.get("aoi_labels", {}))
    return listas, regioes


def regioes_novas(resultados):
    """Regiões detectadas que não dividem token com a AOI manual, contadas por região."""
    novas = {"limpo": 0, "confuso": 0}
    for arquivo, r in resultados.items():
        manuais = [span for spans in r["aoi_regions"].values() for span in spans]
        for d_ini, d_fim in [span for spans in r["aoi_regions_algorithm"].values() for span in spans]:
            if not any(a <= d_fim and d_ini <= b for a, b in manuais):
                novas[subconjunto(arquivo)] += 1
    return novas


def mascara_numerada(df, bases):
    # Igual aos autores: arquivo começa com o snippet e termina com o número da região.
    padrao = "|".join(rf"^{re.escape(base)}-(?:[1-9]|10)" for base in bases)
    return df["file"].str.contains(padrao, regex=True)


def filtrar(df, mascara):
    x = pd.to_numeric(df.loc[mascara, "Max AOI Perplexity"], errors="coerce")
    y = pd.to_numeric(df.loc[mascara, "mean_eeg"], errors="coerce")
    return df.loc[mascara][(x > 0) & y.notna()]


def bootstrap_agrupado(df, n=10_000, semente=42):
    """Reamostra snippets com reposição, como clustered_bootstrap_spearman dos autores."""
    gerador = np.random.RandomState(semente)  # mesma sequência de np.random.seed + np.random.choice
    grupos = df["snippet_id"].to_numpy()
    ids = pd.unique(grupos)
    linhas = {g: np.flatnonzero(grupos == g) for g in ids}
    x = df["Max AOI Perplexity"].to_numpy(float)
    y = df["mean_eeg"].to_numpy(float)
    rhos = []
    for _ in range(n):
        indices = np.concatenate([linhas[g] for g in gerador.choice(ids, size=len(ids), replace=True)])
        if len(indices) < 3:
            continue
        rhos.append(spearmanr(x[indices], y[indices]).statistic)
    rhos = np.array(rhos)
    return float(np.mean(rhos)), float(np.percentile(rhos, 2.5)), float(np.percentile(rhos, 97.5))


def grupos_rq3(analise, listas):
    sobrepoe_limpo = filtrar(analise, mascara_numerada(analise, listas[("limpo", True)]))
    nao_sobrepoe_limpo = filtrar(analise, mascara_numerada(analise, listas[("limpo", False)]))
    sobrepoe_confuso = filtrar(analise, mascara_numerada(analise, listas[("confuso", True)]))
    nao_sobrepoe_confuso = filtrar(analise, mascara_numerada(analise, listas[("confuso", False)]))
    # Na Figura 5, os autores juntam as duas partes nesta ordem.
    return {
        "confuso": pd.concat([sobrepoe_confuso, nao_sobrepoe_confuso]),
        "limpo": pd.concat([sobrepoe_limpo, nao_sobrepoe_limpo]),
        "sobrepoe_limpo": sobrepoe_limpo,
        "nao_sobrepoe_limpo": nao_sobrepoe_limpo,
        "sobrepoe_confuso": sobrepoe_confuso,
        "nao_sobrepoe_confuso": nao_sobrepoe_confuso,
    }


def tabela_correlacoes(grupos):
    linhas = []
    for nome, df in grupos.items():
        rho, inferior, superior = bootstrap_agrupado(df)
        pontual = spearmanr(df["Max AOI Perplexity"], df["mean_eeg"])
        publicado = ARTIGO_RHO[nome]
        linhas.append(
            {
                "grupo": nome,
                "regioes": len(df),
                "snippets": df["snippet_id"].nunique(),
                "rho_bootstrap": rho,
                "ic_inferior": inferior,
                "ic_superior": superior,
                "rho_pontual": pontual.statistic,
                "p_pontual": pontual.pvalue,
                "rho_artigo": publicado[0],
                "ic_artigo": f"[{publicado[1]}, {publicado[2]}]",
                "confere": all(confere_publicado(v, p) for v, p in zip([rho, inferior, superior], publicado)),
            }
        )
    return pd.DataFrame(linhas)


def tabela_fig4(listas, regioes):
    recalculado = {
        "atomos_confusos_detectados": len(listas[("confuso", True)]),
        "atomos_confusos_nao_detectados": len(listas[("confuso", False)]),
        "atomos_limpos_detectados": len(listas[("limpo", True)]),
        "atomos_limpos_nao_detectados": len(listas[("limpo", False)]),
        "regioes_em_confusos": regioes["confuso"],
        "regioes_em_limpos": regioes["limpo"],
    }
    return pd.DataFrame(
        [{"contagem": k, "recalculado": v, "artigo": ARTIGO_FIG4[k], "confere": v == ARTIGO_FIG4[k]} for k, v in recalculado.items()]
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pacote", type=Path, default=PACOTE)
    parser.add_argument("--saida", type=Path, default=SAIDA)
    parser.add_argument("--pasta", type=Path, default=None, help="caminho exato da pasta de saída (sem data)")
    args = parser.parse_args(argv)

    base = args.pacote / "model_results" / MODELO
    resultados = json.loads((base / "perplexity_results_with_aois.json").read_text())
    publicadas = json.loads((base / "data_driven_aois_comparison" / "aoi_overlap_snippet_lists.json").read_text())
    analise = pd.read_csv(base / "individual_analysis" / "eeg_perplexity_analysis_individual.csv")
    analise["snippet_id"] = analise["file"].str.extract(r"(.*)-\d+$")[0]

    listas, regioes = sobreposicao(resultados)
    chaves = {"limpo": "clean", "confuso": "obf"}
    listas_iguais = all(
        sorted(listas[(s, c)]) == sorted(publicadas["overlap" if c else "no_overlap"][chaves[s]]) for s, c in listas
    )

    fig4 = tabela_fig4(listas, regioes)
    novas = regioes_novas(resultados)
    correlacoes = tabela_correlacoes(grupos_rq3(analise, listas))

    # O registro vem antes da pasta de saída, que contaria como alteração.
    registro = {
        **proveniencia(),
        "config": {
            "pacote_commit": commit_externo(args.pacote),
            "fonte": str(base),
            "reamostragens": 10_000,
            "listas_de_sobreposicao_iguais_as_publicadas": listas_iguais,
            "regioes_novas_por_regiao": novas,
        },
        "seed": 42,
        "model": f"{MODELO} (resultados publicados, sem reexecução)",
        "model_revision": None,
        "date": date.today().isoformat(),
    }

    final = args.pasta or args.saida / f"{date.today().isoformat()}-rq3"
    with pasta_nova(final) as destino:
        fig4.to_csv(destino / "fig4.csv", index=False)
        correlacoes.to_csv(destino / "correlacoes.csv", index=False)
        escrever_json(destino / "run.json", registro)

    print(fig4.to_string(index=False))
    print(f"Listas de sobreposição iguais às publicadas: {listas_iguais}")
    print(f"Regiões sem sobreposição, por região: {novas}")
    print(correlacoes.to_string(index=False))
    print(f"Resultados em {final}")
    return final


if __name__ == "__main__":
    main()
