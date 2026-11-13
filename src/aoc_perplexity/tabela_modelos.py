"""Tabela de perplexidade por modelo, a partir dos arquivos publicados e da nossa reexecução."""

import argparse
import json
import tempfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from aoc_perplexity.rq12 import PACOTE, SAIDA, carregar_pares, wilcoxon_com_efeito
from aoc_perplexity.resultados import commit_externo, escrever_json, pasta_nova, proveniencia

REEXECUCAO = Path("results/replicacao") / "2026-11-02-reexecucao-deepseek-coder-1.3b-base-completo" / "perplexidade.json"
COLUNAS = {"ppl": "Overall Perplexity", "max_aoi": "Max AOI Perplexity"}


def linha(modelo, fonte, pares):
    resultado = {"modelo": modelo, "fonte": fonte, "pares": len(pares)}
    for nome, coluna in COLUNAS.items():
        if f"{coluna}_limpo" not in pares:
            continue
        limpo, confuso = pares[f"{coluna}_limpo"], pares[f"{coluna}_confuso"]
        # Mediana, porque a perplexidade é muito assimétrica.
        resultado[f"{nome}_limpo"] = float(np.median(limpo))
        resultado[f"{nome}_confuso"] = float(np.median(confuso))
        resultado[f"{nome}_r"] = wilcoxon_com_efeito(confuso, limpo).r
    return resultado


def pares_da_reexecucao(caminho):
    dados = json.loads(caminho.read_text())
    stats = pd.DataFrame({"File": list(dados), "Overall Perplexity": [d["summary"]["overall_perplexity"] for d in dados.values()]})
    # O CSV temporário fica fora de results/, para não aparecer ao lado de resultados publicados.
    with tempfile.TemporaryDirectory() as pasta:
        temporario = Path(pasta) / "stats_temporario.csv"
        stats.to_csv(temporario, index=False)
        return carregar_pares(temporario)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pacote", type=Path, default=PACOTE)
    parser.add_argument("--saida", type=Path, default=SAIDA)
    parser.add_argument("--pasta", type=Path, default=None, help="caminho exato da pasta de saída (sem data)")
    parser.add_argument("--reexecucao", type=Path, default=REEXECUCAO.parent, help="pasta da reexecução do deepseek-coder-1.3b-base (padrão: a publicada)")
    args = parser.parse_args(argv)

    linhas = []
    for pasta in sorted((args.pacote / "model_results").iterdir()):
        stats = pasta / "pairwise_analysis" / "pairwise_comparisons" / "perplexity_stats.csv"
        if stats.exists():
            linhas.append(linha(pasta.name, "publicado", carregar_pares(stats)))
    reexecucao = args.reexecucao / "perplexidade.json"
    if not reexecucao.exists():
        parser.error(f"{reexecucao} não existe")
    linhas.append(linha("deepseek-coder-1.3b-base", "reexecução (Mac, fp16)", pares_da_reexecucao(reexecucao)))
    tabela = pd.DataFrame(linhas)

    registro = {
        **proveniencia(),
        "config": {"pacote_commit": commit_externo(args.pacote), "reexecucao": str(args.reexecucao), "estatistica": "mediana por condição; r do Wilcoxon pareado"},
        "seed": None,
        "model": "6 modelos publicados e reexecução do 1.3B",
        "model_revision": None,
        "date": date.today().isoformat(),
    }
    final = args.pasta or args.saida / f"{date.today().isoformat()}-tabela-por-modelo"
    with pasta_nova(final) as destino:
        tabela.to_csv(destino / "tabela.csv", index=False)
        escrever_json(destino / "run.json", registro)
    print(tabela.round(3).to_string(index=False))
    print(f"Resultados em {final}")
    return final


if __name__ == "__main__":
    main()
