"""Sensibilidade das conclusões de Gopstein à anotação da AOI (v1, v2 e v2.1) nos trios com C (docs/metodos.md, seção 3).

Sem inferência: lê os `perplexidade-gopstein.json` das execuções dos trios de C do manifesto e refaz a análise do trio com cada anotação.
"""

import argparse
import json
import warnings
from datetime import date
from pathlib import Path

import pandas as pd

from aoc_perplexity.consolidar import SLUGS
from aoc_perplexity.gopstein import ANOTACOES
from aoc_perplexity.resultados import NOVOS, Manifesto, escrever_json, nome_datado, pasta_nova, proveniencia
from aoc_perplexity.trio import PAPEIS, METRICAS, analisar, pares_do_modelo, pares_validos

SAIDA = NOVOS
VERSOES = list(ANOTACOES)
# Ordem das linhas dos CSVs publicados.
ORDEM = ["qwen2.5-3b", "qwen2.5-7b", "qwen2.5-32b-4bit", "deepseek-7b", "gemma-7b", "yi-9b", "llama-2-7b", "llama-2-13b"]


def trios_de_c(manifesto):
    """{slug: pasta do trio de C} do manifesto; as execuções do trio são as mesmas em todas as anotações."""
    trios = {SLUGS[familia]: pasta for (familia, conjunto), pasta in manifesto.trios().items() if conjunto == "gopstein"}
    return {slug: trios[slug] for slug in ORDEM}


TRIOS = trios_de_c(Manifesto.ler())


def conclusao(linha):
    """Sinal da estimativa e se o IC exclui o zero."""
    sinal = "+" if linha.estimativa > 0 else "-"
    return f"{sinal}, IC {'exclui' if linha.ic_inferior > 0 or linha.ic_superior < 0 else 'inclui'} 0"


def classificar(sensibilidade):
    """Uma linha por (trio, quantidade) do aoi_max: robusto se as três anotações dão a mesma conclusão."""
    linhas = []
    parte = sensibilidade[sensibilidade.metrica == "aoi_max"]
    for (trio, quantidade), grupo in parte.groupby(["trio", "quantidade"], sort=False):
        por_versao = {r.anotacao: conclusao(r) for r in grupo.itertuples()}
        linhas.append({"trio": trio, "quantidade": quantidade, **por_versao, "classe": "robusto" if len(set(por_versao.values())) == 1 else "dependente da anotação"})
    return pd.DataFrame(linhas)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifesto", type=Path, default=None, help="de onde ler os trios de C (padrão: results/manifesto.json)")
    parser.add_argument("--saida", type=Path, default=SAIDA)
    parser.add_argument("--pasta", type=Path, default=None, help="caminho exato da pasta de saída (sem data)")
    args = parser.parse_args(argv)
    trios = trios_de_c(Manifesto.ler(args.manifesto)) if args.manifesto else TRIOS

    sensibilidade, tamanhos = [], []
    for trio, pasta in trios.items():
        execucoes = json.loads((Path(pasta) / "run.json").read_text())["config"]["execucoes"]
        for versao in VERSOES:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # o aviso dos pares fora do aoi_max já sai no trio.py
                pares = {papel: pares_do_modelo(Path(execucoes[papel]), "gopstein", "manual", versao)[0] for papel in PAPEIS}
            resumo, _ = analisar(pares)
            n_pares = {nome: int(pares_validos(pares, coluna).sum()) for nome, coluna in METRICAS.items()}
            resumo.insert(0, "anotacao", versao)
            resumo.insert(0, "trio", trio)
            resumo["n_pares"] = resumo.metrica.map(n_pares)
            sensibilidade.append(resumo)
            base = pares["base"][pares_validos(pares, METRICAS["aoi_max"])]
            tamanhos.append(
                {
                    "trio": trio,
                    "anotacao": versao,
                    "n_pares_aoi_max": n_pares["aoi_max"],
                    "mediana_tokens_aoi_confuso": float(base["n_aoi_confuso"].median()),
                    "mediana_tokens_aoi_limpo": float(base["n_aoi_limpo"].median()),
                }
            )
    sensibilidade = pd.concat(sensibilidade, ignore_index=True)

    robustez = classificar(sensibilidade)
    registro = {
        **proveniencia(),
        "config": {
            "trios": {trio: str(pasta) for trio, pasta in trios.items()},
            "anotacoes": {v: str(p) for v, p in ANOTACOES.items()},
            "reamostragens": 10_000,
            "aoi_max": "pares com na_metrica_aoi = nao saem (opção A); snippet_media usa todos",
            "robusto": "mesmo sinal e mesmo status do IC (exclui ou inclui 0) nas três anotações",
        },
        "seed": 42,
        "model": "ver config.trios",
        "model_revision": "nos run.json das análises listadas em config.trios",
        "date": date.today().isoformat(),
    }
    # Análises publicadas não são sobrescritas: uma nova no mesmo dia ganha número.
    final = args.pasta or nome_datado(args.saida, "sensibilidade-aoi-gopstein")
    with pasta_nova(final) as destino:
        sensibilidade.to_csv(destino / "sensibilidade.csv", index=False)
        pd.DataFrame(tamanhos).to_csv(destino / "tamanhos.csv", index=False)
        robustez.to_csv(destino / "robustez.csv", index=False)
        escrever_json(destino / "run.json", registro)
    print(robustez.to_string(index=False))
    print(f"Resultados em {final}")
    return final


if __name__ == "__main__":
    main()
