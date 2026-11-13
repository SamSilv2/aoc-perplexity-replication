"""Compara duas execuções do mesmo modelo em precisões diferentes, nos arquivos da menor."""

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from aoc_perplexity.aoi import mapear
from aoc_perplexity.gopstein import com_codigo, mapear_por_anotacao


def aoi_max(resultado):
    """Max AOI Perplexity de um arquivo, ou NaN sem AOI."""
    lp = [resultado["results"][i]["log_prob"] for spans in resultado["aoi_regions"].values() for a, b in spans for i in range(a, b + 1)]
    return math.exp(-min(lp)) if lp else math.nan


def carregar(pasta, conjunto):
    arquivo = "perplexidade.json" if conjunto == "bergum" else f"perplexidade-{conjunto}.json"
    return json.loads((pasta / arquivo).read_text())


def comparar(referencia, nova, conjunto, tokenizador):
    """Uma linha por arquivo de `nova`: posição 0, diferença nas demais posições e AOI máxima nas duas."""
    referencia = {a: referencia[a] for a in nova}
    for resultados in (referencia, nova):
        if conjunto == "gopstein":
            com_codigo(resultados, tokenizador)
            mapear_por_anotacao(resultados, tokenizador)
        else:
            mapear(resultados, tokenizador)
    linhas = []
    for arquivo, r in nova.items():
        a, b = referencia[arquivo]["results"], r["results"]
        if len(a) != len(b):
            raise ValueError(f"{arquivo}: {len(a)} posições contra {len(b)}")
        diferencas = [abs(x["log_prob"] - y["log_prob"]) for x, y in zip(a[1:], b[1:])]
        linhas.append(
            {
                "conjunto": conjunto,
                "arquivo": arquivo,
                "posicoes": len(a),
                "logprob_pos0_referencia": a[0]["log_prob"],
                "logprob_pos0_nova": b[0]["log_prob"],
                "dif_media_pos_maior_0": sum(diferencas) / len(diferencas),
                "dif_maxima_pos_maior_0": max(diferencas),
                "aoi_max_referencia": aoi_max(referencia[arquivo]),
                "aoi_max_nova": aoi_max(r),
                "aoi_inclui_pos0": any(a_ == 0 for spans in r["aoi_regions"].values() for a_, _ in spans),
            }
        )
    return pd.DataFrame(linhas)


def resumir(tabela):
    """Uma linha por conjunto: médias da posição 0, diferença nas demais posições e variação da AOI máxima (em ln)."""
    variacao = (tabela["aoi_max_nova"] / tabela["aoi_max_referencia"]).apply(math.log).abs()
    agrupado = tabela.assign(variacao_aoi=variacao).groupby("conjunto")
    return agrupado.agg(
        arquivos=("arquivo", "count"),
        pos0_referencia=("logprob_pos0_referencia", "mean"),
        pos0_nova=("logprob_pos0_nova", "mean"),
        dif_media=("dif_media_pos_maior_0", "mean"),
        dif_maxima=("dif_maxima_pos_maior_0", "max"),
        aoi_ln_mediana=("variacao_aoi", "median"),
        aoi_ln_maxima=("variacao_aoi", "max"),
        aoi_com_pos0=("aoi_inclui_pos0", "sum"),
    ).reset_index()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--referencia", type=Path, required=True, help="execução de referência (ex.: fp16 salvo)")
    parser.add_argument("--nova", type=Path, required=True, help="execução a comparar (menos arquivos)")
    parser.add_argument("--saida", type=Path, required=True, help="CSV por arquivo")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    from aoc_perplexity.trio import repositorio_tokenizador

    registro = json.loads((args.nova / "run.json").read_text())
    repositorio, revisao = repositorio_tokenizador(registro)
    tokenizador = AutoTokenizer.from_pretrained(repositorio, revision=revisao)
    tabelas = [comparar(carregar(args.referencia, c), carregar(args.nova, c), c, tokenizador) for c in registro["config"]["conjuntos"]]
    tabela = pd.concat(tabelas, ignore_index=True)
    tabela.to_csv(args.saida, index=False)
    print(resumir(tabela).to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
