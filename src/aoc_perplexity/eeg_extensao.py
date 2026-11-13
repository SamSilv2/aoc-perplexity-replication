"""RQ2 da extensão (docs/metodos.md, seção 5): Spearman entre a perplexidade máxima na AOI e o FRP médio, só em Java."""

import argparse
import json
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from aoc_perplexity.consolidar import TRIOS
from aoc_perplexity.resultados import NOVOS, Manifesto, escrever_json, nome_datado, pasta_nova, proveniencia
from aoc_perplexity.rq12 import PACOTE, wilcoxon_com_efeito
from aoc_perplexity.trio import PAPEIS, pares_do_modelo

SAIDA = NOVOS
PUBLICADOS = PACOTE / "model_results"
ANALISE = Path("pairwise_analysis/pairwise_comparisons/eeg_perplexity_analysis.csv")
COLUNA = "Max AOI Perplexity"
SUBCONJUNTOS = {"todos": None, "confusos": "confuso", "limpos": "limpo"}
ALFA = 0.05
REAMOSTRAGENS = 10_000
SEMENTE = 42
CHAVES_EEG = ["pair_id", "clean_file", "obf_file", "clean_mean_eeg", "obf_mean_eeg"]


def analises_publicadas(pacote=PUBLICADOS):
    """eeg_perplexity_analysis.csv dos modelos publicados, por nome da pasta."""
    return {p.name: pd.read_csv(p / ANALISE) for p in sorted(pacote.iterdir()) if (p / ANALISE).is_file()}


def eeg_publicado(analises):
    """FRP médio por par; exige o mesmo EEG (e os mesmos pares) em todos os modelos publicados."""
    nomes = list(analises)
    referencia = analises[nomes[0]][CHAVES_EEG].sort_values("pair_id").reset_index(drop=True)
    for nome in nomes[1:]:
        if not analises[nome][CHAVES_EEG].sort_values("pair_id").reset_index(drop=True).equals(referencia):
            raise ValueError(f"o EEG de {nome} difere do de {nomes[0]}")
    return referencia


def pares_publicados(analise):
    """Pares de um modelo publicado no formato de trio.pares_do_modelo (base, versao, Max AOI Perplexity)."""
    partes = analise["pair_id"].str.extract(r"^(?P<base>\d*)-(?P<versao>v\d)$")
    if partes.isna().any(axis=None):
        raise ValueError("pair_id fora do padrão esperado")
    pares = pd.concat([partes, analise[[f"obf_{COLUNA}", f"clean_{COLUNA}"]].rename(columns={f"obf_{COLUNA}": f"{COLUNA}_confuso", f"clean_{COLUNA}": f"{COLUNA}_limpo"})], axis=1)
    return pares.sort_values(["base", "versao"]).reset_index(drop=True)


def tabela_snippets(pares, eeg):
    """Um snippet por linha (confuso e limpo de cada par), com a perplexidade na AOI e o FRP médio."""
    pares = pares.assign(pair_id=pares["base"].astype(str) + "-" + pares["versao"])
    junto = pares.merge(eeg[["pair_id", "clean_mean_eeg", "obf_mean_eeg"]], on="pair_id", how="left", validate="one_to_one")
    if junto["obf_mean_eeg"].isna().any():
        raise ValueError("há pares sem FRP")
    partes = [
        pd.DataFrame({"base": junto["base"].astype(str), "pair_id": junto["pair_id"], "condicao": condicao, "ppl": junto[f"{COLUNA}_{condicao}"], "frp": junto[frp]})
        for condicao, frp in [("confuso", "obf_mean_eeg"), ("limpo", "clean_mean_eeg")]
    ]
    snippets = pd.concat(partes, ignore_index=True)
    return snippets[np.isfinite(snippets[["ppl", "frp"]].to_numpy(dtype=float)).all(axis=1)].reset_index(drop=True)


def correlacoes(snippets, n=REAMOSTRAGENS, semente=SEMENTE):
    """Spearman (ρ e p) nos subconjuntos todos, confusos e limpos, com IC por bootstrap agrupado por snippet-base.

    Cada reamostragem sorteia os snippets-base com reposição e leva todas as linhas deles; os três
    subconjuntos usam o mesmo sorteio.
    """
    ids = pd.unique(snippets["base"])
    subconjuntos = {}
    for nome, condicao in SUBCONJUNTOS.items():
        dados = snippets if condicao is None else snippets[snippets["condicao"] == condicao]
        grupos = dados["base"].to_numpy()
        subconjuntos[nome] = (dados["ppl"].to_numpy(dtype=float), dados["frp"].to_numpy(dtype=float), {g: np.flatnonzero(grupos == g) for g in ids})

    gerador = np.random.RandomState(semente)
    amostras = {nome: [] for nome in subconjuntos}
    for _ in range(n):
        sorteados = gerador.choice(ids, size=len(ids), replace=True)
        for nome, (x, y, linhas) in subconjuntos.items():
            indices = np.concatenate([linhas[g] for g in sorteados])
            amostras[nome].append(spearmanr(x[indices], y[indices]).statistic)

    linhas = []
    for nome, (x, y, _) in subconjuntos.items():
        resultado = spearmanr(x, y)
        linhas.append(
            {
                "subconjunto": nome,
                "rho": float(resultado.statistic),
                "ic_inferior": float(np.nanpercentile(amostras[nome], 2.5)),
                "ic_superior": float(np.nanpercentile(amostras[nome], 97.5)),
                "p": float(resultado.pvalue),
                "n": len(x),
            }
        )
    return pd.DataFrame(linhas)


def delta_com_ic(pares, n=REAMOSTRAGENS, semente=SEMENTE):
    """Δ = média de ln(confuso) − ln(limpo) no aoi_max e IC por bootstrap dos átomos (o mesmo sorteio de trio.analisar)."""
    validos = np.isfinite(pares[[f"{COLUNA}_confuso", f"{COLUNA}_limpo"]].to_numpy(dtype=float)).all(axis=1)
    pares = pares[validos].reset_index(drop=True)
    razoes = np.log(pares[f"{COLUNA}_confuso"].to_numpy(dtype=float)) - np.log(pares[f"{COLUNA}_limpo"].to_numpy(dtype=float))
    grupos = pares["base"].to_numpy()
    ids = pd.unique(grupos)
    linhas = {g: np.flatnonzero(grupos == g) for g in ids}
    gerador = np.random.RandomState(semente)
    amostras = [float(np.mean(razoes[np.concatenate([linhas[g] for g in gerador.choice(ids, size=len(ids), replace=True)])])) for _ in range(n)]
    w = wilcoxon_com_efeito(pares[f"{COLUNA}_confuso"], pares[f"{COLUNA}_limpo"])
    return {
        "delta": float(np.mean(razoes)),
        "ic_inferior": float(np.percentile(amostras, 2.5)),
        "ic_superior": float(np.percentile(amostras, 97.5)),
        "W": w.w,
        "p_wilcoxon": w.p,
        "r": w.r,
        "n_wilcoxon": w.n,
        "n_pares": len(pares),
    }


def concordancia(delta):
    """Wilcoxon (p < 0,05) e bootstrap agrupado (IC exclui zero) dizem o mesmo sobre o Δ?"""
    wilcoxon = delta["p_wilcoxon"] < ALFA
    bootstrap = delta["ic_inferior"] > 0 or delta["ic_superior"] < 0
    return {"wilcoxon_rejeita": wilcoxon, "ic_exclui_zero": bootstrap, "concordam": wilcoxon == bootstrap}


def modelos_da_extensao(trios=TRIOS):
    """(trio, papel, modelo, pasta da execução, pasta do trio) dos trios de Java, na ordem do manifesto."""
    for (trio, conjunto), pasta in trios.items():
        if conjunto != "bergum":
            continue
        registro = json.loads((Path(pasta) / "run.json").read_text())
        for papel in PAPEIS:
            yield trio, papel, registro["model"][papel], Path(registro["config"]["execucoes"][papel]), pasta


ROTULO_PAPEL = {"base": "base", "codigo": "code", "instruct": "instruct"}
ROTULO_SUBCONJUNTO = {"todos": "All (144)", "confusos": "Confusing (72)", "limpos": "Clean (72)"}


def celula_rho(linha):
    """ρ [IC] com duas casas; negrito quando o intervalo exclui zero."""
    texto = f"{linha.rho:.2f} [{linha.ic_inferior:.2f}, {linha.ic_superior:.2f}]".replace("-", "$-$")
    return rf"\textbf{{{texto}}}" if linha.ic_inferior > 0 or linha.ic_superior < 0 else texto


def tabela_correlacoes(correlacao):
    """Tabela LaTeX de ρ [IC 95%] por modelo e subconjunto: os modelos publicados e, por trio, base, code e instruct."""
    largo = correlacao.set_index(["modelo", "subconjunto"])
    colunas = list(SUBCONJUNTOS)

    def linha(rotulo, modelo):
        return " & ".join([rotulo] + [celula_rho(largo.loc[(modelo, c)]) for c in colunas]) + r" \\"

    corpo = [r"\multicolumn{4}{l}{\textit{Published models (Abdelsalam et al.)}} \\"]
    publicados = correlacao[correlacao["papel"] == "publicado"]["modelo"].drop_duplicates()
    corpo += [linha(rf"\texttt{{{m}}}", m) for m in publicados]
    extensao = correlacao[correlacao["papel"] != "publicado"]
    for trio in [t for t, c in TRIOS if c == "bergum"]:
        corpo += [r"\midrule", rf"\multicolumn{{4}}{{l}}{{\textit{{{trio}}}}} \\"]
        for papel in PAPEIS:
            modelo = extensao[(extensao["trio"] == trio) & (extensao["papel"] == papel)]["modelo"].iloc[0]
            corpo.append(linha(rf"\quad {ROTULO_PAPEL[papel]}", modelo))
    return "\n".join([
        r"\begin{tabular}{llll}",
        r"\toprule",
        " & ".join(["Model"] + [ROTULO_SUBCONJUNTO[c] for c in colunas]) + r" \\",
        r"\midrule",
        *corpo,
        r"\bottomrule",
        r"\end{tabular}",
    ]) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifesto", type=Path, default=None, help="de onde ler os trios (padrão: results/manifesto.json)")
    parser.add_argument("--saida", type=Path, default=SAIDA)
    parser.add_argument("--pasta", type=Path, default=None, help="caminho exato da pasta de saída (sem data)")
    parser.add_argument("--reamostragens", type=int, default=REAMOSTRAGENS)
    args = parser.parse_args(argv)
    trios = Manifesto.ler(args.manifesto).trios() if args.manifesto else TRIOS

    analises = analises_publicadas()
    eeg = eeg_publicado(analises)

    modelos = [("publicado", "", nome, pares_publicados(analise), None) for nome, analise in analises.items()]
    trios_usados = {}
    for trio, papel, modelo, execucao, pasta in modelos_da_extensao(trios):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # o aviso dos pares fora do aoi_max já sai no trio.py
            pares, _ = pares_do_modelo(execucao, "bergum", "manual")
        modelos.append((papel, trio, modelo, pares, execucao))
        trios_usados[trio] = str(pasta)

    correlacao, concordancias = [], []
    for papel, trio, modelo, pares, _ in modelos:
        identificacao = {"modelo": modelo, "papel": papel, "trio": trio}
        resultado = correlacoes(tabela_snippets(pares, eeg), n=args.reamostragens)
        correlacao.append(pd.concat([pd.DataFrame([identificacao] * len(resultado)), resultado], axis=1))
        delta = delta_com_ic(pares, n=args.reamostragens)
        concordancias.append({**identificacao, **delta, **concordancia(delta)})
    correlacao = pd.concat(correlacao, ignore_index=True)
    concordancias = pd.DataFrame(concordancias)

    # Os Δ e ICs dos nossos 24 modelos têm de ser os de trio.csv (mesmos pares, mesma semente, mesmo sorteio).
    for trio, pasta in trios_usados.items():
        oficial = pd.read_csv(Path(pasta) / "trio.csv").query("metrica == 'aoi_max'").set_index("quantidade")
        for papel in PAPEIS:
            linha = concordancias[(concordancias["trio"] == trio) & (concordancias["papel"] == papel)].iloc[0]
            esperado = oficial.loc[f"Delta_{papel}"]
            for coluna, nome in [("delta", "estimativa"), ("ic_inferior", "ic_inferior"), ("ic_superior", "ic_superior")]:
                if abs(linha[coluna] - esperado[nome]) > 1e-9:
                    raise ValueError(f"{trio} {papel}: {coluna} difere de {pasta}/trio.csv")

    registro = {
        **proveniencia(),
        "config": {
            "metodo": "docs/metodos.md, seção 5",
            "reamostragens": args.reamostragens,
            "ic": "percentis 2,5 e 97,5; bootstrap agrupado por snippet-base (24 grupos)",
            "metrica_perplexidade": COLUNA,
            "frp": "clean_mean_eeg / obf_mean_eeg do pacote publicado (idêntico nos modelos publicados)",
            "subconjuntos": {"todos": "144 snippets", "confusos": "72", "limpos": "72"},
            "trios": trios_usados,
            "publicados": sorted(analises),
            "alfa_wilcoxon": ALFA,
        },
        "seed": SEMENTE,
        "date": date.today().isoformat(),
    }
    final = args.pasta or nome_datado(args.saida, "eeg-java", numerar=False)
    with pasta_nova(final) as destino:
        correlacao.to_csv(destino / "correlacoes.csv", index=False)
        concordancias.to_csv(destino / "concordancia-wilcoxon-bootstrap.csv", index=False)
        (destino / "tabela-correlacoes.tex").write_text(tabela_correlacoes(correlacao))
        escrever_json(destino / "run.json", registro)
    print(correlacao.round(3).to_string(index=False))
    print(f"Resultados em {final}")
    return final


if __name__ == "__main__":
    main()
