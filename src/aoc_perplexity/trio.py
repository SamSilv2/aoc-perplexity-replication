"""Analisa um trio de modelos: Δ por modelo e D, com IC por bootstrap dos pares-base (docs/metodos.md, seções 3 e 4)."""

import argparse
import hashlib
import json
import tempfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from aoc_perplexity.aoi import mapear, resumo
from aoc_perplexity.gopstein import ANOTACAO_PADRAO, ANOTACOES, com_codigo, mapear_por_anotacao, mapear_por_diff
from aoc_perplexity.resultados import NOVOS, escrever_json, pasta_nova, proveniencia
from aoc_perplexity.rq12 import carregar_pares, wilcoxon_com_efeito

SAIDA = NOVOS
METRICAS = {"aoi_max": "Max AOI Perplexity", "snippet_media": "Overall Perplexity"}
PAPEIS = ["base", "codigo", "instruct"]


def repositorio_tokenizador(registro, revisao=None):
    """Tokenizador da execução: o do modelo, salvo quando o run.json registra outro (docs/metodos.md, seção 2).

    `revisao` troca só a revisão do tokenizador; sem ela, vale a registrada no run.json.
    """
    config = registro.get("config", {})
    return config.get("tokenizador", registro["model"]), revisao or config.get("tokenizador_revisao", registro["model_revision"])


def revisoes_por_papel(valores, opcao):
    """Lê PAPEL=SHA (repetível) em {papel: sha}."""
    revisoes = {}
    for valor in valores or []:
        papel, separador, sha = valor.partition("=")
        if not separador or papel not in PAPEIS or not sha:
            raise ValueError(f"{opcao} espera PAPEL=SHA com PAPEL em {PAPEIS}: {valor!r}")
        revisoes[papel] = sha
    return revisoes


def conferir_revisao(registro, revisao):
    """O trio não carrega o modelo: --revision só confere que a execução usou essa revisão."""
    if revisao is not None and registro["model_revision"] != revisao:
        raise ValueError(f"{registro['model']}: a execução usou a revisão {registro['model_revision']}, não {revisao}")


def montar_pares(resultados, tokenizador, conjunto="bergum", aoi="manual", anotacao=ANOTACAO_PADRAO):
    """Mapeia as AOIs de `resultados` (já com `code`) e junta os pares confuso e limpo.

    Um par com `fora_do_aoi_max` (na_metrica_aoi = nao na anotação) fica com as colunas de AOI em NaN
    e continua com os valores do snippet inteiro; `analisar` o tira só do aoi_max.
    """
    if aoi == "diff":
        resultados = mapear_por_diff(resultados, tokenizador)
    elif conjunto == "gopstein":
        resultados = mapear_por_anotacao(resultados, tokenizador, anotacao=ANOTACOES[anotacao])
    else:
        resultados = mapear(resultados, tokenizador)
    linhas = []
    for arquivo, r in resultados.items():
        linha = {"File": arquivo, **resumo(r), "n_aoi": sum(b - a + 1 for a, b in (s for spans in r["aoi_regions"].values() for s in spans))}
        if r.get("fora_do_aoi_max"):
            # carregar_pares recusa NaN: o valor do snippet entra como espaço reservado e é trocado por NaN abaixo.
            linha["Overall AOI Perplexity"], linha["Max AOI Perplexity"] = linha["Overall Perplexity"], linha["Max Perplexity"]
        linhas.append(linha)
    stats = pd.DataFrame(linhas)
    temporario = Path(tempfile.mkdtemp()) / "stats_temporario.csv"
    stats.to_csv(temporario, index=False)
    try:
        pares = carregar_pares(temporario)
    finally:
        temporario.unlink()
        temporario.parent.rmdir()
    fora = {a.replace("-obf-", "-").replace("-clean-", "-") for a, r in resultados.items() if r.get("fora_do_aoi_max")}
    marcados = (pares["base"] + "-" + pares["versao"]).isin(fora)
    for coluna in [c for c in pares.columns if "AOI" in c]:
        pares.loc[marcados, coluna] = np.nan
    return pares.sort_values(["base", "versao"]).reset_index(drop=True)


def pares_do_modelo(pasta, conjunto="bergum", aoi="manual", anotacao=ANOTACAO_PADRAO, revisao=None, revisao_tokenizador=None):
    """Lê uma execução de perplexidade.py, mapeia as AOIs e devolve os pares.

    aoi="manual" usa as AOIs anotadas (Bergum, ou a anotação de Gopstein escolhida: v1, v2 ou v2.1);
    aoi="diff" usa os tokens que mudam entre as versões.
    """
    from transformers import AutoTokenizer

    registro = json.loads((pasta / "run.json").read_text())
    conferir_revisao(registro, revisao)
    repositorio, revisao_usada = repositorio_tokenizador(registro, revisao_tokenizador)
    tokenizador = AutoTokenizer.from_pretrained(repositorio, revision=revisao_usada)
    arquivo = "perplexidade.json" if conjunto == "bergum" else f"perplexidade-{conjunto}.json"
    resultados = json.loads((pasta / arquivo).read_text())
    if conjunto == "gopstein":
        resultados = com_codigo(resultados, tokenizador)
    return montar_pares(resultados, tokenizador, conjunto, aoi, anotacao), registro


def filtrar_atomos(pares_por_papel, atomos):
    """Mantém só os pares dos átomos (pares-base) listados, nos três modelos; sem lista, devolve tudo como está."""
    if not atomos:
        return pares_por_papel
    atomos = {str(a) for a in atomos}
    faltando = atomos - set(pares_por_papel["base"]["base"].astype(str))
    if faltando:
        raise ValueError(f"átomos inexistentes: {sorted(faltando)}")
    return {papel: pares[pares["base"].astype(str).isin(atomos)].reset_index(drop=True) for papel, pares in pares_por_papel.items()}


def pares_validos(pares_por_papel, coluna):
    """Linhas (iguais nos três modelos) com a métrica definida; as demais saíram da métrica (AOI fora do aoi_max)."""
    mascaras = {papel: np.isfinite(pares[[f"{coluna}_confuso", f"{coluna}_limpo"]].to_numpy(dtype=float)).all(axis=1) for papel, pares in pares_por_papel.items()}
    base = mascaras["base"]
    if any((m != base).any() for m in mascaras.values()):
        raise ValueError(f"os três modelos não têm os mesmos pares válidos em {coluna}")
    return base


def log_razao(pares, coluna):
    return np.log(pares[f"{coluna}_confuso"].to_numpy()) - np.log(pares[f"{coluna}_limpo"].to_numpy())


def contrastes(deltas):
    return {
        "D": deltas["instruct"] - deltas["base"],
        "D_codigo": deltas["codigo"] - deltas["base"],
        "D_instrucao": deltas["instruct"] - deltas["codigo"],
    }


def analisar(pares_por_papel, n=10_000, semente=42):
    chaves = pares_por_papel["base"][["base", "versao"]]
    for pares in pares_por_papel.values():
        if not pares[["base", "versao"]].equals(chaves):
            raise ValueError("os três modelos não têm os mesmos pares")

    resumo_trio, por_modelo = [], []
    for nome, coluna in METRICAS.items():
        # Pares com a métrica indefinida (AOI fora do aoi_max) ficam de fora só desta métrica; o bootstrap reamostra os átomos que sobram.
        validos = pares_validos(pares_por_papel, coluna)
        pares_metrica = {papel: pares[validos].reset_index(drop=True) for papel, pares in pares_por_papel.items()}
        grupos = pares_metrica["base"]["base"].to_numpy()
        ids = pd.unique(grupos)
        linhas_por_id = {g: np.flatnonzero(grupos == g) for g in ids}

        razoes = {papel: log_razao(pares, coluna) for papel, pares in pares_metrica.items()}
        deltas = {papel: float(np.mean(l)) for papel, l in razoes.items()}

        gerador = np.random.RandomState(semente)
        amostras = {k: [] for k in [*PAPEIS, "D", "D_codigo", "D_instrucao"]}
        for _ in range(n):
            indices = np.concatenate([linhas_por_id[g] for g in gerador.choice(ids, size=len(ids), replace=True)])
            d = {papel: float(np.mean(l[indices])) for papel, l in razoes.items()}
            for k, v in {**d, **contrastes(d)}.items():
                amostras[k].append(v)

        for k, estimativa in {**deltas, **contrastes(deltas)}.items():
            resumo_trio.append(
                {
                    "metrica": nome,
                    "quantidade": k if k.startswith("D") else f"Delta_{k}",
                    "estimativa": estimativa,
                    "ic_inferior": float(np.percentile(amostras[k], 2.5)),
                    "ic_superior": float(np.percentile(amostras[k], 97.5)),
                }
            )
        for papel, pares in pares_metrica.items():
            w = wilcoxon_com_efeito(pares[f"{coluna}_confuso"], pares[f"{coluna}_limpo"])
            por_modelo.append(
                {
                    "metrica": nome,
                    "papel": papel,
                    "mediana_limpo": float(pares[f"{coluna}_limpo"].median()),
                    "mediana_confuso": float(pares[f"{coluna}_confuso"].median()),
                    "delta": deltas[papel],
                    "W": w.w,
                    "p": w.p,
                    "r": w.r,
                    "n_wilcoxon": w.n,  # pares com diferença não nula; Δ e bootstrap usam todos
                }
            )
    return pd.DataFrame(resumo_trio), pd.DataFrame(por_modelo)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nome", required=True, help="por exemplo qwen2.5-7b")
    for papel in PAPEIS:
        parser.add_argument(f"--{papel}", type=Path, required=True, help="pasta de uma execução de perplexidade.py")
    parser.add_argument("--conjunto", default="bergum", choices=["bergum", "gopstein"])
    parser.add_argument("--aoi", default=None, choices=["manual", "diff"], help="manual: anotação de Bergum ou de Gopstein (--anotacao)")
    parser.add_argument(
        "--anotacao", default=ANOTACAO_PADRAO, choices=list(ANOTACOES), help="anotação de AOI de Gopstein (v2.1 é a principal; v1 e v2 são sensibilidade)"
    )
    parser.add_argument("--atomos", default=None, help="sensibilidade: lista de átomos (pares-base) a manter, separados por vírgula; sem ela, todos")
    parser.add_argument("--rotulo-atomos", default=None, help="nome do subconjunto de átomos, usado na pasta (obrigatório com --atomos)")
    parser.add_argument("--sufixo", default=None, help="sufixo do nome da pasta, para análises que trocam uma execução (por exemplo template, docs/metodos.md, seção 6)")
    parser.add_argument("--revision", action="append", metavar="PAPEL=SHA", help="confere a revisão do modelo registrada na execução (repetível); sem ela, aceita a do run.json")
    parser.add_argument("--tokenizer-revision", action="append", metavar="PAPEL=SHA", help="revisão do tokenizador (repetível); sem ela, a do run.json da execução")
    parser.add_argument("--saida", type=Path, default=SAIDA)
    parser.add_argument("--pasta", type=Path, default=None, help="caminho exato da pasta de saída (sem data); tem precedência sobre --saida")
    args = parser.parse_args(argv)
    try:
        revisoes, revisoes_tokenizador = revisoes_por_papel(args.revision, "--revision"), revisoes_por_papel(args.tokenizer_revision, "--tokenizer-revision")
    except ValueError as erro:
        parser.error(str(erro))
    atomos = [a.strip() for a in args.atomos.split(",")] if args.atomos else None
    if atomos and not args.rotulo_atomos:
        parser.error("--atomos exige --rotulo-atomos")
    aoi = args.aoi or "manual"

    pares, registros = {}, {}
    for papel in PAPEIS:
        pares[papel], registros[papel] = pares_do_modelo(
            getattr(args, papel), args.conjunto, aoi, args.anotacao, revisoes.get(papel), revisoes_tokenizador.get(papel)
        )
    pares = filtrar_atomos(pares, atomos)
    trio, por_modelo = analisar(pares)
    n_pares = {nome: int(pares_validos(pares, coluna).sum()) for nome, coluna in METRICAS.items()}
    # A anotação só vale para Gopstein com AOI manual; os nomes de pasta v1 seguem o padrão antigo.
    anotacao = None
    if args.conjunto == "gopstein" and aoi == "manual":
        arquivo = ANOTACOES[args.anotacao]
        anotacao = {"versao": args.anotacao, "arquivo": str(arquivo), "sha256": hashlib.sha256(arquivo.read_bytes()).hexdigest()}

    registro = {
        **proveniencia(),
        "config": {
            "execucoes": {papel: str(getattr(args, papel)) for papel in PAPEIS},
            "reamostragens": 10_000,
            "conjunto": args.conjunto,
            "aoi": aoi,
            "anotacao": anotacao,
            "n_pares": n_pares,
            **({"atomos": {"rotulo": args.rotulo_atomos, "lista": atomos}} if atomos else {}),
            "unidade": "pares-base em Java, átomos em C (docs/metodos.md, seção 4)",
            "metrica": "docs/metodos.md, seção 4",
            **({"tokenizador_revisao": revisoes_tokenizador} if revisoes_tokenizador else {}),
        },
        "seed": 42,
        "model": {papel: r["model"] for papel, r in registros.items()},
        "model_revision": {papel: r["model_revision"] for papel, r in registros.items()},
        "date": date.today().isoformat(),
    }
    sufixo = f"-{anotacao['versao']}" if anotacao and anotacao["versao"] != "v1" else ""
    if atomos:
        sufixo += f"-atomos-{args.rotulo_atomos}"
    if args.sufixo:
        sufixo += f"-{args.sufixo}"
    final = args.pasta or args.saida / f"{date.today().isoformat()}-trio-{args.nome}-{args.conjunto}-aoi-{aoi}{sufixo}"
    with pasta_nova(final) as destino:
        trio.to_csv(destino / "trio.csv", index=False)
        por_modelo.to_csv(destino / "por_modelo.csv", index=False)
        escrever_json(destino / "run.json", registro)
    print(por_modelo.round(3).to_string(index=False))
    print(trio.round(3).to_string(index=False))
    print(f"Resultados em {final}")
    return final


if __name__ == "__main__":
    main()
