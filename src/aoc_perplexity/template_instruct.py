"""Verificação do chat template dos instruct (docs/metodos.md, seção 6): o mesmo trio com o instruct sem e com template."""

import argparse
import json
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from aoc_perplexity.consolidar import TRIOS
from aoc_perplexity.resultados import NOVOS, Manifesto, escrever_json, nome_datado, pasta_nova, proveniencia
from aoc_perplexity.trio import METRICAS, log_razao, pares_do_modelo, pares_validos

EXTENSAO = NOVOS
# Entradas publicadas (results/manifesto.json): o consolidado e os trios com o instruct rodado com chat template.
CONSOLIDADO = Manifesto.ler().pasta("consolidado") / "consolidado.csv"
TRIOS_COM_TEMPLATE = Manifesto.ler().trios("template")
LINGUAGENS = {"bergum": "Java", "gopstein": "C"}
QUANTIDADES = {"Delta_instruct": "Δ_I", "D_instrucao": "D_instr,code", "D": "D_instr,base"}
REAMOSTRAGENS = 10_000
SEMENTE = 42


def conferir_tokenizacao_do_template(execucao, conjunto, tokenizador=None, snippets=None):
    """Na execução com template, prefixo e snippet tokenizados em separado dão os IDs do texto inteiro (senão, erro)."""
    from aoc_perplexity import gopstein
    from aoc_perplexity.perplexidade import PACOTE, conferir_tokenizacao_separada, ler_snippets

    registro = json.loads((Path(execucao) / "run.json").read_text())
    prefixo = registro["config"]["chat_template"]["prefixo"]
    if tokenizador is None:
        from transformers import AutoTokenizer

        tokenizador = AutoTokenizer.from_pretrained(registro["config"]["tokenizador"], revision=registro["config"]["tokenizador_revisao"])
    if snippets is None:
        snippets = ler_snippets(PACOTE / "code_snippets") if conjunto == "bergum" else gopstein.ler_snippets()
    for codigo in snippets.values():
        conferir_tokenizacao_separada(tokenizador, prefixo, codigo)
    return len(snippets)


def exclui_zero(inferior, superior):
    return bool(inferior > 0 or superior < 0)


def diferenca_delta_i(pares_sem, pares_com, coluna, n=REAMOSTRAGENS, semente=SEMENTE):
    """Δ_I(com) − Δ_I(sem) com IC, no mesmo sorteio de átomos de `trio.analisar` (mesma semente, mesmos grupos).

    As duas execuções têm de ter os mesmos pares e os mesmos pares válidos na métrica; só o instruct muda.
    """
    chaves = pares_sem[["base", "versao"]]
    if not pares_com[["base", "versao"]].equals(chaves):
        raise ValueError("as duas execuções do instruct não têm os mesmos pares")
    validos = pares_validos({"base": pares_sem, "instruct": pares_com}, coluna)
    sem, com = pares_sem[validos].reset_index(drop=True), pares_com[validos].reset_index(drop=True)
    grupos = sem["base"].to_numpy()
    ids = pd.unique(grupos)
    linhas_por_id = {g: np.flatnonzero(grupos == g) for g in ids}
    razoes = log_razao(com, coluna) - log_razao(sem, coluna)

    gerador = np.random.RandomState(semente)
    amostras = [float(np.mean(razoes[np.concatenate([linhas_por_id[g] for g in gerador.choice(ids, size=len(ids), replace=True)])])) for _ in range(n)]
    return {
        "estimativa": float(np.mean(razoes)),
        "ic_inferior": float(np.percentile(amostras, 2.5)),
        "ic_superior": float(np.percentile(amostras, 97.5)),
    }


def posicao_zero(resultados):
    """Log-probabilidade média da posição 0 (o primeiro token do snippet), em todos os snippets e por condição."""
    por_condicao = {"todos": list(resultados), "limpo": [a for a in resultados if "clean" in a], "confuso": [a for a in resultados if "obf" in a]}
    return {c: (len(arqs), float(np.mean([resultados[a]["results"][0]["log_prob"] for a in arqs]))) for c, arqs in por_condicao.items()}


def comparar(sem, com, diferenca, familia, conjunto):
    """Linhas de comparacao.csv: Δ_I, D_instr,code e D_instr,base, sem e com template, nas duas métricas."""
    sem = sem.set_index(["metrica", "quantidade"])
    com = com.set_index(["metrica", "quantidade"])
    linhas = []
    for metrica in METRICAS:
        for quantidade, rotulo in QUANTIDADES.items():
            a, b = sem.loc[(metrica, quantidade)], com.loc[(metrica, quantidade)]
            d = diferenca[metrica]
            linhas.append(
                {
                    "trio": familia,
                    "linguagem": LINGUAGENS[conjunto],
                    "metrica": metrica,
                    "quantidade": quantidade,
                    "rotulo": rotulo,
                    "sem_estimativa": a["estimativa"],
                    "sem_ic_inferior": a["ic_inferior"],
                    "sem_ic_superior": a["ic_superior"],
                    "com_estimativa": b["estimativa"],
                    "com_ic_inferior": b["ic_inferior"],
                    "com_ic_superior": b["ic_superior"],
                    # D_instr,code e D_instr,base diferem de Δ_I por termos que não mudam com o instruct: a diferença é a mesma.
                    "diferenca": d["estimativa"],
                    "diferenca_ic_inferior": d["ic_inferior"],
                    "diferenca_ic_superior": d["ic_superior"],
                    "sem_exclui_zero": exclui_zero(a["ic_inferior"], a["ic_superior"]),
                    "com_exclui_zero": exclui_zero(b["ic_inferior"], b["ic_superior"]),
                }
            )
    return pd.DataFrame(linhas)


def resumir(comparacao):
    """Resumo dos 8 casos (2 trios × 2 linguagens × 2 métricas) do Δ_I e dos contrastes do instruct que mudam o status do IC."""
    delta_i = comparacao[comparacao["quantidade"] == "Delta_instruct"].copy()
    delta_i["queda"] = delta_i["sem_estimativa"] - delta_i["com_estimativa"]
    maior = delta_i.loc[delta_i["queda"].idxmax()]
    mudancas = comparacao[comparacao["sem_exclui_zero"] != comparacao["com_exclui_zero"]]
    return {
        "casos": int(len(delta_i)),
        "delta_i_cai": int((delta_i["com_estimativa"] < delta_i["sem_estimativa"]).sum()),
        "maior_queda": {"trio": maior["trio"], "linguagem": maior["linguagem"], "metrica": maior["metrica"], "queda": float(maior["queda"])},
        "muda_de_sinal": [f"{r.trio} {r.linguagem} {r.metrica}" for r in delta_i.itertuples() if np.sign(r.sem_estimativa) != np.sign(r.com_estimativa)],
        "muda_status_do_ic": [
            {"trio": r.trio, "linguagem": r.linguagem, "metrica": r.metrica, "quantidade": r.quantidade, "sem": [r.sem_ic_inferior, r.sem_ic_superior], "com": [r.com_ic_inferior, r.com_ic_superior]}
            for r in mudancas.itertuples()
        ],
    }


def conferir_com_o_consolidado(sem, consolidado, familia, conjunto):
    """Os valores 'sem template' são os do consolidado (as análises oficiais)."""
    oficial = consolidado[(consolidado["trio"] == familia) & (consolidado["conjunto"] == conjunto)].set_index(["metrica", "quantidade"])
    for linha in sem.itertuples():
        esperado = oficial.loc[(linha.metrica, linha.quantidade)]
        for coluna in ["estimativa", "ic_inferior", "ic_superior"]:
            if getattr(linha, coluna) != esperado[coluna]:
                raise ValueError(f"{familia} {conjunto} {linha.metrica} {linha.quantidade}: {coluna} difere do consolidado")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifesto", type=Path, default=None, help="entradas: consolidado, trios e trios com template (padrão: results/manifesto.json)")
    parser.add_argument("--saida", type=Path, default=EXTENSAO)
    parser.add_argument("--pasta", type=Path, default=None, help="caminho exato da pasta de saída (sem data)")
    parser.add_argument("--reamostragens", type=int, default=REAMOSTRAGENS)
    args = parser.parse_args(argv)
    manifesto = Manifesto.ler(args.manifesto) if args.manifesto else None
    trios = manifesto.trios() if manifesto else TRIOS
    trios_com_template = manifesto.trios("template") if manifesto else TRIOS_COM_TEMPLATE
    arquivo_consolidado = manifesto.pasta("consolidado") / "consolidado.csv" if manifesto else CONSOLIDADO

    consolidado = pd.read_csv(arquivo_consolidado)
    comparacoes, posicoes, execucoes_usadas = [], [], {}
    for (familia, conjunto), pasta_com in trios_com_template.items():
        registro_sem = json.loads((trios[(familia, conjunto)] / "run.json").read_text())
        registro_com = json.loads((pasta_com / "run.json").read_text())
        for papel in ["base", "codigo"]:
            if registro_sem["config"]["execucoes"][papel] != registro_com["config"]["execucoes"][papel]:
                raise ValueError(f"{familia} {conjunto}: {papel} difere entre o trio oficial e o trio com template")
        execucao_sem, execucao_com = Path(registro_sem["config"]["execucoes"]["instruct"]), Path(registro_com["config"]["execucoes"]["instruct"])
        conferir_tokenizacao_do_template(execucao_com, conjunto)
        execucoes_usadas[f"{familia} {conjunto}"] = {"sem": str(execucao_sem), "com": str(execucao_com), "trio_sem": str(trios[(familia, conjunto)]), "trio_com": str(pasta_com)}

        anotacao = (registro_sem["config"].get("anotacao") or {}).get("versao", "v2.1")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # o aviso dos pares fora do aoi_max já sai no trio.py
            pares_sem, _ = pares_do_modelo(execucao_sem, conjunto, "manual", anotacao)
            pares_com, _ = pares_do_modelo(execucao_com, conjunto, "manual", anotacao)
        diferenca = {metrica: diferenca_delta_i(pares_sem, pares_com, coluna, n=args.reamostragens) for metrica, coluna in METRICAS.items()}

        sem = pd.read_csv(trios[(familia, conjunto)] / "trio.csv")
        com = pd.read_csv(pasta_com / "trio.csv")
        conferir_com_o_consolidado(sem, consolidado, familia, conjunto)
        # A diferença e os trio.csv têm de concordar: a estimativa de Δ_I(com) − Δ_I(sem) é a diferença das estimativas.
        for metrica in METRICAS:
            a = sem[(sem["metrica"] == metrica) & (sem["quantidade"] == "Delta_instruct")]["estimativa"].iloc[0]
            b = com[(com["metrica"] == metrica) & (com["quantidade"] == "Delta_instruct")]["estimativa"].iloc[0]
            if abs((b - a) - diferenca[metrica]["estimativa"]) > 1e-12:
                raise ValueError(f"{familia} {conjunto} {metrica}: a diferença não bate com os trio.csv")
        comparacoes.append(comparar(sem, com, diferenca, familia, conjunto))

        arquivo = "perplexidade.json" if conjunto == "bergum" else "perplexidade-gopstein.json"
        for modo, execucao in [("sem_template", execucao_sem), ("com_template", execucao_com)]:
            for condicao, (n, media) in posicao_zero(json.loads((execucao / arquivo).read_text())).items():
                posicoes.append({"trio": familia, "linguagem": LINGUAGENS[conjunto], "modo": modo, "condicao": condicao, "n_snippets": n, "logprob_posicao0_medio": media})

    comparacao = pd.concat(comparacoes, ignore_index=True)
    posicao0 = pd.DataFrame(posicoes)
    registro = {
        **proveniencia(),
        "config": {
            "metodo": "docs/metodos.md, seção 6",
            "reamostragens": args.reamostragens,
            "ic": "percentis 2,5 e 97,5; bootstrap dos átomos (pares-base), o mesmo sorteio de trio.analisar (semente 42)",
            "anotacao_de_C": "v2.1",
            "valores_sem_template": f"{arquivo_consolidado} (conferidos linha a linha)",
            "trios": execucoes_usadas,
            "resumo": resumir(comparacao),
        },
        "seed": SEMENTE,
        "date": date.today().isoformat(),
    }
    final = args.pasta or nome_datado(args.saida, "template-instruct")
    with pasta_nova(final) as destino:
        comparacao.to_csv(destino / "comparacao.csv", index=False)
        posicao0.to_csv(destino / "posicao0.csv", index=False)
        escrever_json(destino / "run.json", registro)
    print(json.dumps(registro["config"]["resumo"], indent=2, ensure_ascii=False))
    print(f"Resultados em {final}")
    return final


if __name__ == "__main__":
    main()
