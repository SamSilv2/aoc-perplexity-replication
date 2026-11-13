"""Junta as análises de trio numa tabela única (CSV e LaTeX) para o artigo."""

import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd

from aoc_perplexity.resultados import NOVOS, Manifesto, escrever_json, nome_datado, pasta_nova, proveniencia

# Trios publicados (results/manifesto.json): Java e C de cada família; C usa a AOI v2.1 (docs/metodos.md, seção 3).
TRIOS = Manifesto.ler().trios()
# Resultados não relatados (docs/metodos.md, seção 4): ficam no CSV, marcados, e aparecem como travessão na tabela.
INVALIDAS = {("Gemma 7B", "snippet_media")}
# Pares por conjunto quando o run.json do trio não registra n_pares (trios anteriores à AOI v2.1: todos os pares).
PARES_TOTAIS = {"bergum": 72, "gopstein": 63}
NAO_RELATADO = "---"
NOTA_INVALIDAS = r"---: not reported (the first token of \texttt{codegemma-7b-it} has log-probability $\approx -395$)."
# Classificação robusto / dependente da anotação do C (sensibilidade à AOI), lida para marcar a tabela compacta.
ROBUSTEZ = "robustez.csv"
SLUGS = {
    "Qwen2.5 3B": "qwen2.5-3b",
    "Qwen2.5 7B": "qwen2.5-7b",
    "Qwen2.5 32B (4 bits)": "qwen2.5-32b-4bit",
    "DeepSeek 7B": "deepseek-7b",
    "Yi 9B": "yi-9b",
    "Llama-2 7B": "llama-2-7b",
    "Llama-2 13B": "llama-2-13b",
    "Gemma 7B": "gemma-7b",
}
DEPENDENTE = "dependente da anotação"
CONTRASTES = ["D", "D_codigo", "D_instrucao"]
NOTA_DEPENDENTE = r"$^\dagger$ Annotation-dependent (C): sign or interval status (excludes or includes zero) differs among AOI annotations v1, v2 and v2.1; only v2.1 is shown. Bold: interval excludes zero."
QUANTIDADES = ["Delta_base", "Delta_codigo", "Delta_instruct", "D", "D_codigo", "D_instrucao"]
ROTULOS = [r"$\Delta_B$", r"$\Delta_C$", r"$\Delta_I$", r"$D_{\mathrm{instr},\mathrm{base}}$", r"$D_{\mathrm{code},\mathrm{base}}$", r"$D_{\mathrm{instr},\mathrm{code}}$"]


def numero(x):
    """Duas casas decimais; um valor não nulo que arredondaria para 0.00 mostra seu primeiro algarismo significativo, sempre em decimal (0.00008, não 8e-05)."""
    texto = f"{x:.2f}"
    if x == 0 or float(texto) != 0:
        return texto
    curto = f"{x:.1g}"
    if "e" in curto:
        curto = f"{float(curto):.{-int(curto.split('e')[1])}f}"
    return curto


def celula(linha):
    if getattr(linha, "valida", "sim") == "nao":
        return NAO_RELATADO
    texto = f"{numero(linha.estimativa)} [{numero(linha.ic_inferior)}, {numero(linha.ic_superior)}]".replace("-", "$-$")
    # Negrito quando o intervalo exclui zero.
    return rf"\textbf{{{texto}}}" if linha.ic_inferior > 0 or linha.ic_superior < 0 else texto


def tabela_contrastes(tabela, robustez):
    """Tabela compacta do aoi_max: uma linha por família, D, D_código e D_instrução em Java e em C.

    Com `robustez`, † marca em C o contraste que a sensibilidade à anotação classifica como dependente.
    O artigo não usa a marca: a dependência da anotação fica nas ameaças à validade (docs/metodos.md, seção 3).
    """
    parte = tabela[tabela.metrica == "aoi_max"].set_index(["conjunto", "trio", "quantidade"])
    dependentes = set()
    if robustez is not None:
        dependentes = {(linha.trio, linha.quantidade) for linha in robustez.itertuples() if linha.classe == DEPENDENTE}
    familias = [tamanho for tamanho, conjunto in TRIOS if conjunto == "bergum"]
    corpo = []
    for familia in familias:
        celulas = []
        for conjunto in ["bergum", "gopstein"]:
            for q in CONTRASTES:
                texto = celula(parte.loc[(conjunto, familia, q)])
                if conjunto == "gopstein" and (SLUGS[familia], q) in dependentes:
                    texto += r"$^\dagger$"
                celulas.append(texto)
        corpo.append(" & ".join([familia] + celulas) + r" \\")
    rotulos = [ROTULOS[QUANTIDADES.index(q)] for q in CONTRASTES]
    return "\n".join([
        r"{\footnotesize\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r" & \multicolumn{3}{c}{Java (Bergum et al.)} & \multicolumn{3}{c}{C (Gopstein et al.)} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        " & ".join(["Family"] + rotulos * 2) + r" \\",
        r"\midrule",
        *corpo,
        r"\bottomrule",
        *([rf"\multicolumn{{7}}{{@{{}}p{{0.98\textwidth}}@{{}}}}{{{NOTA_DEPENDENTE}}} \\"] if dependentes else []),
        r"\end{tabular}}",
    ]) + "\n"


def tabela_contrastes_coluna(tabela):
    """Mesma tabela compacta em uma coluna: bloco de Java sobre o de C, com as células de `celula()` e sem a marca †."""
    parte = tabela[tabela.metrica == "aoi_max"].set_index(["conjunto", "trio", "quantidade"])
    familias = [tamanho for tamanho, conjunto in TRIOS if conjunto == "bergum"]
    rotulos = [ROTULOS[QUANTIDADES.index(q)] for q in CONTRASTES]
    blocos = []
    for conjunto, titulo in [("bergum", "Java (Bergum et al.)"), ("gopstein", "C (Gopstein et al.)")]:
        blocos += [r"\midrule", rf"\multicolumn{{4}}{{l}}{{\textit{{{titulo}}}}} \\", r"\midrule"]
        blocos += [" & ".join([familia] + [celula(parte.loc[(conjunto, familia, q)]) for q in CONTRASTES]) + r" \\" for familia in familias]
    return "\n".join([
        r"{\scriptsize\setlength{\tabcolsep}{2.5pt}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        " & ".join(["Family"] + rotulos) + r" \\",
        *blocos,
        r"\bottomrule",
        r"\end{tabular}}",
    ]) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifesto", type=Path, default=None, help="de onde ler os trios (padrão: results/manifesto.json)")
    parser.add_argument("--saida", type=Path, default=NOVOS, help="pasta onde criar o consolidado")
    parser.add_argument("--pasta", type=Path, default=None, help="caminho exato da pasta de saída (sem data); tem precedência sobre --saida")
    args = parser.parse_args(argv)
    trios = Manifesto.ler(args.manifesto).trios() if args.manifesto else TRIOS

    linhas = []
    for (tamanho, conjunto), pasta in trios.items():
        trio = pd.read_csv(pasta / "trio.csv")
        registro_trio = json.loads((pasta / "run.json").read_text()) if (pasta / "run.json").exists() else {}
        n_pares = registro_trio.get("config", {}).get("n_pares") or {m: PARES_TOTAIS[conjunto] for m in trio.metrica.unique()}
        trio["n_pares"] = trio.metrica.map(n_pares)
        trio["valida"] = ["nao" if (tamanho, m) in INVALIDAS else "sim" for m in trio.metrica]
        trio.insert(0, "conjunto", conjunto)
        trio.insert(0, "trio", tamanho)
        linhas.append(trio)
    tabela = pd.concat(linhas)

    # Consolidados publicados não são sobrescritos: um novo no mesmo dia ganha número.
    final = args.pasta or nome_datado(args.saida, "consolidado")
    with pasta_nova(final) as destino:
        escrever(tabela, trios, destino)
    print(f"Resultados em {final}")
    return final


def escrever(tabela, trios, destino):
    tabela.to_csv(destino / "consolidado.csv", index=False)

    for metrica in ["aoi_max", "snippet_media"]:
        parte = tabela[tabela.metrica == metrica].set_index(["conjunto", "trio", "quantidade"])
        corpo = []
        for conjunto, nome in [("bergum", "Java (Bergum et al.)"), ("gopstein", "C (Gopstein et al.)")]:
            corpo.append(rf"\multicolumn{{{len(QUANTIDADES) + 1}}}{{l}}{{\textit{{{nome}}}}} \\")
            for (tamanho, c) in trios:
                if c == conjunto:
                    celulas = [celula(parte.loc[(c, tamanho, q)]) for q in QUANTIDADES]
                    corpo.append(" & ".join([tamanho] + celulas) + r" \\")
        if (parte.valida == "nao").any():
            corpo.append(rf"\multicolumn{{{len(QUANTIDADES) + 1}}}{{l}}{{\footnotesize {NOTA_INVALIDAS}}} \\")
        cabecalho = " & ".join(["Family"] + ROTULOS) + r" \\"
        (destino / f"tabela-{metrica}.tex").write_text("\n".join([r"\begin{tabular}{" + "l" * (len(QUANTIDADES) + 1) + "}", r"\toprule", cabecalho, r"\midrule", *corpo, r"\bottomrule", r"\end{tabular}"]) + "\n")

    # Sem a marca de dependência da anotação (o artigo a trata nas ameaças à validade).
    (destino / "tabela-contrastes-aoi_max.tex").write_text(tabela_contrastes(tabela, None))
    (destino / "tabela-contrastes-aoi_max-coluna.tex").write_text(tabela_contrastes_coluna(tabela))

    registro = {
        **proveniencia(),
        "config": {"trios": {f"{t}-{c}": str(p) for (t, c), p in trios.items()}, "anotacao_gopstein": "v2.1", "robustez": None, "invalidas": [f"{t}: {m}" for t, m in sorted(INVALIDAS)]},
        "seed": "herdada das análises (42)",
        "model": "Qwen2.5 3B, 7B, 32B 4 bits; DeepSeek 7B; Yi 9B; Llama-2 7B e 13B; Gemma 7B",
        "model_revision": "nos run.json de cada análise listada em config.trios",
        "date": date.today().isoformat(),
    }
    escrever_json(destino / "run.json", registro)


if __name__ == "__main__":
    main()
