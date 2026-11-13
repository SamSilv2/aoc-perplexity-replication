"""Grade de painéis FRP × perplexidade por modelo (Llama-2 e Qwen2.5, Java): uma seta por snippet-base, do limpo ao confuso (nenhum modelo roda aqui)."""

import argparse
from datetime import date
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import patheffects
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerPatch
from matplotlib.patches import FancyArrowPatch

from aoc_perplexity.consolidar import TRIOS
from aoc_perplexity.eeg_extensao import analises_publicadas, eeg_publicado, modelos_da_extensao, tabela_snippets
from aoc_perplexity.figuras import COR_NEUTRA, CONSOLIDADO, CORRELACOES, FONTES, PAPEIS, estilo, fonte_serifada, salvar, sha256
from aoc_perplexity.resultados import NOVOS, Manifesto, escrever_json, nome_datado, pasta_nova, proveniencia
from aoc_perplexity.trio import pares_do_modelo

VERSOES_POR_GRUPO = 3
MEDIA_GERAL = "media_geral"
SEM_NUMERO = "(vazio)"  # o snippet-base sem número no nome do arquivo é um dos 24 grupos
CONDICOES = ["limpo", "confuso"]
# Cores discretas e de saturação baixa. Expected e opposite têm o mesmo formato (linha, espessura e cabeça cheia) e diferem só na cor;
# partial é cinza médio, mais fina, e desenhada por baixo.
CATEGORIAS = {
    "expected": {"cor": "#4F8A9E", "lw": 1.0, "alpha": 0.85, "cabeca": "cheia", "escala": 4, "zorder": 6, "rotulo": "Expected direction"},
    "partial": {"cor": "#7F7F7F", "lw": 0.9, "alpha": 0.8, "cabeca": "cheia", "escala": 4, "zorder": 4, "rotulo": "Partial"},
    "opposite": {"cor": "#B5655A", "lw": 1.0, "alpha": 0.85, "cabeca": "cheia", "escala": 4, "zorder": 5, "rotulo": "Opposite"},
}
COR_MEDIA = "#4DAF4A"
LW_MEDIA, LW_CONTORNO_MEDIA = 2.6, 3.6
X_MAX = 1e6  # limite superior do eixo x em todas as grades; o que passar disso encosta na borda com "▶"
ROTULO_X = "Max AOI perplexity (geometric mean per base pair, log scale)"
ROTULO_Y = r"Mean FRP ($\mu$V)"
# (figura, tamanho em polegadas, famílias da linha na ordem de consolidar.TRIOS): linhas = famílias, colunas = B, C, I.
GRADES = {
    "eeg-grade-llama2": {"tamanho": (7, 4.8), "familias": ["Llama-2 7B", "Llama-2 13B"]},
    "eeg-grade-qwen25": {"tamanho": (7, 6.8), "familias": ["Qwen2.5 3B", "Qwen2.5 7B", "Qwen2.5 32B (4 bits)"]},
    # Os dois trios de Java com padrões de contraste opostos, ambos com D_code,base e D_instr,code fora de zero (docs/metodos.md, seção 5).
    "eeg-grade-contraste": {"tamanho": (7, 4.8), "familias": ["Llama-2 13B", "Gemma 7B"]},
}
PAPEL_ROTULO = {"base": "B", "codigo": "C", "instruct": "I"}


def estilo_grade(fonte=None):
    """Estilo único de figuras.py, sem hinting do texto: com ele, o espaço de "Llama-2 13B" some nos PNGs rotacionados."""
    return {**estilo(fonte), "text.hinting": "none"}


def nome_curto(modelo, familia):
    """Nome do modelo sem o dono do repositório; o 32B vem de uma conversão MLX em 4 bits."""
    nome = modelo.split("/")[-1].removesuffix("-4bit")
    return f"{nome} (4-bit)" if "4 bits" in familia else nome


def dados_do_modelo(snippets):
    """Pontos por snippet-base e condição, mais a média geral de cada condição.

    x: média geométrica da Max AOI Perplexity (exp da média dos ln) nas versões do grupo; y: média do FRP em µV;
    n: snippets do ponto (3 por grupo; a média geral é a dos 24 grupos e conta 72). Média geral: geométrica em x e
    aritmética em y sobre os grupos.
    """
    snippets = snippets.assign(ln_ppl=np.log(snippets["ppl"]), frp_uv=snippets["frp"] * 1e6)
    grupos = snippets.groupby(["base", "condicao"], sort=False).agg(ln_x=("ln_ppl", "mean"), y=("frp_uv", "mean"), n=("ln_ppl", "size")).reset_index()
    if (grupos["n"] != VERSOES_POR_GRUPO).any():
        raise ValueError(f"cada grupo deve ter {VERSOES_POR_GRUPO} snippets por condição")
    if not (grupos.groupby("condicao")["base"].nunique() == grupos["base"].nunique()).all():
        raise ValueError("as duas condições devem ter os mesmos grupos")
    grupos["grupo"] = grupos["base"].replace("", SEM_NUMERO)
    geral = grupos.groupby("condicao").agg(ln_x=("ln_x", "mean"), y=("y", "mean"), n=("n", "sum")).reset_index()
    geral["grupo"] = MEDIA_GERAL
    todos = pd.concat([grupos, geral], ignore_index=True)
    todos["x"] = np.exp(todos["ln_x"])
    todos["eh_media"] = todos["grupo"] == MEDIA_GERAL
    todos["ordem"] = todos["condicao"].map(CONDICOES.index)
    todos = todos.sort_values(["eh_media", "grupo", "ordem"], kind="stable")
    todos["categoria"] = ""
    todos = todos.reset_index(drop=True)
    categorias = categorias_dos_grupos(todos)
    todos["categoria"] = todos["grupo"].map(categorias).fillna("")
    return todos[["grupo", "condicao", "x", "y", "n", "categoria"]]


def categoria(x_limpo, y_limpo, x_confuso, y_confuso):
    """expected: perplexidade e FRP sobem do limpo ao confuso; opposite: os dois descem; partial: os demais casos."""
    if x_confuso > x_limpo and y_confuso > y_limpo:
        return "expected"
    if x_confuso < x_limpo and y_confuso < y_limpo:
        return "opposite"
    return "partial"


def categorias_dos_grupos(dados):
    """Categoria de cada grupo (a média geral fica de fora)."""
    grupos = dados[dados["grupo"] != MEDIA_GERAL].pivot(index="grupo", columns="condicao", values=["x", "y"])
    return {g: categoria(l.loc[("x", "limpo")], l.loc[("y", "limpo")], l.loc[("x", "confuso")], l.loc[("y", "confuso")]) for g, l in grupos.iterrows()}


def contagem(dados):
    """Grupos por categoria (soma 24)."""
    por_grupo = pd.Series(categorias_dos_grupos(dados))
    return {nome: int((por_grupo == nome).sum()) for nome in CATEGORIAS}


def percentual(k, total=24):
    """Proporção arredondada para inteiro (meio para cima)."""
    return int(100 * k / total + 0.5)


def contagem_e_proporcao(dados):
    """Contagem e percentual inteiro por categoria, como aparecem no painel e no CSV."""
    n = contagem(dados)
    return {**{f"{nome}_n": k for nome, k in n.items()}, **{f"{nome}_pct": percentual(k) for nome, k in n.items()}}


def pontos_alem_da_borda(dados):
    """Pontos (grupo e condição) com x acima de X_MAX: são desenhados encostados na borda, sem remover dado."""
    return int(((dados["grupo"] != MEDIA_GERAL) & (dados["x"] > X_MAX)).sum())


def delta_da_media_geral(dados):
    """ln(x confuso geral) − ln(x limpo geral): é o Δ do aoi_max do modelo."""
    geral = dados[dados["grupo"] == MEDIA_GERAL].set_index("condicao")
    return float(np.log(geral.loc["confuso", "x"]) - np.log(geral.loc["limpo", "x"]))


def rho_confusos(correlacoes, familia, papel):
    """ρ do subconjunto dos confusos do modelo, lido de correlacoes.csv (não recalculado)."""
    linha = correlacoes[(correlacoes["trio"] == familia) & (correlacoes["papel"] == papel) & (correlacoes["subconjunto"] == "confusos")]
    if len(linha) != 1:
        raise ValueError(f"correlacoes.csv sem o ρ de {familia} {papel}")
    return float(linha["rho"].iloc[0])


def carregar(trios=TRIOS):
    """Dados de todos os modelos das grades: dict (família, papel) → dict(modelo, dados)."""
    familias = {f for grade in GRADES.values() for f in grade["familias"]}
    eeg = eeg_publicado(analises_publicadas())
    modelos = {}
    for familia, papel, modelo, execucao, _ in modelos_da_extensao(trios):
        if familia in familias:
            pares, registro = pares_do_modelo(execucao, "bergum", "manual")
            modelos[(familia, papel)] = {"modelo": modelo, "dados": dados_do_modelo(tabela_snippets(pares, eeg)), "execucao": execucao, "revisao": registro["model_revision"]}
    return modelos


def contrastes_da_dupla(consolidado, familias=("Llama-2 13B", "Gemma 7B")):
    """D_code,base (D_codigo) e D_instr,code (D_instrucao) em Java, com IC e se o IC exclui zero, lidos do consolidado."""
    linhas = {}
    for familia in familias:
        for quantidade in ["D_codigo", "D_instrucao"]:
            linha = consolidado[(consolidado["trio"] == familia) & (consolidado["conjunto"] == "bergum") & (consolidado["metrica"] == "aoi_max") & (consolidado["quantidade"] == quantidade)].iloc[0]
            linhas[f"{familia} {quantidade}"] = {"estimativa": float(linha["estimativa"]), "ic_inferior": float(linha["ic_inferior"]), "ic_superior": float(linha["ic_superior"]), "ic_exclui_zero": bool(linha["ic_inferior"] > 0 or linha["ic_superior"] < 0)}
    return linhas


def delta_do_consolidado(consolidado, familia, papel):
    linha = consolidado[(consolidado["trio"] == familia) & (consolidado["conjunto"] == "bergum") & (consolidado["metrica"] == "aoi_max") & (consolidado["quantidade"] == f"Delta_{papel}")]
    return float(linha["estimativa"].iloc[0])


def tabela_grade(modelos, correlacoes):
    """dados-grade.csv: todos os pontos (e a média geral) de todos os modelos, com a categoria, o ρ usado e a contagem e o percentual de cada categoria do modelo."""
    partes = []
    for (familia, papel), item in modelos.items():
        parte = item["dados"].copy()
        marca, tamanho = familia.split(" ", 1)
        parte.insert(0, "modelo", item["modelo"])
        parte.insert(0, "papel", papel)
        parte.insert(0, "tamanho", tamanho)
        parte.insert(0, "familia", marca)
        parte["rho_confusos"] = rho_confusos(correlacoes, familia, papel)
        for coluna, valor in contagem_e_proporcao(item["dados"]).items():
            parte[coluna] = valor
        partes.append(parte)
    return pd.concat(partes, ignore_index=True)


def limites(modelos, chaves):
    """Limites compartilhados pelos painéis: x em log, de uma margem de 1/3 de década antes do menor ponto até X_MAX; y com folga no alto para o texto do canto."""
    dados = pd.concat([modelos[c]["dados"] for c in chaves])
    pontos = dados[dados["grupo"] != MEDIA_GERAL]
    x_min = 10 ** (np.log10(pontos["x"].min()) - 0.33)
    y_min, y_max = pontos["y"].min(), pontos["y"].max()
    amplitude = y_max - y_min
    return (x_min, X_MAX), (y_min - 0.06 * amplitude, y_max + 0.55 * amplitude)


def texto_do_painel(rho, dados):
    """Três linhas do canto: ρ dos confusos e contagem (com percentual) de expected, partial e opposite."""
    c = contagem_e_proporcao(dados)
    return "\n".join(
        [
            rf"$\rho_{{\mathrm{{conf}}}}$ = {rho:.2f}",
            f"expected {c['expected_n']}/24 ({c['expected_pct']}%)",
            f"partial {c['partial_n']}/24 ({c['partial_pct']}%) · opposite {c['opposite_n']}/24 ({c['opposite_pct']}%)",
        ]
    )


def propriedades_da_seta(cor, lw, alpha, cabeca, escala=4):
    """arrowprops de uma seta com cabeça cheia ou vazada (contorno na cor e miolo branco)."""
    props = {"arrowstyle": "-|>", "mutation_scale": escala, "lw": lw, "alpha": alpha, "linestyle": "-", "shrinkA": 1.5, "shrinkB": 1.5}
    if cabeca == "vazada":
        return {**props, "edgecolor": cor, "facecolor": "white"}
    return {**props, "color": cor}


def desenhar_painel(eixo, item, familia, papel, rho):
    dados = item["dados"]
    grupos = dados[dados["grupo"] != MEDIA_GERAL]
    limpo = grupos[grupos["condicao"] == "limpo"].set_index("grupo")
    confuso = grupos[grupos["condicao"] == "confuso"].set_index("grupo").loc[limpo.index]
    media = dados[dados["grupo"] == MEDIA_GERAL].set_index("condicao")

    def ponto(x, y, cheio, cor, tamanho, zorder, **estilo):
        """Marca um ponto; acima de X_MAX ele encosta na borda como "▶" (fora do eixo, sem cortar)."""
        if x > X_MAX:
            eixo.scatter(X_MAX, y, s=tamanho * 1.8, marker=">", color=cor if cheio else "white", edgecolors=cor, linewidths=0.6, clip_on=False, zorder=zorder + 0.2)
        else:
            eixo.scatter(x, y, s=tamanho, marker="o", facecolors=cor if cheio else "none", edgecolors=cor if cheio else COR_NEUTRA, linewidths=0 if cheio else 0.5, zorder=zorder, **estilo)

    eixo.axhline(0, color="#999999", lw=0.5, zorder=1)
    for grupo in limpo.index:
        marca = CATEGORIAS[confuso.loc[grupo, "categoria"]]
        inicio, fim = (min(limpo.loc[grupo, "x"], X_MAX), limpo.loc[grupo, "y"]), (min(confuso.loc[grupo, "x"], X_MAX), confuso.loc[grupo, "y"])
        eixo.annotate("", xy=fim, xytext=inicio, arrowprops=propriedades_da_seta(marca["cor"], marca["lw"], marca["alpha"], marca["cabeca"], marca["escala"]), zorder=marca["zorder"])
        ponto(limpo.loc[grupo, "x"], limpo.loc[grupo, "y"], False, COR_NEUTRA, 7, marca["zorder"] + 0.1)
        ponto(confuso.loc[grupo, "x"], confuso.loc[grupo, "y"], True, marca["cor"], 7, marca["zorder"] + 0.1)

    # Média geral por cima de tudo: seta verde com contorno preto e marcadores maiores (limpo vazado, confuso cheio).
    contorno = [patheffects.withStroke(linewidth=LW_CONTORNO_MEDIA, foreground="black")]
    inicio, fim = (media.loc["limpo", "x"], media.loc["limpo", "y"]), (media.loc["confuso", "x"], media.loc["confuso", "y"])
    eixo.annotate(
        "",
        xy=fim,
        xytext=inicio,
        arrowprops={"arrowstyle": "-|>", "mutation_scale": 6, "color": COR_MEDIA, "lw": LW_MEDIA, "shrinkA": 4, "shrinkB": 4, "path_effects": contorno},
        zorder=8,
    )
    eixo.scatter(*inicio, s=28, facecolors="white", edgecolors=COR_MEDIA, linewidths=1.4, zorder=9, path_effects=[patheffects.withStroke(linewidth=2.8, foreground="black")])
    eixo.scatter(*fim, s=28, color=COR_MEDIA, edgecolors="black", linewidths=0.8, zorder=9)

    eixo.set_xscale("log")
    eixo.set_xticks([1, 1e2, 1e4, 1e6], [r"$10^{0}$", r"$10^{2}$", r"$10^{4}$", r"$10^{6}$"])
    eixo.minorticks_off()
    eixo.grid(True)
    eixo.text(0.03, 0.97, texto_do_painel(rho, dados), transform=eixo.transAxes, ha="left", va="top", fontsize=6, linespacing=1.15, color="#333333")
    # Título: o papel (B, C ou I) na cor da Figura 1 e o nome curto do modelo.
    etiqueta = eixo.text(0, 1.03, rf"$\mathbf{{{PAPEL_ROTULO[papel]}}}$", transform=eixo.transAxes, ha="left", va="bottom", fontsize=8, color=PAPEIS[papel]["cor"])
    eixo.annotate(nome_curto(item["modelo"], familia), xy=(1, 0), xycoords=etiqueta, xytext=(3, 0), textcoords="offset points", ha="left", va="bottom", fontsize=7)


def _seta_da_legenda(legend, orig_handle, xdescent, ydescent, width, height, fontsize):
    meio = ydescent + height / 2
    return FancyArrowPatch((xdescent, meio), (xdescent + width, meio), arrowstyle=orig_handle.get_arrowstyle(), mutation_scale=orig_handle.get_mutation_scale(), shrinkA=0, shrinkB=0)


def manipuladores_da_legenda():
    cinza = {"markersize": 3.5, "ls": ""}
    setas = [
        FancyArrowPatch((0, 0), (1, 0), shrinkA=0, shrinkB=0, **{k: v for k, v in propriedades_da_seta(m["cor"], m["lw"], m["alpha"], m["cabeca"], m["escala"]).items() if k not in {"shrinkA", "shrinkB"}})
        for m in CATEGORIAS.values()
    ]
    media = FancyArrowPatch((0, 0), (1, 0), arrowstyle="-|>", mutation_scale=4, color=COR_MEDIA, lw=LW_MEDIA, shrinkA=0, shrinkB=0, path_effects=[patheffects.withStroke(linewidth=LW_CONTORNO_MEDIA, foreground="black")])
    return [
        Line2D([], [], marker="o", markerfacecolor="none", markeredgecolor=COR_NEUTRA, markeredgewidth=0.7, **cinza),
        Line2D([], [], marker="o", color=COR_NEUTRA, **cinza),
        *setas,
        media,
    ]


def figura_grade(modelos, correlacoes, familias, tamanho):
    """Uma grade (famílias × B, C, I) com eixos x e y compartilhados e uma legenda única acima."""
    figura = Figure(figsize=tamanho, layout="constrained")
    eixos = figura.subplots(len(familias), len(PAPEIS), sharex=True, sharey=True, squeeze=False)
    chaves = [(familia, papel) for familia in familias for papel in PAPEIS]
    (x_min, x_max), (y_min, y_max) = limites(modelos, chaves)
    for linha, familia in enumerate(familias):
        for coluna, papel in enumerate(PAPEIS):
            eixo = eixos[linha, coluna]
            desenhar_painel(eixo, modelos[(familia, papel)], familia, papel, rho_confusos(correlacoes, familia, papel))
            eixo.set_xlim(x_min, x_max)
            eixo.set_ylim(y_min, y_max)
        eixos[linha, 0].set_ylabel(familia, fontsize=8)
    figura.supxlabel(ROTULO_X, fontsize=8)
    figura.supylabel(ROTULO_Y, fontsize=8)
    rotulos = ["Clean", "Confusing", *[m["rotulo"] for m in CATEGORIAS.values()], "Mean over base pairs"]
    figura.legend(
        manipuladores_da_legenda(),
        rotulos,
        loc="outside upper center",
        ncol=6,
        frameon=False,
        handlelength=2.0,
        columnspacing=1.0,
        handletextpad=0.4,
        handler_map={FancyArrowPatch: HandlerPatch(patch_func=_seta_da_legenda)},
    )
    return figura


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifesto", type=Path, default=None, help="entradas: consolidado, EEG (correlacoes.csv) e trios (padrão: results/manifesto.json)")
    parser.add_argument("--saida", type=Path, default=NOVOS, help="pasta onde criar AAAA-MM-DD-figuras (só saída)")
    parser.add_argument("--pasta", type=Path, default=None, help="caminho exato da pasta de saída (sem data)")
    args = parser.parse_args(argv)
    manifesto = Manifesto.ler(args.manifesto) if args.manifesto else None
    trios = manifesto.trios() if manifesto else TRIOS
    arquivo_consolidado = manifesto.pasta("consolidado") / "consolidado.csv" if manifesto else CONSOLIDADO
    arquivo_correlacoes = manifesto.pasta("eeg") / "correlacoes.csv" if manifesto else CORRELACOES

    fonte = fonte_serifada()
    correlacoes = pd.read_csv(arquivo_correlacoes)
    consolidado = pd.read_csv(arquivo_consolidado)
    modelos = carregar(trios)
    for (familia, papel), item in modelos.items():
        if abs(delta_da_media_geral(item["dados"]) - delta_do_consolidado(consolidado, familia, papel)) > 1e-9:
            raise ValueError(f"{familia} {papel}: o Δ da média geral difere do consolidado")
    tabela = tabela_grade(modelos, correlacoes)

    todos = pd.concat([item["dados"].assign(familia=f, papel=p) for (f, p), item in modelos.items()])
    pontos = todos[todos["grupo"] != MEDIA_GERAL]
    hoje = date.today().isoformat()
    registro = {
        **proveniencia(),
        "config": {
            "figuras": {
                "eeg-grade-llama2.pdf": "grade 2 × 3 (Llama-2 7B e 13B × B, C, I)",
                "eeg-grade-qwen25.pdf": "grade 3 × 3 (Qwen2.5 3B, 7B e 32B em 4 bits × B, C, I)",
                "eeg-grade-contraste.pdf": "grade 2 × 3 (Llama-2 13B e Gemma 7B × B, C, I): os trios com contrastes opostos e D_code,base e D_instr,code fora de zero",
                "png": "um PNG de 150 dpi por figura, só para pré-visualização",
            },
            "painel": "uma seta por snippet-base (24), do limpo ao confuso; x = média geométrica da Max AOI Perplexity nas 3 versões, y = média do FRP (µV); média geral geométrica em x e aritmética em y",
            "categorias": "expected: x e y sobem; opposite: x e y descem; partial: os demais casos",
            "fontes": {str(c): sha256(c) for c in [arquivo_consolidado, arquivo_correlacoes]},
            "modelos": {f"{f} {p}": {"model": item["modelo"], "model_revision": item["revisao"], "execucao": str(item["execucao"]), "sha256_perplexidade": sha256(item["execucao"] / "perplexidade.json")} for (f, p), item in modelos.items()},
            "contagens": {f"{f} {p}": contagem_e_proporcao(item["dados"]) for (f, p), item in modelos.items()},
            "x_max_do_eixo": X_MAX,
            "pontos_alem_da_borda": {"total": sum(pontos_alem_da_borda(item["dados"]) for item in modelos.values()), "por_modelo": {f"{f} {p}": pontos_alem_da_borda(item["dados"]) for (f, p), item in modelos.items() if pontos_alem_da_borda(item["dados"])}},
            "contrastes_da_dupla": contrastes_da_dupla(consolidado),
            "estilo": {"expected": CATEGORIAS["expected"], "partial": CATEGORIAS["partial"], "opposite": CATEGORIAS["opposite"], "media_geral": {"cor": COR_MEDIA, "lw": LW_MEDIA, "contorno": LW_CONTORNO_MEDIA}, "text.hinting": "none (o espaço de rótulos rotacionados some nos PNGs com hinting)"},
            "rho_confusos": {f"{f} {p}": rho_confusos(correlacoes, f, p) for (f, p) in modelos},
            "delta_conferido_com": f"{arquivo_consolidado} (aoi_max, tolerância 1e-9)",
            "faixa_x": {"min": float(pontos["x"].min()), "max": float(pontos["x"].max())},
            "faixa_y_uv": {"min": float(pontos["y"].min()), "max": float(pontos["y"].max())},
            "fonte_tipografica": {"usada": fonte, "candidatas": FONTES, "mathtext": "stix"},
            "matplotlib": matplotlib.__version__,
            "metadados": "PDF: CreationDate omitido, SOURCE_DATE_EPOCH fixo, fontes embutidas (Type 42); PNG: sem Software",
        },
        "seed": None,
        "date": hoje,
    }
    final = args.pasta or nome_datado(args.saida, "figuras")
    with pasta_nova(final) as destino:
        tabela.to_csv(destino / "dados-grade.csv", index=False)
        rc = estilo_grade(fonte)
        with matplotlib.rc_context(rc):
            for nome, grade in GRADES.items():
                figura = figura_grade(modelos, correlacoes, grade["familias"], grade["tamanho"])
                salvar(figura, destino / f"{nome}.pdf", rc=rc)
                salvar(figura, destino / f"{nome}.png", rc=rc, dpi=150)
        escrever_json(destino / "run.json", registro)
    print(f"Figuras em {final}")
    return final


if __name__ == "__main__":
    main()
