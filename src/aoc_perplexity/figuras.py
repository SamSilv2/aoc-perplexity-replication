"""Figuras do relatório, lidas só de resultados já versionados (nenhum modelo roda aqui)."""

import argparse
import hashlib
import json
import os
from datetime import date
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.figure import Figure

from aoc_perplexity.consolidar import TRIOS
from aoc_perplexity.eeg_extensao import analises_publicadas, eeg_publicado, modelos_da_extensao, tabela_snippets
from aoc_perplexity.resultados import NOVOS, Manifesto, escrever_json, nome_datado, pasta_nova, proveniencia
from aoc_perplexity.trio import pares_do_modelo

EXTENSAO = Path("results/extensao")
# Entradas publicadas (results/manifesto.json); --manifesto troca por outro conjunto.
CONSOLIDADO = Manifesto.ler().pasta("consolidado") / "consolidado.csv"
CORRELACOES = Manifesto.ler().pasta("eeg") / "correlacoes.csv"
FAMILIAS = [familia for familia, conjunto in TRIOS if conjunto == "bergum"]
# B, C e I: modelo base, de código e instruct do trio. Cores de Okabe e Ito e marcadores distintos (legíveis em tons de cinza).
PAPEIS = {
    "base": {"rotulo": "B", "nome": "base", "cor": "#0072B2", "marcador": "o"},
    "codigo": {"rotulo": "C", "nome": "code", "cor": "#E69F00", "marcador": "s"},
    "instruct": {"rotulo": "I", "nome": "instruct", "cor": "#009E73", "marcador": "^"},
}
LINGUAGENS = {"bergum": {"titulo": "Java", "fonte": "Bergum et al."}, "gopstein": {"titulo": "C", "fonte": "Gopstein et al."}}
EXEMPLO = {"trio": "Qwen2.5 7B", "papel": "base"}
ROTULO_X_DELTA = r"Confusion effect $\Delta$ (mean $\ln \mathrm{PPL}_{\mathrm{conf}} - \ln \mathrm{PPL}_{\mathrm{clean}}$)"
ROTULO_X_PARES = r"$\ell = \ln \mathrm{PPL}_{\mathrm{conf}} - \ln \mathrm{PPL}_{\mathrm{clean}}$ (per pair)"
ROTULO_Y_PARES = r"$\mathrm{FRP}_{\mathrm{conf}} - \mathrm{FRP}_{\mathrm{clean}}$ ($\mu$V)"
COR_NEUTRA = "#4D4D4D"
# Largura de página (Java e C lado a lado), de uma coluna ACM (Java sobre C) e grade 2 × 2 em uma coluna
# (main.tex: Java sobre C, trios divididos em duas metades), em polegadas.
TAMANHO_DELTA = {"pagina": (7, 3.2), "coluna": (3.33, 4.6), "grade": (3.33, 2.9)}
# Serifada próxima da do ACM (Linux Libertine); a primeira instalada vale e vai para o run.json.
FONTES = ["Linux Libertine O", "Libertinus Serif", "STIX Two Text", "DejaVu Serif"]
QUADRANTES = {
    "sup_dir": "ℓ > 0, FRP diff > 0",
    "sup_esq": "ℓ < 0, FRP diff > 0",
    "inf_esq": "ℓ < 0, FRP diff < 0",
    "inf_dir": "ℓ > 0, FRP diff < 0",
    "eixo": "ℓ = 0 ou FRP diff = 0",
}
# Metadados fixos: duas execuções geram o mesmo arquivo.
METADADOS_PDF = {"CreationDate": None}
METADADOS_PNG = {"Software": None}


def fonte_serifada():
    """Primeira fonte de FONTES instalada (DejaVu Serif vem com o matplotlib e fecha a lista)."""
    instaladas = {f.name for f in font_manager.fontManager.ttflist}
    return next(fonte for fonte in FONTES if fonte in instaladas)


def estilo(fonte=None):
    """Estilo único das duas figuras: 8 pt nos rótulos dos eixos, 7 pt em marcas e legenda, só eixos esquerdo e inferior."""
    return {
        "pdf.fonttype": 42,
        "font.family": "serif",
        "font.serif": [fonte or fonte_serifada()],
        "mathtext.fontset": "stix",
        "font.size": 7,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": "#D9D9D9",
        "grid.linewidth": 0.4,
        "axes.axisbelow": True,
    }


def dados_delta(consolidado):
    """Δ do aoi_max e IC de cada modelo (família, papel) em Java e em C: 8 × 3 × 2 linhas."""
    quantidades = {f"Delta_{papel}": papel for papel in PAPEIS}
    parte = consolidado[(consolidado["metrica"] == "aoi_max") & consolidado["quantidade"].isin(quantidades)].copy()
    if (parte["valida"] != "sim").any():
        raise ValueError("há Δ do aoi_max marcado como inválido")
    parte["papel"] = parte["quantidade"].map(quantidades)
    parte["linguagem"] = parte["conjunto"].map({"bergum": "Java", "gopstein": "C"})
    parte["ordem"] = list(zip(parte["conjunto"].map(list(LINGUAGENS).index), parte["trio"].map(FAMILIAS.index), parte["papel"].map(list(PAPEIS).index)))
    parte = parte.sort_values("ordem")
    parte = parte.rename(columns={"trio": "familia"})
    dados = parte[["familia", "linguagem", "conjunto", "papel", "estimativa", "ic_inferior", "ic_superior"]].reset_index(drop=True)
    if len(dados) != len(FAMILIAS) * len(PAPEIS) * len(LINGUAGENS):
        raise ValueError(f"esperava 48 linhas, há {len(dados)}")
    return dados


def figura_delta(dados, formato="pagina"):
    """Figura 1: Δ do aoi_max com IC 95% por família, em dois painéis (Java e C) com a mesma escala.

    `pagina` põe os painéis lado a lado; `coluna` empilha Java sobre C na largura de uma coluna; `grade` também empilha Java
    sobre C, mas divide os trios em duas metades lado a lado (2 × 2 painéis, mesma escala), com o 4 bits do 32B só na legenda
    do artigo. Os três formatos têm os mesmos dados, cores e ordem.
    """
    figura = Figure(figsize=TAMANHO_DELTA[formato], layout="constrained")
    if formato == "grade":
        metade = len(FAMILIAS) // 2
        grupos = [FAMILIAS[:metade], FAMILIAS[metade:]]
        eixos = figura.subplots(2, 2, sharex=True)
        paineis = [(eixos[r][c], conjunto, grupo, c) for r, conjunto in enumerate(LINGUAGENS) for c, grupo in enumerate(grupos)]
    else:
        eixos = figura.subplots(*((1, 2) if formato == "pagina" else (2, 1)), sharex=True, sharey=True)
        paineis = [(eixo, conjunto, FAMILIAS, None) for eixo, conjunto in zip(eixos, LINGUAGENS)]
    deslocamento = {papel: (i - 1) * (0.25 if formato == "grade" else 0.24) for i, papel in enumerate(PAPEIS)}
    for eixo, conjunto, grupo, coluna in paineis:
        linguagem = LINGUAGENS[conjunto]
        for i in range(len(grupo)):
            if i % 2 == 0:
                eixo.axhspan(i - 0.5, i + 0.5, color="#F5F5F5", lw=0, zorder=0)
        for papel, marca in PAPEIS.items():
            linhas = dados[(dados["conjunto"] == conjunto) & (dados["papel"] == papel)].set_index("familia").loc[grupo]
            eixo.errorbar(
                linhas["estimativa"],
                [i + deslocamento[papel] for i in range(len(grupo))],
                xerr=[linhas["estimativa"] - linhas["ic_inferior"], linhas["ic_superior"] - linhas["estimativa"]],
                fmt=marca["marcador"],
                color=marca["cor"],
                markersize=3 if formato == "grade" else 3.5,
                elinewidth=0.8 if formato == "grade" else 0.9,
                capsize=1.2 if formato == "grade" else 1.5,
                capthick=0.8 if formato == "grade" else 0.9,
                label=f"{marca['rotulo']} ({marca['nome']})",
                zorder=3,
            )
        eixo.axvline(0, color="black", lw=0.8, zorder=2)
        # Negrito pelo mathtext (STIX): nem toda fonte serifada instalada tem a face em negrito.
        # Na grade, a linguagem fica no painel da esquerda e a fonte dos dados no da direita.
        if coluna != 1:
            eixo.set_title(rf"$\mathbf{{{linguagem['titulo']}}}$", loc="left")
        if coluna != 0:
            eixo.set_title(linguagem["fonte"], loc="right", fontsize=7, color="#595959")
        eixo.set_yticks(range(len(grupo)), [familia.replace(" (4 bits)", "") for familia in grupo] if formato == "grade" else grupo)
        eixo.set_ylim(len(grupo) - 0.5, -0.5)
        eixo.grid(axis="x")
        eixo.tick_params(axis="y", length=0, pad=1 if formato == "grade" else 3.5)
    figura.supxlabel(ROTULO_X_DELTA, fontsize=8)
    manipuladores, rotulos = paineis[0][0].get_legend_handles_labels()
    figura.legend(manipuladores, rotulos, loc="outside upper center", ncol=3, frameon=False)
    return figura


def quadrante(ell, dif):
    """Quadrante do par: ℓ à direita (> 0) ou à esquerda, diferença de FRP acima (> 0) ou abaixo; zero fica em `eixo`."""
    if ell == 0 or dif == 0:
        return "eixo"
    return ("sup" if dif > 0 else "inf") + ("_dir" if ell > 0 else "_esq")


def dados_eeg_pares(snippets):
    """Um par por linha: ℓ = ln(PPL confuso) − ln(PPL limpo) e FRP confuso − FRP limpo (µV).

    Parte de `eeg_extensao.tabela_snippets`, que já tirou os snippets sem AOI válida; o par some se perde um dos dois.
    """
    largo = snippets.pivot(index="pair_id", columns="condicao", values=["base", "ppl", "frp"])
    largo = largo.dropna()
    largo = largo.loc[sorted(largo.index, key=lambda p: (p.split("-")[0], p.split("-")[1]))]
    ppl_confuso, ppl_limpo = largo[("ppl", "confuso")].astype(float), largo[("ppl", "limpo")].astype(float)
    frp_confuso, frp_limpo = largo[("frp", "confuso")].astype(float) * 1e6, largo[("frp", "limpo")].astype(float) * 1e6
    dados = pd.DataFrame(
        {
            "par": largo.index,
            "ppl_confuso": ppl_confuso.to_numpy(),
            "ppl_limpo": ppl_limpo.to_numpy(),
            "ell": (np.log(ppl_confuso) - np.log(ppl_limpo)).to_numpy(),
            "frp_confuso_uv": frp_confuso.to_numpy(),
            "frp_limpo_uv": frp_limpo.to_numpy(),
            "dif_frp_uv": (frp_confuso - frp_limpo).to_numpy(),
        }
    )
    dados["quadrante"] = [quadrante(e, d) for e, d in zip(dados["ell"], dados["dif_frp_uv"])]
    return dados.reset_index(drop=True)


def contagens(dados):
    """Pares por quadrante (inclui `eixo`); a soma é o número de pares."""
    return {nome: int((dados["quadrante"] == nome).sum()) for nome in QUADRANTES}


def figura_eeg(dados):
    """Figura 2: um ponto por par, ℓ contra a diferença de FRP, com a contagem de pares em cada quadrante e o Δ do modelo (média de ℓ)."""
    figura = Figure(figsize=(3.3, 2.6), layout="constrained")
    eixo = figura.subplots()
    eixo.axhline(0, color="#999999", lw=0.5, zorder=1)
    eixo.axvline(0, color="#999999", lw=0.5, zorder=1)
    delta = float(dados["ell"].mean())
    eixo.axvline(delta, color="#666666", ls=(0, (3, 2)), lw=0.7, zorder=1)
    eixo.scatter(dados["ell"], dados["dif_frp_uv"], s=11, color=COR_NEUTRA, alpha=0.7, lw=0, zorder=2)
    eixo.margins(x=0.06, y=0.16)
    eixo.grid(True)
    eixo.set_xlabel(ROTULO_X_PARES)
    eixo.set_ylabel(ROTULO_Y_PARES)
    eixo.annotate(r"$\Delta$", xy=(delta, 1), xycoords=("data", "axes fraction"), xytext=(2.5, -1), textcoords="offset points", ha="left", va="top", color="#666666")
    n = contagens(dados)
    cantos = {"sup_dir": (0.98, 0.98, "right", "top"), "sup_esq": (0.02, 0.98, "left", "top"), "inf_esq": (0.02, 0.02, "left", "bottom"), "inf_dir": (0.98, 0.02, "right", "bottom")}
    for nome, (x, y, h, v) in cantos.items():
        eixo.text(x, y, f"n = {n[nome]}", transform=eixo.transAxes, ha=h, va=v, color="#333333")
    return figura


def salvar(figura, caminho, rc=None, **opcoes):
    os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
    formato = Path(caminho).suffix.removeprefix(".")
    metadados = {"pdf": METADADOS_PDF, "png": METADADOS_PNG}[formato]
    with matplotlib.rc_context(rc or estilo()):
        figura.savefig(caminho, format=formato, metadata=metadados, **opcoes)


def snippets_do_exemplo(trios=TRIOS):
    """Snippets do Qwen2.5-7B base com o FRP médio, pelo mesmo caminho de eeg_extensao; devolve também o modelo, o registro da execução, a pasta e o total de pares."""
    for trio, papel, modelo, execucao, _ in modelos_da_extensao(trios):
        if (trio, papel) == (EXEMPLO["trio"], EXEMPLO["papel"]):
            pares, registro = pares_do_modelo(execucao, "bergum", "manual")
            return tabela_snippets(pares, eeg_publicado(analises_publicadas())), modelo, registro, execucao, len(pares)
    raise ValueError(f"modelo do exemplo não encontrado: {EXEMPLO}")


def sha256(caminho):
    return hashlib.sha256(Path(caminho).read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifesto", type=Path, default=None, help="entradas: consolidado e trios (padrão: results/manifesto.json)")
    parser.add_argument("--saida", type=Path, default=NOVOS, help="pasta onde criar AAAA-MM-DD-figuras (só saída)")
    parser.add_argument("--pasta", type=Path, default=None, help="caminho exato da pasta de saída (sem data)")
    args = parser.parse_args(argv)
    manifesto = Manifesto.ler(args.manifesto) if args.manifesto else None
    trios = manifesto.trios() if manifesto else TRIOS
    consolidado = manifesto.pasta("consolidado") / "consolidado.csv" if manifesto else CONSOLIDADO
    trio_exemplo = trios[(EXEMPLO["trio"], "bergum")] / "trio.csv"

    fonte = fonte_serifada()
    with matplotlib.rc_context(estilo(fonte)):
        delta = dados_delta(pd.read_csv(consolidado))
        figura1 = figura_delta(delta)
        figura1_coluna = figura_delta(delta, "coluna")
        figura1_grade = figura_delta(delta, "grade")
        snippets, modelo, registro_modelo, execucao, total_pares = snippets_do_exemplo(trios)
        pares = dados_eeg_pares(snippets)
        figura2 = figura_eeg(pares)
        delta_trio = pd.read_csv(trio_exemplo).query("metrica == 'aoi_max' and quantidade == 'Delta_base'")["estimativa"].iloc[0]
        if abs(pares["ell"].mean() - delta_trio) > 1e-9 and len(pares) == total_pares:
            raise ValueError("a média de ℓ difere do Delta_base do trio.csv")

    # Figuras publicadas não são sobrescritas: uma nova no mesmo dia ganha número.
    registro = {
        **proveniencia(),
        "config": {
            "figuras": {
                "delta-aoi_max.pdf": "Figura 1: Δ do aoi_max com IC 95% por família e papel (B, C, I), Java e C, mesma escala; dados-delta.csv",
                "delta-aoi_max-coluna.pdf": "Figura 1 em largura de coluna ACM (3,33 × 4,6 pol., Java sobre C): mesmos dados, escala, cores e ordem de delta-aoi_max.pdf; alternativa à grade",
                "delta-aoi_max-grade.pdf": "Figura 1 em grade 2 × 2 na largura de coluna ACM (3,33 × 2,9 pol.): Java sobre C, trios em duas metades lado a lado, mesma escala; mesmos dados, cores e ordem de delta-aoi_max.pdf; sem o '(4 bits)' no rótulo do 32B; para o main.tex",
                "eeg-java-pares.pdf": f"Figura 2: ℓ = ln(PPL confuso) − ln(PPL limpo) da Max AOI Perplexity × FRP confuso − FRP limpo, um ponto por par de {modelo}; descritiva, sem estatística; dados-eeg-pares.csv",
                "png": "delta-aoi_max.png, delta-aoi_max-coluna.png, delta-aoi_max-grade.png e eeg-java-pares.png (150 dpi), só para pré-visualização",
            },
            "fontes": {str(caminho): sha256(caminho) for caminho in [consolidado, trio_exemplo, execucao / "perplexidade.json"]},
            "exemplo": {"trio": EXEMPLO["trio"], "papel": EXEMPLO["papel"], "model": modelo, "model_revision": registro_modelo["model_revision"]},
            "pares": {"total": total_pares, "usados": len(pares), "sem_aoi_valida": total_pares - len(pares), "por_quadrante": contagens(pares), "delta_media_ell": float(pares["ell"].mean())},
            "fonte_tipografica": {"usada": fonte, "candidatas": FONTES, "mathtext": "stix"},
            "matplotlib": matplotlib.__version__,
            "metadados": "PDF: CreationDate omitido, SOURCE_DATE_EPOCH fixo, fontes embutidas (Type 42); PNG: sem Software",
        },
        "seed": None,
        "date": date.today().isoformat(),
    }
    final = args.pasta or nome_datado(args.saida, "figuras")
    with pasta_nova(final) as destino, matplotlib.rc_context(estilo(fonte)):
        delta.to_csv(destino / "dados-delta.csv", index=False)
        pares.to_csv(destino / "dados-eeg-pares.csv", index=False)
        for figura, nome in [(figura1, "delta-aoi_max"), (figura2, "eeg-java-pares"), (figura1_coluna, "delta-aoi_max-coluna"), (figura1_grade, "delta-aoi_max-grade")]:
            salvar(figura, destino / f"{nome}.pdf")
            salvar(figura, destino / f"{nome}.png", dpi=150)
        escrever_json(destino / "run.json", registro)
    print(f"Figuras em {final}")
    return final


if __name__ == "__main__":
    main()
