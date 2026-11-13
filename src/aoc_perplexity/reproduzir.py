"""Fluxo completo da reprodução: das execuções (opcional) às figuras, numa raiz própria, e conferência com o publicado.

    uv run python -m aoc_perplexity.reproduzir                       # análise a partir das execuções salvas, com conferência
    uv run python -m aoc_perplexity.reproduzir --mostrar             # só imprime os comandos equivalentes
    uv run --extra inferencia python -m aoc_perplexity.reproduzir --etapas todas --limpar-cache   # roda também os modelos

Cada etapa lê as entradas do manifesto da raiz (`<raiz>/manifesto.json`), que começa igual a `results/manifesto.json`
e passa a apontar para o que a própria reprodução gerou. Nenhum caminho precisa ser editado à mão.
Saídas em `<raiz>/<etapa>`, com nomes fixos; a raiz padrão fica em `results/novos/`, fora do Git e do índice.
"""

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

from aoc_perplexity.comparar import aprovado, comparar_arquivo, comparar_pastas, imprimir
from aoc_perplexity.resultados import MANIFESTO, NOVOS, Manifesto

PAPEIS = ["base", "codigo", "instruct"]
GRUPOS_DE_TRIOS = ["principais", "template", "atomos"]
# Etapas na ordem de execução e o que cada uma grava. "trios" cobre os três grupos de trios do manifesto.
ETAPAS = ["inferencia", "replicacao", "trios", "sensibilidade", "consolidado", "eeg", "figuras", "figuras_grade", "template"]
PADRAO = [e for e in ETAPAS if e != "inferencia"]
PASTAS = {
    "sensibilidade": "sensibilidade-aoi-gopstein",
    "consolidado": "consolidado",
    "eeg": "eeg-java",
    "figuras": "figuras",
    "figuras_grade": "figuras-grade",
    "template": "template-instruct",
}
REPLICACAO = {"rq12": "rq1-rq2", "rq3": "rq3", "tabela_modelos": "tabela-por-modelo"}
CACHE_MODELOS = Path("data/cache-modelos")
# Espaço em disco dos pesos (GB, aproximado) por número de parâmetros; os 32B do MLX estão em 4 bits.
TAMANHO_GB = {"1.3b": 3, "3b": 7, "7b": 15, "6.7b": 14, "9b": 18, "13b": 27, "32b": 19}
# Critério por token da reprodução, fixado antes da reexecução do 1.3B (docs/metodos.md).
LIMITE_TOKEN = 0.01
FRACAO_MINIMA = 0.99


def ler_json(caminho):
    return json.loads(Path(caminho).read_text())


def execucoes_do_manifesto(manifesto, replicacao=False):
    """Pastas de execução usadas pelos trios do manifesto, sem repetição, agrupadas por modelo.

    Com `replicacao`, inclui as reexecuções da reprodução (deepseek-coder-1.3b-base e Qwen2.5-Coder-32B em 4 bits).
    """
    vistas = list(manifesto.dados.get("execucoes_replicacao", {}).values()) if replicacao else []
    for grupo in GRUPOS_DE_TRIOS:
        for pasta in manifesto.trios(grupo).values():
            for papel in PAPEIS:
                execucao = ler_json(Path(pasta) / "run.json")["config"]["execucoes"][papel]
                if execucao not in vistas:
                    vistas.append(execucao)
    return sorted(vistas, key=lambda e: ler_json(Path(e) / "run.json")["model"])


def argumentos_inferencia(execucao, destino):
    """perplexidade.py com o modelo, as revisões e as opções registradas na execução publicada."""
    registro = ler_json(Path(execucao) / "run.json")
    config = registro["config"]
    partes = ["--modelo", registro["model"], "--revision", registro["model_revision"]]
    tokenizador = config.get("tokenizador", registro["model"])
    if tokenizador != registro["model"]:
        partes += ["--tokenizador", tokenizador]
    partes += ["--tokenizer-revision", config.get("tokenizador_revisao") or registro["model_revision"]]
    partes += ["--backend", config.get("backend", "torch")]
    if config.get("backend", "torch") == "torch":
        partes += ["--dtype", config["dtype"]]
    if config.get("uma_passagem"):
        partes.append("--uma-passagem")
    if config.get("chat_template"):
        partes.append("--chat-template")
    partes += ["--conjuntos", *config.get("conjuntos", ["bergum"]), "--pasta", str(destino)]
    return partes


def argumentos_trio(pasta, execucoes, destino):
    """trio.py com os argumentos registrados no trio publicado; `execucoes` troca as pastas das execuções."""
    registro = ler_json(Path(pasta) / "run.json")
    config = registro["config"]
    nome = Path(pasta).name.split("-trio-", 1)[1].split(f"-{config['conjunto']}-aoi-", 1)[0]
    partes = ["--nome", nome]
    for papel in PAPEIS:
        partes += [f"--{papel}", str(execucoes.get(config["execucoes"][papel], config["execucoes"][papel]))]
    partes += ["--conjunto", config["conjunto"], "--aoi", config["aoi"]]
    if config.get("anotacao"):
        partes += ["--anotacao", config["anotacao"]["versao"]]
    if config.get("atomos"):
        partes += ["--atomos", ",".join(config["atomos"]["lista"]), "--rotulo-atomos", config["atomos"]["rotulo"]]
    # Confere que cada execução usou a revisão publicada.
    for papel in PAPEIS:
        partes += ["--revision", f"{papel}={registro['model_revision'][papel]}"]
    for papel, sha in (config.get("tokenizador_revisao") or {}).items():
        partes += ["--tokenizer-revision", f"{papel}={sha}"]
    partes += ["--pasta", str(destino)]
    return partes


def tamanho_estimado_gb(modelo):
    nome = modelo.lower()
    for chave in sorted(TAMANHO_GB, key=len, reverse=True):
        if f"-{chave}" in nome:
            return TAMANHO_GB[chave]
    return 15


def conferir_espaco(modelos, cache, limpar):
    """Recusa começar a inferência se o disco não comporta os pesos (todos, ou o maior se o cache for limpo a cada modelo)."""
    necessario = max(map(tamanho_estimado_gb, modelos)) if limpar else sum(map(tamanho_estimado_gb, modelos))
    cache.mkdir(parents=True, exist_ok=True)
    livre = shutil.disk_usage(cache).free / 1e9
    print(f"Pesos: cerca de {necessario} GB em {cache} ({'um modelo por vez' if limpar else 'todos os modelos'}); livres: {livre:.0f} GB")
    if livre < necessario * 1.1:
        raise SystemExit("espaço insuficiente; use --limpar-cache ou outra --cache-modelos")


def rodar(modulo, argumentos, mostrar, ambiente=None, extra=""):
    """Roda `python -m aoc_perplexity.<modulo>` (ou só imprime o comando)."""
    comando = [sys.executable, "-m", f"aoc_perplexity.{modulo}", *map(str, argumentos)]
    impresso = f"uv run --locked{extra} python -m aoc_perplexity.{modulo} " + " ".join(shlex.quote(str(a)) for a in argumentos)
    print(impresso, flush=True)
    if not mostrar:
        subprocess.run(comando, check=True, env=ambiente)


def criterio_por_token(publicada, nova, limite=LIMITE_TOKEN, fracao_minima=FRACAO_MINIMA):
    """Critério da reprodução para uma execução nova: tokens idênticos e |Δ log-prob| ≤ limite em ao menos `fracao_minima` dos tokens."""
    tokens = dentro = 0
    for arquivo in sorted(Path(publicada).glob("perplexidade*.json")):
        antigo, novo = ler_json(arquivo), ler_json(Path(nova) / arquivo.name)
        if antigo.keys() != novo.keys():
            return False, f"{arquivo.name}: snippets diferentes"
        for snippet, resultado in antigo.items():
            a, b = resultado["results"], novo[snippet]["results"]
            # Em C os JSONs não guardam o texto dos tokens; valem as posições e, quando os dois lados têm, o hash dos IDs.
            hashes = [r.get("summary", {}).get("token_ids_sha256") for r in (resultado, novo[snippet])]
            if [(x.get("token"), x["position"]) for x in a] != [(x.get("token"), x["position"]) for x in b] or (all(hashes) and hashes[0] != hashes[1]):
                return False, f"{arquivo.name} {snippet}: tokens diferentes"
            tokens += len(a)
            dentro += sum(abs(x["log_prob"] - y["log_prob"]) <= limite for x, y in zip(a, b))
    if not tokens:
        return False, "nenhum token comparado"
    fracao = dentro / tokens
    return fracao >= fracao_minima, f"{fracao:.1%} dos {tokens} tokens com |Δ log-prob| ≤ {limite} (mínimo {fracao_minima:.0%})"


def conferir(publicada, nova, titulo, falhas, tolerancia_csv=0.0, tolerancia_json=0.0, binarios=True, decide=True):
    """Compara e registra a falha; com `decide=False` (etapas derivadas de uma inferência nova), só informa."""
    linhas = comparar_pastas(publicada, nova, tolerancia_csv, tolerancia_json, binarios)
    imprimir(linhas, f"{titulo}: {nova} × {publicada}" + ("" if decide else " (informativo)"))
    if decide and not aprovado(linhas):
        falhas.append(titulo)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifesto", type=Path, default=MANIFESTO, help="resultados publicados de partida (padrão: results/manifesto.json)")
    parser.add_argument("--raiz", type=Path, default=None, help="pasta da reprodução (padrão: results/novos/reproducao-AAAA-MM-DD)")
    parser.add_argument("--etapas", nargs="+", default=PADRAO, help=f"etapas a rodar, de {ETAPAS}, ou 'todas' (padrão: todas menos inferencia)")
    parser.add_argument("--sem-conferir", action="store_true", help="não compara as saídas com as publicadas")
    parser.add_argument("--tolerancia-csv", type=float, default=0.0, help="maior diferença aceita nos CSVs das etapas de análise (padrão: 0)")
    parser.add_argument("--fracao-minima", type=float, default=FRACAO_MINIMA, help=f"com inferência: fração mínima de tokens com |Δ log-prob| ≤ {LIMITE_TOKEN} (padrão: {FRACAO_MINIMA})")
    parser.add_argument("--mostrar", action="store_true", help="só imprime os comandos, sem rodar")
    parser.add_argument("--cache-modelos", type=Path, default=CACHE_MODELOS, help="cache dos pesos só deste experimento (HF_HUB_CACHE da inferência)")
    parser.add_argument("--limpar-cache", action="store_true", help="apaga do cache do experimento os pesos de cada modelo depois da sua última execução")
    args = parser.parse_args(argv)
    etapas = ETAPAS if args.etapas == ["todas"] else args.etapas
    desconhecidas = set(etapas) - set(ETAPAS)
    if desconhecidas:
        parser.error(f"etapas desconhecidas: {sorted(desconhecidas)}")
    raiz = args.raiz or NOVOS / f"reproducao-{date.today().isoformat()}"
    if raiz.exists() and not args.mostrar:
        parser.error(f"{raiz} já existe; escolha outra --raiz")
    publicado = Manifesto.ler(args.manifesto)
    atual = Manifesto(json.loads(json.dumps(publicado.dados)))
    falhas = []
    if not args.mostrar:
        raiz.mkdir(parents=True)

    def gravar_manifesto():
        if not args.mostrar:
            atual.gravar(raiz / "manifesto.json")

    gravar_manifesto()

    # 1. Inferência: uma execução por pasta publicada (as dos trios e as reexecuções da reprodução), com as mesmas revisões e opções.
    #    Cada execução nova passa pelo critério por token; as etapas seguintes, que dependem dela, são comparadas só para informação,
    #    sem as renderizações (PDF, PNG, LaTeX), que mudam com qualquer diferença numérica.
    execucoes = {}
    inferencia = "inferencia" in etapas
    conferir_analise = not args.sem_conferir and not args.mostrar
    if inferencia:
        lista = execucoes_do_manifesto(publicado, replicacao=True)
        modelos = [ler_json(Path(e) / "run.json")["model"] for e in lista]
        if not args.mostrar:
            conferir_espaco(sorted(set(modelos)), args.cache_modelos, args.limpar_cache)
        ambiente = {**os.environ, "HF_HUB_CACHE": str(args.cache_modelos.resolve())}
        for i, execucao in enumerate(lista):
            destino = raiz / "execucoes" / Path(execucao).parent.name / Path(execucao).name
            rodar("perplexidade", argumentos_inferencia(execucao, destino), args.mostrar, ambiente, " --extra inferencia")
            execucoes[execucao] = destino
            ultimo_uso = modelos[i] not in modelos[i + 1 :]
            if args.limpar_cache and ultimo_uso and not args.mostrar:
                pasta_do_modelo = args.cache_modelos / ("models--" + modelos[i].replace("/", "--"))
                shutil.rmtree(pasta_do_modelo, ignore_errors=True)
                print(f"Cache apagado: {pasta_do_modelo}")
            if conferir_analise:
                ok, detalhe = criterio_por_token(execucao, destino, fracao_minima=args.fracao_minima)
                print(f"== execução {Path(execucao).name}: {'atende' if ok else 'NÃO ATENDE'} ao critério por token: {detalhe}")
                if not ok:
                    falhas.append(f"execução {Path(execucao).name}")
    opcoes = {"tolerancia_csv": args.tolerancia_csv, "binarios": not inferencia, "decide": not inferencia}

    # 2. Replicação (RQ1 a RQ3 e tabela por modelo), a partir do pacote dos autores; a tabela usa a reexecução do 1.3B.
    if "replicacao" in etapas:
        reexecucao_13b = publicado.dados["execucoes_replicacao"]["deepseek-coder-1.3b-base"]
        for modulo, pasta in REPLICACAO.items():
            destino = raiz / "replicacao" / pasta
            extra = ["--reexecucao", execucoes.get(reexecucao_13b, reexecucao_13b)] if modulo == "tabela_modelos" else []
            rodar(modulo, [*extra, "--pasta", destino], args.mostrar)
            if conferir_analise:
                conferir(publicado.pasta(modulo), destino, modulo, falhas, **{**opcoes, "decide": opcoes["decide"] or modulo != "tabela_modelos"})

    # Tabela B.3 (32B em 4 bits contra o fp16 publicado): o auxiliar da pasta publicada, sobre a execução nova.
    quantizada = publicado.dados["execucoes_replicacao"]["Qwen2.5-Coder-32B-4bit"]
    if inferencia and quantizada in execucoes:
        auxiliar = Path(quantizada) / "analise.py"
        destino = raiz / "replicacao" / "quantizacao-32b"
        print(f"uv run --locked python {auxiliar} {execucoes[quantizada]} > {destino / 'saida.txt'}", flush=True)
        if not args.mostrar:
            destino.mkdir(parents=True)
            with open(destino / "saida.txt", "w") as saida:
                subprocess.run([sys.executable, str(auxiliar), str(execucoes[quantizada])], check=True, stdout=saida)
            if conferir_analise:
                imprimir([("saida.txt", *comparar_arquivo(Path(quantizada) / "saida.txt", destino / "saida.txt", binarios=False))], "Tabela B.3 (informativo)")

    # 3. Trios: os principais, os com chat template e os subconjuntos de átomos.
    if "trios" in etapas:
        for grupo in GRUPOS_DE_TRIOS:
            novos = {}
            for chave, pasta in publicado.trios(grupo).items():
                destino = raiz / "trios" / Path(pasta).name
                rodar("trio", argumentos_trio(pasta, execucoes, destino), args.mostrar)
                novos[chave] = destino
                if conferir_analise:
                    conferir(pasta, destino, f"trio {Path(pasta).name}", falhas, **opcoes)
            atual = atual.com(trios={grupo: novos})
            gravar_manifesto()

    # 4. Etapas que leem o manifesto da raiz.
    for etapa, modulo in [("sensibilidade", "sensibilidade"), ("consolidado", "consolidar"), ("eeg", "eeg_extensao"), ("figuras", "figuras"), ("figuras_grade", "figuras_grade"), ("template", "template_instruct")]:
        if etapa not in etapas:
            continue
        destino = raiz / PASTAS[etapa]
        rodar(modulo, ["--manifesto", raiz / "manifesto.json", "--pasta", destino], args.mostrar)
        if conferir_analise:
            conferir(publicado.pasta(etapa), destino, etapa, falhas, **opcoes)
        atual = atual.com(etapas={etapa: destino})
        gravar_manifesto()

    if args.mostrar:
        return 0
    print(f"Reprodução em {raiz}")
    if falhas:
        print(f"Diferenças além do aceito em: {', '.join(falhas)}")
        return 1
    if not args.sem_conferir:
        print("Tudo confere com os resultados publicados." if not inferencia else "As execuções novas atendem ao critério por token; as diferenças das etapas seguintes estão acima, para informação.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
