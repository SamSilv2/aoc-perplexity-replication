"""Compara uma pasta de resultados regenerada com a publicada, separando o tipo de diferença.

    uv run --locked python -m aoc_perplexity.comparar PUBLICADA NOVA [--tolerancia-csv X] [--tolerancia-json Y] [--sem-binarios]

Classes, por arquivo:
- idêntico: mesmos bytes;
- formatação: mesmos valores, escritos de outro jeito (por exemplo, 1.0 × 1, ou a ordem das chaves de um JSON);
- esquema: mesmos valores, com uma migração declarada (cabeçalho renomeado em CSV, chave acrescentada em JSON);
- numérico (tolerância): valores numéricos diferentes, todos dentro da tolerância do tipo de arquivo;
- numérico: valores diferentes além da tolerância, textos diferentes, tipos diferentes (por exemplo, true × 1) ou formas diferentes;
- difere: PDF, PNG ou LaTeX com bytes diferentes;
- não comparado: PDF, PNG ou LaTeX com --sem-binarios (os dados das figuras e tabelas continuam comparados);
- metadados: run.json, que registra data, commit e caminhos;
- ausente: o arquivo existe só de um lado.

A comparação passa se só houver idêntico, formatação, esquema, numérico (tolerância), não comparado e metadados, e se as duas
pastas existirem e tiverem ao menos um arquivo de dados. Os valores são lidos como texto e comparados sem arredondamento;
inf, -inf e nan só são iguais a si mesmos.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

# Migrações entre versões do código, sem mudança de valor.
MIGRACOES_CSV = {"n": "n_wilcoxon"}  # cabeçalho antigo → atual
MIGRACOES_JSON = {"token_ids_sha256"}  # chaves que a versão atual acrescenta ao resumo de cada snippet
BINARIOS = {".pdf", ".png", ".tex"}
ACEITAS = {"idêntico", "formatação", "esquema", "numérico (tolerância)", "não comparado", "metadados"}


def _numero(texto):
    try:
        return float(texto)
    except ValueError:
        return None


def _diferenca_texto(a, b):
    """Diferença entre duas células lidas como texto: 0 se iguais, a diferença absoluta se numéricas, None se textos diferentes."""
    if a == b:
        return 0.0
    x, y = _numero(a), _numero(b)
    if x is None or y is None:
        return None
    if math.isfinite(x) and math.isfinite(y):
        return abs(x - y)
    if math.isnan(x) and math.isnan(y) or x == y:  # mesma grafia diferente de inf ou nan
        return 0.0
    return math.inf


def comparar_csv(publicado, novo, tolerancia):
    with open(publicado, newline="") as f:
        a = list(csv.reader(f))
    with open(novo, newline="") as f:
        b = list(csv.reader(f))
    if not a or not b:
        return "numérico", "arquivo vazio"
    cabecalho_a, cabecalho_b = a[0], b[0]
    renomeadas = {}
    if cabecalho_a != cabecalho_b:
        renomeadas = {c: MIGRACOES_CSV[c] for c in cabecalho_a if c in MIGRACOES_CSV and MIGRACOES_CSV[c] in cabecalho_b}
        if [renomeadas.get(c, c) for c in cabecalho_a] != cabecalho_b:
            return "numérico", f"colunas diferentes: {cabecalho_a} × {cabecalho_b}"
    if len(a) != len(b) or any(len(x) != len(y) for x, y in zip(a, b)):
        return "numérico", f"formas diferentes: {len(a) - 1} × {len(b) - 1} linhas"
    maior = 0.0
    for linha_a, linha_b in zip(a[1:], b[1:]):
        for x, y in zip(linha_a, linha_b):
            d = _diferenca_texto(x, y)
            if d is None:
                return "numérico", f"textos diferentes: {x!r} × {y!r}"
            maior = max(maior, d)
    if maior == 0:
        if renomeadas:
            return "esquema", "cabeçalhos " + ", ".join(f"{x} → {y}" for x, y in renomeadas.items())
        return "formatação", "mesmos valores, escritos de outro jeito"
    classe = "numérico (tolerância)" if maior <= tolerancia else "numérico"
    return classe, f"maior diferença {maior:.3g}" + (" e cabeçalhos migrados" if renomeadas else "")


def _diferenca_json(a, b, migradas):
    """Maior diferença absoluta entre duas estruturas JSON; None se a forma, um texto ou um tipo diferir.

    Chaves de MIGRACOES_JSON presentes só no lado novo são aceitas e anotadas em `migradas`; presentes nos dois, são comparadas.
    """
    if isinstance(a, dict) and isinstance(b, dict):
        so_no_novo = set(b) - set(a)
        if set(a) - set(b) or so_no_novo - MIGRACOES_JSON:
            return None
        migradas |= so_no_novo
        diferencas = [_diferenca_json(a[k], b[k], migradas) for k in a]
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return None
        diferencas = [_diferenca_json(x, y, migradas) for x, y in zip(a, b)]
    elif type(a) is not type(b) and not (type(a) in (int, float) and type(b) in (int, float)):
        return None  # true × 1, texto × número, null × número
    elif isinstance(a, (int, float)) and not isinstance(a, bool):
        if a == b or math.isnan(a) and math.isnan(b):
            return 0.0
        return abs(a - b) if math.isfinite(a) and math.isfinite(b) else math.inf
    else:
        return 0.0 if a == b else None
    if any(d is None for d in diferencas):
        return None
    return max(diferencas, default=0.0)


def comparar_json(publicado, novo, tolerancia):
    migradas = set()
    maior = _diferenca_json(json.loads(publicado.read_text()), json.loads(novo.read_text()), migradas)
    if maior is None:
        return "numérico", "estrutura, tipos ou textos diferentes"
    nota = f"chaves acrescentadas: {', '.join(sorted(migradas))}" if migradas else ""
    if maior == 0:
        return ("esquema", nota) if migradas else ("formatação", "mesmos valores, escritos de outro jeito")
    classe = "numérico (tolerância)" if maior <= tolerancia else "numérico"
    return classe, "; ".join(p for p in [f"maior diferença {maior:.3g}", nota] if p)


def comparar_arquivo(publicado, novo, tolerancia_csv=0.0, tolerancia_json=0.0, binarios=True):
    """(classe, detalhe) de um par de arquivos."""
    if publicado.name == "run.json":
        return "metadados", ""
    if publicado.read_bytes() == novo.read_bytes():
        return "idêntico", ""
    if publicado.suffix == ".csv":
        return comparar_csv(publicado, novo, tolerancia_csv)
    if publicado.suffix == ".json":
        return comparar_json(publicado, novo, tolerancia_json)
    if publicado.suffix in BINARIOS and not binarios:
        return "não comparado", "renderização (os dados correspondentes são comparados)"
    return "difere", "bytes diferentes"


def comparar_pastas(publicada, nova, tolerancia_csv=0.0, tolerancia_json=0.0, binarios=True):
    """Lista de (arquivo, classe, detalhe) para todos os arquivos das duas pastas."""
    publicada, nova = Path(publicada), Path(nova)
    for pasta in (publicada, nova):
        if not pasta.is_dir():
            return [(str(pasta), "ausente", "pasta inexistente")]
    nomes = sorted({p.relative_to(publicada) for p in publicada.rglob("*") if p.is_file()} | {p.relative_to(nova) for p in nova.rglob("*") if p.is_file()})
    if not any(n.name != "run.json" for n in nomes):
        return [(str(publicada), "ausente", "nenhum arquivo de dados para comparar")]
    linhas = []
    for nome in nomes:
        a, b = publicada / nome, nova / nome
        if not a.exists() or not b.exists():
            linhas.append((str(nome), "ausente", "só na nova" if b.exists() else "só na publicada"))
        else:
            linhas.append((str(nome), *comparar_arquivo(a, b, tolerancia_csv, tolerancia_json, binarios)))
    return linhas


def aprovado(linhas):
    return bool(linhas) and all(classe in ACEITAS for _, classe, _ in linhas)


def imprimir(linhas, titulo=None):
    if titulo:
        print(f"== {titulo}")
    for nome, classe, detalhe in linhas:
        print(f"  {classe:22} {nome}" + (f"  ({detalhe})" if detalhe else ""))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("publicada", type=Path)
    parser.add_argument("nova", type=Path)
    parser.add_argument("--tolerancia-csv", type=float, default=0.0, help="maior diferença aceita em CSVs (estatísticas e tabelas; padrão 0)")
    parser.add_argument("--tolerancia-json", type=float, default=0.0, help="maior diferença aceita em JSONs (log-probabilidades e perplexidades por token; padrão 0)")
    parser.add_argument("--sem-binarios", action="store_true", help="não compara PDF, PNG e LaTeX (renderizações)")
    args = parser.parse_args(argv)
    linhas = comparar_pastas(args.publicada, args.nova, args.tolerancia_csv, args.tolerancia_json, not args.sem_binarios)
    imprimir(linhas)
    return 0 if aprovado(linhas) else 1


if __name__ == "__main__":
    sys.exit(main())
