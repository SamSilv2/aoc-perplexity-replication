"""Snippets em C de Gopstein et al. (FSE 2017), AOI anotada e AOI por diff (docs/metodos.md, seção 3)."""

import hashlib
import html
import warnings
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

from aoc_perplexity.aoi import para_tokens

GOPSTEIN = Path("data/external/gopstein")
ANOTACAO = Path("data/anotacoes/gopstein-aoi.csv")
# v1: trecho e ocorrência (por ID); v2 e v2.1: intervalo de caracteres (início:fim), com o hash do snippet (docs/metodos.md, seção 3).
ANOTACOES = {
    "v1": ANOTACAO,
    "v2": Path("data/anotacoes/gopstein-aoi-v2.csv"),
    "v2.1": Path("data/anotacoes/gopstein-aoi-v2.1.csv"),
}
ANOTACAO_PADRAO = "v2.1"


def ler_pares(pasta=GOPSTEIN):
    """Tabela com um par por linha: átomo, versão (v0-v2), IDs e código confuso e limpo.

    Os pares de cada átomo são 3 IDs confusos consecutivos, na ordem dos átomos em code_tags.csv.
    code_tags.csv não está alinhado linha a linha com code.csv, então só a ordem dos átomos é usada.
    """
    codigo = pd.read_csv(pasta / "code.csv", sep="^")
    codigo["Code"] = codigo["Code"].map(html.unescape).str.strip()
    atomos = list(pd.unique(pd.read_csv(pasta / "code_tags.csv")["Tag"]))
    por_id = codigo.set_index("ID")
    confusos = codigo[codigo["Type"] == "Confusing"].sort_values("ID").reset_index(drop=True)
    if len(confusos) != 3 * len(atomos):
        raise ValueError("esperados 3 pares por átomo")
    return pd.DataFrame(
        {
            "atomo": [atomos[i // 3] for i in range(len(confusos))],
            "base": [str(i // 3 + 1) for i in range(len(confusos))],
            "versao": [f"v{i % 3}" for i in range(len(confusos))],
            "id_confuso": confusos["ID"].to_numpy(),
            "id_limpo": confusos["Pair"].to_numpy(),
            "confuso": confusos["Code"].to_numpy(),
            "limpo": por_id.loc[confusos["Pair"], "Code"].to_numpy(),
        }
    )


def ler_snippets(pasta=GOPSTEIN):
    snippets = {}
    for p in ler_pares(pasta).itertuples():
        snippets[f"{p.base}-obf-{p.versao}"] = p.confuso
        snippets[f"{p.base}-clean-{p.versao}"] = p.limpo
    return snippets


def hash_ids(ids):
    """SHA-256 dos IDs de token (inteiros separados por vírgula): confere a tokenização sem guardar o texto."""
    return hashlib.sha256(",".join(str(int(i)) for i in ids).encode()).hexdigest()


def sem_codigo(resultados):
    """Remove o código e o texto dos tokens; o hash dos IDs (summary.token_ids_sha256, quando existe) fica.

    Os resultados de C são gravados sem o código; a análise o recoloca a partir do download (`com_codigo`), conferindo o hash.
    """
    return {
        arquivo: {
            "summary": r["summary"],
            "results": [{k: v for k, v in t.items() if k != "token"} for t in r["results"]],
        }
        for arquivo, r in resultados.items()
    }


def com_codigo(resultados, tokenizador, pasta=GOPSTEIN):
    """Recoloca o código a partir do download fixado e confere o número de tokens e, quando salvo, o hash dos IDs.

    Resultados anteriores ao hash (sem summary.token_ids_sha256) só têm a contagem conferida.
    """
    snippets = ler_snippets(pasta)
    for arquivo, r in resultados.items():
        r["code"] = snippets[arquivo]
        ids = tokenizador(r["code"], add_special_tokens=False)["input_ids"]
        if len(ids) != len(r["results"]):
            raise ValueError(f"{arquivo}: {len(ids)} tokens no código e {len(r['results'])} no resultado")
        esperado = r.get("summary", {}).get("token_ids_sha256")
        if esperado is not None and hash_ids(ids) != esperado:
            raise ValueError(f"{arquivo}: os IDs de token do código diferem dos salvos com o resultado")
    return resultados


def hash_snippet(codigo):
    """Primeiros 16 caracteres hexadecimais do SHA-256 do snippet em UTF-8."""
    return hashlib.sha256(codigo.encode("utf-8")).hexdigest()[:16]


def mapear_por_anotacao(resultados, tokenizador, pasta=GOPSTEIN, anotacao=ANOTACAO):
    """Acrescenta `aoi_regions` = {"ANOT": spans} a partir de uma anotação de AOI.

    Com `inicio` no cabeçalho (v2, v2.1), lê o intervalo de caracteres e confere o hash do snippet.
    Senão (v1, gopstein-aoi.csv), procura o trecho pelo ID de code.csv.
    """
    if "inicio" in pd.read_csv(anotacao, keep_default_na=False, nrows=0).columns:
        return mapear_por_intervalo(resultados, tokenizador, anotacao)
    trechos = pd.read_csv(anotacao, keep_default_na=False).set_index("id")
    ids = {}
    for p in ler_pares(pasta).itertuples():
        ids[f"{p.base}-obf-{p.versao}"] = p.id_confuso
        ids[f"{p.base}-clean-{p.versao}"] = p.id_limpo
    for arquivo, r in resultados.items():
        trecho, ocorrencia = trechos.loc[ids[arquivo], "trecho"], trechos.loc[ids[arquivo], "ocorrencia"]
        inicio = -1
        for _ in range(ocorrencia + 1):
            inicio = r["code"].find(trecho, inicio + 1)
        if inicio == -1:
            raise ValueError(f"trecho anotado não encontrado em {arquivo}: {trecho!r}")
        offsets = tokenizador(r["code"], return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]
        r["aoi_regions"] = {"ANOT": para_tokens(offsets, [(inicio, inicio + len(trecho))])}
    return resultados


def parceiro(arquivo):
    """Nome do outro snippet do par (confuso <-> limpo)."""
    return arquivo.replace("-obf-", "-clean-") if "-obf-" in arquivo else arquivo.replace("-clean-", "-obf-")


def mapear_por_intervalo(resultados, tokenizador, anotacao):
    """AOI por intervalo de caracteres (v2, v2.1). `na_metrica_aoi = nao` tira o par inteiro do aoi_max.

    Um par fora do aoi_max recebe `aoi_regions` vazio e `fora_do_aoi_max`; ele continua no snippet_media.
    O hash do snippet precisa bater com o da anotação, senão o código mudou desde que ela foi feita.
    """
    linhas = pd.read_csv(anotacao, keep_default_na=False, dtype=str).set_index("snippet")
    fora = set()
    for arquivo, r in resultados.items():
        if arquivo not in linhas.index:
            raise ValueError(f"{arquivo}: sem linha na anotação {anotacao}")
        if hash_snippet(r["code"]) != linhas.loc[arquivo, "sha256_snippet"]:
            raise ValueError(f"{arquivo}: o hash do snippet difere do da anotação {anotacao}; o código mudou")
        if linhas.loc[arquivo, "na_metrica_aoi"] == "nao":
            fora |= {arquivo, parceiro(arquivo)}
    for arquivo, r in resultados.items():
        if arquivo in fora:
            r["aoi_regions"] = {"ANOT": []}
            r["fora_do_aoi_max"] = True
            continue
        linha = linhas.loc[arquivo]
        offsets = tokenizador(r["code"], return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]
        spans = para_tokens(offsets, [(int(linha.inicio), int(linha.fim))])
        if not spans:
            raise ValueError(f"{arquivo}: o intervalo {linha.inicio}:{linha.fim} não toca nenhum token")
        r["aoi_regions"] = {"ANOT": spans}
    if fora:
        pares = sorted({a.replace("-obf-", "-").replace("-clean-", "-") for a in fora})
        warnings.warn(f"{len(pares)} pares fora do aoi_max (na_metrica_aoi = nao): {', '.join(pares)}; ficam no snippet_media")
    return resultados


def spans_por_diff(ids_a, ids_b):
    """Intervalos de tokens que mudam entre as duas versões, em cada uma delas.

    Uma inserção pura (0 tokens de um lado) marca o token vizinho nesse lado.
    """
    spans_a, spans_b = [], []
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, ids_a, ids_b, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        for spans, ini, fim, n in [(spans_a, i1, i2, len(ids_a)), (spans_b, j1, j2, len(ids_b))]:
            if fim > ini:
                spans.append([ini, fim - 1])
            else:
                vizinho = min(ini, n - 1)
                spans.append([vizinho, vizinho])
    return spans_a, spans_b


def mapear_por_diff(resultados, tokenizador):
    """Acrescenta `aoi_regions` = {"DIFF": spans} a cada arquivo, comparando confuso e limpo."""
    for arquivo, r in resultados.items():
        if "-obf-" not in arquivo:
            continue
        limpo = resultados[arquivo.replace("-obf-", "-clean-")]
        ids = [tokenizador(x["code"], add_special_tokens=False)["input_ids"] for x in (r, limpo)]
        spans_confuso, spans_limpo = spans_por_diff(*ids)
        r["aoi_regions"] = {"DIFF": spans_confuso}
        limpo["aoi_regions"] = {"DIFF": spans_limpo}
    return resultados
