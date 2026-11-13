"""Calcula a perplexidade por token dos snippets e compara com o JSON publicado.

Segue o método dos autores (main/utils.py:188-322 do pacote): um token inicial
antes do código e uma passagem pelo modelo para cada token.
"""

import argparse
import json
import math
import time
from datetime import date
from pathlib import Path

import pandas as pd

from aoc_perplexity.gopstein import hash_ids
from aoc_perplexity.resultados import commit_externo, escrever_json, pasta_nova, proveniencia

PACOTE = Path("data/external/llm-perplexity-AoC")
SAIDA = Path("results/replicacao")
TOLERANCIA = 0.01


def partes_do_chat_template(tokenizador, codigo):
    """Prefixo e sufixo, em texto, que o chat template oficial põe ao redor do snippet (mensagem do usuário, sem prompt de geração)."""
    texto = tokenizador.apply_chat_template([{"role": "user", "content": codigo}], tokenize=False)
    if texto.count(codigo) != 1:
        raise ValueError("o chat template deve conter o snippet uma só vez e sem alterá-lo")
    inicio = texto.index(codigo)
    return texto[:inicio], texto[inicio + len(codigo) :]


def conferir_tokenizacao_separada(tokenizador, prefixo, codigo):
    """Prefixo e código tokenizados em separado têm de dar os mesmos IDs que o texto inteiro; senão a AOI se desloca."""
    separados = list(tokenizador(prefixo, add_special_tokens=False).input_ids) + list(tokenizador(codigo, add_special_tokens=False).input_ids)
    juntos = list(tokenizador(prefixo + codigo, add_special_tokens=False).input_ids)
    if separados != juntos:
        raise ValueError("a tokenização do prefixo do chat template e do snippet em separado difere da tokenização conjunta")


def perplexidade_por_token(codigo, modelo, tokenizador, dispositivo, uma_passagem=False, chat_template=False):
    """Log-probabilidade de cada token do snippet.

    Com `chat_template`, o contexto é o prefixo do chat template oficial (em vez do token inicial) e o snippet segue com a
    mesma tokenização do modo sem template; o sufixo do template vem depois do snippet, não altera a log-probabilidade de
    um modelo causal e não é alimentado. Só os tokens do snippet saem nos resultados, no mesmo formato.
    """
    import torch  # extra de inferência; as análises importam este módulo sem torch

    input_ids = tokenizador(codigo, return_tensors="pt", add_special_tokens=False)["input_ids"].to(dispositivo)
    tokens = [tokenizador.decode([t]) for t in input_ids[0]]
    ids_snippet = hash_ids(input_ids[0].tolist())

    if chat_template:
        prefixo, _ = partes_do_chat_template(tokenizador, codigo)
        conferir_tokenizacao_separada(tokenizador, prefixo, codigo)
        ids_prefixo = list(tokenizador(prefixo, add_special_tokens=False).input_ids)
        if not ids_prefixo:
            raise ValueError("o prefixo do chat template não tem tokens")
        n_prefixo = len(ids_prefixo)
        input_ids = torch.cat([torch.tensor([ids_prefixo], device=dispositivo), input_ids], dim=1)
        tokens = [tokenizador.decode([t]) for t in ids_prefixo] + tokens
        tokens_saida = tokens[n_prefixo:]
    else:
        # Como os autores: BOS se houver; senão EOS; senão a quebra de linha.
        tem_bos = tokenizador.bos_token_id is not None
        if tem_bos:
            inicio = tokenizador.bos_token_id
        elif tokenizador.eos_token_id is not None:
            inicio = tokenizador.eos_token_id
        else:
            inicio = (tokenizador("\n", add_special_tokens=False).input_ids or [0])[0]
        n_prefixo = 1
        input_ids = torch.cat([torch.tensor([[inicio]], device=dispositivo), input_ids], dim=1)
        tokens = [tokenizador.decode([inicio])] + tokens
        # Os autores só removem o token inicial da lista quando ele é BOS.
        tokens_saida = tokens[1:] if tem_bos else tokens

    max_length = getattr(modelo.config, "max_position_embeddings", 1024)
    resultados = []
    with torch.no_grad():
        if uma_passagem:
            if input_ids.size(1) > max_length:
                # O laço por token usa uma janela de max_length; a passagem única não tem janela.
                raise ValueError(f"--uma-passagem: o snippet com o prefixo tem {input_ids.size(1)} tokens e o contexto do modelo, {max_length}")
            # Modelo causal: uma passagem dá as mesmas previsões, a menos de ruído numérico.
            log_probs_todos = torch.log_softmax(modelo(input_ids).logits[0, n_prefixo - 1 : -1].float(), dim=-1)
            alvos = input_ids[0, n_prefixo:]
            valores = log_probs_todos.gather(1, alvos.unsqueeze(1)).squeeze(1).tolist()
            for i, log_prob in enumerate(valores, start=n_prefixo):
                resultados.append(
                    {"token": tokens[i], "position": i - n_prefixo, "log_prob": log_prob, "perplexity": math.exp(-log_prob)}
                )
        for i in range(n_prefixo, input_ids.size(1) if not uma_passagem else n_prefixo):
            contexto = input_ids[:, max(0, i - max_length + 1) : i]
            logits = modelo(contexto).logits[:, -1, :]
            log_prob = torch.log_softmax(logits, dim=-1)[0, input_ids[0, i].item()].item()
            resultados.append(
                {"token": tokens[i], "position": i - n_prefixo, "log_prob": log_prob, "perplexity": math.exp(-log_prob)}
            )

    log_probs = [r["log_prob"] for r in resultados]
    if not all(math.isfinite(lp) for lp in log_probs):
        raise ValueError("log-probabilidade não finita; rode em bf16 (docs/metodos.md, seção 2)")
    perplexidades = [r["perplexity"] for r in resultados]
    return {
        "code": codigo,
        "tokens": tokens_saida,
        "summary": {
            "overall_perplexity": math.exp(-sum(log_probs) / len(log_probs)),
            "mean_perplexity": sum(perplexidades) / len(perplexidades),
            "max_perplexity": max(perplexidades),
            "token_count": len(resultados),
            "token_ids_sha256": ids_snippet,
        },
        "results": resultados,
    }


class ModeloMLX:
    """Adapta um modelo MLX quantizado à interface usada em perplexidade_por_token."""

    def __init__(self, modelo):
        self.modelo = modelo
        self.config = type("Config", (), {"max_position_embeddings": 32768})()

    def __call__(self, ids):
        import mlx.core as mx
        import numpy as np
        import torch

        logits = self.modelo(mx.array(ids.numpy())).astype(mx.float32)
        return type("Saida", (), {"logits": torch.from_numpy(np.array(logits))})()


def ler_snippets(pasta):
    return {arquivo.stem: arquivo.read_text(encoding="utf-8") for arquivo in sorted(pasta.glob("v*/*.java"))}


def nome_pasta(data, modelo, conjuntos, limite=None, chat_template=False):
    """AAAA-MM-DD-reexecucao-<modelo>-<conjuntos> (bergum, gopstein ou bergum-gopstein); amostras levam amostraN no lugar; o chat template leva -template."""
    sufixo = f"amostra{limite}" if limite else "-".join(c for c in ("bergum", "gopstein") if c in conjuntos)
    return f"{data}-reexecucao-{modelo}-{sufixo}" + ("-template" if chat_template else "")


def comparar(nossos, publicados):
    linhas = []
    for arquivo, nosso in nossos.items():
        publicado = publicados[arquivo]["results"]
        if [r["token"] for r in nosso["results"]] != [r["token"] for r in publicado]:
            linhas.append({"arquivo": arquivo, "posicao": None, "token": None, "tokens_iguais": False})
            continue
        for a, b in zip(nosso["results"], publicado):
            linhas.append(
                {
                    "arquivo": arquivo,
                    "posicao": a["position"],
                    "token": a["token"],
                    "tokens_iguais": True,
                    "log_prob": a["log_prob"],
                    "log_prob_publicado": b["log_prob"],
                    "diferenca": abs(a["log_prob"] - b["log_prob"]),
                }
            )
    return pd.DataFrame(linhas)


def resumir(comparacao):
    por_arquivo = comparacao.groupby("arquivo")["tokens_iguais"].all()
    tokens = comparacao[comparacao["tokens_iguais"]]
    # Sem nenhum arquivo de tokens iguais não há diferença a resumir (a coluna nem existe).
    comparados = len(tokens) > 0
    return {
        "arquivos": int(len(por_arquivo)),
        "arquivos_com_tokens_iguais": int(por_arquivo.sum()),
        "tokens_comparados": int(len(tokens)),
        "fracao_ate_tolerancia": float((tokens["diferenca"] <= TOLERANCIA).mean()) if comparados else None,
        "diferenca_mediana": float(tokens["diferenca"].median()) if comparados else None,
        "diferenca_maxima": float(tokens["diferenca"].max()) if comparados else None,
        "tolerancia": TOLERANCIA,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modelo", default="deepseek-ai/deepseek-coder-1.3b-base")
    parser.add_argument("--publicado", help="pasta em model_results/ com o JSON dos autores")
    parser.add_argument("--limite", type=int, help="número de arquivos, para a amostra")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument(
        "--conjuntos", nargs="+", default=["bergum"], choices=["bergum", "gopstein"], help="bases avaliadas com o modelo carregado uma vez"
    )
    parser.add_argument("--uma-passagem", action="store_true", help="uma passagem por snippet em vez de uma por token")
    parser.add_argument("--chat-template", action="store_true", help="verificação (docs/metodos.md, seção 6): o snippet entra no chat template oficial do tokenizador e só os tokens dele são pontuados")
    parser.add_argument("--revision", help="SHA do modelo no Hugging Face; sem ela, a revisão mais recente (o SHA vai para o run.json)")
    parser.add_argument("--tokenizer-revision", help="SHA do tokenizador; sem ela, a do modelo ou, com --tokenizador, a mais recente do tokenizador")
    parser.add_argument("--tokenizador", help="repositório do tokenizador, quando difere do modelo (docs/metodos.md, seção 2)")
    parser.add_argument("--backend", default="torch", choices=["torch", "mlx"], help="mlx para modelos quantizados")
    parser.add_argument("--pacote", type=Path, default=PACOTE)
    parser.add_argument("--saida", type=Path, default=SAIDA)
    parser.add_argument("--pasta", type=Path, default=None, help="caminho exato da pasta de saída (sem data); tem precedência sobre --saida")
    args = parser.parse_args(argv)
    if args.chat_template and args.backend == "mlx":
        parser.error("--chat-template só vale com o backend torch")

    # Requisitos antes da inferência (que leva minutos a horas): dados e destino livre.
    if "bergum" in args.conjuntos and not (args.pacote / "code_snippets").is_dir():
        parser.error(f"pacote dos autores não encontrado em {args.pacote}; rode make data")
    destino_previsto = args.pasta or args.saida / nome_pasta(date.today().isoformat(), args.modelo.split("/")[-1], args.conjuntos, args.limite, args.chat_template)
    if destino_previsto.exists():
        parser.error(f"{destino_previsto} já existe; resultados não são sobrescritos")

    import torch
    from huggingface_hub import HfApi
    from transformers import AutoModelForCausalLM, AutoTokenizer, __version__ as versao_transformers

    revisao = args.revision or HfApi().model_info(args.modelo).sha
    repo_tokenizador = args.tokenizador or args.modelo
    revisao_tokenizador = args.tokenizer_revision or (HfApi().model_info(repo_tokenizador).sha if args.tokenizador else revisao)
    tokenizador = AutoTokenizer.from_pretrained(repo_tokenizador, revision=revisao_tokenizador)
    if args.backend == "mlx":
        from mlx_lm import load

        import mlx.core as mx

        dispositivo = "cpu"  # os tensores de entrada ficam na CPU; o MLX usa a GPU por conta própria
        inferencia = str(mx.default_device())
        modelo = ModeloMLX(load(args.modelo, revision=revisao)[0])
    else:
        dispositivo = "mps" if torch.backends.mps.is_available() else "cpu"
        modelo = AutoModelForCausalLM.from_pretrained(args.modelo, revision=revisao, dtype=getattr(torch, args.dtype))
        modelo.to(dispositivo).eval()
        inferencia = dispositivo

    from aoc_perplexity import gopstein

    por_conjunto, segundos = {}, {}
    for conjunto in args.conjuntos:
        snippets = ler_snippets(args.pacote / "code_snippets") if conjunto == "bergum" else gopstein.ler_snippets()
        if args.limite:
            snippets = dict(list(snippets.items())[: args.limite])
        inicio = time.perf_counter()
        por_conjunto[conjunto] = {a: perplexidade_por_token(c, modelo, tokenizador, dispositivo, args.uma_passagem, args.chat_template) for a, c in snippets.items()}
        segundos[conjunto] = round(time.perf_counter() - inicio, 1)
    nossos = por_conjunto.get("bergum", {})

    nome_curto = args.modelo.split("/")[-1]
    resumo = None
    comparacao = None
    publicado = args.pacote / "model_results" / (args.publicado or nome_curto) / "perplexity_results.json"
    if nossos and publicado.exists():
        comparacao = comparar(nossos, json.loads(publicado.read_text()))
        resumo = resumir(comparacao)

    template = None
    if args.chat_template:
        partes = {partes_do_chat_template(tokenizador, c) for conjunto in args.conjuntos for c in (r["code"] for r in por_conjunto[conjunto].values())}
        if len(partes) != 1:
            raise ValueError("o prefixo e o sufixo do chat template variam com o snippet")
        (prefixo, sufixo), = partes
        template = {
            "papel_da_mensagem": "user",
            "add_generation_prompt": False,
            "chat_template": tokenizador.chat_template,
            "prefixo": prefixo,
            "sufixo": sufixo,
            "tokens_do_prefixo": len(tokenizador(prefixo, add_special_tokens=False).input_ids),
            "nota": "só os tokens do snippet entram na perplexidade e na AOI; o sufixo vem depois do snippet e não é alimentado ao modelo",
        }

    # O registro vem antes da pasta de saída, que contaria como alteração.
    registro = {
        **proveniencia(),
        "config": {
            "pacote_commit": commit_externo(args.pacote),
            "dtype": "quantização do repositório" if args.backend == "mlx" else args.dtype,
            "backend": args.backend,
            "tokenizador": repo_tokenizador,
            "tokenizador_revisao": revisao_tokenizador,
            "uma_passagem": args.uma_passagem,
            "dispositivo_entrada": dispositivo,
            "dispositivo_inferencia": inferencia,
            "limite": args.limite,
            "torch": torch.__version__,
            "transformers": versao_transformers,
            "conjuntos": args.conjuntos,
            "arquivos": {c: len(r) for c, r in por_conjunto.items()},
            "segundos": segundos,
            "comparacao": resumo,
            **({"chat_template": template} if template else {}),
        },
        "seed": None,
        "model": args.modelo,
        "model_revision": revisao,
        "date": date.today().isoformat(),
    }

    with pasta_nova(destino_previsto) as destino:
        for conjunto, resultados in por_conjunto.items():
            nome = "perplexidade.json" if conjunto == "bergum" else f"perplexidade-{conjunto}.json"
            if conjunto == "gopstein":
                resultados = gopstein.sem_codigo(resultados)
            escrever_json(destino / nome, resultados)
        if comparacao is not None:
            comparacao.to_csv(destino / "comparacao.csv", index=False)
        escrever_json(destino / "run.json", registro)

    print(f"Segundos por conjunto: {segundos}")
    print(json.dumps(resumo, indent=2))
    print(f"Resultados em {destino_previsto}")
    return destino_previsto


if __name__ == "__main__":
    main()
