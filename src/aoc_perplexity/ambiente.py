"""Confere o ambiente e diz o que dá para rodar: Python, dados externos, tokenizadores em cache, torch, MLX e dispositivo.

    uv run --locked python -m aoc_perplexity.ambiente                          # diagnóstico
    uv run --locked python -m aoc_perplexity.ambiente --baixar-tokenizadores   # baixa os tokenizadores nas revisões fixadas (rede)
"""

import argparse
import importlib
import importlib.util
import json
import platform
import sys
from pathlib import Path

PACOTE = Path("data/external/llm-perplexity-AoC")
DADOS = {
    "snippets Java (Bergum et al.)": PACOTE / "code_snippets",
    "resultados publicados e FRP (Abdelsalam et al.)": PACOTE / "model_results",
    "snippets de C (Gopstein et al.)": Path("data/external/gopstein/code.csv"),
    "rótulos de átomo de C (Gopstein et al.)": Path("data/external/gopstein/code_tags.csv"),
}
# Repositórios que exigem aceitar a licença no Hugging Face e um token local (hf auth login), inclusive para o tokenizador.
COM_LICENCA = ("meta-llama/", "google/")


def dados_faltando():
    return [nome for nome, caminho in DADOS.items() if not caminho.exists()]


def tokenizadores_do_manifesto(manifesto=None):
    """(repositório, revisão) dos tokenizadores de todas as execuções do manifesto, sem repetição."""
    from aoc_perplexity.reproduzir import execucoes_do_manifesto
    from aoc_perplexity.resultados import Manifesto
    from aoc_perplexity.trio import repositorio_tokenizador

    vistos = []
    for execucao in execucoes_do_manifesto(manifesto or Manifesto.ler(), replicacao=True):
        chave = repositorio_tokenizador(json.loads((Path(execucao) / "run.json").read_text()))
        if chave not in vistos:
            vistos.append(chave)
    return vistos


def tokenizadores_faltando(manifesto=None):
    """Tokenizadores que não estão no cache local do Hugging Face (sem acessar a rede)."""
    from huggingface_hub import try_to_load_from_cache

    return [(repo, rev) for repo, rev in tokenizadores_do_manifesto(manifesto) if not isinstance(try_to_load_from_cache(repo, "tokenizer_config.json", revision=rev), str)]


def baixar_tokenizadores():
    from transformers import AutoTokenizer

    falhas = []
    for repo, rev in tokenizadores_do_manifesto():
        try:
            AutoTokenizer.from_pretrained(repo, revision=rev)
            print(f"ok     {repo}@{rev[:7]}")
        except OSError as erro:
            falhas.append(repo)
            dica = " (aceite a licença no Hugging Face e rode hf auth login)" if repo.startswith(COM_LICENCA) else ""
            print(f"falhou {repo}@{rev[:7]}{dica}: {str(erro).splitlines()[0]}")
    return 1 if falhas else 0


def backend(nome):
    """(importa?, detalhe) de um backend de inferência."""
    if importlib.util.find_spec(nome) is None:
        return False, "não instalado (make setup)"
    try:
        if nome == "torch":
            torch = importlib.import_module("torch")
            return True, "dispositivo " + ("mps" if torch.backends.mps.is_available() else "cpu (MPS indisponível; CUDA não é usada)")
        importlib.import_module("mlx.core")
        return True, "Metal disponível"
    except Exception as erro:  # o MLX falha ao importar sem Metal
        return False, f"instalado, mas não importa: {str(erro).splitlines()[0]}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baixar-tokenizadores", action="store_true")
    args = parser.parse_args(argv)
    if args.baixar_tokenizadores:
        return baixar_tokenizadores()

    linhas = [f"Python {platform.python_version()} ({sys.platform}, {platform.machine()})"]
    if sys.version_info[:2] != (3, 12):
        linhas.append("  aviso: o projeto fixa Python 3.12")
    for nome, caminho in DADOS.items():
        linhas.append(f"{'ok' if caminho.exists() else 'falta':6} {nome}: {caminho}")
    faltam_dados = dados_faltando()
    if faltam_dados:
        linhas.append("       rode make data (rede; nenhum dos dois repositórios exige login)")

    faltam_tok = tokenizadores_faltando()
    total = len(tokenizadores_do_manifesto())
    linhas.append(f"{total - len(faltam_tok)}/{total} tokenizadores das execuções no cache local")
    for repo, rev in faltam_tok:
        linhas.append(f"  falta {repo}@{rev[:7]}" + (" (exige aceitar a licença e hf auth login)" if repo.startswith(COM_LICENCA) else ""))
    if faltam_tok:
        linhas.append("       rode make tokenizadores (rede)")

    torch_ok, torch_det = backend("torch")
    mlx_ok, mlx_det = backend("mlx")
    linhas.append(f"{'ok' if torch_ok else 'falta':6} torch: {torch_det}")
    linhas.append(f"{'ok' if mlx_ok else 'falta':6} mlx: {mlx_det}")

    analise = not faltam_dados and not faltam_tok
    mps = torch_ok and "mps" in torch_det
    linhas += [
        "",
        "make reproduzir: " + ("pode rodar" if analise else "ainda não (faltam " + " e ".join(p for p, f in [("dados", faltam_dados), ("tokenizadores", faltam_tok)] if f) + ")"),
        "make reproduzir-inferencia: "
        + ("pode rodar (torch com MPS e MLX com Metal)" if analise and mps and mlx_ok else "ainda não (precisa dos itens acima, torch com MPS e MLX com Metal: macOS com Apple Silicon)"),
        "Precisam de rede: make data, make tokenizadores e a inferência (pesos). Llama-2, CodeLlama e Gemma exigem aceitar a licença no Hugging Face.",
    ]
    print("\n".join(linhas))
    return 0


if __name__ == "__main__":
    sys.exit(main())
