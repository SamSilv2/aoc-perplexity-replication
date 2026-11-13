"""Compara o mapeamento de AOIs com os intervalos publicados em 4 tokenizadores.

Rode da raiz: uv run --locked python results/extensao/2026-11-02-validacao-aoi/analise.py  (imprime o que está em saida.txt)
As revisões são as que eram a main dos repositórios em 2026-11-02, quando a pasta foi gerada (o run.json registra "main na data").
"""
import json, copy
from transformers import AutoTokenizer
from aoc_perplexity.aoi import mapear
REVISOES = {
    "deepseek-ai/deepseek-coder-1.3b-base": "c919139c3a9b4070729c8b2cca4847ab29ca8d94",
    "Qwen/Qwen2.5-Coder-32B": "2e12b5f7bc878d424d222e224ed40aee564ec45f",
    "01-ai/Yi-Coder-9B": "bc6e63fd26a654ee42dba737a553310bfa01dc5d",
    "deepseek-ai/deepseek-coder-6.7b-base": "ce2207a8bfef3ee92bd7dd4cc31c52cfa0046912",
}
for modelo, repo in [("deepseek-coder-1.3b-base","deepseek-ai/deepseek-coder-1.3b-base"),("Qwen2.5-Coder-32B","Qwen/Qwen2.5-Coder-32B"),("Yi-Coder-9B","01-ai/Yi-Coder-9B"),("deepseek-coder-6.7b-base","deepseek-ai/deepseek-coder-6.7b-base")]:
    pub = json.load(open(f"data/external/llm-perplexity-AoC/model_results/{modelo}/perplexity_results_with_aois.json"))
    nosso = mapear({k: {"code": v["code"]} for k, v in pub.items()}, AutoTokenizer.from_pretrained(repo, revision=REVISOES[repo]))
    def norm(d): return {k: [list(s) for s in v] for k, v in d.items()}
    dif = [k for k in pub if norm(pub[k]["aoi_regions"]) != norm(nosso[k]["aoi_regions"])]
    print(modelo, "arquivos com AOI diferente:", len(dif), dif[:4])
    for k in dif[:2]: print("   pub", pub[k]["aoi_regions"], "nosso", nosso[k]["aoi_regions"])
