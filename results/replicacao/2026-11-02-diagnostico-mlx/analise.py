"""Compara o MLX (sem quantização e 4 bits) com o torch fp16 no Qwen2.5-Coder-1.5B, 20 arquivos.

Rode da raiz: uv run python results/replicacao/2026-11-02-diagnostico-mlx/analise.py
"""

import json
from pathlib import Path

import numpy as np

D = Path("results/replicacao/2026-11-02-diagnostico-mlx")


def log_probs(nome):
    dados = json.loads((D / nome / "perplexidade.json").read_text())
    return np.array([r["log_prob"] for v in dados.values() for r in v["results"]])


ref = log_probs("torch-fp16")
for nome in ["mlx-sem-quantizacao", "mlx-4bit"]:
    x = log_probs(nome)
    print(f"{nome}: mediana |Δ| {np.median(np.abs(x - ref)):.4f}, média com sinal {np.mean(x - ref):.4f}")
