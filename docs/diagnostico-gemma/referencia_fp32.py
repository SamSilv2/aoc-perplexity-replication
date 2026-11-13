"""Referência do diagnóstico de precisão do Gemma: pesos bf16 do repositório (o checkpoint é bf16), com as camadas lineares
e o embedding calculados em fp32. Não é uma conversão do modelo inteiro para fp32: as demais operações não são forçadas a fp32.

    uv run --extra inferencia python docs/diagnostico-gemma/referencia_fp32.py MODELO REVISAO LIMITE SAIDA

Ex.: ... google/codegemma-7b-it 078cdc5... 10 /tmp/gemma (10 snippets de Java e 10 de C; revisões no README desta pasta).
"""
import json, sys, time
from pathlib import Path
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from aoc_perplexity import gopstein
from aoc_perplexity.perplexidade import PACOTE, ler_snippets, perplexidade_por_token

modelo_id, rev, limite, saida = sys.argv[1], sys.argv[2], int(sys.argv[3]), Path(sys.argv[4])
tok = AutoTokenizer.from_pretrained(modelo_id, revision=rev)
m = AutoModelForCausalLM.from_pretrained(modelo_id, revision=rev, dtype=torch.bfloat16).to("mps").eval()
print("dtype dos pesos:", {p.dtype for p in m.parameters()})

def linear_fp32(self, x):
    return F.linear(x.float(), self.weight.float(), None if self.bias is None else self.bias.float())
def emb_fp32(self, x):
    saida = F.embedding(x, self.weight.float())
    return saida * self.embed_scale.float() if hasattr(self, 'embed_scale') else saida
for mod in m.modules():
    if isinstance(mod, torch.nn.Linear):
        mod.forward = linear_fp32.__get__(mod)
    elif isinstance(mod, torch.nn.Embedding):
        mod.forward = emb_fp32.__get__(mod)
# lm_head é amarrado ao embedding no Gemma; fica em Linear acima.
saidas, seg = {}, {}
for conjunto in ("bergum", "gopstein"):
    sn = ler_snippets(PACOTE / "code_snippets") if conjunto == "bergum" else gopstein.ler_snippets()
    sn = dict(list(sn.items())[:limite])
    t = time.perf_counter()
    saidas[conjunto] = {a: perplexidade_por_token(c, m, tok, "mps", True) for a, c in sn.items()}
    seg[conjunto] = round(time.perf_counter() - t, 1)
destino = saida / f"{modelo_id.split('/')[-1]}-fp32-pesos-bf16"
destino.mkdir(parents=True)
for c, r in saidas.items():
    if c == "gopstein":
        r = gopstein.sem_codigo(r)
    (destino / ("perplexidade.json" if c == "bergum" else f"perplexidade-{c}.json")).write_text(json.dumps(r, indent=2, ensure_ascii=False) + "\n")
(destino / "run.json").write_text(json.dumps({"model": modelo_id, "model_revision": rev, "config": {"conjuntos": ["bergum", "gopstein"], "dtype": "float32 (pesos bf16)", "segundos": seg}}, indent=2) + "\n")
print("ok", destino, seg)
