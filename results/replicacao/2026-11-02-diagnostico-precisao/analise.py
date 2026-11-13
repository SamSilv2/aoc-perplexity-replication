"""Diagnóstico da #5: compara fp16, fp32 e bf16 com o JSON publicado. Rode da raiz: uv run python results/replicacao/2026-11-02-diagnostico-precisao/analise.py"""
import json, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, "src")
from aoc_perplexity.rq12 import carregar_pares, wilcoxon_com_efeito
R = "results/replicacao/2026-11-02-reexecucao-deepseek-coder-1.3b-base-completo"
S = "results/replicacao/2026-11-02-diagnostico-precisao"
pub = json.load(open("data/external/llm-perplexity-AoC/model_results/deepseek-coder-1.3b-base/perplexity_results.json"))
c = pd.read_csv(f"{R}/comparacao.csv")
ulp = np.spacing(np.abs(c["log_prob_publicado"]).astype(np.float16)).astype(float)
c["ulps"] = c["diferenca"] / ulp
print("diferenca em ULPs do fp16: quantis", c["ulps"].quantile([.5,.9,.99,1]).round(2).to_dict())
print("fracao <= 1 ULP", (c["ulps"] <= 1.0001).mean().round(4), " <= 2 ULP", (c["ulps"] <= 2.0001).mean().round(4))
print("|log_prob| medio dos que passam de 0,01:", c.loc[c.diferenca>0.01, "log_prob_publicado"].abs().mean().round(2), "dos demais:", c.loc[c.diferenca<=0.01, "log_prob_publicado"].abs().mean().round(2))
print("spearman token a token", spearmanr(c.log_prob, c.log_prob_publicado).statistic)
# fp32 como referencia: quem fica mais perto?
for nome in ["float32", "bfloat16"]:
    import glob
    d = f"{S}/{nome}/comparacao.csv"
    o = pd.read_csv(d)
    print(nome, "vs publicado: mediana", o.diferenca.median(), "fracao<=0,01", round((o.diferenca<=0.01).mean(),4))
f32 = pd.read_csv(f"{S}/float32/comparacao.csv")
print("fp16 nosso vs fp32: mediana", (c.log_prob - f32.log_prob).abs().median(), " publicado vs fp32:", (c.log_prob_publicado - f32.log_prob).abs().median())
# estatistica no nivel do snippet
nosso = json.load(open(f"{R}/perplexidade.json"))
linhas = []
for k in nosso:
    for metrica in ["overall_perplexity", "max_perplexity"]:
        linhas.append({"arquivo": k, "metrica": metrica, "nosso": nosso[k]["summary"][metrica], "publicado": pub[k]["summary"][metrica]})
s = pd.DataFrame(linhas); s["rel"] = (s.nosso/s.publicado - 1).abs()
print("diferenca relativa no snippet: ", s.groupby("metrica")["rel"].describe()[["50%","max"]].to_string())
for fonte in ["nosso", "publicado"]:
    for metrica in ["overall_perplexity", "max_perplexity"]:
        v = s[s.metrica==metrica].set_index("arquivo")[fonte]
        bases = sorted({k.replace("-obf-","-X-").replace("-clean-","-X-") for k in v.index})
        conf = [v[b.replace("-X-","-obf-")] for b in bases]; limp = [v[b.replace("-X-","-clean-")] for b in bases]
        r = wilcoxon_com_efeito(conf, limp)
        print(f"RQ1 snippet {metrica:20} {fonte:9} n={r.n} W={r.w} p={r.p:.4f} r={r.r:.3f}")
