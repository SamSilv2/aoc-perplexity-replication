# Resultados

- `replicacao/`: reprodução de Abdelsalam et al. (RQ1 a RQ3, tabela por modelo, reexecuções e diagnósticos de precisão).
- `extensao/`: execuções, trios, consolidado, EEG, figuras e verificações da extensão. O guia é [`extensao/INDICE.md`](extensao/INDICE.md).
- [`manifesto.json`](manifesto.json): as pastas que cada etapa usa (trios principais, trios com chat template, trios de subconjuntos de átomos, consolidado, EEG, sensibilidade, template, figuras e replicação). É a entrada padrão de todos os módulos e de `aoc_perplexity.reproduzir`.
- `novos/`: onde os módulos gravam resultados novos (fora do Git). As pastas publicadas não são sobrescritas.

Os comandos e os critérios de comparação estão em [`../docs/reproducao.md`](../docs/reproducao.md); de onde vem cada pasta, em [`../docs/proveniencia.md`](../docs/proveniencia.md).

## Pastas

Cada pasta tem um `run.json` com o commit, a configuração, a semente, o modelo e a revisão (campos em [`proveniencia.md`](../docs/proveniencia.md)). Os nomes seguem `AAAA-MM-DD-assunto`. `perplexidade.py` cria `AAAA-MM-DD-reexecucao-<modelo>-<conjuntos>`, com o sufixo `-bergum`, `-gopstein` ou `-bergum-gopstein`; algumas execuções mais antigas usam `-completo` ([`extensao/INDICE.md`](extensao/INDICE.md)). `bergum` é o conjunto de Java (Bergum et al.) e `gopstein`, o de C.

## Nomes dos contrastes

Os CSVs e o código usam nomes curtos; o artigo usa dois índices ([`docs/metodos.md`](../docs/metodos.md), seção 4). Os valores são os mesmos.

| CSV / código | Artigo | Definição |
|---|---|---|
| `D_codigo` | $D_{\mathrm{code},\mathrm{base}}$ | $\Delta_C - \Delta_B$ |
| `D_instrucao` | $D_{\mathrm{instr},\mathrm{code}}$ | $\Delta_I - \Delta_C$ |
| `D` | $D_{\mathrm{instr},\mathrm{base}}$ | $\Delta_I - \Delta_B$ |

As métricas são `aoi_max` (perplexidade máxima na AOI, principal) e `snippet_media` (perplexidade média do snippet, secundária).
