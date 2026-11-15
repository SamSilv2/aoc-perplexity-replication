# aoc-perplexity-replication

Pacote de replicação do artigo *Code Specialization, Instruction Tuning, and Atoms of Confusion: Comparing Perplexity Across General-Purpose Base, Code-Specialized Base, and Instruct Models* ([PDF](paper/aoc-perplexity.pdf); fonte LaTeX em [`paper/`](paper/)).

Átomos de confusão são trechos de código válidos, mas difíceis de entender. Abdelsalam et al. (ICSE 2026) mostraram que a perplexidade de LLMs de código é maior na região do átomo (a área de interesse, AOI) e se associa a respostas cerebrais de programadores (o FRP, medido por EEG). Este pacote:

1. **reproduz** parte daquele estudo a partir do pacote dos autores: RQ1 a RQ3 do Qwen2.5-Coder-32B e reexecução do `deepseek-coder-1.3b-base`;
2. **estende** a análise a 8 trios de modelos relacionados (base geral, base de código e instruct de código) de 5 famílias, em Java (Bergum et al.) e em C (Gopstein et al.), e pergunta se a especialização em código e o ajuste para instruções mudam o efeito do átomo na perplexidade.

## Uso mínimo

Requisitos: `git`, `make`, `curl`, `shasum` e [`uv`](https://docs.astral.sh/uv/) (que instala o Python 3.12). A análise não usa GPU e foi verificada em macOS arm64; a inferência, em macOS com Apple Silicon e 32 GiB. Os tokenizadores de Llama-2, CodeLlama e Gemma exigem aceitar a licença no Hugging Face e `hf auth login`.

```bash
git clone https://github.com/SamSilv2/aoc-perplexity-replication.git
cd aoc-perplexity-replication
make setup-analise   # dependências da análise, nas versões de uv.lock
make data            # pacote dos autores e snippets de C, nos commits fixados
make tokenizadores   # tokenizadores nas revisões registradas
make verificar       # mostra o que falta e o que dá para rodar
make reproduzir      # regenera os resultados a partir das execuções salvas e compara com os publicados
```

Resultado esperado de `make reproduzir`: as 45 pastas regeneradas (replicação, 36 trios, sensibilidade, consolidado, EEG, figuras e chat template) conferem com as publicadas, em `results/novos/reproducao-AAAA-MM-DD/`, em cerca de 7 minutos. Para rodar também os modelos, os requisitos de memória e disco e o mapa entre o artigo e os arquivos, veja [`docs/reproducao.md`](docs/reproducao.md).

## Documentação

| | |
|---|---|
| [`docs/reproducao.md`](docs/reproducao.md) | requisitos, comandos, critérios de comparação, inferência, mapa figura/tabela → arquivo, testes |
| [`docs/metodos.md`](docs/metodos.md) | desenho do experimento: modelos, dados, métricas, estatística, verificações e limitações |
| [`docs/proveniencia.md`](docs/proveniencia.md) | o que os `run.json` registram e a relação dos resultados com o código desta versão |
| [`data/README.md`](data/README.md) | fontes, versões e termos dos dados, inclusive o material de terceiros distribuído aqui |
| [`results/README.md`](results/README.md) | organização dos resultados e o manifesto que liga as etapas |

## Licença

| O quê | Licença |
|---|---|
| Código (`src/`, `tests/`, `Makefile`, `pyproject.toml`, `analise.py` e `referencia_fp32.py`) | MIT ([`LICENSE`](LICENSE)) |
| Documentação, resultados e anotações (`docs/`, `results/`, `data/anotacoes/`, READMEs) | CC BY 4.0 ([`LICENSE-CC-BY-4.0.md`](LICENSE-CC-BY-4.0.md)) |
| `results/**/perplexidade.json` das execuções em Java | CC BY-SA 4.0, porque contêm os snippets de Bergum et al. |
| Trechos de snippets de C de Gopstein et al. nas anotações | material de terceiros, distribuído com atribuição |
| `paper/` | direitos dos autores; não coberto pelas licenças acima |

O material de terceiros está descrito em [`data/README.md`](data/README.md).

## Como citar

Veja [`CITATION.cff`](CITATION.cff).
