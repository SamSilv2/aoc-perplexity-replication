# Reprodução

Comandos, requisitos e critérios para regenerar os resultados do artigo e conferi-los com os publicados. O desenho do experimento está em [`metodos.md`](metodos.md); de onde vem cada resultado publicado, em [`proveniencia.md`](proveniencia.md).

## Requisitos

- `git`, `make`, `curl`, `shasum` e [`uv`](https://docs.astral.sh/uv/); o `uv` instala o Python 3.12 e as dependências fixadas em `uv.lock`. Os alvos do `Makefile` usam `uv --locked`, que recusa um lockfile divergente em vez de atualizá-lo; nos comandos diretos, use também `uv run --locked`.
- Rede para `make data`, `make tokenizadores` e a inferência. Nenhum dos dois repositórios de dados exige login.
- Conta no Hugging Face com a licença aceita em `meta-llama/Llama-2-7b-hf`, `meta-llama/Llama-2-13b-hf`, `meta-llama/CodeLlama-7b-hf`, `meta-llama/CodeLlama-7b-Instruct-hf`, `meta-llama/CodeLlama-13b-hf`, `meta-llama/CodeLlama-13b-Instruct-hf`, `google/gemma-7b`, `google/codegemma-7b` e `google/codegemma-7b-it`, e um token local (`hf auth login`). Vale também para o perfil de análise, que precisa dos tokenizadores desses modelos. A liberação do acesso pode demorar; peça antes.

| Perfil | Instalação | O que roda | Verificado em |
|---|---|---|---|
| análise | `make setup-analise` | replicação, trios, sensibilidade, consolidado, EEG, figuras e chat template, a partir das execuções salvas; não usa GPU | macOS arm64, Python 3.12 |
| inferência | `make setup` | o acima e os modelos (`perplexidade.py`: torch/MPS em fp16 e MLX em 4 bits) | macOS com Apple Silicon e 32 GiB, Python 3.12 |

O backend torch usa MPS ou CPU; CUDA não é selecionada nem foi testada. O MLX só funciona em Apple Silicon (Metal), e o trio Qwen2.5 32B usa MLX. Outros sistemas não foram verificados.

## Preparação

```bash
make setup-analise   # ou make setup, para a inferência
make data            # pacote de Abdelsalam et al. (git, commit fixado) e snippets de Gopstein et al. (SHA-256 conferido)
make tokenizadores   # tokenizadores de todas as execuções, nas revisões dos run.json
make verificar       # o que está disponível e o que dá para rodar
```

- `make data` baixa os snippets de C numa pasta temporária, confere o SHA-256 de cada arquivo e só então os move para `data/external/gopstein/`; qualquer falha termina com erro e não deixa arquivos. Para usar uma cópia local já obtida (por exemplo, quando o acesso passa pelos autores): `make data UPSTREAM_URL=/caminho/do/clone GOPSTEIN_URL=file:///caminho/da/pasta`; o commit e os hashes continuam conferidos.
- `make verificar` lista os dados e os tokenizadores que faltam, quais exigem licença, se torch e MLX importam e qual dispositivo o torch usa.

## Reprodução a partir das execuções salvas

```bash
make reproduzir
```

Sem rodar modelos, regenera tudo em `results/novos/reproducao-AAAA-MM-DD/` e compara arquivo a arquivo com o publicado:

| Etapa | Módulo | Saída na raiz | Comparada com |
|---|---|---|---|
| replicacao | `rq12`, `rq3`, `tabela_modelos` | `replicacao/` | `results/replicacao/2026-10-30-rq1-rq2`, `2026-11-02-rq3`, `2026-11-02-tabela-por-modelo` |
| trios | `trio` (36: 16 principais, 4 com chat template, 16 de subconjuntos de átomos) | `trios/` | a pasta de mesmo nome em `results/extensao/` |
| sensibilidade | `sensibilidade` | `sensibilidade-aoi-gopstein/` | `2026-11-05-sensibilidade-aoi-gopstein` |
| consolidado | `consolidar` | `consolidado/` | `2026-11-12-consolidado` |
| eeg | `eeg_extensao` | `eeg-java/` | `2026-11-12-eeg-java` |
| figuras | `figuras` | `figuras/` | `2026-11-11-figuras` |
| figuras_grade | `figuras_grade` | `figuras-grade/` | `2026-11-12-figuras` |
| template | `template_instruct` | `template-instruct/` | `2026-11-06-template-instruct` |

Cada etapa lê as entradas de `<raiz>/manifesto.json`, que começa igual a [`results/manifesto.json`](../results/manifesto.json) e passa a apontar para o que a própria reprodução gerou. Nenhum caminho precisa ser editado.

- **Só algumas etapas:** `make reproduzir ARGS="--etapas trios consolidado"`.
- **Outra execução no mesmo dia:** a raiz padrão já existe e é recusada; use `make reproduzir ARGS="--raiz results/novos/reproducao-2"`.
- **Só os comandos:** `make reproduzir ARGS="--mostrar"` (análise) ou `make reproduzir-inferencia ARGS="--mostrar"` (com as 31 execuções). A lista, com as revisões fixadas, é a receita de cada resultado.
- **Um módulo sozinho:** `uv run --locked python -m aoc_perplexity.<módulo> --help` mostra as opções, que variam por módulo. As análises aceitam `--manifesto` (entradas; padrão `results/manifesto.json`), `--saida` (pasta onde criar `AAAA-MM-DD-…`; padrão `results/novos/`) e `--pasta` (destino exato); `trio` e `perplexidade` recebem as execuções e o modelo por argumento. Exemplos: `uv run --locked python -m aoc_perplexity.consolidar --manifesto results/novos/reproducao-AAAA-MM-DD/manifesto.json` (análise) e `uv run --locked --extra inferencia python -m aoc_perplexity.perplexidade --modelo Qwen/Qwen2.5-3B --revision 3aab1f1954e9cc14eb9509a215f9e5ca08227a9b --dtype float16 --uma-passagem --conjuntos bergum gopstein` (inferência). Resultados publicados nunca são sobrescritos, e cada pasta só aparece no destino quando está completa, com o `run.json`.
- **Tempo:** cerca de 7 minutos num Mac com Apple Silicon.

### Critérios de comparação

A conferência (`aoc_perplexity.comparar`, também usável sozinha: `uv run --locked python -m aoc_perplexity.comparar PUBLICADA NOVA`) lê os valores como texto, sem arredondar, e classifica cada arquivo:

| Classe | Significado | Aceita |
|---|---|---|
| idêntico | mesmos bytes | sim |
| formatação | mesmos valores, escritos de outro jeito | sim |
| esquema | mesmos valores, com uma migração declarada: o cabeçalho `n` → `n_wilcoxon` em CSV, a chave `token_ids_sha256` acrescentada ao resumo em JSON | sim |
| numérico (tolerância) | valores diferentes, dentro da tolerância do tipo de arquivo | sim |
| numérico | valores além da tolerância, textos ou tipos diferentes (por exemplo `true` × `1`, `inf` × `-inf`) | não |
| difere | PDF, PNG ou LaTeX com bytes diferentes | não |
| não comparado | PDF, PNG ou LaTeX quando há inferência nova (as tabelas e os dados das figuras continuam comparados) | sim |
| metadados | `run.json` (data, commit, caminhos) | não comparado |
| ausente | arquivo ou pasta só de um lado, ou pasta sem arquivos de dados | não |

A tolerância padrão é zero. Resultado esperado de `make reproduzir`: tudo idêntico, salvo o `por_modelo.csv` do trio Qwen2.5 3B em Java, classificado como esquema (o arquivo publicado foi gerado antes de a coluna `n` ser renomeada; os valores são os mesmos).

### Figuras

A conferência binária dos PDFs e PNGs vale para o ambiente em que foram gerados: fonte STIX Two Text (a primeira instalada da lista `figuras.FONTES`; o `run.json` de cada pasta de figuras registra a usada), matplotlib 3.11.2 (fixado em `uv.lock`) e macOS arm64. Com outra fonte, os PDFs e PNGs mudam, mas os dados das figuras (`dados-delta.csv`, `dados-eeg-pares.csv`, `dados-grade.csv`) continuam idênticos; nesse caso, confira os dados e trate as renderizações como inspeção visual (`uv run --locked python -m aoc_perplexity.comparar PUBLICADA NOVA --sem-binarios`).

## Reprodução com inferência

```bash
make setup
make reproduzir-inferencia
```

Roda antes 31 execuções, com o modelo, a revisão, o tokenizador, a precisão e as opções de cada execução publicada: as 29 dos trios (24 modelos; o instruct do Qwen2.5 3B e do DeepSeek 7B também com chat template) e as 2 da reprodução (`deepseek-coder-1.3b-base` e `Qwen2.5-Coder-32B` em 4 bits). Depois segue com todas as etapas sobre as execuções novas; a tabela por modelo usa a reexecução nova do 1.3B, e a Tabela B.3 é recalculada pelo `analise.py` da pasta publicada do 32B.

- **Critério por token** (o da reprodução, fixado antes da reexecução do 1.3B): cada execução nova precisa ter os mesmos tokens da publicada e |Δ log-probabilidade| ≤ 0,01 em pelo menos 99% dos tokens (`--fracao-minima` muda a fração). A reexecução publicada do 1.3B não atingiu esse critério em relação ao JSON dos autores (84,5%, [`metodos.md`](metodos.md), seção 1); diferenças de hardware e de versão de kernel podem reprová-lo também.
- **Etapas derivadas:** comparadas só para informação, com a maior diferença de cada arquivo; PDFs, PNGs e LaTeX não são comparados. Estatísticas derivadas de log-probabilidades com pequenas diferenças não têm um critério binário próprio.
- **Memória:** os modelos rodam um por vez. O trio Llama-2 13B em fp16 precisou de 32 GiB de memória unificada (pico de swap de 10,7 GB, memória livre mínima de 10%); com menos memória, provavelmente não cabe. O Qwen2.5-Coder-32B em 4 bits levou cerca de 1 hora, com swap de 14 GB.
- **Disco:** os pesos vão para `data/cache-modelos/` (`HF_HUB_CACHE` só da inferência). O alvo usa `--limpar-cache`: os pesos de cada modelo são apagados desse cache depois da sua última execução, e basta espaço para o maior modelo (cerca de 27 GB). Sem limpar, os 25 modelos ocupam cerca de 400 GB. O fluxo exige espaço livre 10% acima da estimativa antes de começar e não toca no cache global do Hugging Face.
- **Tempo:** a inferência registrada nos `run.json` soma cerca de 4 horas num Mac com Apple Silicon, sem contar os downloads. O CodeLlama-13b levou cerca de 20 minutos em Java, 8 vezes os outros modelos do trio.
- **Não validado:** este fluxo não foi executado de ponta a ponta na versão pública; as execuções publicadas foram geradas pelos mesmos comandos, com versões anteriores do código ([`proveniencia.md`](proveniencia.md)).

## Mapa do artigo

`make paper` compila `paper/main.tex` em `paper/build/main.pdf`. Requer uma distribuição TeX (por exemplo, TeX Live ou MacTeX) com `latexmk`, `pdflatex`, `bibtex`, a classe `acmart` e TikZ/PGF. O artigo lê as figuras e tabelas das pastas publicadas por caminhos fixos; uma reprodução não muda o PDF. Para conferir a integração, compare os arquivos da tabela abaixo com os da raiz da reprodução.

| No artigo | Fonte | Como entra |
|---|---|---|
| Figura 1 | `results/extensao/2026-11-11-figuras/delta-aoi_max-grade.pdf` (dados em `dados-delta.csv`) | arquivo incluído |
| Tabela 1 | `results/extensao/2026-11-12-consolidado/tabela-contrastes-aoi_max-coluna.tex` | arquivo incluído |
| Figura A.1 | desenhada em `paper/anexos.tex` | — |
| Tabelas A.1 e A.2 (modelos e revisões) | `model` e `model_revision` dos `run.json` das execuções | transcrita |
| Tabela B.1 (RQ1 a RQ3) | `results/replicacao/2026-10-30-rq1-rq2/` (`rq1.csv`, `rq2.csv`) e `2026-11-02-rq3/correlacoes.csv` | transcrita |
| Tabela B.2 (reexecução do 1.3B) | `results/replicacao/2026-11-02-tabela-por-modelo/tabela.csv` (medianas) e `2026-11-02-diagnostico-precisao/saida.txt` (W, p, r) | transcrita |
| Tabela B.3 (32B em 4 bits) | `results/replicacao/2026-11-02-reexecucao-Qwen2.5-Coder-32B-4bit-completo/saida.txt` | transcrita |
| Tabela C.1 (`aoi_max`) | `results/extensao/2026-11-12-consolidado/tabela-aoi_max.tex` | arquivo incluído |
| Tabela C.2 (`snippet_media`) | `results/extensao/2026-11-12-consolidado/tabela-snippet_media.tex` | arquivo incluído |
| Tabela C.3 (EEG) | `results/extensao/2026-11-12-eeg-java/tabela-correlacoes.tex` | arquivo incluído |
| Figuras C.1 e C.2 (grades EEG) | `results/extensao/2026-11-12-figuras/eeg-grade-llama2.pdf` e `eeg-grade-qwen25.pdf` | arquivo incluído |
| Sensibilidade à anotação (Seção 5) | `results/extensao/2026-11-05-sensibilidade-aoi-gopstein/` | números no texto |
| Átomos confirmados (Seção 5) | `results/extensao/2026-11-05-trio-*-atomos-confirmados*` | números no texto |
| Chat template (Seção 5) | `results/extensao/2026-11-06-template-instruct/` | números no texto |

## Testes

```bash
make test           # tudo; o que depende de algo ausente é pulado, com o motivo
make test-unidade   # só o que não precisa de dados externos, tokenizadores nem torch
```

Marcadores: `dados` (pacote dos autores e snippets de C), `tokenizador` (todos os tokenizadores das execuções no cache local, de `make tokenizadores`) e `inferencia` (torch). Um teste marcado é pulado quando o pré-requisito falta; os testes não acessam a rede. Contagens verificadas em macOS arm64:

| Perfil | Aprovados | Pulados |
|---|---|---|
| análise, sem `make data` | 123 | 30 |
| análise, com `make data`, sem tokenizadores (cache frio) | 134 | 19 |
| análise, com `make data` e tokenizadores | 151 | 2 (torch) |
| inferência, com `make data`, sem tokenizadores (cache frio) | 150 | 19 |
| inferência, com `make data` e tokenizadores | 169 | 0 |

`make test-unidade` roda 123 testes (2 pulados sem torch) e deixa de fora os 28 marcados.

Os testes de `inferencia` usam modelos simulados e tokenizadores reais; não validam a inferência dos modelos.

## Auxiliares fora do fluxo

Registram como algumas pastas foram geradas e rodam a partir da raiz do repositório, depois de `make data`:

- `docs/diagnostico-gemma/referencia_fp32.py`: a referência do diagnóstico do Gemma, com a receita no README da pasta.
- `results/replicacao/2026-11-02-diagnostico-precisao/analise.py` e `2026-11-02-diagnostico-mlx/analise.py`: precisão do 1.3B e adaptador MLX.
- `results/extensao/2026-11-02-validacao-aoi/analise.py` e `validacao-diff/analise.py`: mapeamento das AOIs em tokens e AOI por diff, com as revisões dos tokenizadores fixadas; o segundo grava em `results/novos/validacao-diff/` (ou na pasta passada como argumento).
- A derivação da anotação v2.1 está em `data/anotacoes/gopstein-v2.1/changelog-v2-para-v2.1.csv`, mudança a mudança, com a regra aplicada; não é etapa executável.
