# Dados

## Externos (baixados por `make data`, fora do Git)

| Pasta | Conteúdo | Fonte e versão | Condições |
|---|---|---|---|
| `data/external/llm-perplexity-AoC/` | snippets Java de Bergum et al., AOIs manuais, FRP agregado por par e resultados publicados dos 6 modelos de Abdelsalam et al. | [brains-on-code/llm-perplexity-AoC](https://github.com/brains-on-code/llm-perplexity-AoC), commit `720e1bfb7b79d54f8834d4670ab8e5d80d44f84d` | CC BY-SA 4.0; não é redistribuído aqui |
| `data/external/gopstein/` | `code.csv` e `code_tags.csv`: os 126 snippets em C e os rótulos de átomo de Gopstein et al. (FSE 2017) | [dgopstein/atoms-of-confusion](https://github.com/dgopstein/atoms-of-confusion), `snippet_study/data/`, commit `7aed29b562fc8e9bfdab59800572555dea3a51d4`; SHA-256 conferido pelo `Makefile` | o repositório não declara licença; os arquivos não são redistribuídos aqui |

Mesmo sem rodar modelos, as análises precisam desses dados: os snippets Java e o EEG estão só no pacote dos autores, e os resultados de C (`perplexidade-gopstein.json`) são gravados sem o código, que a análise recoloca a partir do download.

Os pesos e tokenizadores dos modelos vêm do Hugging Face, nas revisões registradas nos `run.json`; cada modelo tem a sua licença ([`../docs/metodos.md`](../docs/metodos.md), seção 2).

## Material de terceiros distribuído neste pacote

| Onde | O que contém | Origem e termos |
|---|---|---|
| `results/**/perplexidade.json` (execuções em Java) | o código de cada snippet Java e o texto dos tokens | snippets de Bergum et al., do pacote de Abdelsalam et al., sob CC BY-SA 4.0; esses arquivos são distribuídos sob a mesma licença, com esta atribuição |
| `results/**/perplexidade-gopstein.json` (execuções em C) | só posições, log-probabilidades e o hash dos IDs dos tokens, sem código | — |
| `data/anotacoes/*.csv`, `data/anotacoes/gopstein-v2.1/changelog-v2-para-v2.1.csv` e `relatorio-v2.1.md` | trechos dos snippets de C (as AOIs) | Gopstein et al., FSE 2017, commit `7aed29b`; o repositório original não declara licença; distribuídos com atribuição, para que as anotações possam ser conferidas |
| `data/anotacoes/gopstein-v2.1/gopstein-aoi-v2.1-conferencia.html` | os 126 snippets de C inteiros, com a AOI destacada | idem |

## Deste estudo

- [`anotacoes/`](anotacoes/): AOIs dos snippets de C (v1, v2 e a principal, v2.1) e a correspondência entre os átomos da base e os do artigo original.

## Dados pessoais

A análise usa só o FRP agregado publicado pelos autores (um valor médio por snippet, sem dados individuais de participantes). Nenhum arquivo deste pacote contém dados pessoais.
