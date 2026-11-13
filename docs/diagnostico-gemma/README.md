# Diagnóstico do Gemma em fp16

A log-prob da posição 0 do `codegemma-7b-it` é ≈ −395 nos 270 arquivos (nos outros dois modelos fica entre −10 e −20). O diagnóstico testa se o valor vem do arredondamento do fp16 ([`../metodos.md`](../metodos.md), seção 4).

## Como foi medido
- Entradas: as execuções da amostra (fp16 refeito, bf16 e referência) não são distribuídas; para repetir o diagnóstico, regenere-as com os comandos abaixo. Ficam os CSVs da comparação.
- Amostra: os 10 primeiros arquivos de cada conjunto (Java e C), 20 snippets por modelo, uma passagem, MPS.
- **fp16** e **bf16**: `perplexidade.py` com `--dtype float16` ou `--dtype bfloat16`, `--limite 10`, `--uma-passagem`, `--conjuntos bergum gopstein` e a revisão da execução vigente (`--revision`; tabela abaixo). O fp16 refeito é idêntico ao salvo (diferença 0,000 nos três modelos).
- **Referência fp32**: o modelo em fp32 não cabe em 32 GiB (cerca de 34 GB de pesos). A referência usa os pesos bf16 do repositório (o checkpoint já é bf16) com as camadas lineares (inclusive a saída) e o embedding, com o fator de escala, calculados em fp32; as demais operações não foram forçadas a fp32. Mesmo tokenizador, BOS acrescentado pelo código. Script: [`referencia_fp32.py`](referencia_fp32.py) (`MODELO REVISAO LIMITE SAIDA`).
- **Comparação**: `uv run python -m aoc_perplexity.precisao --referencia A --nova B --saida C.csv`, nos arquivos da execução menor; `tests/test_precisao.py`. Os CSVs por arquivo estão nesta pasta (`<modelo>-<referência>-vs-<nova>.csv`).
- Colunas: média e máximo de |Δ log-prob| nas posições > 0, por arquivo e depois entre os arquivos; variação do Max AOI Perplexity em |Δln| (0,05 ≈ 5 %).

| modelo | revisão |
|---|---|
| `google/gemma-7b` | `ff6768d9368919a1f025a54f9f5aa0ee591730bb` |
| `google/codegemma-7b` | `2ec970089fe8f083d10d1ce6bac30538ac4badaa` |
| `google/codegemma-7b-it` | `078cdc51070553d1636d645c9a238f3b0914459a` |

## fp16 salvo contra bf16
| modelo | conjunto | pos. 0 fp16 | pos. 0 bf16 | média \|Δ\| | máx. \|Δ\| | AOI máx., mediana \|Δln\| | AOI máx., máx. \|Δln\| |
|---|---|---|---|---|---|---|---|
| gemma-7b | bergum | -14,8 | -9,5 | 0,306 | 3,25 | 0,142 | 2,022 |
| gemma-7b | gopstein | -15,3 | -10,4 | 0,130 | 2,08 | 0,416 | 1,227 |
| codegemma-7b | bergum | -10,1 | -9,9 | 0,178 | 1,23 | 0,066 | 0,224 |
| codegemma-7b | gopstein | -12,1 | -11,9 | 0,163 | 1,62 | 0,246 | 0,539 |
| codegemma-7b-it | bergum | -394,5 | -386,0 | 0,327 | 1,24 | 0,442 | 0,736 |
| codegemma-7b-it | gopstein | -397,0 | -388,0 | 0,254 | 1,67 | 0,477 | 1,460 |

## Cada precisão contra a referência
fp16:

| modelo | conjunto | pos. 0 fp32 | pos. 0 fp16 | média \|Δ\| | máx. \|Δ\| | AOI máx., máx. \|Δln\| |
|---|---|---|---|---|---|---|
| gemma-7b | bergum | -12,8 | -14,8 | 0,029 | 0,31 | 0,071 |
| gemma-7b | gopstein | -13,2 | -15,3 | 0,016 | 0,41 | 0,086 |
| codegemma-7b | bergum | -10,0 | -10,1 | 0,025 | 0,15 | 0,027 |
| codegemma-7b | gopstein | -12,0 | -12,1 | 0,017 | 0,16 | 0,079 |
| codegemma-7b-it | bergum | -395,2 | -394,5 | 0,045 | 0,22 | 0,153 |
| codegemma-7b-it | gopstein | -397,5 | -397,0 | 0,028 | 0,17 | 0,150 |

bf16:

| modelo | conjunto | pos. 0 fp32 | pos. 0 bf16 | média \|Δ\| | máx. \|Δ\| | AOI máx., máx. \|Δln\| |
|---|---|---|---|---|---|---|
| gemma-7b | bergum | -12,8 | -9,5 | 0,305 | 3,08 | 2,094 |
| gemma-7b | gopstein | -13,2 | -10,4 | 0,131 | 2,02 | 1,243 |
| codegemma-7b | bergum | -10,0 | -9,9 | 0,170 | 1,16 | 0,197 |
| codegemma-7b | gopstein | -12,0 | -11,9 | 0,165 | 1,66 | 0,543 |
| codegemma-7b-it | bergum | -395,2 | -386,0 | 0,326 | 1,17 | 0,817 |
| codegemma-7b-it | gopstein | -397,5 | -388,0 | 0,253 | 1,61 | 1,512 |

## Posição 0 nos 270 arquivos do fp16 salvo
| modelo | conjunto | arquivos | pos. 0, média |
|---|---|---|---|
| gemma-7b | bergum | 144 | -15,4 |
| gemma-7b | gopstein | 126 | -15,0 |
| codegemma-7b | bergum | 144 | -10,4 |
| codegemma-7b | gopstein | 126 | -11,9 |
| codegemma-7b-it | bergum | 144 | -394,6 |
| codegemma-7b-it | gopstein | 126 | -396,8 |


A AOI não inclui a posição 0 em nenhum arquivo (`aoi_com_pos0` = 0 nos 270 arquivos dos três modelos).

## Leitura
- **A posição 0 de −395 não muda com a precisão testada.** A referência dá −395,2 (Java) e −397,5 (C) no `codegemma-7b-it`; o fp16 dá −394,5 e −397,0 e o bf16, −386,0 e −388,0. Dentro desta amostra e desta implementação, o diagnóstico não sustenta atribuir o valor só ao arredondamento do fp16; não exclui outras causas numéricas, porque a referência não é um fp32 completo.
- **O fp16 é a precisão mais próxima da referência** nos três modelos: média de |Δ| entre 0,016 e 0,045 e máximo de 0,15 a 0,41, contra 0,13 a 0,33 e 1,2 a 3,1 no bf16. O bf16 tem menos bits de mantissa, o que já se viu com o deepseek (seção 1 de [`../metodos.md`](../metodos.md)).
- **Sem inf/nan em fp16**, então a regra de precisão ([`../metodos.md`](../metodos.md), seção 2) não manda trocar.
- No `gemma-7b` o fp16 erra a posição 0 em cerca de 2 nats (−14,8 contra −12,8 em Java); as demais posições ficam a 0,03 em média.
- Entre o fp16 salvo e o bf16, a média de |Δ| passa de 0,05 e a AOI máxima muda em todos os modelos (primeira tabela); a referência mostra que a maior parte desse desvio vem do bf16.

## Consequência
O fp16 e as execuções do Gemma ficam como estão, sem refazer em bf16. O `run.json` do Gemma continua sem `tokenizador` e `tokenizador_revisao` (o analisador usa o repositório do modelo, que é o tokenizador correto nos três).

## Efeito no `snippet_media`
Com −395 numa posição, o `snippet_media` do `codegemma-7b-it` é dominado pela posição 0: a mediana de Overall Perplexity do instruct fica em ≈ 6×10⁸ (Bergum, limpo) e ≈ 7×10³ (Gopstein, limpo), contra ≈ 15 e ≈ 4,5 na base. O Δ de `snippet_media` do instruct e os contrastes D, D_código e D_instrução dessa métrica não são comparáveis com o resto. A troca de precisão não corrige isso. O `aoi_max` não usa a posição 0 e não é afetado, a menos do ruído acima (fp16 contra a referência: |Δln| máximo de 0,09 no Gemma base e de 0,15 no instruct).

Resultado: o trio Gemma é relatado só com o `aoi_max` ([`../metodos.md`](../metodos.md), seção 4).
