# Anotações

AOI dos 126 snippets de Gopstein et al. (FSE 2017), feitas para este estudo. Critério de Bergum et al.: no confuso, o menor trecho que contém o átomo; no limpo, o trecho equivalente. Processo e uso na análise: [`docs/metodos.md`](../../docs/metodos.md), seção 3. Os arquivos trazem trechos dos snippets originais ([`../README.md`](../README.md)).

- `gopstein-aoi-v2.1.csv`: **anotação principal** (v2.1), por intervalo de caracteres (`inicio`, `fim`, fim exclusivo) e `sha256_snippet` (16 primeiros caracteres hexadecimais do SHA-256 do snippet em UTF-8; se divergir do snippet baixado, a análise para com erro). `na_metrica_aoi = nao` marca os 3 pares de código morto (átomo 17), que saem do `aoi_max` e ficam no `snippet_media`. `revisao` e `alternativa` registram os 6 casos ambíguos. Os demais campos estão na seção 6 do relatório.
- `gopstein-aoi-v2.csv`: a anotação v2 (segunda anotação independente, com a regra de cada átomo). Só sensibilidade. Mesmo formato da v2.1.
- `gopstein-aoi.csv`: a v1, anotação preliminar por LLM, por ID de `code.csv`, `trecho` e `ocorrencia` (0 = primeira aparição). Só sensibilidade.
- `gopstein-atomos-artigo.csv`: correspondência entre os 21 átomos da base e os candidatos testados no artigo original, com o status (confirmado, não confirmado ou indentação).
- `gopstein-v2.1/`: como a v2.1 foi obtida da v2: `regras-v2.1.md` (regras R1 a R10, escritas antes das checagens), `changelog-v2-para-v2.1.csv` (cada mudança, com a regra aplicada), `relatorio-v2.1.md` (tamanhos, casos marcados, processo de anotação e ambiguidades) e `gopstein-aoi-v2.1-conferencia.html` (página para conferir cada AOI no snippet).

A análise lê a anotação com `python -m aoc_perplexity.trio --anotacao {v1,v2,v2.1}` (padrão v2.1). Revisão humana: exemplos de 26 pares, validados por um dos autores (seção 7 do relatório).
