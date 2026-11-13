# Proveniência

O que cada `run.json` registra, de onde vêm os insumos e por que os resultados publicados valem com o código desta versão.

## Campos do `run.json`

Toda pasta de resultado tem um `run.json` (a exceção é `results/extensao/validacao-diff/`, documentada no [índice](../results/extensao/INDICE.md)):

- `commit` e `codigo_alterado`: o commit do código que gerou a pasta e se havia alteração não versionada em `src`, `tests`, `pyproject.toml` ou `uv.lock`. Gerado fora de um clone Git, `commit` é nulo e `versao` identifica o pacote;
- `model` e `model_revision` (e `config.tokenizador` e `config.tokenizador_revisao`, quando o tokenizador é outro): o modelo e a revisão do Hugging Face;
- `config`: argumentos e entradas (pastas das execuções, anotação com SHA-256, número de pares, átomos, versões de torch e transformers, `pacote_commit` do pacote dos autores);
- `seed` e `date`.

Os `run.json` publicados são mantidos como foram gravados. Os valores de `commit` neles identificam versões de um repositório de trabalho que não é publicado; não são consultáveis aqui. Alguns registros citam decisões pelo número (por exemplo, `"decisao": "0020"`); a correspondência está no fim de [`metodos.md`](metodos.md).

## Insumos

| Insumo | Versão | Conferência |
|---|---|---|
| Pacote de Abdelsalam et al. (`brains-on-code/llm-perplexity-AoC`) | commit `720e1bfb7b79d54f8834d4670ab8e5d80d44f84d` | `make data` faz checkout desse commit; os `run.json` registram `pacote_commit` |
| Snippets de Gopstein et al. (`dgopstein/atoms-of-confusion`, `snippet_study/data/code.csv` e `code_tags.csv`) | commit `7aed29b562fc8e9bfdab59800572555dea3a51d4` | SHA-256 dos dois arquivos no `Makefile`; a análise confere ainda o hash de cada snippet anotado e o hash dos IDs dos tokens |
| Pesos e tokenizadores do Hugging Face | revisão em cada `run.json` | `trio.py --revision` recusa uma execução com outra revisão; `make tokenizadores` baixa as revisões registradas |

## Relação com o código desta versão

**Análises.** O código desta versão regenera, a partir das execuções salvas, todas as pastas que o artigo usa, com resultado idêntico ([`reproducao.md`](reproducao.md)); a única diferença é de esquema, no `por_modelo.csv` do trio Qwen2.5 3B em Java, que traz a coluna `n` com o nome antigo (hoje `n_wilcoxon`; os valores são os mesmos, e o arquivo publicado não foi alterado). A conferência se repete com `make reproduzir`.

**Execuções (log-probabilidades).** Há duas verificações, de naturezas diferentes:

- *Conferência histórica, feita antes da publicação.* Com acesso ao histórico interno, a função de pontuação de cada uma das 28 versões de `perplexidade.py` citadas nos `run.json` foi executada lado a lado com a desta versão, com um modelo simulado de logits determinísticos, o tokenizador real do Qwen2.5-Coder-3B e 12 snippets (6 de Java e 6 de C). Tokens, posições, log-probabilidades e resumos foram idênticos em todas as versões, no laço por token, na passagem única e, nas 6 versões que o têm, com chat template; a versão atual só acrescenta ao resumo o hash dos IDs dos tokens. A verificação não cobre o adaptador MLX, usado no trio Qwen2.5 32B, nem modelos reais. Ela não pode ser repetida com este pacote, porque depende das versões antigas.
- *Verificação disponível agora.* `make reproduzir-inferencia` roda os modelos com o código desta versão e aplica o critério por token a cada execução ([`reproducao.md`](reproducao.md)). Ainda não foi executada de ponta a ponta.

## `codigo_alterado: true`

Cinco pastas registram `codigo_alterado: true`: as execuções de 2026-11-02 do `Qwen2.5-Coder-3B`, do `Qwen2.5-Coder-3B-Instruct`, do `Qwen2.5-Coder-32B-4bit` e do `Qwen2.5-Coder-32B-Instruct-4bit`, e o trio `2026-11-02-trio-qwen2.5-3b-gopstein-aoi-manual`. Nessa versão, a flag contava qualquer arquivo não versionado na árvore, inclusive pastas de resultado recém-criadas; a versão atual só olha `src`, `tests`, `pyproject.toml` e `uv.lock`. Pelo histórico interno, não havia diferença de código entre o commit registrado e o commit que versionou essas pastas, e o padrão (o modelo base de cada trio com `false`, os seguintes com `true`) é o esperado de pastas criadas em sequência. Não há um `git status` gravado no momento das execuções; a explicação vem do histórico, não de um registro da época.
