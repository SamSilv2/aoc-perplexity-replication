# Regras da AOI v2.1 (escritas antes das checagens, 2026-11-04)

Cada regra vale para os 126 snippets, inclusive os `ambas`. Toda mudança entra em `changelog-v2-para-v2.1.csv`.

## Item 1. Tamanho e minimalidade

- **R1 (tamanho).** O tamanho da AOI é o número de tokens que tocam o intervalo de caracteres, com a mesma função `para_tokens` do repositório. Ele é medido com dois tokenizadores (Qwen2.5-Coder-7B e Llama-2-7b-hf), nos dois lados, em v1, v2 e v2.1.
- **R2 (razão).** Para cada par, calcula-se n_limpo / n_confuso com o tokenizador do Qwen. O par é marcado se a razão for maior que 2 ou menor que 0,5 em pelo menos um dos dois tokenizadores. A marcação não muda nada sozinha: o caso passa pelos testes R3 e R4 e, se continuar grande, recebe uma justificativa em `observacao`.
- **R3 (minimalidade, confuso).** Na borda da AOI, um comando inteiro ou uma subexpressão inteira sai quando, sem ele, o trecho ainda contém todos os tokens do átomo e o cálculo que o átomo altera. São três os casos:
  - **R3a.** Uma declaração ou um alvo de atribuição no início (`int V2 = `, `V2 = `) sai quando não faz parte do cálculo. Exceções:
    - Type conversion, porque o tipo declarado é o que provoca a conversão;
    - os átomos de comando inteiro (Dead, Curly braces, Indentation e Re-purposed), em que o comando é o próprio átomo.
  - **R3b (nível igual nas duas pontas).** Se a AOI começa no nível de expressão, ela também termina no nível de expressão. O `;` final de um comando-expressão sai.
  - **R3c.** Um `{` final que abre um corpo deixado fora da AOI sai.
- **R4 (minimalidade, limpo).** A AOI do limpo não vai além do equivalente ao trecho confuso.
  - Na borda, sai um comando sem correspondente no trecho confuso.
  - Saem também os pedaços que correspondem a algo que, no confuso, fica fora da AOI, como o alvo da atribuição.
  - R3a, R3b e R3c valem também para o limpo.
  - Pelo princípio já aprovado, as palavras de controle que são o equivalente do operador confuso ficam na AOI: `if`/`else` em lugar de `?:`, `&&` e `||`.
- **R5 (o que fazer).** Um caso marcado é reduzido ao menor trecho contíguo que satisfaz a regra do átomo. Se não der para reduzir sem perder parte do cálculo, o trecho maior fica e a razão vai para `observacao`.
- **R6 (padrão por átomo).** Os pares v0, v1 e v2 do mesmo átomo seguem o mesmo padrão. Se a estrutura de uma versão obrigar a uma regra diferente, vale para as três a opção que funciona em todas. Se nenhuma funcionar nas três, o desvio fica documentado.

## Item 2. Código morto

- **R7.** Para a métrica principal, há duas opções:
  - **(A)** excluir o par do aoi_max, com n = 60;
  - **(B)** usar como AOI do limpo o comando equivalente que sobra, com n = 63.

  A escolha é a opção em que o ℓ do par compara duas versões do mesmo cálculo. A AOI do confuso é o trecho que não existe no limpo (R3: o que torna o trecho morto também existe no limpo e fica fora).

## Item 3. Lista de revisão

- **R8.** Cada um dos 20 casos é resolvido pela regra escrita do átomo e por R3 a R6. O desfecho fica em `revisao`:
  - `mantido`;
  - `alterado (item)`;
  - `ambiguo`: vale a opção mais curta que respeita a regra, e a outra fica em `alternativa`.

## Item 4. Sensibilidade

- **R9.** O cálculo usa `aoc_perplexity.trio`, sem alteração e com as mesmas sementes (10.000 reamostragens, semente 42), sobre os `perplexidade-gopstein.json` publicados, em três anotações:
  - v1: igual à publicada, e a reprodução do `trio.csv` publicado precisa dar o mesmo resultado;
  - v2;
  - v2.1.

  Nos pares excluídos, o aoi_max usa 60 pares e o snippet_media usa 63.
- **R10.** Uma conclusão muda quando o D troca de sinal ou quando o IC passa a excluir ou a incluir o zero. Uma mudança é atribuída ao tamanho quando:
  - a correlação de Spearman entre os pares, de Δℓ com Δ ln(n_confuso / n_limpo), é |ρ| ≥ 0,5;
  - e a conclusão volta a ser a da v1 quando se removem os pares cuja razão de tamanho mudou mais de 2 vezes.
