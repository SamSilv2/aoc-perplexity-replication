# AOI v2.1 de Gopstein: relatório

2026-11-04. Entrada: a anotação v2 (`data/anotacoes/gopstein-aoi-v2.csv`). Comparação: a v1 (`data/anotacoes/gopstein-aoi.csv`, anotação preliminar).

- Saídas: `data/anotacoes/gopstein-aoi-v2.1.csv`, `changelog-v2-para-v2.1.csv` (cada mudança da v2 para a v2.1, com a regra), `gopstein-aoi-v2.1-conferencia.html` e este relatório.
- Não houve inferência: só os `perplexidade-gopstein.json` já publicados e os tokenizadores fixados nos `run.json`.

## 1. Regras (escritas antes das checagens)

Texto integral em `regras-v2.1.md`, gravado antes de qualquer medida. Resumo:

- **R1:** tamanho em tokens, com `para_tokens` e os tokenizadores do Qwen2.5-Coder-7B e do Llama-2-7b-hf.
- **R2:** um par é marcado quando a razão limpo/confuso passa de 2 ou fica abaixo de 0,5.
- **R3 (confuso):** sai da borda todo comando ou subexpressão inteira que não seja necessária para conter o átomo:
  - R3a: o alvo de atribuição que não faz parte do cálculo;
  - R3b: o `;` final, quando a AOI começa no nível de expressão;
  - R3c: o `{` de um corpo que fica fora da AOI.
- **R4 (limpo):** a AOI não vai além do equivalente ao trecho confuso. As palavras de controle que substituem o operador confuso ficam.
- **R5:** um caso marcado é reduzido ao menor trecho que satisfaz a regra. Se não houver redução possível, o caso é mantido e justificado.
- **R6:** v0, v1 e v2 de cada átomo seguem o mesmo padrão.
- **R7 (código morto):** vale a opção em que o ℓ compara versões do mesmo cálculo.
- **R8:** a lista de revisão é resolvida pela regra. Num caso ambíguo, fica a opção mínima.
- **R9 e R10:** sensibilidade e atribuição de uma mudança ao tamanho da AOI.

## 2. Tamanho das AOIs

Medianas sobre os 63 pares. Nos limpos de código morto, v2 e v2.1 não têm AOI. As unidades são caracteres (car) e tokens.

| versão | obf car | obf qwen | obf llama | clean car | clean qwen | clean llama | razão mediana (Qwen) | pares marcados (R2) |
|---|---|---|---|---|---|---|---|---|
| v1 | 10 | 6 | 7 | 9 | 6 | 6 | 1.00 | 4 |
| v2 | 16 | 10 | 10 | 21 | 13 | 14 | 1.14 | 20 |
| v2.1 | 12 | 8 | 8 | 15 | 10 | 10 | 1.00 | 10 |

Mediana por átomo, em tokens do Qwen:

| atomo | v1 obf | v1 clean | v2 obf | v2 clean | v2.1 obf | v2.1 clean |
|---|---|---|---|---|---|---|
| add_CONDITION_atom | 5.0 | 10.0 | 5.0 | 10.0 | 5.0 | 10.0 |
| add_PARENTHESIS_atom | 7.0 | 8.0 | 7.0 | 8.0 | 7.0 | 8.0 |
| move_POST_INC_DEC_atom | 3.0 | 3.0 | 6.0 | 30.0 | 6.0 | 5.0 |
| move_PRE_INC_DEC_atom | 3.0 | 3.0 | 6.0 | 14.0 | 6.0 | 4.0 |
| replace_CONSTANTVARIABLE_atom | 5.0 | 4.0 | 18.0 | 4.0 | 5.0 | 4.0 |
| remove_INDENTATION_atom | 3.0 | 3.0 | 22.0 | 22.0 | 22.0 | 22.0 |
| replace_MACRO_atom | 14.0 | 10.0 | 14.0 | 10.0 | 14.0 | 10.0 |
| replace_Mixed_Pointer_Integer_Arithmetic | 5.0 | 6.0 | 6.0 | 6.0 | 6.0 | 6.0 |
| replace_Ternary_Operator | 11.0 | 5.0 | 11.0 | 29.0 | 11.0 | 29.0 |
| replace_Arithmetic_As_Logic | 17.0 | 10.0 | 17.0 | 10.0 | 17.0 | 10.0 |
| replace_Comma_Operator | 13.0 | 7.0 | 13.0 | 18.0 | 13.0 | 18.0 |
| move_Preprocessor_Directives_Inside_Statements | 6.0 | 6.0 | 29.0 | 12.0 | 23.0 | 7.0 |
| Constant Assignment | 7.0 | 5.0 | 5.0 | 12.0 | 5.0 | 11.0 |
| Logic as Control Flow | 9.0 | 7.0 | 9.0 | 25.0 | 9.0 | 22.0 |
| Re-purposed variables | 16.0 | 11.0 | 41.0 | 39.0 | 41.0 | 37.0 |
| Swapped subscripts | 5.0 | 7.0 | 5.0 | 7.0 | 5.0 | 7.0 |
| Dead, unreachable, repeated | 6.0 | 6.0 | 13.0 | nan | 6.0 | nan |
| Literal encoding | 3.0 | 3.0 | 3.0 | 3.0 | 3.0 | 3.0 |
| Curly braces | 10.0 | 12.0 | 14.0 | 16.0 | 14.0 | 16.0 |
| Type conversion | 7.0 | 7.0 | 7.0 | 7.0 | 7.0 | 7.0 |
| Indentation | 6.0 | 6.0 | 25.0 | 25.0 | 25.0 | 25.0 |

Leitura:
- A v2.1 desfaz cerca de metade do aumento da v2.
- No confuso, a mediana fica em 8 tokens (v1: 6, v2: 10). No limpo, em 10 tokens (v1: 6, v2: 13).
- A razão mediana limpo/confuso volta a 1,0.
- O que continua acima da v1 é, em grande parte, deliberado. Nas versões novas, os átomos de estrutura (indentação, chaves, ternário, Logic as Control Flow, Re-purposed) usam a estrutura inteira, enquanto na v1 eles marcavam uma linha só.

## 3. Casos marcados e desfecho

### 3a. Razão de tamanho (R2)

São os pares marcados na v2, na v2.1 ou nas duas.

| par | átomo | razão v2 (Qwen/Llama) | razão v2.1 (Qwen/Llama) | desfecho |
|---|---|---|---|---|
| 3-v0 | move_POST_INC_DEC_atom | 2.60 / 2.80 | 1.00 / 1.00 | reduzida no limpo |
| 3-v1 | move_POST_INC_DEC_atom | 5.00 / 6.00 | 0.83 / 0.83 | reduzida no limpo |
| 3-v2 | move_POST_INC_DEC_atom | 5.00 / 6.17 | 0.83 / 0.83 | reduzida no limpo |
| 4-v0 | move_PRE_INC_DEC_atom | 2.33 / 2.50 | 0.67 / 0.67 | reduzida no limpo |
| 4-v1 | move_PRE_INC_DEC_atom | 2.17 / 2.50 | 0.83 / 0.83 | reduzida no limpo |
| 4-v2 | move_PRE_INC_DEC_atom | 2.33 / 2.50 | 0.67 / 0.67 | reduzida no limpo |
| 5-v0 | replace_CONSTANTVARIABLE_atom | 0.22 / 0.21 | 0.80 / 0.80 | reduzida no confuso |
| 5-v1 | replace_CONSTANTVARIABLE_atom | 0.22 / 0.21 | 1.00 / 1.00 | reduzida no confuso |
| 5-v2 | replace_CONSTANTVARIABLE_atom | 0.29 / 0.26 | 1.00 / 1.00 | reduzida no confuso |
| 9-v0 | replace_Ternary_Operator | 2.64 / 3.00 | 2.64 / 3.00 | mantida (R5): Razão > 2: o if/else é o equivalente de ?: (R4); as atribuições ficam no meio e não podem sair sem quebrar a continuidade. |
| 9-v1 | replace_Ternary_Operator | 2.21 / 2.56 | 2.21 / 2.56 | mantida (R5): Razão > 2: o if/else aninhado é o equivalente dos ?: aninhados (R4); nada sai da borda. |
| 9-v2 | replace_Ternary_Operator | 2.64 / 3.09 | 2.64 / 3.09 | mantida (R5): Razão > 2: o if/else é o equivalente de ?: (R4); nada sai da borda. |
| 12-v1 | move_Preprocessor_Directives_Inside_Statements | 0.14 / 0.13 | 0.17 / 0.17 | reduzida no confuso; mantida (R5): Razão < 0,5: as diretivas no meio da expressão são o próprio átomo; nada sai sem perdê-lo. No limpo, as diretivas estão fora do cálculo.  |
| 12-v2 | move_Preprocessor_Directives_Inside_Statements | 0.30 / 0.29 | 0.30 / 0.29 | mantida (R5): Razão < 0,5: o if/else é o que faz as diretivas parecerem condicionais; sem ele a AOI perde o átomo.  |
| 13-v0 | Constant Assignment | 2.40 / 2.60 | 2.20 / 2.40 | reduzida no limpo; mantida (R5): Razão > 2: o cálculo do limpo tem duas partes (atribuição e teste) em comandos separados; nenhuma sai sem perder o cálculo. |
| 13-v1 | Constant Assignment | 2.60 / 3.00 | 2.40 / 2.80 | reduzida no limpo; mantida (R5): Razão > 2: o cálculo do limpo tem duas partes (atribuição e teste) em comandos separados; nenhuma sai sem perder o cálculo. |
| 14-v0 | Logic as Control Flow | 5.00 / 5.71 | 4.86 / 5.57 | reduzida no limpo; mantida (R5): Razão > 2: o menor trecho contíguo do primeiro ao último teste passa pelo primeiro corpo (continuidade); a opção maior está em alternativa. |
| 14-v1 | Logic as Control Flow | 1.92 / 2.23 | 1.69 / 1.92 | reduzida no limpo |
| 14-v2 | Logic as Control Flow | 2.44 / 2.67 | 2.44 / 2.67 | mantida (R5): Razão > 2: o while com o if/break é o equivalente de && no teste do laço (R4); do primeiro teste ao último pedaço. |
| 20-v1 | Type conversion | 4.50 / 5.25 | 4.50 / 5.25 | mantida (R5): Razão > 2: a conversão explícita do limpo é condicional; tirar o teste ou um ramo perde a equivalência. |

### 3b. AOIs reduzidas por minimalidade (R3, R4) e padrão (R6): 24 snippets

`⏎` marca uma quebra de linha.

| snippet | v2 | v2.1 | regra |
|---|---|---|---|
| 3-clean-v0 | `V2 = V1 + 3;⏎   V1++;` | `V1 + 3` | 1c/1e R4 + R6: no limpo, a expressão que usa o valor; o comando de incremento separado só entra, em v1 e v2, junto com os corpos do if, que não fazem parte do cálculo; v0 segue o mesmo padrão. |
| 3-clean-v1 | `if (V1 == 0) {⏎      printf("true ");⏎   }⏎   else {⏎      printf("false ");⏎   }⏎   V1++;` | `V1 == 0` | 1c/1e/3 R4 + R6: no limpo, a expressão que usa o valor; o comando de incremento separado só entra, em v1 e v2, junto com os corpos do if, que não fazem parte do cálculo; v0 segue o mesmo padrão. |
| 3-clean-v2 | `if (V1 == 1) {⏎      printf("true ");⏎   }⏎   else {⏎      printf("false ");⏎   }⏎   V1--;` | `V1 == 1` | 1c/1e/3 R4 + R6: no limpo, a expressão que usa o valor; o comando de incremento separado só entra, em v1 e v2, junto com os corpos do if, que não fazem parte do cálculo; v0 segue o mesmo padrão. |
| 4-clean-v0 | `++V1;⏎   V2 = 5 - V1;` | `5 - V1` | 1c/1e R4 + R6: no limpo, a expressão que usa o valor; o comando de incremento separado só entra, em v1 e v2, junto com os corpos do if, que não fazem parte do cálculo; v0 segue o mesmo padrão. |
| 4-clean-v1 | `V1--;⏎   if (V1 == 1) {` | `V1 == 1` | 1c/1e R4 + R6: no limpo, a expressão que usa o valor; o comando de incremento separado só entra, em v1 e v2, junto com os corpos do if, que não fazem parte do cálculo; v0 segue o mesmo padrão. |
| 4-clean-v2 | `V2 = 9 - V1;⏎   --V1;` | `9 - V1` | 1c/1e R4 + R6: no limpo, a expressão que usa o valor; o comando de incremento separado só entra, em v1 e v2, junto com os corpos do if, que não fazem parte do cálculo; v0 segue o mesmo padrão. |
| 5-obf-v0 | `int V1 = 3;⏎   int V2 = V1 + 2;` | `V1 + 2` | 1b/1c/3 R3 + R5: a declaração da constante é um comando inteiro na borda; sem ela a AOI ainda contém o uso da constante, que é o átomo. |
| 5-obf-v1 | `int V1 = 2;⏎   int V2 = 2 * V1;` | `2 * V1` | 1b/1c/3 R3 + R5: a declaração da constante é um comando inteiro na borda; sem ela a AOI ainda contém o uso da constante, que é o átomo. |
| 5-obf-v2 | `int V1 = 2;⏎   printf("%f\n", 2.5 * V1);` | `2.5 * V1` | 1b/1c/3 R3 + R5: a declaração da constante é um comando inteiro na borda; sem ela a AOI ainda contém o uso da constante, que é o átomo. |
| 8-obf-v2 | `int *V2 = V1 + 1;⏎   V2 = V2 + 2;` | `V1 + 1;⏎   V2 = V2 + 2` | 1c R3a + R3b: o alvo da declaração não faz parte da conta de ponteiro. |
| 8-clean-v2 | `int *V2 = &V1[1];⏎   V2 = &V2[2];` | `&V1[1];⏎   V2 = &V2[2]` | 1c R4 (R3a + R3b). |
| 11-clean-v0 | `V1 = 2;⏎   V2 = 1;` | `2;⏎   V2 = 1` | 1c/3 R4: o alvo V1 fica fora da AOI no confuso; nível de expressão nas duas pontas. |
| 12-obf-v0 | `char *V1 = "abcdef"⏎   #define M1⏎   "abcdef";` | `"abcdef"⏎   #define M1⏎   "abcdef"` | 1c R3a + R3b: a diretiva está no inicializador; o alvo não faz parte do cálculo. |
| 12-clean-v0 | `char *V1 = "abcdef"⏎   "abcdef";` | `"abcdef"⏎   "abcdef"` | 1c R4 (R3a + R3b). |
| 12-obf-v1 | `int V2 = 1⏎   #define M1 1⏎   +⏎   #define M2 2⏎   V1;` | `1⏎   #define M1 1⏎   +⏎   #define M2 2⏎   V1` | 1c R3a + R3b: a diretiva está no inicializador; o alvo não faz parte do cálculo. |
| 11-clean-v1 | `V1 *= 2;⏎⏎   int V2 = (V1 += 1);` | `V1 *= 2;⏎⏎   int V2 = (V1 += 1)` | 1c/3 R4 (R3b): nível de expressão nas duas pontas, como o confuso. |
| 11-clean-v2 | `V1 += 1;⏎⏎   int V2 = (V1 *= 2);` | `V1 += 1;⏎⏎   int V2 = (V1 *= 2)` | 1c/3 R4 (R3b): nível de expressão nas duas pontas, como o confuso. |
| 13-clean-v0 | `V1 = 1;⏎   if (1) {` | `V1 = 1;⏎   if (1)` | 1c R4 (R3c): o '{' abre o corpo, que fica fora. |
| 13-clean-v1 | `V1 = 0;⏎⏎   if (V1) {` | `V1 = 0;⏎⏎   if (V1)` | 1c R4 (R3c): o '{' abre o corpo, que fica fora. |
| 13-clean-v2 | `V1 = 4;⏎⏎   while (!4) {` | `V1 = 4;⏎⏎   while (!4)` | 1c R4 (R3c): o '{' abre o corpo, que fica fora. |
| 14-clean-v0 | `if (++V1) {⏎        V1 = V1 * 2;⏎        V2 = V2 * 2;⏎   } else if (++V2) {` | `if (++V1) {⏎        V1 = V1 * 2;⏎        V2 = V2 * 2;⏎   } else if (++V2)` | 1c/3 R4 (R3c): o '{' abre o segundo corpo, que fica fora. |
| 14-clean-v1 | `if (V1 == V2) {⏎      ++V1;⏎   } else {⏎      ++V2;⏎   }` | `if (V1 == V2) {⏎      ++V1;⏎   } else {⏎      ++V2` | 1c/1e R4: a regra do átomo termina no último pedaço (++V2), como em v0 e v2; nível de expressão (R3b). |
| 15-clean-v1 | `for (int V2 = 0; V2 < 2; V2++) {⏎         V3 = 4 * V1 + V2;⏎         V1 = V2;` | `for (int V2 = 0; V2 < 2; V2++) {⏎         V3 = 4 * V1 + V2;` | 1c R4: comando na borda sem correspondente no trecho confuso. |
| 17-obf-v0 | `V1 = 3;⏎   V1 = 2;` | `V1 = 3;` | 2 R3 + R7: a atribuição seguinte, que torna V1 = 3 morta, existe igual no limpo; a AOI é o trecho que não existe no limpo, como em v1 e v2. |

Consistência por átomo (R6). Estes eram os desvios na v2, todos corrigidos:
- **3 e 4 (inc/dec):** a v2 marcava do primeiro ao último comando do cálculo. Isso dava dois comandos vizinhos em 3-v0 e 4-v0/v1/v2, mas o if inteiro com os corpos em 3-v1/v2. Agora os 6 limpos usam a expressão que usa o valor.
- **8:** v0 e v1 estavam no nível de expressão e v2 no de comando. Agora v2 também está no nível de expressão.
- **12:** v0 e v1 incluíam o alvo da declaração, que agora sai. v2 continua no nível da estrutura, porque é o if/else que faz as diretivas parecerem condicionais. Esse desvio é justificado.
- **14 (limpo):** v1 terminava no `}` e v0/v2 no último pedaço. Agora os três terminam no último pedaço.
- **17:** v0 incluía a atribuição seguinte e v1/v2 não. Agora os três marcam só o trecho que não existe no limpo.

Nenhum outro átomo tinha desvio.

### 3c. Lista de revisão: 20 snippets (R8)

| snippet | revisao | alternativa |
|---|---|---|
| 3-clean-v1 | ambiguo (duvidoso): escolhida a opção mínima; a outra em alternativa | Do teste ao incremento, com os corpos do if (v2). |
| 3-clean-v2 | ambiguo (duvidoso): escolhida a opção mínima; a outra em alternativa | Do teste ao decremento, com os corpos do if (v2). |
| 5-obf-v0 | ambiguo (duvidoso): escolhida a opção mínima; a outra em alternativa | Da declaração até a conta (v2). |
| 5-obf-v1 | ambiguo (duvidoso): escolhida a opção mínima; a outra em alternativa | Da declaração até a conta (v2). |
| 5-obf-v2 | ambiguo (duvidoso): escolhida a opção mínima; a outra em alternativa | Da declaração até a conta (v2). |
| 9-obf-v0 | mantido (ternario_virgula) |  |
| 9-clean-v0 | mantido (ternario_virgula) |  |
| 9-obf-v1 | mantido (ternario_virgula) |  |
| 9-clean-v1 | mantido (ternario_virgula) |  |
| 9-obf-v2 | mantido (ternario_virgula) |  |
| 9-clean-v2 | mantido (ternario_virgula) |  |
| 11-obf-v0 | mantido (ternario_virgula) |  |
| 11-clean-v0 | alterado (ternario_virgula; item 1c/3) |  |
| 11-obf-v1 | mantido (ternario_virgula) |  |
| 11-clean-v1 | alterado (ternario_virgula; item 1c/3) |  |
| 11-obf-v2 | mantido (ternario_virgula) |  |
| 11-clean-v2 | alterado (ternario_virgula; item 1c/3) |  |
| 14-clean-v0 | ambiguo (duvidoso): escolhida a opção mínima; a outra em alternativa | O if/else inteiro, com o segundo corpo. |
| 15-obf-v1 | mantido (novo) |  |
| 17-obf-v0 | alterado (novo; item 2) | V1 = 3; V1 = 2; (v2: inclui o que torna o trecho morto). |

## 4. Código morto (átomo 17, 3 pares)

**Recomendação: opção A.** O par sai do aoi_max nos dois lados (n = 60). A exclusão vale só para o aoi_max: o snippet_media continua com os 63 pares.

**Por que A:**
- O limpo de código morto não tem o cálculo que o confuso tem: o trecho morto foi apagado.
- Na opção B, o "comando equivalente que permanece" é um vizinho que existe igual nos dois lados: `V1 = 2;`, `V1 = 1` e `int V1 = 0;` (os trechos da v1). O ℓ do par passaria a comparar o trecho morto com um comando diferente, e não duas formas do mesmo cálculo. É o oposto do princípio da AOI.
- O snippet_media não depende de AOI. Nele, o par mede o programa inteiro com e sem o código morto, o que continua válido.
- Na sensibilidade (seção 5), a coluna v2.1-B mostra a opção B. A opção não muda o sinal de nenhum D. Com a A, o IC do D do Yi 9B exclui o zero, e com a B não exclui. A conclusão desse D depende, portanto, de 3 pares.

**17-obf-v0, minimalidade (R3):**
- `V1 = 2;`, que torna `V1 = 3;` morta, existe igual no limpo. Por isso sai da AOI, que agora é só `V1 = 3;`.
- Fica o mesmo padrão de v1 (`if (0) {...}`) e de v2 (`V1 = V1;`): a AOI é o trecho que não existe no limpo.
- A opção anterior está em `alternativa`. Como o par está fora do aoi_max, a escolha não afeta a métrica principal.

## 5. Sensibilidade

A sensibilidade das conclusões à anotação (v1, v2 e v2.1) foi calculada nos 8 trios de C, pelo mesmo caminho da análise principal (`trio.analisar`, 10.000 reamostragens dos átomos, semente 42), em `results/extensao/2026-11-05-sensibilidade-aoi-gopstein/` (`uv run python -m aoc_perplexity.sensibilidade`): `sensibilidade.csv` traz os contrastes por trio e anotação, `tamanhos.csv` o tamanho mediano da AOI em tokens e `robustez.csv` a classificação de cada contraste (robusto se o sinal e o status do IC são iguais nas três anotações). O `snippet_media` não depende da AOI e é idêntico nas três.

## 6. Campos

- `nivel` foi renomeado para **`nivel_aprox`**. É uma estimativa automática por palavras-chave:
  - BLOCK: tem `{` ou quebra de linha e também if/for/while/else;
  - STAT: tem `;`, quebra de linha ou `#define`;
  - NEXP: o resto.

  **O campo não separa OP, WEXP e NEXP**, e nos trechos mistos (atribuição seguida do teste) pode dar BLOCK. Não serve para comparar níveis com Bergum.
- Ficam como estavam: `trecho`, `inicio`, `fim` (fim exclusivo), `ocorrencia` e `sha256_snippet` (16 primeiros caracteres hexadecimais do SHA-256 do snippet).
- Em todas as linhas, `trecho` é igual a `código[inicio:fim]` e a ocorrência número `ocorrencia` de `trecho` começa em `inicio`. As duas coisas foram conferidas.
- `origem` ganha o valor `regra_v2.1`, para as AOIs que a v2.1 reduziu e que não coincidem com nenhuma das duas anotações. Quando a redução coincide com a v1, a origem passa a ser `equipe` ou `ambas`.
- O desfecho da lista de revisão está em `revisao`. Nos outros snippets, o campo fica vazio.

## 7. Processo de anotação

- **Anotações.** Duas, ambas por LLM: a preliminar (v1, `gopstein-aoi.csv`, por trecho) e uma independente, por intervalo de caracteres.
- **Critério e validação humana.** Um dos autores definiu o princípio da AOI e a regra de cada átomo e validou exemplos de 26 pares, sempre vendo o par confuso/limpo completo com a AOI destacada: 6 pares de expressão (1-v0, 2-v0, 7-v0, 10-v0, 16-v0 e 18-v0) e 2 pares de cada um dos outros 10 tipos (8-v1/v2, 20-v0/v1, 19-v0/v2, 6-v0 e 21-v0, 3-v0/v1, 14-v0/v2, 13-v0/v2, 5-v0/v2, 12-v0/v2, 15-v0/v1). Para ternário, vírgula e código morto, a regra foi fixada por escrito, sem exemplos validados.
- **Divergências.** Resolvidas por regra, sem revisão humana caso a caso (58 snippets de origem `independente` na v2 e 24 reduções na v2.1): na v2, a regra de cada átomo (com 1 escolha da v1 e 2 trechos novos); na v2.1, as regras R1 a R8, escritas antes das checagens e aplicadas aos 126 snippets.
- **Regras revistas sem nova validação por exemplos:** inc/dec, CONSTANTVARIABLE, Logic as Control Flow (fim no último pedaço) e código morto.

## 8. O que ficou ambíguo

Nos casos abaixo, as duas opções cabem na regra. Ficou a opção mínima, e a outra está em `alternativa`.

| snippet | escolhida (v2.1) | alternativa |
|---|---|---|
| 3-clean-v1 | `V1 == 0` | Do teste ao incremento, com os corpos do if (v2). |
| 3-clean-v2 | `V1 == 1` | Do teste ao decremento, com os corpos do if (v2). |
| 5-obf-v0 | `V1 + 2` | Da declaração até a conta (v2). |
| 5-obf-v1 | `2 * V1` | Da declaração até a conta (v2). |
| 5-obf-v2 | `2.5 * V1` | Da declaração até a conta (v2). |
| 14-clean-v0 | `if (++V1) {⏎        V1 = V1 * 2;⏎        V2 = V2 * 2;⏎   } else if (++V2)` | O if/else inteiro, com o segundo corpo. |

Outros pontos em aberto:
- **3-clean-v1, 3-clean-v2 e os demais limpos de inc/dec.** A AOI não inclui o comando de incremento separado. Isso reduz o tamanho, mas o limpo deixa de conter a parte "incrementa" do cálculo. Os quatro limpos sem corpo no meio (3-v0, 4-v0, 4-v1, 4-v2) poderiam incluí-lo sem custo. Ficaram de fora só para manter o mesmo padrão nos três pares (R6).
- **Pares ainda fora da faixa de razão (10).** A razão continua acima de 2 em 9-v0/v1/v2, 13-v0/v1, 14-v0/v2, 20-v1 e abaixo de 0,5 em 12-v1/v2, porque o equivalente do limpo é uma estrutura ou um operador. Esses casos estão justificados em `observacao` e ainda pesam no Δ.
- **Átomo 11 (vírgula).** O limpo v0 começa num literal (`2;⏎   V2 = 1`), porque o alvo `V1 =` sai pela R4. O trecho é mínimo, mas tem um recorte pouco natural.
- **`nivel_aprox`** é só uma aproximação (seção 6).

