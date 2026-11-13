# Métodos

Desenho final do experimento e as verificações que o sustentam. Os comandos estão em [`reproducao.md`](reproducao.md); as fontes e as condições dos dados, em [`../data/README.md`](../data/README.md).

**Termos.** *Átomo de confusão*: pequeno padrão de código válido que costuma confundir programadores (por exemplo, `a = b++ + 2`). *AOI* (área de interesse): a região do snippet que contém o átomo. *FRP* (potencial relacionado à fixação): resposta do EEG medida a partir de quando o olhar se fixa num ponto; o pacote dos autores traz o FRP médio de cada snippet na AOI. *Trio*: três checkpoints relacionados, a base geral (B), a base de código (C), que continua o pré-treinamento da B, e o instruct de código (I), ajustado a partir da C.

## 1. Reprodução de Abdelsalam et al.

- **Escopo.** Reprodução computacional parcial: as RQ1 a RQ3 do Qwen2.5-Coder-32B são recalculadas a partir dos arquivos publicados; o `deepseek-coder-1.3b-base` é reexecutado localmente (fp16, MPS). O 32B em fp16 (65,5 GB de pesos) não cabe no hardware usado (Mac com Apple Silicon e 32 GiB de memória unificada).
- **EEG.** Usa-se o FRP agregado publicado (um valor por snippet, confuso e limpo de cada par); o EEG bruto não está no pacote, então não há FRP para regiões novas, como as detectadas automaticamente na RQ3.
- **Pares de Java.** Os 72 pares (144 snippets) vêm de 24 pares-base em 3 versões (`v0`, `v1`, `v2`): os mesmos 24 identificadores aparecem nas 3 versões, que mudam valores e às vezes operadores ou a ordem dos termos. Um par-base tem o identificador vazio no nome do arquivo (`-clean-v0.java`, `-obf-v0.java`); o pacote o trata como 0. As categorias de átomo têm de 2 a 7 pares-base.
- **Testes do artigo original.** O Wilcoxon da RQ1 e os valores p do Spearman da RQ2 tratam os 72 pares como independentes. A reprodução mantém esse método para comparar os números e acrescenta um bootstrap agrupado por par-base (seção 4), com o qual os seis modelos publicados mantêm intervalos acima de zero para Δ e para ρ nos confusos.
- **Critérios de aceitação** (fixados antes das análises): estatísticas iguais às publicadas depois do arredondamento; na reexecução, tokens idênticos e |Δ log-probabilidade| ≤ 0,01 em pelo menos 99% dos tokens.
- **Resultado da reexecução do 1.3B.** Tokens idênticos nos 144 arquivos (3.588 tokens), mas só 84,5% dentro da tolerância (mediana 0,0017; máximo 0,11); o critério não é atendido e o desvio é relatado, não reajustado. A diferença tem forma de ruído numérico: Spearman token a token de 0,99998; os maiores desvios ficam em tokens improváveis; tomando fp32 como referência, o nosso fp16 fica a 0,0025 (mediana) e o publicado a 0,0031; a RQ1 no snippet dá W = 863 contra 864 e r = 0,298 nos dois. Precisão (fração ≤ 0,01; mediana): fp16 84,5%, 0,0017; fp32 81,7%, 0,0031; bf16 39,7%, 0,0186. Hardware e versões dos autores são desconhecidos.
- **RQ3.** A sobreposição entre regiões detectadas e átomos é recalculada a partir dos intervalos de tokens publicados; o algoritmo de detecção não é reexecutado. Bootstrap por snippet, 10.000 reamostragens, semente 42, como no notebook dos autores. Conferem as 6 contagens da Figura 4, as listas de snippets e a Figura 5; na Tabela 3, 3 de 4 grupos. Achados de método: o ρ relatado é a média das reamostragens, não o ρ da amostra; o grupo "com sobreposição" reúne snippets, não regiões; a Figura 4 mistura snippets e regiões; o texto troca 85 e 69 regiões novas entre confusos e limpos; 227 regiões detectadas, 199 com linha de EEG e 194 com valor válido.
- **Discrepâncias com o texto do artigo original** (os arquivos de resultado conferem com o recálculo): RQ2 nos limpos, ρ = 0,063 contra 0,05; RQ3 com sobreposição, limpos, média bootstrap 0,42 contra 0,38 (ρ pontual 0,41).

## 2. Modelos da extensão

Oito trios de cinco famílias. Critério: a base de código continua o pré-treinamento da base geral, o instruct é ajuste da base de código e os três compartilham o tokenizador; sem isso, os contrastes misturariam pós-treinamento com arquitetura.

| Trio | Base geral (B) | Base de código (C) | Instruct de código (I) |
|---|---|---|---|
| Qwen2.5 3B | `Qwen/Qwen2.5-3B` | `Qwen/Qwen2.5-Coder-3B` | `Qwen/Qwen2.5-Coder-3B-Instruct` |
| Qwen2.5 7B | `Qwen/Qwen2.5-7B` | `Qwen/Qwen2.5-Coder-7B` | `Qwen/Qwen2.5-Coder-7B-Instruct` |
| Qwen2.5 32B (4 bits) | `mlx-community/Qwen2.5-32B-4bit` | `mlx-community/Qwen2.5-Coder-32B-4bit` | `mlx-community/Qwen2.5-Coder-32B-Instruct-4bit` |
| DeepSeek 7B | `deepseek-ai/deepseek-llm-7b-base` | `deepseek-ai/deepseek-coder-7b-base-v1.5` | `deepseek-ai/deepseek-coder-7b-instruct-v1.5` |
| Yi 9B | `01-ai/Yi-9B` | `01-ai/Yi-Coder-9B` | `01-ai/Yi-Coder-9B-Chat` |
| Llama-2 7B | `meta-llama/Llama-2-7b-hf` | `meta-llama/CodeLlama-7b-hf` | `meta-llama/CodeLlama-7b-Instruct-hf` |
| Llama-2 13B | `meta-llama/Llama-2-13b-hf` | `meta-llama/CodeLlama-13b-hf` | `meta-llama/CodeLlama-13b-Instruct-hf` |
| Gemma 7B | `google/gemma-7b` | `google/codegemma-7b` | `google/codegemma-7b-it` |

- **Correspondência com o artigo original.** O Qwen2.5-Coder-32B tem o trio 32B (em 4 bits) e os trios menores da mesma família; o Yi-Coder-9B vem do Yi-9B; os deepseek-coder 1.3B e 6.7B foram treinados do zero, sem base geral, e o trio DeepSeek usa a geração v1.5; o Artigenz é ajuste do deepseek-coder-6.7b e não forma trio; o Llama 3.2 não tem versão de código, e a família Llama entra pelo Llama-2/CodeLlama. O Gemma não está no artigo original. Modelos MoE e as gerações Qwen3 não têm trio completo, com base de código, que caiba em 32 GiB sem quantização.
- **Acesso e licenças.** Qwen2.5 7B e Yi 9B: Apache 2.0; Qwen2.5 3B: Qwen Research; DeepSeek: licença DeepSeek; Llama-2 e Gemma: licenças próprias, que exigem aceitar os termos no Hugging Face e um token local. O `deepseek-llm-7b-base` só tem pesos `.bin` (pickle), carregados com `weights_only`.
- **Revisões.** Cada execução registra a revisão do Hugging Face no `run.json`; todas as execuções de um mesmo modelo usam a mesma revisão (Tabela A.2 do artigo).
- **Tokenização.** Antes de cada trio, confere-se que os três tokenizadores produzem os mesmos tokens nos snippets. Exceção: o `Yi-Coder-9B-Chat` tokeniza o primeiro token sem o prefixo de espaço (`int`, não `▁int`) em todos os snippets; o vocabulário é idêntico ao do `Yi-Coder-9B`, cujo tokenizador ele usa (`--tokenizador`, com a revisão no `run.json`). `add_prefix_space=True` não resolve, porque põe espaço depois de cada quebra de linha.
- **Precisão.** Pesos em fp16, como os autores: na reexecução do 1.3B, o bf16 errou cerca de 10 vezes mais que o fp16 em relação ao fp32. Um modelo com `inf` ou `nan` em fp16 rodaria em bf16, com registro; nenhum precisou.
- **Qwen2.5 32B.** Os três modelos em 4 bits (MLX, `--backend mlx`, cerca de 18 GB cada). Antes do trio, o Coder-32B em 4 bits foi comparado com o JSON publicado em fp16: tokens iguais nos 144 arquivos; mediana |Δ log-prob| 0,28, viés −0,30, Spearman 0,96; perplexidade cerca de 40% maior. No Qwen2.5-Coder-1.5B, o MLX sem quantização concorda com o torch (mediana 0,024), então o desvio vem da quantização, não do adaptador. A RQ1 com a perplexidade máxima na AOI se mantém em 4 bits (Tabela B.3); os efeitos no nível do snippet perdem significância. Em Java, o Δ do Coder-32B cai de 2,75 (fp16 publicado) para 2,47. A mesma quantização nos três modelos não garante que o erro se cancele nos contrastes.
- **Llama-2 13B.** fp16 em torch/MPS; os pesos ocupam cerca de 24 GiB, perto do limite recomendado do MPS em 32 GiB. Rodou sem erro de memória, com swap (requisitos em [`reproducao.md`](reproducao.md)).
- **Entrada.** O snippet cru, depois do token inicial (BOS), sem chat template, nos três modelos (a sensibilidade a essa escolha está na seção 6). Uma passagem pelo modelo por snippet: no Qwen2.5-Coder-1.5B, a diferença para o laço de uma passagem por token dos autores é ruído numérico (mediana 0,0009 em Java e 0,0003 em C; `results/extensao/2026-11-02-equivalencia-uma-passagem/`). O trio Qwen2.5 7B e o 1.3B da reprodução usam o laço; os três modelos de um trio usam sempre o mesmo método.

## 3. Dados de C

- **Base.** Os 126 snippets em C (63 pares, 21 átomos × 3 pares) de Gopstein et al. (FSE 2017), num commit fixado. Todos os trios rodam Java e C.
- **Átomos.** O estudo original testou 19 candidatos com participantes e confirmou 15 (Tabela 2; McNemar com correção para dados pareados agrupados); os não confirmados são Dead, Unreachable, Repeated (p = 0,059), Arithmetic as Logic (0,248), Pointer Arithmetic (0,752) e Constant Variables (1,000). Os 2 átomos de indentação da base também foram testados, mas excluídos da análise publicada. Os 21 átomos entram na análise principal; o mapeamento, conferido pelo conteúdo dos snippets, está em `data/anotacoes/gopstein-atomos-artigo.csv`.
- **AOI.** A base não tem AOI. A anotação principal, v2.1 (`data/anotacoes/gopstein-aoi-v2.1.csv`), segue o critério de Bergum: no confuso, o menor trecho que contém o átomo; no limpo, o trecho equivalente. Processo: duas anotações por LLM (a v1, por trecho, e uma independente, por intervalo de caracteres, que deu a v2); um dos autores definiu o critério e a regra de cada átomo e validou exemplos de 26 pares; as divergências foram resolvidas por regras escritas antes das checagens (na v2, a regra do átomo; na v2.1, as regras R1 a R8, aplicadas aos 126 snippets). Quatro regras revistas na v2.1 não foram validadas por exemplos: inc/dec, CONSTANTVARIABLE, Logic as Control Flow e código morto. Seis snippets têm duas opções que cabem na regra; fica a mais curta, e a outra está registrada. A v1 e a v2 só entram como sensibilidade (seção 6). Detalhes em [`../data/anotacoes/`](../data/anotacoes/).
- **Código morto.** O átomo 17 não tem trecho equivalente no limpo: os 3 pares saem do `aoi_max` (60 pares, 20 átomos) e ficam no `snippet_media` (63 pares), igualmente nos três modelos do trio.
- **Integridade.** Cada AOI registra o SHA-256 do snippet; se divergir do snippet baixado, a análise para. Os resultados de C são gravados sem o código, que a análise recoloca a partir do download, conferindo o hash dos IDs dos tokens.
- **AOI por diff.** Descartada como métrica: em Java cobre 37% da AOI manual e mantém a RQ1 em só 2 de 5 modelos (`results/extensao/validacao-diff/`).

## 4. Métricas e estatística

Fixadas em 2026-11-02, antes de rodar qualquer modelo da extensão.

- Para cada par confuso/limpo: ℓ = ln(perplexidade do confuso) − ln(perplexidade do limpo). O logaritmo da razão torna Δ independente da escala de perplexidade de cada modelo, que muda com o tokenizador e a quantização.
- Δ(modelo) = média de ℓ nos pares (72 em Java; em C, 60 no `aoi_max` e 63 no `snippet_media`).
- Métrica principal: perplexidade máxima na AOI (`aoi_max`, a do artigo original). Secundária: perplexidade média do snippet (`snippet_media`).
- Contrastes dentro do trio, com o primeiro índice entrando com sinal positivo: D_code,base = Δ_C − Δ_B, D_instr,code = Δ_I − Δ_C e D_instr,base = Δ_I − Δ_B = D_code,base + D_instr,code. Nos CSVs e no código: `D_codigo`, `D_instrucao` e `D`.
- IC de 95% por percentil, com 10.000 reamostragens e semente 42, de grupos compartilhados pelos três modelos do trio: os 24 pares-base em Java e os átomos em C, com as 3 versões juntas (versões do mesmo par-base são parecidas; reamostrá-las separadamente estreitaria o intervalo). Os intervalos são nominais, sem correção para comparações múltiplas.
- Descritivo por modelo: medianas e Wilcoxon pareado (W, p, r), como na RQ1.
- Os contrastes são interpretados dentro de cada trio; comparações entre trios e tamanhos são descritivas.
- **Gemma 7B: só `aoi_max`.** No `codegemma-7b-it`, a log-probabilidade da posição 0 é cerca de −395 em todos os snippets (nos outros dois modelos, entre −10 e −20) e domina o `snippet_media` (mediana do instruct em Java, limpo, ≈ 6×10⁸). O `aoi_max` não inclui a posição 0. No CSV, as linhas do `snippet_media` do Gemma ficam marcadas (`valida = nao`); na tabela, aparece um travessão. Um diagnóstico com 20 snippets ([`diagnostico-gemma/`](diagnostico-gemma/)) não sustenta atribuir o valor só ao arredondamento do fp16; a referência usada não é um fp32 completo.

## 5. EEG (RQ2 na extensão)

- Spearman entre a perplexidade máxima na AOI e o FRP médio publicado, por snippet, nos 24 modelos de Java e, como referência, nos 6 publicados, nos subconjuntos todos (144), confusos (72) e limpos (72). Só Java: não há EEG para C. O EEG é idêntico nos 6 modelos publicados (os testes conferem) e vale para todos.
- Mesmo cálculo do notebook dos autores, sem remover o par 16-v1; o código reproduz os valores publicados do Qwen2.5-Coder-32B (0,309, 0,474 e 0,063).
- IC de 95% por bootstrap agrupado por par-base (24 grupos, 10.000 reamostragens, semente 42), com o mesmo sorteio nos três subconjuntos. Os ρ compartilham o mesmo EEG: são correlações do mesmo sinal com perplexidades diferentes, não evidências independentes, e não são comparados entre modelos nem entre etapas de treino.
- Complemento: nos 30 modelos, compara-se o Wilcoxon (p < 0,05) com o bootstrap agrupado do Δ do `aoi_max` (IC exclui zero).

## 6. Verificações de sensibilidade

- **Anotação da AOI de C.** Os contrastes são recalculados com as anotações v1, v2 e v2.1 nos 8 trios; um contraste é *robusto* se sinal e status do IC coincidem nas três (`results/extensao/2026-11-05-sensibilidade-aoi-gopstein/`). Doze dos 24 contrastes principais de C mudam de sinal ou de status. O artigo trata isso como ameaça à validade, sem marcas nas tabelas.
- **Subconjuntos de átomos.** Só os 15 confirmados (45 pares) e os confirmados mais a indentação (51 pares), nos 8 trios de C, com a AOI v2.1 e o `aoi_max`. No subconjunto confirmado, os 24 Δ de C aumentam; só o D_code,base do Gemma, perto de zero, troca de sinal (0,006 → −0,005); D_instr,base e D_instr,code do Qwen2.5 3B passam a excluir zero; D_instr,base e D_code,base do Llama-2 13B deixam de excluir.
- **Chat template.** O instruct do Qwen2.5 3B e do DeepSeek 7B roda também com o chat template oficial (o snippet como mensagem do usuário; só os tokens do snippet entram na perplexidade e na AOI, com a mesma tokenização; o contexto é o prefixo do template, no lugar do BOS), em Java e C, com as mesmas revisões. Com o template, o Δ_I cai em 6 dos 8 casos, sem trocar de sinal (maior queda: Qwen2.5 3B, Java, `aoi_max`, 0,49); 5 dos 16 contrastes que envolvem o instruct, a maioria perto de zero, trocam de sinal (`results/extensao/2026-11-06-template-instruct/`).

## 7. Limitações

- Uma família por fornecedor; snippets curtos e sintéticos; EEG só para Java.
- A perplexidade máxima na AOI depende de quantos tokens a AOI tem, o que varia entre versões, linguagens e tokenizadores.
- Java e C diferem ao mesmo tempo em linguagem, estilo e categorias de átomo; o efeito menor em C não é atribuído só à linguagem.
- A AOI de C é uma suposição: Gopstein et al. validaram os átomos pelo erro no snippet inteiro, não por região.
- O instruct é sempre ajustado a partir da base de código, então D_instr,code compara numa ordem fixa; as receitas de pós-treinamento diferem entre famílias.

## Numeração usada nos resultados

Os `run.json` e alguns nomes de pasta citam as decisões pelo número do registro em que foram tomadas. Correspondência:

| Número | Data | Onde está aqui |
|---|---|---|
| 0001 escopo da reprodução | 2026-10-30 | seção 1 |
| 0002 FRP agregado | 2026-10-30 | seção 1 |
| 0003 unidade de reamostragem | 2026-10-30 | seções 1 e 4 |
| 0004 modelos da extensão | 2026-11-02 | seção 2 |
| 0005 precisão | 2026-11-02 | seção 2 |
| 0006 Qwen2.5 32B em 4 bits | 2026-11-02 | seção 2 |
| 0007 métrica da extensão | 2026-11-02 | seção 4 |
| 0008 C (Gopstein et al.) e uma passagem por snippet | 2026-11-02 | seções 2 e 3 |
| 0009, 0011 e 0015 C nos demais trios | 2026-11-03 e 2026-11-04 | seção 3 |
| 0010 tokenizador do Yi-Coder-9B-Chat | 2026-11-03 | seção 2 |
| 0012 snippets de C no repositório | 2026-11-04 | [`../data/README.md`](../data/README.md) |
| 0013 AOI v2.1 | 2026-11-04 | seções 3 e 6 |
| 0014 Gemma só com `aoi_max` | 2026-11-04 | seção 4 |
| 0016 notação dos contrastes | 2026-11-05 | seção 4 |
| 0017 anotação de C só nas ameaças à validade | 2026-11-05 | seção 6 |
| 0018 trio Llama-2 13B | 2026-11-04 | seção 2 |
| 0019 átomos de Gopstein conferidos | 2026-11-05 | seções 3 e 6 |
| 0020 RQ2 nos modelos de Java | 2026-11-05 | seção 5 |
| 0021 chat template | 2026-11-05 | seção 6 |

As regras da anotação v2.1 foram escritas antes das checagens que as aplicaram ([`../data/anotacoes/gopstein-v2.1/regras-v2.1.md`](../data/anotacoes/gopstein-v2.1/regras-v2.1.md)).
