# Índice de `results/extensao`

Uma linha por pasta publicada. Quais pastas cada etapa usa está em [`../manifesto.json`](../manifesto.json); o mapa entre o artigo e os arquivos, em [`docs/reproducao.md`](../../docs/reproducao.md). As pastas não são renomeadas nem movidas, porque os `run.json` apontam umas para as outras pelos caminhos atuais.

Tipos: **execução** (saída de `perplexidade.py`: log-probabilidades por token e `run.json`), **trio** (análise de base, código e instruct de uma família: `trio.csv` e `por_modelo.csv`), **consolidado** (tabela única dos trios), **análise** (EEG, chat template, sensibilidade), **validação** (conferência de método) e **figuras** (PDFs, PNGs de pré-visualização e os dados de cada figura).

## Leitura dos nomes

- **Conjuntos:** `bergum` é Java (144 snippets) e `gopstein`, C (126). Execuções recentes levam o sufixo dos conjuntos rodados (`-bergum`, `-gopstein`, `-bergum-gopstein`). O sufixo antigo **`-completo`** significa "todos os snippets dos conjuntos rodados": só Java nas execuções do Yi (o C do Yi está nas pastas `-gopstein`) e Java e C nas do Qwen2.5 3B e 32B, DeepSeek, Gemma e Llama-2. O `config.conjuntos` (ou `config.arquivos`) do `run.json` diz o que cada pasta tem.
- **Anotação de C:** trios de C terminados em `-v2.1` usam a anotação principal; os sem sufixo usam a v1 e só entram na sensibilidade, que recalcula v1, v2 e v2.1 em `2026-11-05-sensibilidade-aoi-gopstein` (`sensibilidade.csv`, `tamanhos.csv` e `robustez.csv`). Para regenerar um trio v1, use `trio.py --anotacao v1`.
- **Outros sufixos:** `-atomos-confirmados` e `-atomos-confirmados-indentacao` são os subconjuntos de átomos; `-template`, o instruct rodado com chat template.
- **Consolidado:** o vigente é `2026-11-12-consolidado` (tabelas do artigo). `2026-11-05-consolidado-3` e `2026-11-05-eeg-java` têm os mesmos valores do consolidado e da análise de EEG vigentes e ficam porque são as fontes, com SHA-256, registradas nos `run.json` das figuras.
- **`validacao-diff`:** é a única pasta sem `run.json`: a tabela foi gerada por `analise.py`, na própria pasta, que registra os tokenizadores e as revisões usados (a main dos repositórios em 2026-11-02).
- **Gemma:** o `snippet_media` do trio Gemma não é relatado ([`docs/metodos.md`](../../docs/metodos.md), seção 4).

## Pastas

| pasta | tipo | modelo(s) | conjunto(s) | situação |
|---|---|---|---|---|
| `2026-11-02-equivalencia-uma-passagem` | validação | Qwen2.5-Coder-1.5B | bergum (Java) e gopstein (C) | vigente (justifica a uma passagem) |
| `2026-11-02-reexecucao-Qwen2.5-32B-4bit-completo` | execução | Qwen2.5-32B-4bit (MLX, 4 bits) | bergum (Java) e gopstein (C) | vigente (entra nos trios 32B) |
| `2026-11-02-reexecucao-Qwen2.5-3B-completo` | execução | Qwen2.5-3B | bergum (Java) e gopstein (C) | vigente (entra nos trios 3B) |
| `2026-11-02-reexecucao-Qwen2.5-Coder-32B-4bit-completo` | execução | Qwen2.5-Coder-32B-4bit (MLX, 4 bits) | bergum (Java) e gopstein (C) | vigente (entra nos trios 32B) |
| `2026-11-02-reexecucao-Qwen2.5-Coder-32B-Instruct-4bit-completo` | execução | Qwen2.5-Coder-32B-Instruct-4bit (MLX, 4 bits) | bergum (Java) e gopstein (C) | vigente (entra nos trios 32B) |
| `2026-11-02-reexecucao-Qwen2.5-Coder-3B-Instruct-completo` | execução | Qwen2.5-Coder-3B-Instruct | bergum (Java) e gopstein (C) | vigente (entra nos trios 3B) |
| `2026-11-02-reexecucao-Qwen2.5-Coder-3B-completo` | execução | Qwen2.5-Coder-3B | bergum (Java) e gopstein (C) | vigente (entra nos trios 3B) |
| `2026-11-02-trio-qwen2.5-3b-bergum-aoi-manual` | trio | Qwen2.5 3B / Coder-3B / Coder-3B-Instruct | bergum (Java) | vigente |
| `2026-11-02-trio-qwen2.5-3b-gopstein-aoi-manual` | trio | Qwen2.5 3B / Coder-3B / Coder-3B-Instruct | gopstein (C) | sensibilidade (AOI v1; a principal é a v2.1) |
| `2026-11-02-validacao-aoi` | validação | só tokenizadores (deepseek-coder-1.3b-base e 6.7b-base, Qwen2.5-Coder-32B, Yi-Coder-9B) | bergum (Java) | vigente (mapeamento de AOI) |
| `2026-11-03-reexecucao-Yi-9B-completo` | execução | Yi-9B | bergum (Java) | vigente (entra no trio Yi, Java) |
| `2026-11-03-reexecucao-Yi-9B-gopstein` | execução | Yi-9B | gopstein (C) | vigente (entra no trio Yi, C) |
| `2026-11-03-reexecucao-Yi-Coder-9B-Chat-completo` | execução | Yi-Coder-9B-Chat (tokenizador do Yi-Coder-9B) | bergum (Java) | vigente (entra no trio Yi, Java) |
| `2026-11-03-reexecucao-Yi-Coder-9B-Chat-gopstein` | execução | Yi-Coder-9B-Chat (tokenizador do Yi-Coder-9B) | gopstein (C) | vigente (entra no trio Yi, C) |
| `2026-11-03-reexecucao-Yi-Coder-9B-completo` | execução | Yi-Coder-9B | bergum (Java) | vigente (entra no trio Yi, Java) |
| `2026-11-03-reexecucao-Yi-Coder-9B-gopstein` | execução | Yi-Coder-9B | gopstein (C) | vigente (entra no trio Yi, C) |
| `2026-11-03-reexecucao-codegemma-7b-completo` | execução | codegemma-7b | bergum (Java) e gopstein (C) | vigente (entra nos trios Gemma) |
| `2026-11-03-reexecucao-codegemma-7b-it-completo` | execução | codegemma-7b-it | bergum (Java) e gopstein (C) | vigente (entra nos trios Gemma); posição 0 ≈ −395 (`docs/diagnostico-gemma/`) |
| `2026-11-03-reexecucao-deepseek-coder-7b-base-v1.5-completo` | execução | deepseek-coder-7b-base-v1.5 | bergum (Java) e gopstein (C) | vigente (entra nos trios DeepSeek) |
| `2026-11-03-reexecucao-deepseek-coder-7b-instruct-v1.5-completo` | execução | deepseek-coder-7b-instruct-v1.5 | bergum (Java) e gopstein (C) | vigente (entra nos trios DeepSeek) |
| `2026-11-03-reexecucao-deepseek-llm-7b-base-completo` | execução | deepseek-llm-7b-base | bergum (Java) e gopstein (C) | vigente (entra nos trios DeepSeek) |
| `2026-11-03-reexecucao-gemma-7b-completo` | execução | gemma-7b | bergum (Java) e gopstein (C) | vigente (entra nos trios Gemma) |
| `2026-11-03-trio-deepseek-7b-bergum-aoi-manual` | trio | deepseek-llm-7b / deepseek-coder-7b / -instruct | bergum (Java) | vigente |
| `2026-11-03-trio-deepseek-7b-gopstein-aoi-manual` | trio | deepseek-llm-7b / deepseek-coder-7b / -instruct | gopstein (C) | sensibilidade (AOI v1; a principal é a v2.1) |
| `2026-11-03-trio-gemma-7b-bergum-aoi-manual` | trio | gemma-7b / codegemma-7b / codegemma-7b-it | bergum (Java) | vigente; `snippet_media` do instruct inválido |
| `2026-11-03-trio-gemma-7b-gopstein-aoi-manual` | trio | gemma-7b / codegemma-7b / codegemma-7b-it | gopstein (C) | sensibilidade (AOI v1; a principal é a v2.1) |
| `2026-11-03-trio-qwen2.5-32b-4bit-bergum-aoi-manual` | trio | Qwen2.5 32B / Coder-32B / Coder-32B-Instruct (4 bits) | bergum (Java) | vigente |
| `2026-11-03-trio-qwen2.5-32b-4bit-gopstein-aoi-manual` | trio | Qwen2.5 32B / Coder-32B / Coder-32B-Instruct (4 bits) | gopstein (C) | sensibilidade (AOI v1; a principal é a v2.1) |
| `2026-11-03-trio-yi-9b-bergum-aoi-manual` | trio | Yi-9B / Yi-Coder-9B / Yi-Coder-9B-Chat | bergum (Java) | vigente |
| `2026-11-05-consolidado-3` | consolidado | Qwen2.5 3B, 7B, 32B; DeepSeek 7B; Yi 9B; Llama-2 7B e 13B; Gemma 7B (16 trios: 8 em Java e 8 em C, de 5 famílias) | bergum e gopstein (C com a AOI v2.1) | mesmos valores do vigente; fonte das figuras (`config.fontes`) |
| `2026-11-12-consolidado` | consolidado | como `2026-11-05-consolidado-3` | bergum e gopstein (C com a AOI v2.1) | vigente (tabelas do relatório) |
| `2026-11-04-reexecucao-CodeLlama-7b-Instruct-hf-completo` | execução | CodeLlama-7b-Instruct-hf | bergum (Java) e gopstein (C) | vigente (entra nos trios Llama-2) |
| `2026-11-04-reexecucao-CodeLlama-7b-hf-completo` | execução | CodeLlama-7b-hf | bergum (Java) e gopstein (C) | vigente (entra nos trios Llama-2) |
| `2026-11-04-reexecucao-Llama-2-7b-hf-completo` | execução | Llama-2-7b-hf | bergum (Java) e gopstein (C) | vigente (entra nos trios Llama-2) |
| `2026-11-05-reexecucao-CodeLlama-13b-Instruct-hf-bergum-gopstein` | execução | CodeLlama-13b-Instruct-hf (fp16, torch/MPS) | bergum (Java) e gopstein (C) | vigente (entra no trio 13B) |
| `2026-11-05-reexecucao-CodeLlama-13b-hf-bergum-gopstein` | execução | CodeLlama-13b-hf (fp16, torch/MPS) | bergum (Java) e gopstein (C) | vigente (entra no trio 13B) |
| `2026-11-05-reexecucao-Llama-2-13b-hf-bergum-gopstein` | execução | Llama-2-13b-hf (fp16, torch/MPS) | bergum (Java) e gopstein (C) | vigente (entra no trio 13B) |
| `2026-11-05-reexecucao-Qwen2.5-Coder-3B-Instruct-bergum-gopstein-template` | execução | Qwen2.5-Coder-3B-Instruct (fp16, torch/MPS, uma passagem, com chat template) | bergum (Java) e gopstein (C) | verificação (mesma revisão da execução vigente; só entra nos trios `-template`) |
| `2026-11-05-reexecucao-deepseek-coder-7b-instruct-v1.5-bergum-gopstein-template` | execução | deepseek-coder-7b-instruct-v1.5 (fp16, torch/MPS, uma passagem, com chat template) | bergum (Java) e gopstein (C) | verificação (mesma revisão da execução vigente; só entra nos trios `-template`) |
| `2026-11-04-reexecucao-Qwen2.5-7B-bergum-gopstein` | execução | Qwen2.5-7B | bergum (Java) e gopstein (C) | vigente (entra no trio 7B) |
| `2026-11-04-reexecucao-Qwen2.5-Coder-7B-Instruct-bergum-gopstein` | execução | Qwen2.5-Coder-7B-Instruct | bergum (Java) e gopstein (C) | vigente (entra no trio 7B) |
| `2026-11-04-reexecucao-Qwen2.5-Coder-7B-bergum-gopstein` | execução | Qwen2.5-Coder-7B | bergum (Java) e gopstein (C) | vigente (entra no trio 7B) |
| `2026-11-05-sensibilidade-aoi-gopstein` | validação | os 8 trios com C (os 7 anteriores e o Llama-2 13B) | gopstein (C) | vigente (contrastes na v1, v2 e v2.1; robusto ou dependente da anotação) |
| `2026-11-05-eeg-java` | análise | os 24 modelos de Java dos 8 trios e os 6 publicados | bergum (Java), AOI manual | mesmos `correlacoes.csv` e `concordancia-wilcoxon-bootstrap.csv` do vigente; fonte de `2026-11-12-figuras` |
| `2026-11-12-eeg-java` | análise | os 24 modelos de Java dos 8 trios e os 6 publicados | bergum (Java), AOI manual | vigente: Spearman perplexidade × FRP com IC por bootstrap agrupado por snippet-base (`correlacoes.csv`, `concordancia-wilcoxon-bootstrap.csv`, `tabela-correlacoes.tex`) |
| `2026-11-12-figuras` | figuras | os 24 modelos de Java dos 8 trios | bergum (Java) | vigente: `eeg-grade-llama2.pdf` e `eeg-grade-qwen25.pdf` (Figuras C.1 e C.2 do artigo) e `eeg-grade-contraste.pdf` (não usada no artigo), com `dados-grade.csv`; PNGs só para pré-visualização |
| `2026-11-11-figuras` | figuras | os 8 trios | bergum (Java) e gopstein (C) | vigente: `delta-aoi_max-grade.pdf` (Figura 1 do relatório, grade 2 × 2 na largura de uma coluna ACM, Java sobre C) e as versões `delta-aoi_max.pdf` e `delta-aoi_max-coluna.pdf`, com os dados |
| `2026-11-06-template-instruct` | análise | Qwen2.5 3B e DeepSeek 7B (instruct sem e com template) | bergum (Java) e gopstein (C) | vigente (`comparacao.csv`: Δ_I, D_instr,code e D_instr,base sem e com template; `posicao0.csv`; conferência da tokenização separada do template) |
| `2026-11-04-trio-deepseek-7b-gopstein-aoi-manual-v2.1` | trio | deepseek-llm-7b / deepseek-coder-7b / -instruct | gopstein (C) | vigente (AOI v2.1) |
| `2026-11-04-trio-gemma-7b-gopstein-aoi-manual-v2.1` | trio | gemma-7b / codegemma-7b / codegemma-7b-it | gopstein (C) | vigente (AOI v2.1); `snippet_media` não relatado |
| `2026-11-04-trio-llama-2-7b-bergum-aoi-manual` | trio | Llama-2-7b / CodeLlama-7b / CodeLlama-7b-Instruct | bergum (Java) | vigente |
| `2026-11-04-trio-llama-2-7b-gopstein-aoi-manual` | trio | Llama-2-7b / CodeLlama-7b / CodeLlama-7b-Instruct | gopstein (C) | sensibilidade (AOI v1; a principal é a v2.1) |
| `2026-11-04-trio-llama-2-7b-gopstein-aoi-manual-v2.1` | trio | Llama-2-7b / CodeLlama-7b / CodeLlama-7b-Instruct | gopstein (C) | vigente (AOI v2.1) |
| `2026-11-05-trio-llama-2-13b-bergum-aoi-manual` | trio | Llama-2-13b / CodeLlama-13b / CodeLlama-13b-Instruct | bergum (Java) | vigente |
| `2026-11-05-trio-llama-2-13b-gopstein-aoi-manual` | trio | Llama-2-13b / CodeLlama-13b / CodeLlama-13b-Instruct | gopstein (C) | sensibilidade (AOI v1; a principal é a v2.1) |
| `2026-11-05-trio-llama-2-13b-gopstein-aoi-manual-v2.1` | trio | Llama-2-13b / CodeLlama-13b / CodeLlama-13b-Instruct | gopstein (C) | vigente |
| `2026-11-05-trio-qwen2.5-3b-bergum-aoi-manual-template` | trio | Qwen2.5-3B / Qwen2.5-Coder-3B / Qwen2.5-Coder-3B-Instruct com template | bergum (Java) | verificação (só o instruct muda) |
| `2026-11-05-trio-qwen2.5-3b-gopstein-aoi-manual-v2.1-template` | trio | Qwen2.5-3B / Qwen2.5-Coder-3B / Qwen2.5-Coder-3B-Instruct com template | gopstein (C, AOI v2.1) | verificação (só o instruct muda) |
| `2026-11-05-trio-deepseek-7b-bergum-aoi-manual-template` | trio | deepseek-llm-7b / deepseek-coder-7b / -instruct com template | bergum (Java) | verificação (só o instruct muda) |
| `2026-11-05-trio-deepseek-7b-gopstein-aoi-manual-v2.1-template` | trio | deepseek-llm-7b / deepseek-coder-7b / -instruct com template | gopstein (C, AOI v2.1) | verificação (só o instruct muda) |
| `2026-11-04-trio-qwen2.5-32b-4bit-gopstein-aoi-manual-v2.1` | trio | Qwen2.5 32B / Coder-32B / Coder-32B-Instruct (4 bits) | gopstein (C) | vigente (AOI v2.1) |
| `2026-11-04-trio-qwen2.5-3b-gopstein-aoi-manual-v2.1` | trio | Qwen2.5 3B / Coder-3B / Coder-3B-Instruct | gopstein (C) | vigente (AOI v2.1) |
| `2026-11-04-trio-qwen2.5-7b-bergum-aoi-manual` | trio | Qwen2.5 7B / Coder-7B / Coder-7B-Instruct | bergum (Java) | vigente |
| `2026-11-04-trio-qwen2.5-7b-gopstein-aoi-manual` | trio | Qwen2.5 7B / Coder-7B / Coder-7B-Instruct | gopstein (C) | sensibilidade (AOI v1; a principal é a v2.1) |
| `2026-11-04-trio-qwen2.5-7b-gopstein-aoi-manual-v2.1` | trio | Qwen2.5 7B / Coder-7B / Coder-7B-Instruct | gopstein (C) | vigente |
| `2026-11-04-trio-yi-9b-gopstein-aoi-manual` | trio | Yi-9B / Yi-Coder-9B / Yi-Coder-9B-Chat | gopstein (C) | sensibilidade (AOI v1; a principal é a v2.1) |
| `2026-11-04-trio-yi-9b-gopstein-aoi-manual-v2.1` | trio | Yi-9B / Yi-Coder-9B / Yi-Coder-9B-Chat | gopstein (C) | vigente (AOI v2.1) |
| `validacao-diff` | validação | Qwen2.5-Coder-32B, deepseek-coder-1.3b-base e 6.7b-base, Artigenz-Coder-DS-6.7B, Yi-Coder-9B | bergum (Java) | vigente (AOI por diff descartada como métrica) |
| `2026-11-05-trio-deepseek-7b-gopstein-aoi-manual-v2.1-atomos-confirmados` | sensibilidade por átomos | deepseek-7b | gopstein (AOI v2.1) | confirmados (15 átomos, 45 pares) |
| `2026-11-05-trio-deepseek-7b-gopstein-aoi-manual-v2.1-atomos-confirmados-indentacao` | sensibilidade por átomos | deepseek-7b | gopstein (AOI v2.1) | confirmados + indentação (17 átomos, 51 pares) |
| `2026-11-05-trio-gemma-7b-gopstein-aoi-manual-v2.1-atomos-confirmados` | sensibilidade por átomos | gemma-7b | gopstein (AOI v2.1) | confirmados (15 átomos, 45 pares) |
| `2026-11-05-trio-gemma-7b-gopstein-aoi-manual-v2.1-atomos-confirmados-indentacao` | sensibilidade por átomos | gemma-7b | gopstein (AOI v2.1) | confirmados + indentação (17 átomos, 51 pares) |
| `2026-11-05-trio-llama-2-13b-gopstein-aoi-manual-v2.1-atomos-confirmados` | sensibilidade por átomos | llama-2-13b | gopstein (AOI v2.1) | confirmados (15 átomos, 45 pares) |
| `2026-11-05-trio-llama-2-13b-gopstein-aoi-manual-v2.1-atomos-confirmados-indentacao` | sensibilidade por átomos | llama-2-13b | gopstein (AOI v2.1) | confirmados + indentação (17 átomos, 51 pares) |
| `2026-11-05-trio-llama-2-7b-gopstein-aoi-manual-v2.1-atomos-confirmados` | sensibilidade por átomos | llama-2-7b | gopstein (AOI v2.1) | confirmados (15 átomos, 45 pares) |
| `2026-11-05-trio-llama-2-7b-gopstein-aoi-manual-v2.1-atomos-confirmados-indentacao` | sensibilidade por átomos | llama-2-7b | gopstein (AOI v2.1) | confirmados + indentação (17 átomos, 51 pares) |
| `2026-11-05-trio-qwen2.5-32b-4bit-gopstein-aoi-manual-v2.1-atomos-confirmados` | sensibilidade por átomos | qwen2.5-32b-4bit | gopstein (AOI v2.1) | confirmados (15 átomos, 45 pares) |
| `2026-11-05-trio-qwen2.5-32b-4bit-gopstein-aoi-manual-v2.1-atomos-confirmados-indentacao` | sensibilidade por átomos | qwen2.5-32b-4bit | gopstein (AOI v2.1) | confirmados + indentação (17 átomos, 51 pares) |
| `2026-11-05-trio-qwen2.5-3b-gopstein-aoi-manual-v2.1-atomos-confirmados` | sensibilidade por átomos | qwen2.5-3b | gopstein (AOI v2.1) | confirmados (15 átomos, 45 pares) |
| `2026-11-05-trio-qwen2.5-3b-gopstein-aoi-manual-v2.1-atomos-confirmados-indentacao` | sensibilidade por átomos | qwen2.5-3b | gopstein (AOI v2.1) | confirmados + indentação (17 átomos, 51 pares) |
| `2026-11-05-trio-qwen2.5-7b-gopstein-aoi-manual-v2.1-atomos-confirmados` | sensibilidade por átomos | qwen2.5-7b | gopstein (AOI v2.1) | confirmados (15 átomos, 45 pares) |
| `2026-11-05-trio-qwen2.5-7b-gopstein-aoi-manual-v2.1-atomos-confirmados-indentacao` | sensibilidade por átomos | qwen2.5-7b | gopstein (AOI v2.1) | confirmados + indentação (17 átomos, 51 pares) |
| `2026-11-05-trio-yi-9b-gopstein-aoi-manual-v2.1-atomos-confirmados` | sensibilidade por átomos | yi-9b | gopstein (AOI v2.1) | confirmados (15 átomos, 45 pares) |
| `2026-11-05-trio-yi-9b-gopstein-aoi-manual-v2.1-atomos-confirmados-indentacao` | sensibilidade por átomos | yi-9b | gopstein (AOI v2.1) | confirmados + indentação (17 átomos, 51 pares) |

Para saber de que execuções um trio vem, leia `config.execucoes` no `run.json` do trio. Para saber de que trios um consolidado vem, `config.trios` no `run.json` dele.

