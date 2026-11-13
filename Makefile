UPSTREAM_URL := https://github.com/brains-on-code/llm-perplexity-AoC.git
UPSTREAM_COMMIT := 720e1bfb7b79d54f8834d4670ab8e5d80d44f84d
UPSTREAM_DIR := data/external/llm-perplexity-AoC
GOPSTEIN_COMMIT := 7aed29b562fc8e9bfdab59800572555dea3a51d4
GOPSTEIN_URL := https://raw.githubusercontent.com/dgopstein/atoms-of-confusion/$(GOPSTEIN_COMMIT)/snippet_study/data
GOPSTEIN_DIR := data/external/gopstein
GOPSTEIN_SHA_CODE := 81eaa88c44bf3fe9570feff4097d31bde021c7ee1ce4f585328058df1360c74e
GOPSTEIN_SHA_TAGS := a09b846224207054eb035928b1e58c8a32079e4bd48ecab2930a861a5e64a4bf

.PHONY: setup setup-analise data data-gopstein tokenizadores verificar test test-unidade reproduzir reproduzir-inferencia paper

# Análise e inferência (torch e MLX), exatamente nas versões de uv.lock. Inferência validada só em macOS com Apple Silicon.
setup:
	uv sync --locked --extra inferencia

# Só análise: trios, consolidado, EEG e figuras a partir das execuções salvas (sem torch nem MLX).
setup-analise:
	uv sync --locked

# Pacote dos autores (Abdelsalam et al.) e snippets de Gopstein et al., nos commits fixados.
data: data-gopstein
	@if [ ! -d "$(UPSTREAM_DIR)/.git" ]; then git clone "$(UPSTREAM_URL)" "$(UPSTREAM_DIR)"; fi
	git -C "$(UPSTREAM_DIR)" checkout --quiet --detach "$(UPSTREAM_COMMIT)"
	@echo "Pacote em $(UPSTREAM_DIR), commit $(UPSTREAM_COMMIT)"

# Baixa numa pasta temporária, confere o SHA-256 de cada arquivo e só então move para o destino:
# qualquer falha termina com erro e não deixa um conjunto parcial.
data-gopstein:
	@set -e; tmp=$$(mktemp -d); trap 'rm -rf "$$tmp"' EXIT; \
	for f in code.csv code_tags.csv; do \
		curl -sfL -o "$$tmp/$$f" "$(GOPSTEIN_URL)/$$f" || { echo "ERRO: falhou o download de $$f" >&2; exit 1; }; \
	done; \
	test "$$(shasum -a 256 "$$tmp/code.csv" | cut -d' ' -f1)" = "$(GOPSTEIN_SHA_CODE)" || { echo "ERRO: SHA-256 de code.csv não confere" >&2; exit 1; }; \
	test "$$(shasum -a 256 "$$tmp/code_tags.csv" | cut -d' ' -f1)" = "$(GOPSTEIN_SHA_TAGS)" || { echo "ERRO: SHA-256 de code_tags.csv não confere" >&2; exit 1; }; \
	mkdir -p "$(GOPSTEIN_DIR)"; mv "$$tmp/code.csv" "$$tmp/code_tags.csv" "$(GOPSTEIN_DIR)/"; \
	echo "Gopstein em $(GOPSTEIN_DIR), commit $(GOPSTEIN_COMMIT)"

# Tokenizadores de todas as execuções, nas revisões dos run.json (alguns MB cada; Llama-2, CodeLlama e Gemma exigem
# aceitar a licença no Hugging Face e hf auth login). A análise e os testes marcados com "tokenizador" precisam deles.
tokenizadores:
	uv run --locked python -m aoc_perplexity.ambiente --baixar-tokenizadores

# O que está disponível (dados, tokenizadores, torch, MLX e dispositivo) e o que dá para rodar.
verificar:
	uv run --locked python -m aoc_perplexity.ambiente

# Todos os testes; os que dependem de dados, tokenizadores ou torch são pulados, com o motivo, quando faltam.
test:
	uv run --locked pytest -rs

# Só os testes que não precisam de dados externos, rede nem torch.
test-unidade:
	uv run --locked pytest -m "not dados and not tokenizador and not inferencia"

# Regenera tudo a partir das execuções salvas (replicação, 36 trios, sensibilidade, consolidado, EEG, figuras,
# chat template) em results/novos/ e compara arquivo a arquivo com o publicado.
# Ex.: make reproduzir ARGS="--etapas trios consolidado"
reproduzir:
	uv run --locked python -m aoc_perplexity.reproduzir $(ARGS)

# Igual, rodando antes os modelos: os pesos vão para data/cache-modelos e são apagados depois de cada modelo.
# Horas num Mac com Apple Silicon e 32 GiB; veja docs/reproducao.md.
reproduzir-inferencia:
	uv run --locked --extra inferencia python -m aoc_perplexity.reproduzir --etapas todas --limpar-cache $(ARGS)

paper:
	mkdir -p paper/build
	cd paper && latexmk -pdf -interaction=nonstopmode -outdir=build main.tex
