import math
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.inferencia
torch = pytest.importorskip("torch")  # extra de inferência (make setup)

from aoc_perplexity.perplexidade import comparar, nome_pasta, partes_do_chat_template, perplexidade_por_token, resumir

VOCAB = 4


class TokenizadorFalso:
    """Cada caractere é um token; 'a'=1, 'b'=2, 'c'=3."""

    def __init__(self, bos=0, eos=None):
        self.bos_token_id = bos
        self.eos_token_id = eos

    def __call__(self, texto, return_tensors=None, add_special_tokens=False):
        ids = [" abc".index(c) for c in texto]
        return {"input_ids": torch.tensor([ids])} if return_tensors else SimpleNamespace(input_ids=ids)

    def decode(self, ids):
        return "<s>" if ids[0] == 0 else " abc"[ids[0]]


class ModeloFalso:
    """Prevê o próximo token com probabilidades fixas: 0,5 para 'a' e 1/6 para os demais."""

    config = SimpleNamespace(max_position_embeddings=16)

    def __call__(self, contexto):
        logits = torch.log(torch.tensor([1 / 6, 1 / 2, 1 / 6, 1 / 6])).repeat(1, contexto.size(1), 1)
        return SimpleNamespace(logits=logits)


def test_perplexidade_usa_bos_e_calcula_resumos():
    saida = perplexidade_por_token("ab", ModeloFalso(), TokenizadorFalso(), "cpu")
    assert saida["tokens"] == ["a", "b"]
    assert [r["position"] for r in saida["results"]] == [0, 1]
    assert saida["results"][0]["log_prob"] == pytest.approx(math.log(1 / 2))
    assert saida["results"][1]["perplexity"] == pytest.approx(6)
    assert saida["summary"]["overall_perplexity"] == pytest.approx(math.sqrt(2 * 6))
    assert saida["summary"]["mean_perplexity"] == pytest.approx(4)
    assert saida["summary"]["max_perplexity"] == pytest.approx(6)


def test_sem_bos_usa_eos_e_mantem_token_inicial_como_os_autores():
    saida = perplexidade_por_token("ab", ModeloFalso(), TokenizadorFalso(bos=None, eos=3), "cpu")
    assert saida["tokens"] == ["c", "a", "b"]
    assert saida["summary"]["token_count"] == 2


def resultado(tokens, log_probs):
    return {"results": [{"token": t, "position": i, "log_prob": lp} for i, (t, lp) in enumerate(zip(tokens, log_probs))]}


def test_comparacao_mede_diferenca_e_detecta_tokens_diferentes():
    nossos = {"x": resultado(["a", "b"], [-1.0, -2.0]), "y": resultado(["a"], [-1.0])}
    publicados = {"x": resultado(["a", "b"], [-1.005, -2.5]), "y": resultado(["b"], [-1.0])}
    resumo = resumir(comparar(nossos, publicados))
    assert resumo["arquivos"] == 2
    assert resumo["arquivos_com_tokens_iguais"] == 1
    assert resumo["tokens_comparados"] == 2
    assert resumo["fracao_ate_tolerancia"] == pytest.approx(0.5)
    assert resumo["diferenca_maxima"] == pytest.approx(0.5)


def test_resumo_sem_tokens_iguais():
    resumo = resumir(comparar({"x": resultado(["a"], [-1.0])}, {"x": resultado(["b"], [-1.0])}))
    assert resumo["arquivos_com_tokens_iguais"] == 0
    assert resumo["tokens_comparados"] == 0
    assert resumo["diferenca_maxima"] is None


def test_uma_passagem_igual_ao_laco():
    laco = perplexidade_por_token("abca", ModeloFalso(), TokenizadorFalso(), "cpu")
    unica = perplexidade_por_token("abca", ModeloFalso(), TokenizadorFalso(), "cpu", uma_passagem=True)
    assert [r["log_prob"] for r in unica["results"]] == pytest.approx([r["log_prob"] for r in laco["results"]])
    assert unica["tokens"] == laco["tokens"]


def test_pasta_da_execucao_leva_o_sufixo_do_conjunto():
    assert nome_pasta("2026-11-05", "Foo-7B", ["bergum"]) == "2026-11-05-reexecucao-Foo-7B-bergum"
    assert nome_pasta("2026-11-05", "Foo-7B", ["gopstein"]) == "2026-11-05-reexecucao-Foo-7B-gopstein"
    assert nome_pasta("2026-11-05", "Foo-7B", ["gopstein", "bergum"]) == "2026-11-05-reexecucao-Foo-7B-bergum-gopstein"
    assert nome_pasta("2026-11-05", "Foo-7B", ["bergum", "gopstein"], limite=10) == "2026-11-05-reexecucao-Foo-7B-amostra10"


class TokenizadorComTemplate(TokenizadorFalso):
    """Template falso: prefixo "c" (token 3) e sufixo "b" ao redor da mensagem do usuário."""

    def __init__(self, prefixo="c", sufixo="b", **kwargs):
        super().__init__(**kwargs)
        self.prefixo, self.sufixo = prefixo, sufixo

    def apply_chat_template(self, mensagens, tokenize=False):
        assert not tokenize and mensagens[0]["role"] == "user"
        return self.prefixo + mensagens[0]["content"] + self.sufixo


class ModeloContextual:
    """O próximo token depende do último: favorece (último + 1) mod 4. Guarda o tamanho dos contextos vistos."""

    config = SimpleNamespace(max_position_embeddings=16)

    def __init__(self):
        self.tamanhos = []

    def __call__(self, contexto):
        self.tamanhos.append(contexto.size(1))
        logits = torch.full((1, contexto.size(1), VOCAB), -3.0)
        for i in range(contexto.size(1)):
            logits[0, i, (int(contexto[0, i]) + 1) % VOCAB] = 0.0
        return SimpleNamespace(logits=logits)


def test_sem_a_opcao_nada_muda():
    padrao = perplexidade_por_token("abca", ModeloContextual(), TokenizadorComTemplate(), "cpu")
    explicito = perplexidade_por_token("abca", ModeloContextual(), TokenizadorComTemplate(), "cpu", chat_template=False)
    assert padrao == explicito
    # O início continua sendo o BOS: o primeiro token ('a' = 1) vem depois de 0, que favorece 1.
    assert padrao["tokens"] == ["a", "b", "c", "a"] and padrao["results"][0]["log_prob"] > -0.3
    assert nome_pasta("2026-11-05", "Foo-7B", ["bergum"]) == "2026-11-05-reexecucao-Foo-7B-bergum"


def test_com_template_so_os_tokens_do_snippet_sao_pontuados():
    sem = perplexidade_por_token("abca", ModeloContextual(), TokenizadorComTemplate(), "cpu")
    com = perplexidade_por_token("abca", ModeloContextual(), TokenizadorComTemplate(), "cpu", chat_template=True)
    assert com["tokens"] == ["a", "b", "c", "a"] == [r["token"] for r in com["results"]]
    assert [r["position"] for r in com["results"]] == [0, 1, 2, 3] and com["summary"]["token_count"] == 4
    assert [r["token"] for r in com["results"]] == [r["token"] for r in sem["results"]]
    # A posição 0 muda: o contexto é o prefixo ("c" = 3, que favorece 0), não o BOS; as demais seguem o token anterior e não mudam.
    assert com["results"][0]["log_prob"] < sem["results"][0]["log_prob"] - 1
    assert [r["log_prob"] for r in com["results"][1:]] == pytest.approx([r["log_prob"] for r in sem["results"][1:]])


def test_template_alimenta_so_prefixo_e_snippet_nao_o_sufixo():
    modelo = ModeloContextual()
    perplexidade_por_token("abca", modelo, TokenizadorComTemplate(prefixo="cab", sufixo="bbbb"), "cpu", uma_passagem=True, chat_template=True)
    assert modelo.tamanhos == [3 + 4]


def test_template_uma_passagem_igual_ao_laco():
    t = TokenizadorComTemplate(prefixo="cab")
    laco = perplexidade_por_token("abca", ModeloContextual(), t, "cpu", chat_template=True)
    unica = perplexidade_por_token("abca", ModeloContextual(), t, "cpu", uma_passagem=True, chat_template=True)
    assert [r["log_prob"] for r in unica["results"]] == pytest.approx([r["log_prob"] for r in laco["results"]])
    assert unica["tokens"] == laco["tokens"] == ["a", "b", "c", "a"]


def test_partes_do_template_e_recusa_de_template_que_altera_o_snippet():
    assert partes_do_chat_template(TokenizadorComTemplate(prefixo="ca", sufixo="b"), "abc") == ("ca", "b")
    with pytest.raises(ValueError):
        partes_do_chat_template(TokenizadorComTemplate(prefixo="abc", sufixo=""), "abc")  # o snippet aparece duas vezes
    falso = TokenizadorComTemplate()
    falso.apply_chat_template = lambda mensagens, tokenize=False: "c" + mensagens[0]["content"].strip() + "b"
    with pytest.raises(ValueError):
        partes_do_chat_template(falso, "abc ")


def test_pasta_com_template_leva_sufixo():
    assert nome_pasta("2026-11-05", "Foo-7B-Instruct", ["bergum", "gopstein"], chat_template=True) == "2026-11-05-reexecucao-Foo-7B-Instruct-bergum-gopstein-template"


# Revisões das execuções com chat template (results/extensao/2026-11-05-reexecucao-*-template).
REVISOES_TEMPLATE = {
    "Qwen/Qwen2.5-Coder-3B-Instruct": "488639f1ff808d1d3d0ba301aef8c11461451ec5",
    "deepseek-ai/deepseek-coder-7b-instruct-v1.5": "2a050a4c59d687a85324d32e147517992117ed30",
}


@pytest.mark.tokenizador
@pytest.mark.parametrize("modelo", list(REVISOES_TEMPLATE))
def test_tokenizadores_reais_pontuam_exatamente_os_tokens_do_snippet(modelo):
    from transformers import AutoTokenizer

    tokenizador = AutoTokenizer.from_pretrained(modelo, revision=REVISOES_TEMPLATE[modelo])

    class Uniforme:
        config = SimpleNamespace(max_position_embeddings=4096)

        def __call__(self, contexto):
            return SimpleNamespace(logits=torch.zeros(1, contexto.size(1), len(tokenizador)))

    codigo = "int x = (a == b) ? 1 : 2;\nfor (int i = 0; i < 3; i++) {\n    x += i;\n}\n"
    sem = perplexidade_por_token(codigo, Uniforme(), tokenizador, "cpu", uma_passagem=True)
    com = perplexidade_por_token(codigo, Uniforme(), tokenizador, "cpu", uma_passagem=True, chat_template=True)
    esperados = [tokenizador.decode([t]) for t in tokenizador(codigo, add_special_tokens=False)["input_ids"]]
    assert [r["token"] for r in com["results"]] == [r["token"] for r in sem["results"]] == esperados
    assert [r["position"] for r in com["results"]] == list(range(len(esperados)))
    prefixo, sufixo = partes_do_chat_template(tokenizador, codigo)
    assert prefixo and sufixo and prefixo + codigo + sufixo == tokenizador.apply_chat_template([{"role": "user", "content": codigo}], tokenize=False)


def test_uma_passagem_recusa_snippet_maior_que_o_contexto():
    # ModeloFalso tem contexto 16; o snippet de 16 caracteres mais o BOS dá 17 tokens.
    with pytest.raises(ValueError, match="contexto do modelo"):
        perplexidade_por_token("ab" * 8, ModeloFalso(), TokenizadorFalso(), "cpu", uma_passagem=True)
    # O laço por token usa janela e continua aceitando.
    assert perplexidade_por_token("ab" * 8, ModeloFalso(), TokenizadorFalso(), "cpu")["summary"]["token_count"] == 16


def test_resumo_guarda_hash_dos_ids_do_snippet():
    from aoc_perplexity.gopstein import hash_ids

    saida = perplexidade_por_token("abca", ModeloFalso(), TokenizadorFalso(), "cpu")
    assert saida["summary"]["token_ids_sha256"] == hash_ids([1, 2, 3, 1])
    com = perplexidade_por_token("abca", ModeloContextual(), TokenizadorComTemplate(), "cpu", chat_template=True)
    assert com["summary"]["token_ids_sha256"] == saida["summary"]["token_ids_sha256"]


class TokenizadorQueFunde(TokenizadorComTemplate):
    """Junta 'ba' num só token (id 0) quando tokeniza o texto inteiro, como um BPE que atravessa a fronteira."""

    def __call__(self, texto, return_tensors=None, add_special_tokens=False):
        ids = []
        i = 0
        while i < len(texto):
            if texto[i : i + 2] == "ba":
                ids.append(0)
                i += 2
            else:
                ids.append(" abc".index(texto[i]))
                i += 1
        return {"input_ids": torch.tensor([ids])} if return_tensors else SimpleNamespace(input_ids=ids)


def test_template_recusa_tokenizacao_separada_diferente_da_conjunta():
    from aoc_perplexity.perplexidade import conferir_tokenizacao_separada

    conferir_tokenizacao_separada(TokenizadorFalso(), "cab", "abca")
    # Prefixo "cb" + snippet "acca": separados dão ...b|a..., juntos fundem "ba".
    with pytest.raises(ValueError, match="tokenização conjunta"):
        conferir_tokenizacao_separada(TokenizadorQueFunde(), "cb", "acca")
    with pytest.raises(ValueError, match="tokenização conjunta"):
        perplexidade_por_token("acca", ModeloContextual(), TokenizadorQueFunde(prefixo="cb"), "cpu", chat_template=True)
