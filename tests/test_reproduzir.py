import json
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from aoc_perplexity import comparar, reproduzir, resultados
from aoc_perplexity.resultados import Manifesto


def test_manifesto_publicado_aponta_para_pastas_existentes():
    manifesto = Manifesto.ler()
    for grupo, esperado in [("principais", 16), ("template", 4), ("atomos", 16)]:
        trios = manifesto.trios(grupo)
        assert len(trios) == esperado
        assert all((pasta / "trio.csv").is_file() for pasta in trios.values())
    for etapa in manifesto.dados["etapas"]:
        assert manifesto.pasta(etapa).is_dir(), etapa
    for execucao in reproduzir.execucoes_do_manifesto(manifesto):
        assert (Path(execucao) / "run.json").is_file(), execucao


def test_mostrar_lista_todas_as_etapas_com_revisoes_fixadas(capsys):
    assert reproduzir.main(["--etapas", "todas", "--mostrar", "--raiz", "results/novos/teste-mostrar"]) == 0
    linhas = capsys.readouterr().out.splitlines()
    trios = [l for l in linhas if "aoc_perplexity.trio " in l]
    inferencias = [l for l in linhas if "aoc_perplexity.perplexidade " in l]
    assert len(trios) == 36  # 16 principais, 4 com template e 16 de subconjuntos de átomos
    assert all(l.count("--revision ") == 3 and "--pasta " in l for l in trios)
    assert sum("--atomos " in l for l in trios) == 16
    assert len(inferencias) == len(reproduzir.execucoes_do_manifesto(Manifesto.ler(), replicacao=True)) == 31
    assert all("--revision " in l and "--tokenizer-revision " in l and "--pasta " in l for l in inferencias)
    assert sum("--chat-template" in l for l in inferencias) == 2
    for modulo in ["sensibilidade", "consolidar", "eeg_extensao", "figuras", "figuras_grade", "template_instruct"]:
        assert any(f"aoc_perplexity.{modulo} --manifesto " in l for l in linhas), modulo
    assert not Path("results/novos/teste-mostrar").exists()


def test_trios_da_reproducao_usam_as_execucoes_novas():
    pasta = Manifesto.ler().trios()[("Qwen2.5 3B", "bergum")]
    antigas = json.loads((pasta / "run.json").read_text())["config"]["execucoes"]
    novas = {antigas["base"]: Path("raiz/execucoes/base")}
    argumentos = reproduzir.argumentos_trio(pasta, novas, Path("raiz/trios/x"))
    assert argumentos[argumentos.index("--base") + 1] == "raiz/execucoes/base"
    assert argumentos[argumentos.index("--codigo") + 1] == antigas["codigo"]


def test_pasta_nova_nao_deixa_saida_incompleta(tmp_path):
    destino = tmp_path / "saida"
    with pytest.raises(RuntimeError):
        with resultados.pasta_nova(destino) as temporaria:
            (temporaria / "parcial.csv").write_text("x")
            raise RuntimeError("falhou no meio")
    assert not destino.exists() and list(tmp_path.iterdir()) == []
    with resultados.pasta_nova(destino) as temporaria:
        (temporaria / "run.json").write_text("{}")
    assert (destino / "run.json").is_file()
    with pytest.raises(FileExistsError):
        with resultados.pasta_nova(destino):
            pass


def test_proveniencia_sem_git_registra_a_versao(monkeypatch, tmp_path):
    monkeypatch.setattr(resultados, "RAIZ_PROJETO", tmp_path)  # pasta sem .git
    registro = resultados.proveniencia()
    assert registro == {"commit": None, "codigo_alterado": None, "versao": resultados.__version__}


def test_proveniencia_ignora_repositorio_ancestral(monkeypatch, tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    projeto = tmp_path / "projeto"
    projeto.mkdir()
    monkeypatch.setattr(resultados, "RAIZ_PROJETO", projeto)
    assert resultados.proveniencia()["commit"] is None


def test_comparador_distingue_esquema_de_numero(tmp_path):
    publicada, nova = tmp_path / "a", tmp_path / "b"
    publicada.mkdir()
    nova.mkdir()
    pd.DataFrame({"n": [1, 2], "x": [0.5, 0.25]}).to_csv(publicada / "por_modelo.csv", index=False)
    pd.DataFrame({"n_wilcoxon": [1, 2], "x": [0.5, 0.25]}).to_csv(nova / "por_modelo.csv", index=False)
    pd.DataFrame({"x": [1.0]}).to_csv(publicada / "trio.csv", index=False)
    pd.DataFrame({"x": [1.0 + 1e-6]}).to_csv(nova / "trio.csv", index=False)
    (publicada / "run.json").write_text('{"date": "1"}')
    (nova / "run.json").write_text('{"date": "2"}')
    classes = {nome: classe for nome, classe, _ in comparar.comparar_pastas(publicada, nova)}
    assert classes == {"por_modelo.csv": "esquema", "trio.csv": "numérico", "run.json": "metadados"}
    assert not comparar.aprovado(comparar.comparar_pastas(publicada, nova))
    assert comparar.aprovado(comparar.comparar_pastas(publicada, nova, tolerancia_csv=1e-5))


def test_saidas_padrao_ficam_fora_dos_resultados_publicados():
    from aoc_perplexity import consolidar, eeg_extensao, rq12, sensibilidade, template_instruct, trio

    for modulo in [eeg_extensao, sensibilidade, trio, rq12]:
        assert modulo.SAIDA == resultados.NOVOS
    assert template_instruct.EXTENSAO == resultados.NOVOS
    assert "results/novos/" in Path(".gitignore").read_text()
    assert consolidar.TRIOS == Manifesto.ler().trios()


def _par(tmp_path, nome, publicado, novo):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(exist_ok=True)
    b.mkdir(exist_ok=True)
    (a / nome).write_text(publicado)
    (b / nome).write_text(novo)
    return a, b


def test_comparador_nao_arredonda_valores_proximos(tmp_path):
    a, b = _par(tmp_path, "t.csv", "x\n0.123456789011\n", "x\n0.123456789012\n")
    assert [c for _, c, _ in comparar.comparar_pastas(a, b)] == ["numérico"]
    assert comparar.aprovado(comparar.comparar_pastas(a, b, tolerancia_csv=1e-11))


def test_comparador_distingue_infinitos_e_ausentes(tmp_path):
    a, b = _par(tmp_path, "t.csv", "x\ninf\n", "x\n-inf\n")
    assert not comparar.aprovado(comparar.comparar_pastas(a, b, tolerancia_csv=1e9))
    a, b = _par(tmp_path, "u.csv", "x\n\n", "x\nnan\n")
    assert not comparar.aprovado(comparar.comparar_pastas(a, b, tolerancia_csv=1e9))


def test_comparador_distingue_booleano_de_numero(tmp_path):
    a, b = _par(tmp_path, "t.json", '{"x": true}', '{"x": 1}')
    assert [c for _, c, _ in comparar.comparar_pastas(a, b, tolerancia_json=1e9)] == ["numérico"]


def test_comparador_recusa_pastas_inexistentes_ou_sem_dados(tmp_path):
    assert not comparar.aprovado(comparar.comparar_pastas(tmp_path / "nao", tmp_path / "existe"))
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "run.json").write_text("{}")
    (tmp_path / "b" / "run.json").write_text("{}")
    assert not comparar.aprovado(comparar.comparar_pastas(tmp_path / "a", tmp_path / "b"))


def test_hash_dos_tokens_acrescentado_e_migracao_declarada(tmp_path):
    publicado = json.dumps({"s": {"results": [{"log_prob": -1.0}], "summary": {"ppl": 2.0}}})
    novo = json.dumps({"s": {"results": [{"log_prob": -1.0}], "summary": {"ppl": 2.0, "token_ids_sha256": "abc"}}})
    a, b = _par(tmp_path, "perplexidade.json", publicado, novo)
    assert [c for _, c, _ in comparar.comparar_pastas(a, b)] == ["esquema"]
    outra = json.dumps({"s": {"results": [{"log_prob": -1.0}], "summary": {"ppl": 2.0, "outra": 1}}})
    a, b = _par(tmp_path, "perplexidade.json", publicado, outra)
    assert not comparar.aprovado(comparar.comparar_pastas(a, b))
