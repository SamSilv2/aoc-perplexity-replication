import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(not all(shutil.which(p) for p in ["make", "curl", "shasum"]), reason="precisa de make, curl e shasum")


def baixar(origem, destino):
    return subprocess.run(["make", "-s", "data-gopstein", f"GOPSTEIN_URL=file://{origem}", f"GOPSTEIN_DIR={destino}"], capture_output=True, text=True)


def test_download_parcial_falha_e_nao_deixa_arquivos(tmp_path):
    origem = tmp_path / "origem"
    origem.mkdir()
    (origem / "code_tags.csv").write_text("só um dos dois arquivos\n")
    saida = baixar(origem, tmp_path / "destino")
    assert saida.returncode != 0 and "code.csv" in saida.stderr
    assert not (tmp_path / "destino").exists()


def test_arquivo_com_hash_diferente_falha_e_nao_deixa_arquivos(tmp_path):
    origem = tmp_path / "origem"
    origem.mkdir()
    for nome in ["code.csv", "code_tags.csv"]:
        (origem / nome).write_text("conteúdo diferente do publicado\n")
    saida = baixar(origem, tmp_path / "destino")
    assert saida.returncode != 0 and "SHA-256" in saida.stderr
    assert not (tmp_path / "destino").exists()


@pytest.mark.dados
def test_arquivos_corretos_sao_aceitos(tmp_path):
    saida = baixar(Path("data/external/gopstein").resolve(), tmp_path / "destino")
    assert saida.returncode == 0, saida.stderr
    assert sorted(p.name for p in (tmp_path / "destino").iterdir()) == ["code.csv", "code_tags.csv"]
