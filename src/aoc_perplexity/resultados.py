"""Onde ficam os resultados e como registrá-los.

- `results/manifesto.json` lista as pastas publicadas que cada etapa lê (execuções, trios, consolidado, EEG, figuras).
  As etapas recebem `--manifesto` para ler outro conjunto, por exemplo o gerado por `aoc_perplexity.reproduzir`.
- Resultados novos vão para `results/novos/` (fora do Git) por padrão: as pastas publicadas não são sobrescritas.
- `pasta_nova` grava numa pasta temporária e só a renomeia no fim, para não deixar saída sem `run.json`.
- `proveniencia` registra o commit quando o projeto é um clone Git; sem Git (por exemplo, um ZIP), registra a versão do pacote.
"""

import json
import os
import shutil
import subprocess
from contextlib import contextmanager
from datetime import date
from pathlib import Path

from aoc_perplexity import __version__

RAIZ_PROJETO = Path(__file__).resolve().parents[2]
MANIFESTO = Path("results/manifesto.json")
NOVOS = Path("results/novos")
CAMINHOS_DE_CODIGO = ["src", "tests", "pyproject.toml", "uv.lock"]


def _git(*argumentos):
    return subprocess.run(["git", "-C", str(RAIZ_PROJETO), *argumentos], capture_output=True, text=True, check=False)


def proveniencia():
    """commit, codigo_alterado e versao do código que gera o resultado.

    Só usa o Git se a raiz do repositório for a do projeto (evita registrar o commit de um repositório ancestral).
    Sem Git ou sem commit, `commit` e `codigo_alterado` ficam nulos e `versao` identifica o pacote.
    """
    registro = {"commit": None, "codigo_alterado": None, "versao": __version__}
    try:
        raiz = _git("rev-parse", "--show-toplevel")
    except FileNotFoundError:  # git não instalado
        return registro
    if raiz.returncode != 0 or Path(raiz.stdout.strip()).resolve() != RAIZ_PROJETO:
        return registro
    head = _git("rev-parse", "HEAD")
    if head.returncode != 0:
        return registro
    registro["commit"] = head.stdout.strip()
    registro["codigo_alterado"] = bool(_git("status", "--porcelain", "--", *CAMINHOS_DE_CODIGO).stdout)
    return registro


def commit_externo(pasta):
    """Commit de um checkout de dados externo (por exemplo, o pacote dos autores), ou None se não for um clone Git."""
    try:
        saida = subprocess.run(["git", "-C", str(pasta), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return None
    return saida.stdout.strip() if saida.returncode == 0 else None


def nome_datado(saida, assunto, numerar=True):
    """`AAAA-MM-DD-assunto`; com `numerar`, um nome já usado no dia ganha `-2`, `-3`..."""
    hoje = date.today().isoformat()
    destino = Path(saida) / f"{hoje}-{assunto}"
    sufixo = 1
    while numerar and destino.exists():
        sufixo += 1
        destino = Path(saida) / f"{hoje}-{assunto}-{sufixo}"
    return destino


@contextmanager
def pasta_nova(destino):
    """Grava numa pasta temporária ao lado de `destino` e a renomeia só se tudo der certo. Não sobrescreve."""
    destino = Path(destino)
    if destino.exists():
        raise FileExistsError(f"{destino} já existe; resultados não são sobrescritos")
    destino.parent.mkdir(parents=True, exist_ok=True)
    temporaria = destino.parent / f".incompleto-{destino.name}-{os.getpid()}"
    shutil.rmtree(temporaria, ignore_errors=True)
    temporaria.mkdir()
    try:
        yield temporaria
        temporaria.rename(destino)
    except BaseException:
        shutil.rmtree(temporaria, ignore_errors=True)
        raise


def escrever_json(caminho, dados):
    Path(caminho).write_text(json.dumps(dados, indent=2, ensure_ascii=False) + "\n")


class Manifesto:
    """Pastas de resultados por papel. Os caminhos do arquivo são relativos à raiz do projeto."""

    def __init__(self, dados, caminho=None):
        self.dados = dados
        self.caminho = Path(caminho) if caminho else None

    @classmethod
    def ler(cls, caminho=MANIFESTO):
        caminho = Path(caminho)
        if not caminho.is_absolute() and not caminho.exists():
            caminho = RAIZ_PROJETO / caminho
        return cls(json.loads(caminho.read_text()), caminho)

    def pasta(self, chave):
        """Pasta de uma etapa (consolidado, eeg, sensibilidade, template, figuras, figuras_grade)."""
        return Path(self.dados["etapas"][chave])

    def trios(self, grupo="principais"):
        """{(família, conjunto): pasta} na ordem do arquivo. Grupos: principais, template, atomos."""
        return {tuple(chave.split("|")): Path(pasta) for chave, pasta in self.dados["trios"][grupo].items()}

    def com(self, trios=None, etapas=None):
        """Cópia com grupos de trios ({grupo: {chave: pasta}}) ou etapas ({etapa: pasta}) trocados (usada por `reproduzir`)."""
        dados = json.loads(json.dumps(self.dados))
        for grupo, valor in (trios or {}).items():
            dados["trios"][grupo] = {"|".join(k): str(v) for k, v in valor.items()}
        for etapa, valor in (etapas or {}).items():
            dados["etapas"][etapa] = str(valor)
        return Manifesto(dados)

    def gravar(self, caminho):
        escrever_json(caminho, self.dados)
        self.caminho = Path(caminho)
