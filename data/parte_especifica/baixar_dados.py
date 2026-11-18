"""Baixa o TMDB 5000 Movie Dataset (Kaggle) para esta pasta.

Uso:  python data/parte_especifica/baixar_dados.py

Os CSVs já estão versionados no repositório; este script serve só para
reproduzir o download do zero. Não precisa de login no Kaggle.
"""
# [IA] Claude Code – script de download (urllib + zipfile)
from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

URL = "https://www.kaggle.com/api/v1/datasets/download/tmdb/tmdb-movie-metadata"
DESTINO = Path(__file__).resolve().parent
ARQUIVOS = ("tmdb_5000_movies.csv", "tmdb_5000_credits.csv")


def main() -> None:
    print(f"Baixando {URL} ...")
    with urllib.request.urlopen(URL) as resp:
        conteudo = resp.read()
    with zipfile.ZipFile(io.BytesIO(conteudo)) as zf:
        for nome in ARQUIVOS:
            (DESTINO / nome).write_bytes(zf.read(nome))
            print(f"  -> {DESTINO / nome}")


if __name__ == "__main__":
    main()
