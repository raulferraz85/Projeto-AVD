"""Carga do TMDB 5000 (Parte Específica).

Só faz a parte *estrutural/técnica*: lê os CSVs, converte as colunas em JSON
para listas de nomes, converte a data e (opcionalmente) junta os créditos.

NÃO toma nenhuma decisão de tratamento (zeros em budget/revenue, runtime == 0,
vote_count == 0, faltantes, outliers). Isso é decisão do grupo e deve ser
feita e justificada no eda.ipynb.
"""
# [IA] Claude Code – leitura dos CSVs, parsing das colunas JSON e join com credits
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
DATA_DIR = RAIZ / "data" / "parte_especifica"
OUT_DIR = RAIZ / "out" / "parte_especifica"

ARQ_FILMES = DATA_DIR / "tmdb_5000_movies.csv"
ARQ_CREDITOS = DATA_DIR / "tmdb_5000_credits.csv"

# Colunas que vêm como texto JSON: '[{"id": 28, "name": "Action"}, ...]'
COLUNAS_JSON = [
    "genres",
    "keywords",
    "production_companies",
    "production_countries",
    "spoken_languages",
]

VARS_QUANT = ["budget", "revenue", "popularity", "runtime", "vote_average", "vote_count"]


def _nomes(texto_json: str, chave: str = "name") -> list[str]:
    """'[{"id":1,"name":"Action"}]' -> ['Action'] (vazio/NaN -> [])."""
    if not isinstance(texto_json, str) or not texto_json:
        return []
    return [item[chave] for item in json.loads(texto_json)]


def carregar_filmes(com_creditos: bool = False) -> pd.DataFrame:
    """Lê tmdb_5000_movies.csv e devolve um DataFrame com:

    - colunas JSON convertidas em listas de nomes (ex.: genres -> ['Action', 'Adventure']);
    - production_countries / spoken_languages como listas de códigos ISO;
    - release_date como datetime e coluna auxiliar `ano`;
    - se com_creditos=True: `diretor` (str) e `elenco_top3` (list[str]) vindos do credits.

    Os valores numéricos ficam exatamente como no arquivo original.
    """
    df = pd.read_csv(ARQ_FILMES)

    for col in ["genres", "keywords", "production_companies"]:
        df[col] = df[col].apply(_nomes)
    df["production_countries"] = df["production_countries"].apply(_nomes, chave="iso_3166_1")
    df["spoken_languages"] = df["spoken_languages"].apply(_nomes, chave="iso_639_1")

    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["ano"] = df["release_date"].dt.year.astype("Int64")

    if com_creditos:
        cred = pd.read_csv(ARQ_CREDITOS, usecols=["movie_id", "cast", "crew"])
        cred["diretor"] = cred["crew"].apply(_diretor)
        cred["elenco_top3"] = cred["cast"].apply(lambda t: _nomes(t)[:3])
        df = df.merge(
            cred[["movie_id", "diretor", "elenco_top3"]],
            left_on="id", right_on="movie_id", how="left",
        ).drop(columns="movie_id")

    return df


def _diretor(texto_json: str) -> str | None:
    if not isinstance(texto_json, str):
        return None
    for pessoa in json.loads(texto_json):
        if pessoa.get("job") == "Director":
            return pessoa["name"]
    return None


def explodir(df: pd.DataFrame, coluna: str) -> pd.DataFrame:
    """Uma linha por (filme, item da lista). Ex.: explodir(df, 'genres') para contar gêneros.

    Atenção: um filme com 3 gêneros aparece 3 vezes — não somar métricas por filme
    depois de explodir sem levar isso em conta.
    """
    return df.explode(coluna)


def pasta_saida(*partes: str) -> Path:
    """Cria (se preciso) e devolve out/parte_especifica/<partes>."""
    p = OUT_DIR.joinpath(*partes)
    p.mkdir(parents=True, exist_ok=True)
    return p
