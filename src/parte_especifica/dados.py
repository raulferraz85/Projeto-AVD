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

import numpy as np
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


# ------------------------------------------------------------------------------------------
# Tratamento — as escolhas ficam em DECISOES (justificativas no eda.ipynb, seções 2 e 3).
# Todo mundo usa:  df = tratar(carregar_filmes())
# ------------------------------------------------------------------------------------------
# [IA] Claude Code – estrutura parametrizada do tratamento
DECISOES = {
    # budget/revenue == 0 significam "não informado".
    #   "nan"    -> vira NaN (linha fica no dataset; só sai das contas que usam a coluna)
    #   "manter" -> deixa o 0
    "zeros_financeiros": "nan",
    # budget/revenue positivos mas abaixo deste valor (US$) são tratados como erro -> NaN.
    #   None -> não mexe.  Ex.: 1000
    "limite_implausivel": 1000,
    # runtime == 0  ->  "nan" | "manter"
    "runtime_zero": "nan",
    # vote_count == 0: a nota 0 não é uma avaliação real.
    #   "nan"    -> vote_average vira NaN nesses filmes
    #   "manter" -> deixa a nota 0
    "sem_votos": "nan",
    # colunas descartadas da análise (ex.: ["homepage", "tagline"]). Lista vazia = não descarta nada.
    "remover_colunas": ["homepage"],
    # filmes com status != "Released" (Rumored / Post Production) ficam sem retorno calculado
    "retorno_so_lancados": True,
    # remove as linhas sem release_date (necessária para ano/decada)
    "remover_sem_data": True,
}


def tratar(df: pd.DataFrame, decisoes: dict | None = None, verbose: bool = False) -> pd.DataFrame:
    """Aplica o tratamento de dados definido em DECISOES (ou no dict `decisoes`, para testar alternativas).

    Só remove linhas se remover_sem_data=True; nos demais casos valores inválidos viram NaN
    e flags indicam o que estava faltando.
    Cria também as colunas derivadas usadas na pergunta de pesquisa:
      - orcamento_informado, receita_informada (bool)
      - retorno = revenue / budget (NaN se um dos dois não for válido)
    Gêneros: cada filme conta em TODOS os seus gêneros (decisão do EDA, seção 1.2).
    Para analisar por gênero use explodir(df, "genres").
    """
    d = {**DECISOES, **(decisoes or {})}
    out = df.copy()
    log = {}

    if d.get("remover_sem_data"):
        sem_data = out["release_date"].isna()
        log["linhas sem data removidas"] = int(sem_data.sum())
        out = out[~sem_data].reset_index(drop=True)

    for col in ["budget", "revenue"]:
        out[col] = out[col].astype(float)
        invalido = pd.Series(False, index=out.index)
        if d["zeros_financeiros"] == "nan":
            invalido |= out[col] == 0
        if d["limite_implausivel"] is not None:
            invalido |= (out[col] > 0) & (out[col] < d["limite_implausivel"])
        log[f"{col} -> NaN"] = int(invalido.sum())
        out.loc[invalido, col] = np.nan

    if d["runtime_zero"] == "nan":
        z = out["runtime"] == 0
        log["runtime 0 -> NaN"] = int(z.sum())
        out.loc[z, "runtime"] = np.nan

    if d["sem_votos"] == "nan":
        z = out["vote_count"] == 0
        log["vote_average sem votos -> NaN"] = int(z.sum())
        out.loc[z, "vote_average"] = np.nan

    if d["remover_colunas"]:
        out = out.drop(columns=d["remover_colunas"])
        log["colunas removidas"] = list(d["remover_colunas"])

    out["orcamento_informado"] = out["budget"].notna() & (out["budget"] > 0)
    out["receita_informada"] = out["revenue"].notna() & (out["revenue"] > 0)
    validos = out["orcamento_informado"] & out["receita_informada"]
    if d.get("retorno_so_lancados"):
        nao_lancado = validos & (out["status"] != "Released")
        log["não lançados fora do retorno"] = int(nao_lancado.sum())
        validos &= ~nao_lancado
    out["retorno"] = np.where(validos, out["revenue"] / out["budget"], np.nan)
    log["filmes com retorno calculável"] = int(validos.sum())

    if verbose:
        for k, v in log.items():
            print(f"{k:32s} {v}")
    return out


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
