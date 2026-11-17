"""Utilitários compartilhados da Parte Comum (Wine Quality).

Carrega o dataset (vinho tinto + branco), define constantes e estilo de gráficos.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

RAIZ = Path(__file__).resolve().parents[2]
DATA_DIR = RAIZ / "data" / "parte_comum"
OUT_DIR = RAIZ / "out" / "parte_comum"

SEED = 42

# Nomes originais (UCI) -> nomes usados no projeto (snake_case, sem espaços)
FEATURES = [
    "fixed_acidity",
    "volatile_acidity",
    "citric_acid",
    "residual_sugar",
    "chlorides",
    "free_sulfur_dioxide",
    "total_sulfur_dioxide",
    "density",
    "ph",
    "sulphates",
    "alcohol",
]
TARGET = "quality"
TIPO = "tipo"  # 'tinto' | 'branco' (NÃO é usado como feature do K-Means)


def carregar_dados(variante: str = "ambos") -> pd.DataFrame:
    """Lê os CSVs do UCI Wine Quality.

    variante: 'ambos' (tinto + branco, padrão), 'tinto' ou 'branco'.
    Retorna DataFrame com as 11 features físico-químicas, `quality` e `tipo`.
    """
    partes = []
    if variante in ("ambos", "tinto"):
        partes.append(_ler_csv(DATA_DIR / "winequality-red.csv", "tinto"))
    if variante in ("ambos", "branco"):
        partes.append(_ler_csv(DATA_DIR / "winequality-white.csv", "branco"))
    if not partes:
        raise ValueError("variante deve ser 'ambos', 'tinto' ou 'branco'")
    return pd.concat(partes, ignore_index=True)


def _ler_csv(caminho: Path, tipo: str) -> pd.DataFrame:
    df = pd.read_csv(caminho, sep=";")
    df.columns = FEATURES + [TARGET]
    df[TIPO] = tipo
    return df


def pasta_saida(*partes: str) -> Path:
    """Cria (se preciso) e devolve out/parte_comum/<partes>."""
    p = OUT_DIR.joinpath(*partes)
    p.mkdir(parents=True, exist_ok=True)
    return p


def estilo_graficos() -> None:
    sns.set_theme(style="whitegrid", context="notebook", palette="deep")
    plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight"})


def salvar(fig, caminho: Path) -> Path:
    fig.savefig(caminho)
    plt.close(fig)
    return caminho
