"""Apoio à discussão crítica: as notas de qualidade vizinhas são distinguíveis pelos dados?

1) Tamanho de efeito (d de Cohen) entre notas vizinhas, variável a variável.
2) K-Means reavaliado com classes agrupadas (baixa ≤5 / média 6 / alta ≥7), k = nº de grupos.

Uso:  python src/parte_comum/granularidade.py
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans

import metricas
from common import FEATURES, SEED, TARGET, TIPO, carregar_dados, estilo_graficos, pasta_saida, salvar
from kmeans import N_INIT, padronizar


def cohen_d(a: pd.Series, b: pd.Series) -> float:
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((b.mean() - a.mean()) / sp) if sp > 0 else 0.0


def efeito_entre_vizinhas(df: pd.DataFrame, notas_min_n: int = 20) -> pd.DataFrame:
    """|d| entre cada par de notas consecutivas (ex.: 5 vs 6). |d|<0,2 = efeito desprezível;
    0,2–0,5 pequeno; 0,5–0,8 médio; >0,8 grande (convenção de Cohen)."""
    notas = [n for n, c in df[TARGET].value_counts().sort_index().items() if c >= notas_min_n]
    linhas = []
    for a, b in zip(notas[:-1], notas[1:]):
        ga, gb = df[df[TARGET] == a], df[df[TARGET] == b]
        d = {f: abs(cohen_d(ga[f], gb[f])) for f in FEATURES}
        linhas.append({"par": f"{a} vs {b}", "n_a": len(ga), "n_b": len(gb), "d_medio": np.mean(list(d.values())), "d_maximo": max(d.values()),
                       "variavel_d_maximo": max(d, key=d.get), **d})
    return pd.DataFrame(linhas).set_index("par")


def agrupar_notas(q: pd.Series) -> pd.Series:
    return pd.cut(q, bins=[0, 5, 6, 10], labels=[1, 2, 3]).astype(int)  # 1=baixa(≤5) 2=média(6) 3=alta(≥7)


def main() -> None:
    estilo_graficos()
    pd.set_option("display.width", 220, "display.max_columns", 30)
    saida = pasta_saida("granularidade")
    df = carregar_dados("ambos")
    resumo: dict = {}

    # (1) tamanho de efeito entre notas vizinhas — total e dentro de cada tipo de vinho
    tabs = {"todos": efeito_entre_vizinhas(df)}
    for t in ("tinto", "branco"):
        tabs[t] = efeito_entre_vizinhas(df[df[TIPO] == t])
    for nome, tab in tabs.items():
        tab.to_csv(saida / f"efeito_vizinhas_{nome}.csv")
        print(f"\n== |d| entre notas vizinhas ({nome}) ==")
        print(tab[["n_a", "n_b", "d_medio", "d_maximo", "variavel_d_maximo"]].round(2))
    resumo["d_medio_vizinhas_todos"] = tabs["todos"]["d_medio"].round(3).to_dict()
    resumo["d_maximo_vizinhas_todos"] = tabs["todos"]["d_maximo"].round(3).to_dict()

    fig, ax = plt.subplots(figsize=(11, 3.8))
    sns.heatmap(tabs["todos"][FEATURES], annot=True, fmt=".2f", cmap="YlOrRd", vmin=0, vmax=1, ax=ax)
    ax.set_title("|d de Cohen| entre notas vizinhas, por variável (todos os vinhos)")
    ax.set_ylabel("")
    salvar(fig, saida / "efeito_vizinhas_heatmap.png")

    # (2) K-Means com classes agrupadas
    X, _, _ = padronizar(df)
    y3 = agrupar_notas(df[TARGET])
    sai = {}
    for k in (2, 3):
        km = KMeans(n_clusters=k, n_init=N_INIT, random_state=SEED).fit(X)
        av = metricas.avaliar(X, y3, km.labels_)
        sai[f"k{k}"] = av
        print(f"\n== Classes agrupadas (3 níveis), K-Means k={k} ==")
        print(av["contingencia"])
        print(av["por_classe"].round(3))
        print({kk: round(v, 4) if isinstance(v, float) else v for kk, v in {**av["global"], **av["nao_supervisionadas"]}.items()})
        av["contingencia"].to_csv(saida / f"agrupadas_k{k}_matriz_retangular.csv")
        av["confusao"].to_csv(saida / f"agrupadas_k{k}_matriz_confusao.csv")
        av["por_classe"].to_csv(saida / f"agrupadas_k{k}_metricas_por_classe.csv")
        resumo[f"agrupadas_k{k}"] = {**av["global"], **av["nao_supervisionadas"], "baseline": av["baseline_classe_majoritaria"]}

    (saida / "resumo.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False), "utf-8")
    print(f"\nArquivos salvos em {saida}")


if __name__ == "__main__":
    main()
