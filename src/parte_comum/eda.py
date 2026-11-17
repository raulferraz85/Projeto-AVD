"""EDA do Wine Quality nos quatro pilares: estrutural, distribucional, relacional e outliers.

Uso:  python src/parte_comum/eda.py
Saídas (tabelas CSV + figuras PNG) em out/parte_comum/eda/.
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from common import FEATURES, TARGET, TIPO, carregar_dados, estilo_graficos, pasta_saida, salvar

# ----------------------------------------------------------------------------------------
# 1) ESTRUTURAL
# ----------------------------------------------------------------------------------------


def estrutural(df: pd.DataFrame) -> dict:
    """Dimensões, tipos, nulos, duplicatas, cardinalidade e consistência física básica."""
    info = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "nulos": df.isna().sum(),
            "valores_unicos": df.nunique(),
            "min": df.select_dtypes("number").min(),
            "max": df.select_dtypes("number").max(),
        }
    )
    resumo = {
        "linhas": int(df.shape[0]),
        "colunas": int(df.shape[1]),
        "nulos_total": int(df.isna().sum().sum()),
        "duplicatas": int(df.duplicated().sum()),
        "duplicatas_pct": round(100 * df.duplicated().mean(), 2),
        "n_classes": int(df[TARGET].nunique()),
        "classes": sorted(int(c) for c in df[TARGET].unique()),
        "free_so2_maior_que_total": int((df["free_sulfur_dioxide"] > df["total_sulfur_dioxide"]).sum()),
        "por_tipo": df[TIPO].value_counts().to_dict(),
    }
    return {"info": info, "resumo": resumo}


# ----------------------------------------------------------------------------------------
# 2) DISTRIBUCIONAL
# ----------------------------------------------------------------------------------------


def distribucional(df: pd.DataFrame) -> dict:
    """Estatísticas descritivas, assimetria, curtose e teste de normalidade por variável."""
    desc = df[FEATURES].describe().T
    desc["assimetria"] = df[FEATURES].skew()
    desc["curtose"] = df[FEATURES].kurt()
    # Shapiro-Wilk perde sentido com n grande (rejeita tudo); D'Agostino-Pearson é mais estável
    desc["p_normalidade"] = [stats.normaltest(df[c]).pvalue for c in FEATURES]
    freq = df[TARGET].value_counts().sort_index()
    dist_classes = pd.DataFrame({"n": freq, "pct": (100 * freq / freq.sum()).round(2)})
    por_tipo = pd.crosstab(df[TARGET], df[TIPO])
    return {"descritiva": desc, "classes": dist_classes, "classes_por_tipo": por_tipo}


def figs_distribucional(df: pd.DataFrame, saida) -> None:
    # histogramas + KDE de cada variável
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    for ax, col in zip(axes.ravel(), FEATURES):
        sns.histplot(df[col], kde=True, ax=ax, bins=40)
        ax.set_title(f"{col}\nassimetria={df[col].skew():.2f}")
        ax.set_xlabel("")
    axes.ravel()[-1].axis("off")
    fig.suptitle("Distribuição das variáveis físico-químicas", y=1.01, fontsize=14)
    fig.tight_layout()
    salvar(fig, saida / "dist_histogramas.png")

    # distribuição da classe (qualidade), total e por tipo
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.countplot(data=df, x=TARGET, ax=axes[0], color="#4C72B0")
    for c in axes[0].containers:
        axes[0].bar_label(c)
    axes[0].set_title("Quantidade de vinhos por nota de qualidade")
    sns.countplot(data=df, x=TARGET, hue=TIPO, ax=axes[1])
    axes[1].set_title("Nota de qualidade por tipo de vinho")
    fig.tight_layout()
    salvar(fig, saida / "dist_classes.png")


# ----------------------------------------------------------------------------------------
# 3) RELACIONAL
# ----------------------------------------------------------------------------------------


def relacional(df: pd.DataFrame) -> dict:
    """Correlações (Pearson e Spearman) entre todas as variáveis, incluindo a classe."""
    cols = FEATURES + [TARGET]
    pearson = df[cols].corr(method="pearson")
    spearman = df[cols].corr(method="spearman")
    # associação de cada feature com a classe (Spearman, pois a nota é ordinal)
    com_classe = pd.DataFrame(
        {
            "spearman_com_quality": spearman[TARGET].drop(TARGET),
            "pearson_com_quality": pearson[TARGET].drop(TARGET),
        }
    ).sort_values("spearman_com_quality", key=np.abs, ascending=False)
    # Kruskal-Wallis: a feature muda entre as notas de qualidade?
    kw = []
    for c in FEATURES:
        grupos = [g[c].values for _, g in df.groupby(TARGET)]
        h, p = stats.kruskal(*grupos)
        kw.append({"variavel": c, "H": h, "p_valor": p})
    kw = pd.DataFrame(kw).set_index("variavel")
    # pares de variáveis fortemente correlacionadas (multicolinearidade)
    pares = []
    for i, a in enumerate(FEATURES):
        for b in FEATURES[i + 1 :]:
            pares.append({"var_a": a, "var_b": b, "pearson": pearson.loc[a, b], "spearman": spearman.loc[a, b]})
    pares = pd.DataFrame(pares).sort_values("pearson", key=np.abs, ascending=False)
    # quanto as features separam tinto x branco (explica o 'efeito tipo' no K-Means)
    sep_tipo = pd.DataFrame(
        {
            "diferenca_padronizada": {
                c: (df.loc[df[TIPO] == "branco", c].mean() - df.loc[df[TIPO] == "tinto", c].mean()) / df[c].std()
                for c in FEATURES
            }
        }
    ).sort_values("diferenca_padronizada", key=np.abs, ascending=False)
    return {
        "pearson": pearson,
        "spearman": spearman,
        "com_classe": com_classe,
        "kruskal": kw,
        "pares": pares,
        "sep_tipo": sep_tipo,
    }


def figs_relacional(df: pd.DataFrame, rel: dict, saida) -> None:
    for nome in ("pearson", "spearman"):
        fig, ax = plt.subplots(figsize=(10, 8))
        m = rel[nome]
        sns.heatmap(
            m, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, center=0,
            mask=np.triu(np.ones_like(m, dtype=bool), k=1), ax=ax, cbar_kws={"shrink": 0.8},
        )
        ax.set_title(f"Correlação de {nome.capitalize()} (variáveis + quality)")
        salvar(fig, saida / f"corr_{nome}.png")

    # boxplots de cada variável por nota de qualidade
    fig, axes = plt.subplots(3, 4, figsize=(18, 11))
    for ax, col in zip(axes.ravel(), FEATURES):
        sns.boxplot(data=df, x=TARGET, y=col, ax=ax, color="#8DA0CB", fliersize=2)
        ax.set_title(col)
        ax.set_xlabel("")
    axes.ravel()[-1].axis("off")
    fig.suptitle("Variáveis físico-químicas por nota de qualidade", y=1.01, fontsize=14)
    fig.tight_layout()
    salvar(fig, saida / "rel_boxplots_por_classe.png")

    # variáveis x tipo de vinho
    fig, axes = plt.subplots(3, 4, figsize=(18, 11))
    for ax, col in zip(axes.ravel(), FEATURES):
        sns.boxplot(data=df, x=TIPO, y=col, ax=ax, hue=TIPO, fliersize=2, legend=False)
        ax.set_title(col)
        ax.set_xlabel("")
    axes.ravel()[-1].axis("off")
    fig.suptitle("Variáveis físico-químicas por tipo de vinho", y=1.01, fontsize=14)
    fig.tight_layout()
    salvar(fig, saida / "rel_boxplots_por_tipo.png")

    # pairplot das variáveis mais associadas à classe
    top = list(rel["com_classe"].index[:4])
    g = sns.pairplot(df.sample(2000, random_state=42), vars=top, hue=TARGET, palette="viridis",
                     plot_kws={"s": 12, "alpha": 0.6}, corner=True, diag_kind="kde")
    g.figure.suptitle("Pairplot — 4 variáveis mais associadas à qualidade (amostra de 2000)", y=1.02)
    salvar(g.figure, saida / "rel_pairplot_top4.png")


# ----------------------------------------------------------------------------------------
# 4) OUTLIERS
# ----------------------------------------------------------------------------------------


def outliers(df: pd.DataFrame, limite_z: float = 3.0, k_iqr: float = 1.5) -> dict:
    """Detecta outliers por IQR (Tukey) e z-score, por variável e por linha."""
    q1, q3 = df[FEATURES].quantile(0.25), df[FEATURES].quantile(0.75)
    iqr = q3 - q1
    inf, sup = q1 - k_iqr * iqr, q3 + k_iqr * iqr
    mascara_iqr = (df[FEATURES] < inf) | (df[FEATURES] > sup)
    z = (df[FEATURES] - df[FEATURES].mean()) / df[FEATURES].std()
    mascara_z = z.abs() > limite_z
    tabela = pd.DataFrame(
        {
            "limite_inf_iqr": inf,
            "limite_sup_iqr": sup,
            "n_outliers_iqr": mascara_iqr.sum(),
            "pct_iqr": (100 * mascara_iqr.mean()).round(2),
            "n_outliers_z3": mascara_z.sum(),
            "pct_z3": (100 * mascara_z.mean()).round(2),
            "max_|z|": z.abs().max().round(1),
        }
    )
    # linhas com muitos outliers simultâneos (candidatas a erro de medição)
    n_por_linha = mascara_z.sum(axis=1)
    linhas_multi = df.loc[n_por_linha >= 3].assign(n_outliers_z3=n_por_linha[n_por_linha >= 3])
    # outliers por nota de qualidade: eles se concentram nas classes raras?
    por_classe = pd.DataFrame(
        {
            "n_vinhos": df.groupby(TARGET).size(),
            "com_algum_outlier_iqr": mascara_iqr.any(axis=1).groupby(df[TARGET]).sum(),
        }
    )
    por_classe["pct"] = (100 * por_classe["com_algum_outlier_iqr"] / por_classe["n_vinhos"]).round(1)
    # extremos mais críticos (z mais alto)
    extremos = (
        z.abs().stack().sort_values(ascending=False).head(10).rename("|z|").reset_index()
        .rename(columns={"level_0": "linha", "level_1": "variavel"})
    )
    extremos["valor"] = [df.loc[r, v] for r, v in zip(extremos["linha"], extremos["variavel"])]
    return {
        "tabela": tabela,
        "linhas_multi": linhas_multi,
        "por_classe": por_classe,
        "extremos": extremos,
        "n_linhas_com_outlier_iqr": int(mascara_iqr.any(axis=1).sum()),
        "pct_linhas_com_outlier_iqr": round(100 * mascara_iqr.any(axis=1).mean(), 1),
        "limites": (inf, sup),
    }


def figs_outliers(df: pd.DataFrame, out: dict, saida) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    for ax, col in zip(axes.ravel(), FEATURES):
        sns.boxplot(x=df[col], ax=ax, color="#66C2A5", fliersize=2.5, orient="h")
        ax.set_title(f"{col}  ({out['tabela'].loc[col, 'pct_iqr']}% fora do IQR)")
        ax.set_xlabel("")
    axes.ravel()[-1].axis("off")
    fig.suptitle("Boxplots — outliers (critério de Tukey, 1,5·IQR)", y=1.01, fontsize=14)
    fig.tight_layout()
    salvar(fig, saida / "outliers_boxplots.png")


# ----------------------------------------------------------------------------------------
# Orquestração
# ----------------------------------------------------------------------------------------


def main() -> None:
    estilo_graficos()
    df = carregar_dados("ambos")
    saida = pasta_saida("eda")

    est = estrutural(df)
    est["info"].to_csv(saida / "estrutural_info.csv")
    (saida / "estrutural_resumo.json").write_text(json.dumps(est["resumo"], indent=2, ensure_ascii=False), "utf-8")

    dist = distribucional(df)
    dist["descritiva"].to_csv(saida / "distribucional_descritiva.csv")
    dist["classes"].to_csv(saida / "distribucional_classes.csv")
    dist["classes_por_tipo"].to_csv(saida / "distribucional_classes_por_tipo.csv")
    figs_distribucional(df, saida)

    rel = relacional(df)
    for k in ("pearson", "spearman", "com_classe", "kruskal", "pares", "sep_tipo"):
        rel[k].to_csv(saida / f"relacional_{k}.csv")
    figs_relacional(df, rel, saida)

    out = outliers(df)
    out["tabela"].to_csv(saida / "outliers_tabela.csv")
    out["por_classe"].to_csv(saida / "outliers_por_classe.csv")
    out["extremos"].to_csv(saida / "outliers_extremos.csv", index=False)
    out["linhas_multi"].to_csv(saida / "outliers_linhas_multi.csv")
    figs_outliers(df, out, saida)

    pd.set_option("display.width", 200, "display.max_columns", 30)
    print("== ESTRUTURAL ==\n", json.dumps(est["resumo"], indent=2, ensure_ascii=False))
    print("\n== DISTRIBUCIONAL ==\n", dist["descritiva"][["mean", "std", "min", "max", "assimetria", "curtose"]].round(3))
    print(dist["classes"])
    print("\n== RELACIONAL ==\n", rel["com_classe"].round(3))
    print(rel["pares"].head(6).round(3))
    print(rel["sep_tipo"].round(2))
    print("\n== OUTLIERS ==\n", out["tabela"])
    print(out["por_classe"])
    print(f"Linhas com >=1 outlier (IQR): {out['n_linhas_com_outlier_iqr']} ({out['pct_linhas_com_outlier_iqr']}%)")
    print(out["extremos"])
    print(f"Linhas com >=3 outliers z>3 simultâneos: {len(out['linhas_multi'])}")
    print(f"\nArquivos salvos em {saida}")


if __name__ == "__main__":
    main()
