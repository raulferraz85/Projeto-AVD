"""Padronização + K-Means em duas passadas (k livre via cotovelo; k fixo = nº de classes).

Uso:  python src/parte_comum/kmeans.py
Saídas em out/parte_comum/kmeans/ (figuras PNG, tabelas CSV, resumo JSON).
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import metricas
from common import FEATURES, SEED, TARGET, TIPO, carregar_dados, estilo_graficos, pasta_saida, salvar

K_RANGE = range(2, 11)
N_INIT = 20


# ----------------------------------------------------------------------------------------
# Pré-processamento
# ----------------------------------------------------------------------------------------


def padronizar(df: pd.DataFrame, tratamento: str = "nenhum") -> tuple[np.ndarray, StandardScaler, pd.DataFrame]:
    """Z-score (média 0, desvio 1) nas 11 features numéricas. A classe e o tipo NÃO entram.

    tratamento: 'nenhum' (padrão) ou 'winsor' (corta cada variável nos percentis 1% e 99% antes
    de padronizar — usado só como análise de sensibilidade a outliers).
    """
    base = df[FEATURES].copy()
    if tratamento == "winsor":
        base = base.clip(base.quantile(0.01), base.quantile(0.99), axis=1)
    elif tratamento != "nenhum":
        raise ValueError("tratamento deve ser 'nenhum' ou 'winsor'")
    scaler = StandardScaler().fit(base)
    return scaler.transform(base), scaler, base


# ----------------------------------------------------------------------------------------
# Passada 1 — k livre (Método do Cotovelo)
# ----------------------------------------------------------------------------------------


def varrer_k(X: np.ndarray, k_range=K_RANGE) -> pd.DataFrame:
    from sklearn.metrics import silhouette_score

    linhas = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=N_INIT, random_state=SEED).fit(X)
        linhas.append(
            {
                "k": k,
                "inercia": km.inertia_,
                "silhouette": silhouette_score(X, km.labels_, sample_size=5000, random_state=SEED),
            }
        )
    return pd.DataFrame(linhas).set_index("k")


def k_cotovelo(inercias: pd.Series) -> int:
    """Cotovelo = ponto da curva de inércia mais distante da reta que liga o primeiro ao último ponto
    (critério geométrico, equivalente ao 'kneedle'). Curvas e eixos são normalizados para [0,1]."""
    ks = np.array(inercias.index, dtype=float)
    y = inercias.values.astype(float)
    xn = (ks - ks.min()) / (ks.max() - ks.min())
    yn = (y - y.min()) / (y.max() - y.min())
    # distância de cada ponto à reta entre (0, yn[0]) e (1, yn[-1])
    p0, p1 = np.array([xn[0], yn[0]]), np.array([xn[-1], yn[-1]])
    d = p1 - p0
    dist = np.abs(d[0] * (yn - p0[1]) - d[1] * (xn - p0[0])) / np.hypot(*d)
    return int(ks[int(np.argmax(dist))])


def fig_cotovelo(varredura: pd.DataFrame, k_escolhido: int, saida) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].plot(varredura.index, varredura["inercia"], "o-")
    axes[0].axvline(k_escolhido, color="crimson", ls="--", label=f"cotovelo: k = {k_escolhido}")
    axes[0].set(title="Método do Cotovelo", xlabel="k", ylabel="Inércia (WCSS)")
    axes[0].legend()
    axes[1].plot(varredura.index, varredura["silhouette"], "s-", color="#55A868")
    axes[1].set(title="Silhouette médio por k (apoio)", xlabel="k", ylabel="silhouette")
    fig.tight_layout()
    salvar(fig, saida / "passada1_cotovelo.png")


# ----------------------------------------------------------------------------------------
# Visualizações dos grupos
# ----------------------------------------------------------------------------------------


def fig_pca(X: np.ndarray, clusters: np.ndarray, y_real: pd.Series, titulo: str, caminho) -> PCA:
    pca = PCA(n_components=2, random_state=SEED).fit(X)
    Z = pca.transform(X)
    var = pca.explained_variance_ratio_ * 100
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, cores, nome, pal in (
        (axes[0], clusters, "Clusters do K-Means", "tab10"),
        (axes[1], y_real.values, "Classe real (quality)", "viridis"),
    ):
        sc = sns.scatterplot(x=Z[:, 0], y=Z[:, 1], hue=cores, palette=pal, s=10, alpha=0.6, ax=ax, linewidth=0)
        ax.set(title=nome, xlabel=f"PC1 ({var[0]:.1f}%)", ylabel=f"PC2 ({var[1]:.1f}%)")
        ax.legend(title="", markerscale=2, fontsize=8, loc="best")
    fig.suptitle(titulo, y=1.02)
    fig.tight_layout()
    salvar(fig, caminho)
    return pca


def fig_perfil(km: KMeans, scaler: StandardScaler, titulo: str, caminho) -> pd.DataFrame:
    """Heatmap dos centroides em z-score (quanto cada cluster se afasta da média geral)."""
    z = pd.DataFrame(km.cluster_centers_, columns=FEATURES)
    z.index.name = "cluster"
    fig, ax = plt.subplots(figsize=(11, 0.55 * len(z) + 2.5))
    sns.heatmap(z, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-2.5, vmax=2.5, ax=ax)
    ax.set_title(titulo)
    salvar(fig, caminho)
    return z


def fig_confusao(cm_ou_cont: pd.DataFrame, titulo: str, caminho, normalizar_linhas: bool = True) -> None:
    dados = cm_ou_cont.astype(float)
    pct = dados.div(dados.sum(axis=1).replace(0, np.nan), axis=0).fillna(0) * 100
    fig, axes = plt.subplots(1, 2, figsize=(max(11, 1.2 * dados.shape[1] + 7), 4.8))
    sns.heatmap(dados, annot=True, fmt=".0f", cmap="Blues", ax=axes[0], cbar=False)
    axes[0].set(title="Contagem", xlabel=dados.columns.name or "cluster → classe mapeada", ylabel="classe real")
    sns.heatmap(pct, annot=True, fmt=".0f", cmap="Oranges", ax=axes[1], cbar=False)
    axes[1].set(title="% da classe real (linha)", xlabel=dados.columns.name or "cluster → classe mapeada", ylabel="")
    fig.suptitle(titulo, y=1.02)
    fig.tight_layout()
    salvar(fig, caminho)


# ----------------------------------------------------------------------------------------
# Execução de uma passada / de um experimento
# ----------------------------------------------------------------------------------------


def passada(X: np.ndarray, y: pd.Series, k: int) -> tuple[KMeans, dict]:
    km = KMeans(n_clusters=k, n_init=N_INIT, random_state=SEED).fit(X)
    return km, metricas.avaliar(X, y, km.labels_)


def _resumo_passada(av: dict, k: int, km: KMeans) -> dict:
    g = av["global"]
    return {
        "k": k,
        "inercia": float(km.inertia_),
        "mapa_cluster_para_classe": {str(c): m for c, m in av["mapa"].items()},
        **g,
        **av["nao_supervisionadas"],
        "baseline_classe_majoritaria": av["baseline_classe_majoritaria"],
    }


def salvar_passada(nome: str, km: KMeans, av: dict, X, y, df, scaler, saida) -> None:
    av["contingencia"].to_csv(saida / f"{nome}_matriz_retangular.csv")
    av["confusao"].to_csv(saida / f"{nome}_matriz_confusao.csv")
    av["por_classe"].to_csv(saida / f"{nome}_metricas_por_classe.csv")
    pd.crosstab(km.labels_, df[TIPO].values, rownames=["cluster"]).to_csv(saida / f"{nome}_cluster_x_tipo.csv")
    fig_pca(X, km.labels_, y, f"{nome.replace('passada', 'Passada ')} — k = {km.n_clusters} (projeção PCA)", saida / f"{nome}_pca.png")
    fig_perfil(km, scaler, f"{nome.replace('passada', 'Passada ')} — perfil dos centroides (z-score)", saida / f"{nome}_perfil_centroides.png")
    fig_confusao(av["contingencia"], f"{nome.replace('passada', 'Passada ')} — matriz de confusão retangular (classe real × cluster→classe mapeada)", saida / f"{nome}_matriz_retangular.png")
    fig_confusao(av["confusao"], f"{nome.replace('passada', 'Passada ')} — matriz de confusão quadrada (real × rótulo mapeado)", saida / f"{nome}_matriz_confusao.png")


def experimento(variante: str = "ambos", tratamento: str = "nenhum", salvar_saidas: bool = True, nome_pasta: str = "kmeans") -> dict:
    """Roda Passada 1 e Passada 2 para uma variante do dataset e um tratamento de outliers."""
    df = carregar_dados(variante)
    y = df[TARGET]
    n_classes = int(y.nunique())
    X, scaler, _ = padronizar(df, tratamento)

    varredura = varrer_k(X)
    k1 = k_cotovelo(varredura["inercia"])
    km1, av1 = passada(X, y, k1)
    km2, av2 = passada(X, y, n_classes)

    if salvar_saidas:
        saida = pasta_saida(nome_pasta)
        varredura.to_csv(saida / "passada1_varredura_k.csv")
        fig_cotovelo(varredura, k1, saida)
        salvar_passada("passada1", km1, av1, X, y, df, scaler, saida)
        salvar_passada("passada2", km2, av2, X, y, df, scaler, saida)

    return {
        "variante": variante,
        "tratamento": tratamento,
        "n": len(df),
        "n_classes": n_classes,
        "varredura": varredura,
        "passada1": {"km": km1, "av": av1, "resumo": _resumo_passada(av1, k1, km1)},
        "passada2": {"km": km2, "av": av2, "resumo": _resumo_passada(av2, n_classes, km2)},
        "df": df,
        "X": X,
        "scaler": scaler,
    }


def sem_padronizar_para_comparacao(df: pd.DataFrame, k: int) -> dict:
    """Mostra por que padronizar é necessário: K-Means nos dados brutos (escalas diferentes)."""
    X = df[FEATURES].values
    km = KMeans(n_clusters=k, n_init=N_INIT, random_state=SEED).fit(X)
    var = pd.Series(X.var(axis=0), index=FEATURES)
    # contribuição de cada variável para a distância euclidiana ao quadrado entre centroides
    cent = pd.DataFrame(km.cluster_centers_, columns=FEATURES)
    contrib = ((cent - cent.mean()) ** 2).sum()
    contrib = (100 * contrib / contrib.sum()).sort_values(ascending=False)
    av = metricas.avaliar(X, df[TARGET], km.labels_)
    return {"variancias": var, "contribuicao_pct": contrib, "ARI": av["nao_supervisionadas"]["ARI"], "NMI": av["nao_supervisionadas"]["NMI"]}


def fig_padronizacao(df: pd.DataFrame, X_pad: np.ndarray, saida) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    sns.boxplot(data=df[FEATURES], orient="h", ax=axes[0], fliersize=1.5, color="#8DA0CB")
    axes[0].set(title="Antes: escalas muito diferentes", xscale="symlog")
    sns.boxplot(data=pd.DataFrame(X_pad, columns=FEATURES), orient="h", ax=axes[1], fliersize=1.5, color="#66C2A5")
    axes[1].set(title="Depois: z-score (média 0, desvio 1)")
    axes[1].set_yticklabels([])
    fig.tight_layout()
    salvar(fig, saida / "padronizacao_antes_depois.png")


# ----------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------


def main() -> None:
    estilo_graficos()
    pd.set_option("display.width", 220, "display.max_columns", 30)
    saida = pasta_saida("kmeans")

    # ---- experimento principal: tinto+branco, sem remoção de outliers, z-score
    res = experimento("ambos", "nenhum")
    df, X = res["df"], res["X"]
    fig_padronizacao(df, X, saida)
    p1, p2 = res["passada1"], res["passada2"]

    print("== PASSADA 1 (k livre) ==  k =", p1["resumo"]["k"])
    print(res["varredura"].round(3))
    for nome, p in (("PASSADA 1", p1), ("PASSADA 2", p2)):
        print(f"\n== {nome}: k={p['resumo']['k']} ==")
        print("mapa cluster->classe:", p["av"]["mapa"])
        print(p["av"]["contingencia"])
        print(p["av"]["por_classe"].round(3))
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in p["resumo"].items() if k != "mapa_cluster_para_classe"})
        print(pd.crosstab(p["km"].labels_, df[TIPO].values, rownames=["cluster"]))

    # ---- por que padronizar
    bruto = sem_padronizar_para_comparacao(df, p2["resumo"]["k"])
    print("\n== SEM padronizar (k=%d) ==" % p2["resumo"]["k"])
    print("ARI bruto=%.4f | ARI padronizado=%.4f" % (bruto["ARI"], p2["resumo"]["ARI"]))
    print(bruto["contribuicao_pct"].round(2))
    bruto["variancias"].to_csv(saida / "padronizacao_variancias_brutas.csv", header=["variancia"])
    bruto["contribuicao_pct"].to_csv(saida / "padronizacao_contribuicao_distancia_bruta.csv", header=["pct"])

    # ---- sensibilidade (outliers e tipo de vinho)
    linhas = []
    cenarios = [
        ("tinto+branco, sem tratamento (PRINCIPAL)", res),
        ("tinto+branco, winsorizado 1–99%", experimento("ambos", "winsor", salvar_saidas=False)),
        ("somente tinto", experimento("tinto", "nenhum", salvar_saidas=False)),
        ("somente branco", experimento("branco", "nenhum", salvar_saidas=False)),
    ]
    for nome, r in cenarios:
        for pas in ("passada1", "passada2"):
            s = r[pas]["resumo"]
            linhas.append(
                {
                    "cenario": nome, "passada": pas[-1], "n": r["n"], "n_classes": r["n_classes"], "k": s["k"],
                    "acuracia": s["acuracia_global"], "baseline_maioria": s["baseline_classe_majoritaria"],
                    "precisao_macro": s["precisao_macro"], "recall_macro": s["recall_macro"],
                    "silhouette": s["silhouette"], "ARI": s["ARI"], "NMI": s["NMI"],
                }
            )
    sens = pd.DataFrame(linhas)
    sens.to_csv(saida / "sensibilidade_cenarios.csv", index=False)
    print("\n== SENSIBILIDADE ==")
    print(sens.round(3).to_string(index=False))

    resumo = {
        "passada1": p1["resumo"],
        "passada2": p2["resumo"],
        "k_cotovelo": p1["resumo"]["k"],
        "sem_padronizar": {"ARI": bruto["ARI"], "NMI": bruto["NMI"]},
    }
    (saida / "resumo.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False), "utf-8")
    print(f"\nArquivos salvos em {saida}")


if __name__ == "__main__":
    main()
