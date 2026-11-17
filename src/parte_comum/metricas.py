"""Mapeamento cluster -> classe majoritária, matriz de confusão e métricas por classe.

O K-Means NÃO usa os rótulos. Os rótulos reais só entram aqui, DEPOIS do agrupamento, para
medir quanto os grupos descobertos coincidem com as classes conhecidas.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_score,
)


def mapear_clusters(y_real: pd.Series, clusters: np.ndarray) -> dict[int, int]:
    """Cada cluster recebe a classe real majoritária entre seus membros (empate -> menor classe)."""
    tab = pd.crosstab(pd.Series(clusters, name="cluster"), y_real.values)
    return {int(c): int(tab.loc[c].idxmax()) for c in tab.index}


def matriz_contingencia(y_real: pd.Series, clusters: np.ndarray, mapa: dict[int, int]) -> pd.DataFrame:
    """Linhas = classe real; colunas = clusters (rotulados 'C<i>→<classe mapeada>').

    É a matriz de confusão RETANGULAR (n_classes x k): fica quadrada só quando k = n_classes.
    """
    tab = pd.crosstab(y_real.values, clusters)
    tab.index.name = "classe_real"
    ordem = sorted(tab.columns, key=lambda c: (mapa[c], c))
    tab = tab[ordem]
    tab.columns = [f"C{c}→{mapa[c]}" for c in ordem]
    return tab


def matriz_confusao(y_real: pd.Series, clusters: np.ndarray, mapa: dict[int, int]) -> pd.DataFrame:
    """Matriz de confusão QUADRADA: linhas = classe real, colunas = rótulo mapeado.

    Todas as classes reais aparecem nos dois eixos; classes que nunca são majoritárias em nenhum
    cluster aparecem com coluna zerada (o modelo nunca "prevê" aquela classe).
    """
    classes = sorted(int(c) for c in y_real.unique())
    pred = pd.Series(clusters).map(mapa).values
    cm = pd.crosstab(y_real.values, pred).reindex(index=classes, columns=classes, fill_value=0)
    cm.index.name = "real"
    cm.columns.name = "mapeado"
    return cm


def metricas_por_classe(cm: pd.DataFrame) -> pd.DataFrame:
    """Acurácia (um-contra-todos), precisão e recall por classe, a partir da matriz quadrada."""
    m = cm.values.astype(float)
    total = m.sum()
    tp = np.diag(m)
    fp = m.sum(axis=0) - tp
    fn = m.sum(axis=1) - tp
    tn = total - tp - fp - fn
    with np.errstate(divide="ignore", invalid="ignore"):
        precisao = np.where(tp + fp > 0, tp / (tp + fp), np.nan)  # nan = classe nunca prevista
        recall = np.where(tp + fn > 0, tp / (tp + fn), np.nan)
        f1 = np.where(
            np.nan_to_num(precisao) + np.nan_to_num(recall) > 0,
            2 * np.nan_to_num(precisao) * np.nan_to_num(recall) / (np.nan_to_num(precisao) + np.nan_to_num(recall)),
            0.0,
        )
    out = pd.DataFrame(
        {
            "suporte": (tp + fn).astype(int),
            "TP": tp.astype(int),
            "FP": fp.astype(int),
            "FN": fn.astype(int),
            "TN": tn.astype(int),
            "acuracia_ovr": (tp + tn) / total,
            "precisao": precisao,
            "recall": recall,
            "f1": f1,
        },
        index=cm.index,
    )
    out.index.name = "classe"
    return out


def resumo_global(cm: pd.DataFrame, por_classe: pd.DataFrame) -> dict:
    """Acurácia global (traço/total) e médias macro (precisão indefinida conta como 0)."""
    total = cm.values.sum()
    return {
        "acuracia_global": float(np.trace(cm.values) / total),
        "precisao_macro": float(np.nan_to_num(por_classe["precisao"]).mean()),
        "recall_macro": float(np.nan_to_num(por_classe["recall"]).mean()),
        "f1_macro": float(por_classe["f1"].mean()),
        "classes_nunca_previstas": [int(c) for c in por_classe.index[por_classe["TP"] + por_classe["FP"] == 0]],
    }


def metricas_nao_supervisionadas(X: np.ndarray, y_real: pd.Series, clusters: np.ndarray) -> dict:
    """Métricas independentes do mapeamento (não dependem de escolher uma classe por cluster)."""
    return {
        "silhouette": float(silhouette_score(X, clusters, sample_size=min(len(X), 5000), random_state=42)),
        "ARI": float(adjusted_rand_score(y_real, clusters)),
        "NMI": float(normalized_mutual_info_score(y_real, clusters)),
        "homogeneidade": float(homogeneity_score(y_real, clusters)),
    }


def avaliar(X: np.ndarray, y_real: pd.Series, clusters: np.ndarray) -> dict:
    """Pipeline completo de avaliação de uma passada."""
    mapa = mapear_clusters(y_real, clusters)
    cont = matriz_contingencia(y_real, clusters, mapa)
    cm = matriz_confusao(y_real, clusters, mapa)
    pc = metricas_por_classe(cm)
    return {
        "mapa": mapa,
        "contingencia": cont,
        "confusao": cm,
        "por_classe": pc,
        "global": resumo_global(cm, pc),
        "nao_supervisionadas": metricas_nao_supervisionadas(X, y_real, clusters),
        "baseline_classe_majoritaria": float(y_real.value_counts(normalize=True).max()),
    }
