"""Teste de Kruskal-Wallis implementado só com numpy/pandas (sem scipy).

O enunciado só permite pandas, numpy, matplotlib, seaborn, plotly, scikit-learn e
streamlit, e o scipy não está na lista. Por isso o teste usado na EDA (pilar relacional)
foi escrito aqui, seguindo a mesma fórmula do scipy.stats.kruskal (resultados conferidos:
idênticos ao scipy).

Kruskal-Wallis: os grupos (vinhos de cada nota de qualidade) vêm da mesma distribuição?
É a versão não paramétrica da ANOVA: trabalha com postos (ranks) em vez dos valores,
então não exige normalidade e é pouco sensível a outliers.
"""
# [IA] Claude Code – reimplementação do kruskal sem scipy (fórmula do scipy.stats)
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def qui2_sf(x: float, gl: int) -> float:
    """p-valor: P(X > x) para X ~ qui-quadrado com `gl` graus de liberdade.

    Usa a fórmula fechada da cauda da qui-quadrado, que existe quando gl é inteiro.
    """
    if x <= 0:
        return 1.0
    if gl % 2 == 0:
        # gl par (gl = 2m):  P = e^{-x/2} · Σ_{i=0..m-1} (x/2)^i / i!
        termo, soma = 1.0, 1.0
        for i in range(1, gl // 2):
            termo *= (x / 2) / i
            soma += termo
        return math.exp(-x / 2) * soma
    # gl ímpar (gl = 2m+1): P = 2·(1 − Φ(√x)) + 2·φ(√x) · Σ_{i=1..m} x^{(2i−1)/2} / (1·3·…·(2i−1))
    # Φ = acumulada da normal padrão (via erfc) e φ = densidade da normal padrão
    z = math.sqrt(x)
    p = math.erfc(z / math.sqrt(2))  # = 2·(1 − Φ(z))
    termo, soma = z, 0.0
    for i in range(1, (gl - 1) // 2 + 1):
        if i > 1:
            termo *= x / (2 * i - 1)
        soma += termo
    return p + 2 * math.exp(-x / 2) / math.sqrt(2 * math.pi) * soma


def kruskal(*grupos) -> tuple[float, float]:
    """Teste de Kruskal-Wallis entre grupos independentes. Retorna (H, p-valor)."""
    grupos = [np.asarray(g, dtype=float) for g in grupos]
    todos = np.concatenate(grupos)
    n = todos.size

    # 1) junta todos os valores e dá um posto a cada um (1 = menor);
    #    valores empatados recebem a média dos postos que ocupariam
    postos = pd.Series(todos).rank(method="average").to_numpy()

    # 2) H = 12 / (n·(n+1)) · Σ R_i² / n_i  −  3·(n+1)
    #    R_i = soma dos postos do grupo i, n_i = tamanho do grupo i.
    #    Se os grupos fossem iguais, cada um teria posto médio parecido e H ficaria perto de 0.
    soma, ini = 0.0, 0
    for g in grupos:
        soma += postos[ini : ini + g.size].sum() ** 2 / g.size
        ini += g.size
    h = 12.0 / (n * (n + 1)) * soma - 3 * (n + 1)

    # 3) correção de empates: divide H por 1 − Σ(t³ − t) / (n³ − n),
    #    onde t é o tamanho de cada grupo de valores repetidos (comum aqui: medidas com 1-2 casas)
    t = pd.Series(todos).value_counts().to_numpy(dtype=float)
    h /= 1 - (t**3 - t).sum() / (n**3 - n)

    # 4) sob H0, H segue qui-quadrado com (k − 1) graus de liberdade (k = número de grupos)
    return h, qui2_sf(h, len(grupos) - 1)
