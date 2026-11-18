"""Testes estatísticos implementados só com numpy/pandas (sem scipy).

O enunciado só permite pandas, numpy, matplotlib, seaborn, plotly, scikit-learn e
streamlit; por isso os dois testes usados na EDA foram reescritos aqui, seguindo as
mesmas fórmulas do scipy.stats (resultados conferidos: idênticos ao scipy).

- normaltest: D'Agostino-Pearson (K² = Z_assimetria² + Z_curtose², ~ qui² com 2 g.l.)
- kruskal:    Kruskal-Wallis (H com correção de empates, ~ qui² com k-1 g.l.)
"""
# [IA] Claude Code – reimplementação de normaltest/kruskal sem scipy (fórmulas do scipy.stats)
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def qui2_sf(x: float, gl: int) -> float:
    """P(X > x) para X ~ qui-quadrado com `gl` graus de liberdade (fórmula fechada para gl inteiro)."""
    if x <= 0:
        return 1.0
    if gl % 2 == 0:
        # gl = 2m:  e^{-x/2} · Σ_{i<m} (x/2)^i / i!
        termo, soma = 1.0, 1.0
        for i in range(1, gl // 2):
            termo *= (x / 2) / i
            soma += termo
        return math.exp(-x / 2) * soma
    # gl = 2m+1:  2·(1 − Φ(√x)) + 2·φ(√x) · Σ_{i=1..m} x^{(2i−1)/2} / (1·3·…·(2i−1))
    z = math.sqrt(x)
    p = math.erfc(z / math.sqrt(2))
    termo, soma = z, 0.0
    for i in range(1, (gl - 1) // 2 + 1):
        if i > 1:
            termo *= x / (2 * i - 1)
        soma += termo
    return p + 2 * math.exp(-x / 2) / math.sqrt(2 * math.pi) * soma


def _z_assimetria(a: np.ndarray) -> float:
    n = a.size
    d = a - a.mean()
    b2 = (d**3).mean() / (d**2).mean() ** 1.5
    y = b2 * math.sqrt((n + 1) * (n + 3) / (6.0 * (n - 2)))
    beta2 = 3.0 * (n**2 + 27 * n - 70) * (n + 1) * (n + 3) / ((n - 2.0) * (n + 5) * (n + 7) * (n + 9))
    w2 = -1 + math.sqrt(2 * (beta2 - 1))
    delta = 1 / math.sqrt(0.5 * math.log(w2))
    alpha = math.sqrt(2.0 / (w2 - 1))
    y = 1.0 if y == 0 else y
    return delta * math.log(y / alpha + math.sqrt((y / alpha) ** 2 + 1))


def _z_curtose(a: np.ndarray) -> float:
    n = a.size
    d = a - a.mean()
    b2 = (d**4).mean() / (d**2).mean() ** 2
    e = 3.0 * (n - 1) / (n + 1)
    var_b2 = 24.0 * n * (n - 2) * (n - 3) / ((n + 1) * (n + 1.0) * (n + 3) * (n + 5))
    x = (b2 - e) / math.sqrt(var_b2)
    sqrt_beta1 = 6.0 * (n * n - 5 * n + 2) / ((n + 7) * (n + 9)) * math.sqrt(6.0 * (n + 3) * (n + 5) / (n * (n - 2) * (n - 3)))
    A = 6.0 + 8.0 / sqrt_beta1 * (2.0 / sqrt_beta1 + math.sqrt(1 + 4.0 / sqrt_beta1**2))
    termo1 = 1 - 2 / (9.0 * A)
    denom = 1 + x * math.sqrt(2 / (A - 4.0))
    if denom == 0:
        return float("nan")
    termo2 = math.copysign(((1 - 2.0 / A) / abs(denom)) ** (1 / 3.0), denom)
    return (termo1 - termo2) / math.sqrt(2 / (9.0 * A))


def normaltest(valores) -> tuple[float, float]:
    """Teste de normalidade de D'Agostino-Pearson. Retorna (K², p-valor)."""
    a = np.asarray(valores, dtype=float)
    k2 = _z_assimetria(a) ** 2 + _z_curtose(a) ** 2
    return k2, qui2_sf(k2, 2)


def kruskal(*grupos) -> tuple[float, float]:
    """Teste de Kruskal-Wallis entre grupos independentes. Retorna (H, p-valor)."""
    grupos = [np.asarray(g, dtype=float) for g in grupos]
    todos = np.concatenate(grupos)
    n = todos.size
    postos = pd.Series(todos).rank(method="average").to_numpy()
    h, ini = 0.0, 0
    for g in grupos:
        h += postos[ini : ini + g.size].sum() ** 2 / g.size
        ini += g.size
    h = 12.0 / (n * (n + 1)) * h - 3 * (n + 1)
    empates = pd.Series(todos).value_counts().to_numpy(dtype=float)
    h /= 1 - (empates**3 - empates).sum() / (n**3 - n)
    return h, qui2_sf(h, len(grupos) - 1)
