# Projeto AVD — EDA e Agrupamento (Wine Quality)

Parte Comum: análise exploratória, padronização e K-Means (duas passadas) sobre o
[UCI Wine Quality](https://archive.ics.uci.edu/dataset/186) (vinho tinto + branco, 6 497 vinhos, 7 classes de nota: 3–9).

## Como reproduzir

```bash
pip install -r requirements.txt
python src/parte_comum/eda.py            # EDA  -> out/parte_comum/eda/
python src/parte_comum/kmeans.py         # padronização + K-Means (passadas 1 e 2) + métricas -> out/parte_comum/kmeans/
python src/parte_comum/granularidade.py  # apoio à discussão crítica -> out/parte_comum/granularidade/
python src/parte_comum/relatorio.py      # gera relatorio.pdf a partir de out/
```

Só usamos as bibliotecas permitidas pelo enunciado (pandas, numpy, matplotlib, seaborn, plotly, scikit-learn, streamlit).

Os notebooks em `notebooks/parte_comum/` (`eda.ipynb`, `kmeans_metricas.ipynb`) reutilizam as funções de
`src/parte_comum/` e já estão executados. Semente fixa (42), `n_init = 20`: resultados determinísticos.

## Estrutura

| Caminho | Conteúdo |
|---|---|
| `data/parte_comum/` | CSVs originais do UCI (`winequality-red/white.csv`, `.names`) |
| `src/parte_comum/` | `common.py` (carga/estilo), `eda.py`, `kmeans.py`, `metricas.py`, `granularidade.py`, `relatorio.py`, `estatistica.py` (testes de normalidade e Kruskal-Wallis sem scipy), `pdf_simples.py` (gerador de PDF só com matplotlib) |
| `notebooks/parte_comum/` | `eda.ipynb`, `kmeans_metricas.ipynb` |
| `out/parte_comum/` | figuras e tabelas (CSV/JSON) geradas |
| `relatorio.pdf` | relatório (Parte Comum completa; Parte Específica como marcador) |

## Decisões de projeto

- **Dataset:** tinto + branco juntos (coluna auxiliar `tipo`, **não** usada no K-Means) → 7 classes (3–9). Sensibilidade com só tinto (6 classes) e só branco no relatório.
- **Outliers:** não removidos (22,7 % das linhas têm algum; concentram-se nas classes raras; são plausíveis). Winsorização 1–99 % como teste de sensibilidade.
- **Duplicatas** (18 %): mantidas, com justificativa no relatório.
- **Padronização:** z-score (`StandardScaler`) das 11 variáveis.
- **k da Passada 1:** cotovelo por critério geométrico (distância máxima à corda) → k = 4; silhouette como apoio.
- **Métricas:** cluster → classe majoritária; matriz retangular (real × cluster) e quadrada (real × mapeado); acurácia (global e um-contra-todos), precisão e recall por classe; baseline, ARI e NMI para contextualizar.

## Resultado principal

| | k | Acurácia | Baseline (“sempre 6”) | ARI | NMI | Silhouette |
|---|---|---|---|---|---|---|
| Passada 1 (cotovelo) | 4 | 0,458 | 0,437 | 0,022 | 0,045 | 0,249 |
| Passada 2 (k = 7) | 7 | 0,459 | 0,437 | 0,035 | 0,063 | 0,182 |

O K-Means agrupa sobretudo por **tipo de vinho** e perfil químico; as notas vizinhas (5×6, 6×7) não são separáveis pelos dados
(d de Cohen médio ≈ 0,2). Detalhes e discussão crítica em `relatorio.pdf`.

## Pendências (dependem do grupo)

- `docs/DECLARACAO_IA.pdf` assinada por todos.
- Parte Específica: dataset próprio, `src/parte_especifica/eda.ipynb` e `dashboard.py` (Streamlit).
- Nomes dos integrantes na capa do relatório e seção 7 do relatório.
