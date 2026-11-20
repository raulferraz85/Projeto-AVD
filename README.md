# Projeto AVD 2026.2 — Análise e Visualização de Dados

Projeto final da disciplina, em duas partes:

- **Parte Comum:** EDA, padronização e agrupamento K-Means no [UCI Wine Quality](https://archive.ics.uci.edu/dataset/186)
  (vinho tinto + branco, 6 497 vinhos, notas de 3 a 9).
- **Parte Específica:** EDA e dashboard sobre o [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
  (4 803 filmes), respondendo a uma pergunta de pesquisa do grupo.

Entregáveis: este repositório, o `relatorio.pdf` e o dashboard em Streamlit.

## Integrantes

| Papel | Nome | Responsabilidades |
|---|---|---|
| Pessoa 1 | _(preencher)_ | Dados e tratamento, identidade visual, fechamento (relatório, slides, README) |
| Pessoa 2 | _(preencher)_ | EDA distribucional, gráficos exploratório e explanatório, resposta à pergunta |
| Pessoa 3 | _(preencher)_ | EDA relacional, gráficos exploratório e explanatório, storytelling |
| Pessoa 4 | _(preencher)_ | Estrutura do dashboard, página da Parte Comum, gráfico exploratório |

## Como rodar

Python 3.11 ou mais recente (testado com 3.12). Todos os comandos são executados a partir da raiz do projeto.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Dashboard** (Parte Específica + página da Parte Comum):

```bash
streamlit run src/parte_especifica/dashboard.py
```

Para alternar entre claro e escuro: menu ⋮ → Settings → Theme.

**Parte Comum** (gera as figuras e tabelas em `out/parte_comum/` e depois o PDF):

```bash
python src/parte_comum/eda.py            # EDA
python src/parte_comum/kmeans.py         # padronização + K-Means (passadas 1 e 2) + métricas
python src/parte_comum/granularidade.py  # apoio à discussão crítica
python src/parte_comum/relatorio.py      # gera relatorio.pdf
```

**Parte Específica:** abrir `src/parte_especifica/eda.ipynb` e executar todas as células.
O notebook deve ser aberto com a pasta `src/parte_especifica/` como diretório de trabalho.

Usamos só as bibliotecas permitidas pelo enunciado: pandas, numpy, matplotlib, seaborn, plotly, scikit-learn e streamlit.
Os pacotes de Jupyter do `requirements.txt` servem apenas para abrir e executar os notebooks.

## Estrutura do repositório

| Caminho | Conteúdo |
|---|---|
| `data/parte_comum/` | CSVs originais do UCI (`winequality-red/white.csv`, `.names`) |
| `data/parte_especifica/` | CSVs do TMDB 5000 (`movies`, `credits`) e `baixar_dados.py` |
| `src/parte_comum/` | `common.py` (carga e estilo), `eda.py`, `kmeans.py`, `metricas.py`, `granularidade.py`, `relatorio.py`, `estatistica.py` (testes sem scipy), `pdf_simples.py` |
| `notebooks/parte_comum/` | `eda.ipynb`, `kmeans_metricas.ipynb` (já executados) |
| `src/parte_especifica/` | `eda.ipynb`, `dados.py`, `estilo.py`, `graficos.py`, `notas.py`, `dashboard.py`, `vitrine_estilo.py` |
| `.streamlit/config.toml` | tema do dashboard (cores e fontes da identidade visual) |
| `out/` | figuras e tabelas geradas |
| `docs/` | declaração de uso de IA |
| `relatorio.pdf` | relatório final |

---

## Parte Comum — Wine Quality

### Decisões de projeto

- **Dataset:** tinto e branco juntos (coluna auxiliar `tipo`, **não** usada no K-Means), o que dá 7 classes (3 a 9). A sensibilidade com só tinto (6 classes) e só branco está no relatório.
- **Outliers:** não removidos. Estão em 22,7 % das linhas, concentram-se nas classes raras e são plausíveis. A winsorização 1–99 % entra como teste de sensibilidade.
- **Duplicatas** (18 %): mantidas, com justificativa no relatório.
- **Padronização:** z-score (`StandardScaler`) das 11 variáveis.
- **k da Passada 1:** cotovelo por critério geométrico (maior distância à corda), resultando em k = 4. A silhouette serve de apoio.
- **Métricas:** cada cluster recebe a classe majoritária. Calculamos a matriz retangular (real × cluster) e a quadrada (real × mapeado), a acurácia (global e um-contra-todos) e a precisão e o recall por classe. Baseline, ARI e NMI dão contexto.
- **Reprodutibilidade:** semente fixa (42) e `n_init = 20`.

### Resultado principal

| | k | Acurácia | Baseline (“sempre 6”) | ARI | NMI | Silhouette |
|---|---|---|---|---|---|---|
| Passada 1 (cotovelo) | 4 | 0,458 | 0,437 | 0,022 | 0,045 | 0,249 |
| Passada 2 (k = 7) | 7 | 0,459 | 0,437 | 0,035 | 0,063 | 0,182 |

O K-Means agrupa sobretudo por **tipo de vinho** e perfil químico. As notas vizinhas (5×6, 6×7) não são separáveis pelos dados
(d de Cohen médio ≈ 0,2). Detalhes e discussão crítica em `relatorio.pdf`.

---

## Parte Específica — TMDB 5000

**Pergunta de pesquisa:** Filmes com orçamento maior têm retorno (receita ÷ orçamento) maior, ou isso varia por gênero?
**Variáveis centrais:** `budget`, `revenue`, `genres`
**Público-alvo:** gestor de estúdio ou produtora

### Dados e tratamento

A carga e o tratamento estão em `src/parte_especifica/dados.py`. Todo o grupo usa `tratar(carregar_filmes())`.
As escolhas ficam no dicionário `DECISOES`, e as justificativas, nas seções 2 a 4 do `eda.ipynb`.

- `budget` e `revenue` iguais a 0 significam “não informado” e viram NaN. A ausência **não é aleatória** (MAR), então as conclusões valem para os filmes com dados financeiros, que são mais populares e em sua maioria de língua inglesa.
- Valores positivos abaixo de US$ 1 000 são tratados como erro de digitação ou de unidade e viram NaN.
- `runtime` = 0 vira NaN. `vote_average` vira NaN quando `vote_count` = 0.
- A única linha removida é o filme sem data de lançamento (4 803 → 4 802 filmes).
- Outliers de orçamento, receita e retorno são mantidos: são filmes reais, não erros.
- Gêneros não são exclusivos: cada filme conta em todos os seus gêneros. Nas análises por gênero, entram só os gêneros com pelo menos 30 filmes.
- `retorno = revenue ÷ budget` é calculado só quando os dois valores são válidos. Retorno não é lucro nem margem.

### Gráficos e notas

| Gráfico | Tipo | Autor | Status |
|---|---|---|---|
| Cobertura financeira por gênero | Exploratório | Pessoa 1 | pronto |
| Dispersão orçamento × retorno por gênero | Exploratório | Pessoa 2 | pronto |
| Concentração de orçamento e retorno por gênero | Exploratório | Pessoa 3 | pronto |
| _(a definir)_ | Exploratório | Pessoa 4 | pendente |
| Mediana do retorno por faixa de orçamento e gênero | Explanatório | Pessoa 2 | pronto |
| Correlação de Spearman por gênero (IC bootstrap 95 %) | Explanatório | Pessoa 3 | pronto |

Os gráficos ficam no `eda.ipynb`. A versão em plotly, com a identidade visual do projeto, está em `graficos.py`, e as notas analíticas, em `notas.py`.

### Resposta à pergunta

**Não de forma simples.** Nos 3 210 filmes com retorno calculável, a correlação de Spearman entre orçamento e retorno
é fraca e levemente negativa (−0,136). Por faixa de orçamento, a relação forma um “U”: até US$ 20M (mediana de 2,88×)
e acima de US$ 100M (2,76×) superam as faixas do meio (1,99×). O que mais muda com o orçamento alto é a chance de a
receita não cobrir o custo: 7,9 % acima de US$ 100M, contra cerca de um em cada quatro filmes nas outras faixas.

**Varia por gênero.** A associação é negativa e fraca em romance, drama, suspense e comédia, e próxima de zero em ação
e aventura. Retorno não é lucro, e parte da correlação negativa vem de o orçamento estar no denominador do retorno.

Texto completo, com as ressalvas, no fim do `eda.ipynb` e na página Início do dashboard.

### Storytelling

_(resumo pendente — o texto completo da Pessoa 3 está no `eda.ipynb`, seção 6.2, e na página Explicação do dashboard)_

### Dashboard

`streamlit run src/parte_especifica/dashboard.py`

| Página | Conteúdo |
|---|---|
| Início | pergunta, público-alvo e números gerais da base |
| Exploração | gráficos exploratórios com notas |
| Explicação | gráficos explanatórios e storytelling |
| Dados e tratamento | faltantes, valores implausíveis e decisões de tratamento |
| Parte Comum | resultados do K-Means no Wine Quality |

**Identidade visual:** a paleta, as fontes e o formato brasileiro dos números estão em `estilo.py` (`aplicar_estilo(fig)`) e em `.streamlit/config.toml`.
Cada gênero tem uma cor fixa em todos os gráficos. Para ver os componentes com dados fictícios: `streamlit run src/parte_especifica/vitrine_estilo.py`.

---

## Uso de IA

Células e arquivos de código feitos com apoio de IA estão marcados com `# [IA] <ferramenta> – <finalidade>`.
A IA ajudou com cálculo, gráficos e estrutura de código. Perguntas, decisões de tratamento, escolha de gráficos, interpretações e notas são do grupo.
A declaração assinada por todos fica em `docs/DECLARACAO_IA.pdf`.

## Matriz de contribuição

| Atividade | Pessoa 1 | Pessoa 2 | Pessoa 3 | Pessoa 4 |
|---|---|---|---|---|
| Parte Comum — EDA | | | | |
| Parte Comum — K-Means e métricas | | | | |
| Parte Comum — discussão crítica | | | | |
| Parte Específica — escolha do dataset e da pergunta | | | | |
| Parte Específica — EDA estrutural, faltantes e outliers | | | | |
| Parte Específica — EDA distribucional | | | | |
| Parte Específica — EDA relacional | | | | |
| Gráficos exploratórios e notas | | | | |
| Gráficos explanatórios e notas | | | | |
| Storytelling e resposta à pergunta | | | | |
| Identidade visual | | | | |
| Dashboard | | | | |
| Relatório | | | | |
| Slides e apresentação | | | | |
| README | | | | |

_(preencher com P = principal, A = apoio)_

## Pendências

- [ ] Nomes dos integrantes (aqui e na capa do relatório)
- [ ] Gráfico exploratório e nota da Pessoa 4
- [x] Resposta à pergunta (resumo aqui)
- [ ] Resumo do storytelling aqui
- [ ] Matriz de contribuição
- [ ] `docs/DECLARACAO_IA.pdf` assinada por todos
- [ ] `relatorio.pdf` com a Parte Específica
- [ ] Slides da apresentação (03/12)
