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
| Pessoa 1 | Julia Maria Santos Teixeira | Dados e tratamento, identidade visual, fechamento (relatório, slides, README) |
| Pessoa 2 | Raul Vitor Ferraz Silva | EDA distribucional, gráficos exploratório e explanatório, resposta à pergunta |
| Pessoa 3 | Lucas Rodrigues Campos | EDA relacional, gráficos exploratório e explanatório, storytelling |

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
python src/parte_comum/relatorio.py      # gera relatorio.pdf (Parte Comum + Parte Específica)
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
| `src/parte_especifica/` | `eda.ipynb`, `dados.py`, `estilo.py`, `graficos.py`, `notas.py`, `dashboard.py`, `relatorio_especifica.py` (seção 7 do relatório), `vitrine_estilo.py` |
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

**Enquadramento:** O dataset se enquadra em um problema de classificação. A variável-alvo é binária e derivada do retorno (receita ÷ orçamento): o filme recupera o orçamento quando o retorno é igual ou maior que 1× e não recupera quando fica abaixo disso, o que só pode ser definido para os 3.210 filmes com orçamento e receita válidos. O orçamento e o gênero são as variáveis que ajudam a explicar essa classe, e a relação entre eles e o retorno é justamente o que a pergunta de pesquisa investiga: o retorno contínuo mostra se gastar mais rende mais, e a versão binária mostra se gastar mais muda a chance de não recuperar o orçamento. A análise é exploratória e explanatória: descrevemos como o retorno e a classe variam por faixa de orçamento e por gênero, sem treinar um modelo de classificação.

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
| Violino: distribuição do retorno por gênero | Exploratório | Pessoa 2 | pronto |
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

Texto completo, com as ressalvas, no fim do `eda.ipynb`, na seção 7.10 do `relatorio.pdf` e no fim da página Explicação do dashboard. O Início mostra a versão curta (`RESPOSTA_CURTA` em `notas.py`).

### Storytelling

**Público-alvo:** gestores de estúdio ou produtora, que precisam avaliar como orçamento, gênero e retorno bruto se relacionam ao pensar o portfólio.

**O que a exploração mostrou:** nos gêneros analisados, a associação de Spearman entre orçamento e retorno é negativa em Comédia, Drama, Romance e Suspense; em Ação e Aventura, fica próxima de zero, com intervalos bootstrap que incluem zero. O gráfico exploratório mostra a concentração e a dispersão de filmes por orçamento e retorno em cada gênero. O padrão é descritivo: não demonstra causalidade nem diferença formal entre gêneros, que também compartilham filmes.

**Por que essa mensagem:** orçamentos maiores, por si só, não garantem maior retorno bruto relativo. A relação deve ser lida junto com gênero e dispersão para orientar perguntas sobre o portfólio. Retorno (receita ÷ orçamento) não é lucro nem margem, e o orçamento também aparece no denominador do retorno, o que limita a interpretação da associação.

**Atributos pré-atentivos no gráfico explanatório:** a posição horizontal mostra a associação de Spearman e a linha vertical marca ausência de associação; o segmento horizontal mostra o intervalo bootstrap de 95%; a posição vertical identifica os gêneros; o tamanho dos pontos codifica a quantidade de filmes; e a cor identifica cada gênero conforme a paleta fixa do projeto.

### Dashboard

`streamlit run src/parte_especifica/dashboard.py`

| Página | Conteúdo |
|---|---|
| Início | pergunta, resposta curta em destaque, números da base e cards de navegação |
| Exploração | gráficos exploratórios (um por vez) com notas e indicadores do recorte |
| Explicação | gráficos explanatórios, storytelling e resposta completa |
| Década a década | animação do percurso das medianas por década, com troca de métrica (retorno, lucro bruto, receita) e de escala (log, linear) |
| Dados e tratamento | faltantes, valores implausíveis, decisões de tratamento e a base tratada |
| Parte Comum | resultados do K-Means no Wine Quality, com as passadas 1 e 2 uma por vez ou lado a lado (PCA, classe × cluster, matriz de confusão, perfil dos centroides) |

O botão **Baixar relatório (PDF)**, na barra lateral de todas as páginas e no fim da página Parte Comum, baixa o `relatorio.pdf`.

**Filtros:** a barra lateral das páginas Exploração, Explicação, Década a década e Dados filtra por gênero, faixa de orçamento,
ano de lançamento e mínimo de filmes por gênero. Os cards de indicadores comparam o recorte com a base inteira, e a página avisa
quando há filtro ativo, já que as notas e o storytelling descrevem a base inteira.

**Títulos dos gráficos:** cada autor escreve o achado do seu gráfico em `TITULOS_ACHADO` (`notas.py`); enquanto estiver vazio,
o dashboard usa um título descritivo.

**Identidade visual:** a paleta, as fontes e o formato brasileiro dos números estão em `estilo.py` (`aplicar_estilo(fig)`) e em `.streamlit/config.toml`.
Cada gênero tem uma cor fixa em todos os gráficos. Para ver os componentes com dados fictícios: `streamlit run src/parte_especifica/vitrine_estilo.py`.

---

## Uso de IA

Células e arquivos de código feitos com apoio de IA estão marcados com `# [IA] <ferramenta> – <finalidade>`.
Perguntas, decisões de tratamento, escolha de gráficos, interpretações e notas são do grupo.
A declaração assinada por todos fica em `docs/DECLARACAO_IA.pdf`.

**Pessoa 1 (Julia) — Claude Code**

As decisões foram todas dela: tratamento dos faltantes e outliers (`DECISOES` em `dados.py`), corte de 30 filmes por gênero,
gráfico exploratório e nota, identidade visual, organização do dashboard e do relatório. A IA foi usada só para:

- tirar dúvidas;
- entender erros;
- apoio com pandas, scikit-learn e Streamlit (trechos marcados com `# [IA]` no `eda.ipynb`, em `dados.py`, `estilo.py`, `graficos.py`, `dashboard.py` e nos scripts do relatório);
- revisão de ortografia e clareza dos textos.

**Pessoa 2 (Raul) — Codex e Claude Code**

As decisões foram todas dele: a escolha dos três gráficos (dispersão orçamento × retorno por gênero, barras agrupadas
por faixa de orçamento e violino do retorno por gênero), as variáveis, o objetivo e as faixas de orçamento de cada gráfico,
as notas analíticas e os títulos-achado, o texto da EDA distribucional, a resposta à pergunta de pesquisa e o resumo da
discussão da Parte Comum. A IA foi usada só para:

- implementar em código os gráficos especificados por ele (Codex: `eda.ipynb`, seção 5 e "Visualizações selecionadas — Pessoa 2", e o violino; Claude Code: versão em plotly do violino em `graficos.py`);
- agregações e contagens em pandas (Codex);
- gráficos da Parte Comum: histogramas, correlações, boxplots, cotovelo, PCA, matrizes de confusão, perfil dos centroides e efeito entre notas vizinhas (Codex, em `eda.py`, `kmeans.py`, `granularidade.py` e `common.py`);
- comparação entre as passadas do K-Means e botão de download do relatório na página Wine Quality do dashboard (Claude Code);
- conferir se os números citados nos textos batiam com os dados (percentuais, Spearman, d de Cohen e centroides);
- revisão de ortografia e clareza dos textos, correção de acentos e troca dos rótulos "Pessoa 4" depois da saída de um integrante (Claude Code);
- indicar onde pesquisar tipos de gráfico (a escolha foi dele) e explicar por que a escala logarítmica não pode começar em 0.

**Pessoa 3 (Lucas) — Codex e Gemini**

As decisões analíticas, narrativas e de validação foram dele: a pergunta analítica sobre como a associação entre orçamento
e retorno bruto relativo varia entre gêneros, o recorte nos seis gêneros, a escolha da visão conjunta de
concentração/dispersão e da correlação de Spearman com intervalo de confiança via bootstrap, os limites interpretativos
(associação descritiva, sem causalidade nem teste formal de diferença entre gêneros), a mensagem de negócio do storytelling
e os títulos-achado. A IA foi usada para:

- apoio na implementação e integração da análise relacional e dos gráficos `p3_exploratorio_densidade` e `p3_explanatorio_spearman` no notebook, no dashboard e no relatório (Codex);
- apoio na redação e no preenchimento de `TITULOS_ACHADO`, do resumo relacional do relatório e do storytelling (Codex, commit `9704b4c`);
- estruturar prompts para o Codex, revisar a consistência entre resumo relacional, títulos-achado, storytelling e atributos pré-atentivos, e apoiar a conferência final e o fluxo de commit/push (Gemini).

## Matriz de contribuição

| Atividade | Pessoa 1 | Pessoa 2 | Pessoa 3 |
|---|---|---|---|
| Parte Comum — EDA | A | P | |
| Parte Comum — K-Means e métricas | | P | |
| Parte Comum — discussão crítica | | P | |
| Parte Específica — escolha do dataset e da pergunta | P | A | A |
| Parte Específica — EDA estrutural, faltantes e outliers | P | | |
| Parte Específica — EDA distribucional | | P | |
| Parte Específica — EDA relacional | | | P |
| Gráficos exploratórios e notas | P | P | P |
| Gráficos explanatórios e notas | | P | P |
| Storytelling e resposta à pergunta | A | P | P |
| Identidade visual | P | | |
| Dashboard | P | A | |
| Relatório | P | A | A |
| Slides e apresentação | P | A | A |
| README | P | A | |

P = principal, A = apoio.

## Pendências

- [x] Nomes dos integrantes (aqui e na capa do relatório)
- [x] Gráfico exploratório e nota do violino (Pessoa 2)
- [x] Resposta à pergunta (resumo aqui)
- [ ] Resumo do storytelling aqui
- [x] Matriz de contribuição
- [ ] `docs/DECLARACAO_IA.pdf` assinada por todos
- [x] `relatorio.pdf`: preencher os trechos marcados **[PREENCHER]** na seção 7 (textos em `TEXTOS`, `src/parte_especifica/relatorio_especifica.py`) e os nomes na capa
- [x] Títulos-achado dos gráficos (`TITULOS_ACHADO` em `notas.py`), cada autor o seu
- [ ] Slides da apresentação (03/12)
