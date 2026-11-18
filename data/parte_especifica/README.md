# Dataset da Parte Específica — TMDB 5000 Movie Dataset

Fonte: <https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata> (aprovado pela professora no checkpoint).

| Arquivo | Linhas × colunas | Conteúdo |
|---|---|---|
| `tmdb_5000_movies.csv` | 4 803 × 20 | metadados do filme (orçamento, receita, gêneros, notas, etc.) |
| `tmdb_5000_credits.csv` | 4 803 × 4 | elenco (`cast`) e equipe (`crew`) em JSON; liga com `movies.id` por `movie_id` |

Para baixar de novo: `python data/parte_especifica/baixar_dados.py`.

Carga no código: `from dados import carregar_filmes` (ver `src/parte_especifica/dados.py`).
