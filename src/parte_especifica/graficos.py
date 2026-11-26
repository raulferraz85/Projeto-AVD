"""Gráficos da Parte Específica em plotly, com a identidade visual do projeto (estilo.py).

Cada função recebe o DataFrame já tratado (tratar(carregar_filmes())) e devolve uma figura plotly.
O conteúdo (tipo de gráfico, variáveis, gêneros, faixas, títulos) é o que cada autor definiu no
eda.ipynb; aqui só muda o visual (paleta, fontes, formato brasileiro, modo claro/escuro).
As notas analíticas de cada gráfico ficam em notas.py.

Uso no dashboard:
    from graficos import GRAFICOS
    for g in GRAFICOS: grafico(g["funcao"](df)); nota(NOTAS[g["id"]])

Filtros do dashboard: toda função aceita `generos` (None = sem filtro; senão mostra só os gêneros do
autor que estão no filtro) e `min_filmes` (gêneros com menos filmes que isso no recorte ficam de fora).
Sem esses argumentos, cada gráfico sai igual ao do eda.ipynb. Se o recorte esvaziar o gráfico: SemDados.
"""
# [IA] Claude Code – conversão dos gráficos do eda.ipynb para plotly com o estilo do projeto
# [IA] Claude Code – parâmetros de filtro (generos, min_filmes) para os filtros do dashboard
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dados import explodir
from estilo import (CORES, GENEROS_PT, LIMITES_FAIXAS, ROTULOS_FAIXAS, FONTE_NUMERO, FONTE_TEXTO, _br,
                    aplicar_estilo, eixo_log, fmt_mult, fmt_pct, fmt_usd, mapa_cores_generos, modo_atual)


class SemDados(Exception):
    """O recorte (filtros do dashboard) não deixou nenhum gênero com filmes suficientes para o gráfico."""


def _escolher(padrao: list[str], generos) -> list[str]:
    """Gêneros do autor (na ordem dele) que também estão no filtro; generos=None = sem filtro."""
    return [g for g in padrao if generos is None or g in generos]


def _grade(n: int, max_colunas: int) -> tuple[int, int]:
    colunas = min(n, max_colunas)
    return -(-n // colunas), colunas  # (linhas, colunas)


def _cor_genero(genero_en: str, modo: str) -> str:
    """Cor fixa do gênero (estilo.py); gêneros fora dos 6 com cor ficam em cinza de contexto."""
    return mapa_cores_generos(modo).get(genero_en, CORES[modo]["contexto"])


def _base_genero(df: pd.DataFrame, generos: list[str], so_positivo: bool = False, min_filmes: int = 0) -> pd.DataFrame:
    """Uma linha por (filme, gênero) só com filmes de retorno válido e os gêneros pedidos.

    Gêneros com menos de `min_filmes` filmes ficam de fora (corte usado pelo filtro do dashboard).
    """
    filtro = df["retorno"].gt(0) if so_positivo else df["retorno"].notna()
    base = explodir(df.loc[filtro, ["id", "budget", "retorno", "genres"]], "genres").dropna(subset=["genres"])
    base = base.drop_duplicates(["id", "genres"])
    base = base.loc[base["genres"].isin(generos)]
    contagem = base["genres"].value_counts()
    base = base.loc[base["genres"].map(contagem).ge(min_filmes)].copy()
    if base.empty:
        raise SemDados
    return base


def _presentes(base: pd.DataFrame, generos: list[str]) -> list[str]:
    return [g for g in generos if g in set(base["genres"])]


def _rotulos_titulos_subplot(fig: go.Figure, modo: str, titulo_x: str, titulo_y: str) -> None:
    """Títulos dos painéis no estilo do projeto + UM título por eixo para a figura toda (como supxlabel/supylabel)."""
    c = CORES[modo]
    fig.update_annotations(font=dict(family=FONTE_TEXTO, size=13, color=c["texto"]))
    fonte = dict(family=FONTE_TEXTO, size=12, color=c["texto_suave"])
    fig.add_annotation(text=titulo_x, xref="paper", yref="paper", x=0.5, y=0, yshift=-46, showarrow=False, font=fonte)
    fig.add_annotation(text=titulo_y, xref="paper", yref="paper", x=0, y=0.5, xshift=-58, textangle=-90, showarrow=False, font=fonte)
    fig.update_layout(margin=dict(l=76, b=64, t=fig.layout.margin.t + 18))


# ==========================================================================================
# Pessoa 1 — exploratório: cobertura financeira (filmes com retorno calculável) por gênero
# ==========================================================================================
MIN_FILMES_GENERO = 30  # corte mínimo de filmes por gênero (decisão da Pessoa 1)


def p1_tabela_cobertura(df: pd.DataFrame, min_filmes: int = MIN_FILMES_GENERO) -> pd.DataFrame:
    """Por gênero: nº de filmes e % com orçamento, receita e os dois (= retorno calculável) após tratar().

    Gêneros com menos de `min_filmes` filmes ficam de fora. Ordem: menor cobertura primeiro.
    """
    base = df[["id", "genres", "orcamento_informado", "receita_informada", "retorno"]].copy()
    base["retorno_calculavel"] = base["retorno"].notna()
    base = explodir(base, "genres").dropna(subset=["genres"]).drop_duplicates(["id", "genres"])
    tab = base.groupby("genres").agg(
        filmes=("id", "size"),
        com_retorno=("retorno_calculavel", "sum"),
        pct_orcamento=("orcamento_informado", "mean"),
        pct_receita=("receita_informada", "mean"),
        pct_retorno=("retorno_calculavel", "mean"),
    ).reset_index().rename(columns={"genres": "genero"})
    tab["gênero"] = tab["genero"].map(lambda g: GENEROS_PT.get(g, g))
    tab = tab.loc[tab["filmes"].ge(min_filmes)]
    return tab.sort_values("pct_retorno").reset_index(drop=True)


def p1_exploratorio_cobertura(df: pd.DataFrame, modo: str | None = None, generos=None,
                              min_filmes: int = MIN_FILMES_GENERO) -> go.Figure:
    modo = modo or modo_atual()
    c = CORES[modo]
    tab = p1_tabela_cobertura(df, min_filmes)
    if generos is not None:
        tab = tab.loc[tab["genero"].isin(generos)]
    if tab.empty:
        raise SemDados
    geral = df["retorno"].notna().mean()
    fig = go.Figure(go.Bar(
        x=tab["pct_retorno"], y=tab["gênero"], orientation="h",
        marker_color=[_cor_genero(g, modo) for g in tab["genero"]],
        text=[fmt_pct(v, 0) for v in tab["pct_retorno"]], textposition="outside", cliponaxis=False,
        textfont=dict(family=FONTE_NUMERO, size=11, color=c["texto"]),
        customdata=np.column_stack([tab["filmes"], tab["com_retorno"], tab["pct_orcamento"], tab["pct_receita"]]),
        hovertemplate=("<b>%{y}</b><br>filmes no gênero: %{customdata[0]:,.0f}<br>"
                       "com retorno calculável: %{customdata[1]:,.0f} (%{x:.1%})<br>"
                       "com orçamento: %{customdata[2]:.1%} · com receita: %{customdata[3]:.1%}<extra></extra>"),
    ))
    fig.add_vline(x=geral, line=dict(dash="dash", color=c["texto_suave"], width=1.2))
    fig.add_annotation(x=geral, y=1, yref="paper", yanchor="bottom", xanchor="left", xshift=4, showarrow=False,
                       text=f"todos os filmes: {fmt_pct(geral, 0)}", font=dict(family=FONTE_TEXTO, size=11, color=c["texto_suave"]))
    aplicar_estilo(fig, "Quanto de cada gênero tem orçamento e receita informados",
                   f"% de filmes com retorno calculável (orçamento e receita válidos) · gêneros com ao menos {min_filmes} filmes",
                   modo=modo, altura=max(260, min(620, 152 + 26 * len(tab))))  # base toda = 620 px, altura original
    fig.update_xaxes(title_text="Filmes com retorno calculável (% do gênero)", range=[0, 1.08], tickformat=".0%",
                     tickvals=[0, 0.25, 0.5, 0.75, 1], ticktext=["0%", "25%", "50%", "75%", "100%"])
    fig.update_yaxes(title_text="Gênero", showgrid=False, tickfont=dict(family=FONTE_TEXTO, size=12, color=c["texto"]))
    fig.update_layout(separators=",.", bargap=0.25)
    return fig


# ==========================================================================================
# Pessoa 2 — exploratório: dispersão orçamento × retorno por gênero (cell "Visualizações selecionadas")
# ==========================================================================================
GENEROS_P2 = ["Action", "Comedy", "Drama", "Adventure"]


def p2_exploratorio_dispersao(df: pd.DataFrame, modo: str | None = None, generos=None, min_filmes: int = 0) -> go.Figure:
    modo = modo or modo_atual()
    base = _base_genero(df, _escolher(GENEROS_P2, generos), min_filmes=min_filmes)
    generos = _presentes(base, GENEROS_P2)
    base["orcamento_mi"] = base["budget"] / 1e6
    linhas, colunas = _grade(len(generos), 2)
    fig = make_subplots(rows=linhas, cols=colunas, shared_xaxes=True, shared_yaxes=True,
                        subplot_titles=[GENEROS_PT[g] for g in generos],
                        horizontal_spacing=0.06, vertical_spacing=0.12)
    for k, genero in enumerate(generos):
        d = base.loc[base["genres"].eq(genero)]
        fig.add_trace(go.Scatter(
            x=d["orcamento_mi"], y=d["retorno"], mode="markers", name=GENEROS_PT[genero], showlegend=False,
            marker=dict(color=_cor_genero(genero, modo), size=5, opacity=0.35),
            hovertemplate="orçamento: US$ %{x:,.1f}M<br>retorno: %{y:.2f}×<extra>" + GENEROS_PT[genero] + "</extra>",
        ), row=k // colunas + 1, col=k % colunas + 1)
    aplicar_estilo(fig, "Orçamento e retorno dos filmes por gênero",
                   "Cada ponto é um filme · eixos em escala logarítmica · mesma escala "
                   + ("nos quatro painéis" if len(generos) == 4 else "em todos os painéis"),
                   modo=modo, altura=200 + 210 * linhas)
    eixo_log(fig, "x", sufixo="", valores=[0.01, 0.1, 1, 10, 100])
    eixo_log(fig, "y", valores=[0.01, 0.1, 1, 10, 100, 1000, 10000])
    fig.update_xaxes(ticktext=[fmt_usd(v * 1e6) for v in [0.01, 0.1, 1, 10, 100]])
    _rotulos_titulos_subplot(fig, modo, "Orçamento (US$, escala logarítmica)", "Retorno (receita ÷ orçamento, escala logarítmica)")
    return fig


# ==========================================================================================
# Pessoa 2 — explanatório: mediana do retorno por faixa de orçamento e gênero (barras agrupadas)
# ==========================================================================================
def p2_tabela_faixas(df: pd.DataFrame, generos=None, min_filmes: int = 0) -> pd.DataFrame:
    """Mediana do retorno e nº de filmes por faixa e gênero (mesmo cálculo do notebook)."""
    base = _base_genero(df, _escolher(GENEROS_P2, generos), min_filmes=min_filmes)
    base["faixa"] = pd.cut(base["budget"], bins=[0, *LIMITES_FAIXAS, np.inf], labels=ROTULOS_FAIXAS, right=True)
    return base.groupby(["faixa", "genres"], observed=True)["retorno"].agg(mediana="median", filmes="size").reset_index()


def p2_explanatorio_faixas(df: pd.DataFrame, modo: str | None = None, generos=None, min_filmes: int = 0) -> go.Figure:
    modo = modo or modo_atual()
    c = CORES[modo]
    tab = p2_tabela_faixas(df, generos, min_filmes)
    maximos = tab.groupby("faixa", observed=True)["mediana"].transform("max")
    tab["maior"] = tab["mediana"].eq(maximos)
    fig = go.Figure()
    for genero in _presentes(tab, GENEROS_P2):
        d = tab.loc[tab["genres"].eq(genero)].set_index("faixa").reindex(ROTULOS_FAIXAS)
        fig.add_trace(go.Bar(
            x=ROTULOS_FAIXAS, y=d["mediana"], name=GENEROS_PT[genero], marker_color=_cor_genero(genero, modo),
            marker_line=dict(color=[c["texto"] if m is True else "rgba(0,0,0,0)" for m in d["maior"]], width=2),
            text=[fmt_mult(v) if m is True else "" for v, m in zip(d["mediana"], d["maior"])],
            textposition="outside", textfont=dict(family=FONTE_NUMERO, size=12, color=c["texto"]), cliponaxis=False,
            constraintext="none",
            customdata=d["filmes"],
            hovertemplate="%{x}<br>mediana: %{y:.2f}×<br>filmes: %{customdata}<extra>" + GENEROS_PT[genero] + "</extra>",
        ))
    # item de legenda para o contorno (mesmo papel do "Maior mediana da faixa" do notebook)
    fig.add_trace(go.Bar(x=[None], y=[None], name="Maior mediana da faixa",
                         marker=dict(color="rgba(0,0,0,0)", line=dict(color=c["texto"], width=2))))
    aplicar_estilo(fig, "Mediana do retorno por faixa de orçamento e gênero",
                   "Mediana de receita ÷ orçamento · contorno e rótulo = maior mediana da faixa", modo=modo, altura=480)
    fig.update_layout(barmode="group", bargap=0.22, bargroupgap=0.06)
    fig.update_xaxes(title_text="Faixa de orçamento", showgrid=False)
    fig.update_yaxes(title_text="Mediana do retorno (receita ÷ orçamento)", ticksuffix="×", rangemode="tozero")
    return fig


# ==========================================================================================
# Pessoa 3 — exploratório: concentração (contagem por bin, log) de orçamento × retorno por gênero
# ==========================================================================================
GENEROS_P3 = ["Drama", "Comedy", "Thriller", "Action", "Adventure", "Romance"]


def p3_exploratorio_densidade(df: pd.DataFrame, modo: str | None = None, generos=None, min_filmes: int = 0) -> go.Figure:
    modo = modo or modo_atual()
    c = CORES[modo]
    base = _base_genero(df, _escolher(GENEROS_P3, generos), so_positivo=True, min_filmes=min_filmes)
    generos = _presentes(base, GENEROS_P3)
    lx, ly = np.log10(base["budget"]), np.log10(base["retorno"])
    # = np.logspace do notebook, em log10 (a folga só entra se o recorte deixar todos os valores iguais)
    x_bins = np.linspace(lx.min() - 0.5 * (lx.min() == lx.max()), lx.max() + 0.5 * (lx.min() == lx.max()), 34)
    y_bins = np.linspace(ly.min() - 0.5 * (ly.min() == ly.max()), ly.max() + 0.5 * (ly.min() == ly.max()), 34)
    xc, yc = (x_bins[:-1] + x_bins[1:]) / 2, (y_bins[:-1] + y_bins[1:]) / 2

    titulos = [f"{GENEROS_PT[g]} · n={_br(base['genres'].eq(g).sum(), 0)}" for g in generos]
    linhas, colunas = _grade(len(generos), 3)
    fig = make_subplots(rows=linhas, cols=colunas, shared_xaxes=True, shared_yaxes=True, subplot_titles=titulos,
                        horizontal_spacing=0.09, vertical_spacing=0.12)
    for k, genero in enumerate(generos):
        d = base.loc[base["genres"].eq(genero)]
        cont, _, _ = np.histogram2d(np.log10(d["budget"]), np.log10(d["retorno"]), bins=[x_bins, y_bins])
        z = np.where(cont > 0, np.log10(np.maximum(cont, 1)), np.nan).T  # escala de cor log (como LogNorm), bins vazios sem cor
        # [IA] Claude Code – barra de cor de cada painel (escala própria), com marcas em nº de filmes
        linha, coluna = k // colunas + 1, k % colunas + 1
        painel = fig.get_subplot(linha, coluna)
        dx, dy = painel.xaxis.domain, painel.yaxis.domain
        marcas = [v for v in (1, 2, 5, 10, 20, 50, 100) if v <= max(cont.max(), 1)]
        fig.add_trace(go.Heatmap(
            x=xc, y=yc, z=z, customdata=cont.T, showscale=True, hoverongaps=False,
            colorscale=[[0, c["cartao"]], [1, _cor_genero(genero, modo)]], zmin=0,
            colorbar=dict(x=dx[1] + 0.004, xanchor="left", y=(dy[0] + dy[1]) / 2, yanchor="middle",
                          len=0.85 * (dy[1] - dy[0]), thickness=8, outlinewidth=0,
                          tickvals=np.log10(marcas), ticktext=[str(v) for v in marcas],
                          tickfont=dict(family=FONTE_NUMERO, size=9, color=c["texto_suave"]),
                          title=dict(text="filmes", side="top", font=dict(family=FONTE_TEXTO, size=10, color=c["texto_suave"]))),
            hovertemplate="orçamento ≈ US$ %{x:.2f} (log10)<br>retorno ≈ %{y:.2f} (log10)<br>filmes: %{customdata:.0f}<extra>"
                          + GENEROS_PT[genero] + "</extra>",
        ), row=linha, col=coluna)
        fig.add_hline(y=0, line=dict(dash="dash", color=c["texto_suave"], width=1), row=k // colunas + 1, col=k % colunas + 1)

    marcas_x = [1e4, 1e6, 1e8]  # poucas marcas para não encostar entre painéis
    marcas_y = [0.01, 0.1, 1, 10, 100, 1000, 10000]
    fig.update_xaxes(tickvals=np.log10(marcas_x), ticktext=[fmt_usd(v) for v in marcas_x], showgrid=False)
    fig.update_yaxes(tickvals=np.log10(marcas_y), ticktext=[f"{_br(v, 0 if v >= 1 else (1 if v >= 0.1 else 2))}×" for v in marcas_y], showgrid=False)
    aplicar_estilo(fig, "Concentração e dispersão de orçamento e retorno por gênero",
                   "Filmes por região (cor mais intensa = mais filmes; cada painel tem sua própria barra de cor) · linha tracejada = 1×",
                   modo=modo, altura=200 + 210 * linhas)
    _rotulos_titulos_subplot(fig, modo, "Orçamento (US$, escala logarítmica)", "Retorno bruto (receita ÷ orçamento, escala logarítmica)")
    return fig


# ==========================================================================================
# Pessoa 3 — explanatório: Spearman por gênero com IC bootstrap de 95%
# ==========================================================================================
def _bootstrap_spearman(x, y, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(x)
    estimativas = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        estimativas[b] = pd.Series(x[idx]).rank().corr(pd.Series(y[idx]).rank())
    return np.nanquantile(estimativas, [0.025, 0.975])


def p3_tabela_spearman(df: pd.DataFrame, generos=None, min_filmes: int = 0) -> pd.DataFrame:
    """Mesmo cálculo da célula da Pessoa 3 (mesmas sementes) — os números batem com o storytelling."""
    base = _base_genero(df, _escolher(GENEROS_P3, generos), so_positivo=True, min_filmes=min_filmes)
    linhas = []
    for i, genero in enumerate(GENEROS_P3):  # semente pela posição na lista do autor, mesmo com filtro
        d = base.loc[base["genres"].eq(genero)]
        if d.empty:
            continue
        x, y = d["budget"].to_numpy(), d["retorno"].to_numpy()
        rho = pd.Series(x).rank().corr(pd.Series(y).rank())
        inf, sup = _bootstrap_spearman(x, y, seed=42 + i)
        linhas.append({"genero": genero, "gênero": GENEROS_PT[genero], "rho": rho, "ic_inf": inf, "ic_sup": sup, "filmes": len(d)})
    return pd.DataFrame(linhas).sort_values("rho").reset_index(drop=True)


TITULO_P3_EXPLANATORIO = "Associação negativa em quatro gêneros; próxima de zero em dois"


def p3_explanatorio_spearman(df: pd.DataFrame, modo: str | None = None, generos=None, min_filmes: int = 0,
                             titulo: str = TITULO_P3_EXPLANATORIO) -> go.Figure:
    """`titulo` traz a conclusão da Pessoa 3 para a base inteira; o dashboard troca por um neutro com filtro ativo."""
    modo = modo or modo_atual()
    c = CORES[modo]
    tab = p3_tabela_spearman(df, generos, min_filmes)
    max_n = tab["filmes"].max()
    fig = go.Figure()
    for _, r in tab.iterrows():
        cor = _cor_genero(r["genero"], modo)
        fig.add_trace(go.Scatter(
            x=[r["rho"]], y=[r["gênero"]], mode="markers", name=r["gênero"], showlegend=False,
            marker=dict(color=cor, size=8 + 14 * r["filmes"] / max_n),
            error_x=dict(type="data", symmetric=False, array=[r["ic_sup"] - r["rho"]], arrayminus=[r["rho"] - r["ic_inf"]],
                         color=cor, thickness=2.5, width=0),
            hovertemplate=(f"<b>{r['gênero']}</b><br>rho: {_br(r['rho'], 3)}<br>"
                           f"IC 95%: {_br(r['ic_inf'], 3)} a {_br(r['ic_sup'], 3)}<br>filmes: {_br(r['filmes'], 0)}<extra></extra>"),
        ))
    # [IA] Claude Code – legenda do tamanho do ponto (mesma fórmula do tamanho acima)
    referencias = sorted({int(tab["filmes"].min()), int(tab["filmes"].max())})
    if len(referencias) == 2 and referencias[0] < 1000 < referencias[1]:
        referencias.insert(1, 1000)
    for i, n in enumerate(referencias):
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers", name=f"{_br(n, 0)} filmes", showlegend=True, hoverinfo="skip",
            marker=dict(color=c["contexto"], size=8 + 14 * n / max_n),
            legendgroup="tamanho", legendgrouptitle_text="Tamanho do ponto:" if i == 0 else None,
        ))
    fig.add_vline(x=0, line=dict(dash="dash", color=c["texto_suave"], width=1.2))
    aplicar_estilo(fig, titulo,
                   "Correlação de Spearman entre orçamento e retorno · segmento = IC bootstrap de 95% · tamanho = nº de filmes",
                   modo=modo, altura=max(300, 170 + 50 * len(tab)))
    fig.update_xaxes(title_text="Correlação de Spearman entre orçamento e retorno", range=[-1, 1], zeroline=False)
    fig.update_yaxes(title_text="Gênero", showgrid=False, tickfont=dict(family=FONTE_TEXTO, size=13, color=c["texto"]))
    # legenda de tamanho dentro do gráfico, no canto vazio (à direita do zero), em vez de acima do título
    fig.update_layout(legend=dict(orientation="v", x=0.99, xanchor="right", y=0.03, yanchor="bottom",
                                  itemsizing="trace", tracegroupgap=0), margin_t=96)
    return fig


# ==========================================================================================
# Pessoa 2 — exploratório (violino): distribuição do retorno por gênero (violino) e parcela abaixo de 1×
# ==========================================================================================
# [IA] Claude Code – implementação em plotly do violino escolhido pelo grupo (tipo de gráfico, variáveis e objetivo são do grupo)
def p4_tabela_violino(df: pd.DataFrame, generos=None, min_filmes: int = MIN_FILMES_GENERO) -> pd.DataFrame:
    """Filmes com retorno e % com retorno < 1× (receita menor que o orçamento) por gênero, do maior % ao menor."""
    base = explodir(df.loc[df["retorno"].gt(0), ["id", "retorno", "genres"]], "genres").dropna(subset=["genres"])
    base = base.drop_duplicates(["id", "genres"])
    if generos is not None:
        base = base.loc[base["genres"].isin(generos)]
    tab = base.groupby("genres")["retorno"].agg(filmes="size", pct_abaixo=lambda r: r.lt(1).mean(), mediana="median")
    tab = tab.loc[tab["filmes"].ge(max(min_filmes, 1))].reset_index().rename(columns={"genres": "genero"})
    tab["gênero"] = tab["genero"].map(lambda g: GENEROS_PT.get(g, g))
    return tab.sort_values("pct_abaixo", ascending=False).reset_index(drop=True)


def p4_exploratorio_violino(df: pd.DataFrame, modo: str | None = None, generos=None,
                            min_filmes: int = MIN_FILMES_GENERO) -> go.Figure:
    modo = modo or modo_atual()
    c = CORES[modo]
    min_filmes = max(min_filmes, MIN_FILMES_GENERO)
    tab = p4_tabela_violino(df, generos, min_filmes)
    if tab.empty:
        raise SemDados
    base = explodir(df.loc[df["retorno"].gt(0), ["id", "retorno", "genres"]], "genres").dropna(subset=["genres"])
    base = base.drop_duplicates(["id", "genres"])
    fig = go.Figure()
    # o violino estima a densidade em escala linear; como o retorno é muito assimétrico, desenhamos log10(retorno)
    # e trocamos as marcas do eixo por valores em ×
    for k, r in tab.iterrows():
        d = base.loc[base["genres"].eq(r["genero"]), "retorno"]
        destaque = k < 3  # os 3 gêneros com maior % de filmes abaixo de 1×
        cor = c["prejuizo"] if destaque else c["contexto"]
        fig.add_trace(go.Violin(
            x=np.log10(d), y=[f"{r['gênero']} ({fmt_pct(r['pct_abaixo'], 0)})"] * len(d), orientation="h",
            name=r["gênero"], showlegend=False, spanmode="hard", width=0.9, points=False,
            line=dict(color=cor, width=1.2), fillcolor=cor, opacity=0.75 if destaque else 0.6,
            box=dict(visible=True, width=0.25, fillcolor=c["cartao"], line=dict(color=c["texto"], width=1)),
            meanline=dict(visible=False),
            hovertemplate=(f"<b>{r['gênero']}</b><br>filmes: {_br(r['filmes'], 0)}<br>"
                           f"abaixo de 1×: {fmt_pct(r['pct_abaixo'], 1)}<br>"
                           f"retorno mediano: {fmt_mult(r['mediana'])}<extra></extra>"),
        ))
    fig.add_vline(x=0, line=dict(dash="dash", color=c["texto_suave"], width=1.2))
    fig.add_annotation(x=0, y=1, yref="paper", yanchor="bottom", xanchor="left", xshift=4, showarrow=False,
                       text="empate (1×)", font=dict(family=FONTE_TEXTO, size=11, color=c["texto_suave"]))
    valores = [0.1, 0.5, 1, 2, 5, 10, 100, 1000]
    aplicar_estilo(fig, "Distribuição do retorno por gênero",
                   f"Violino do retorno (receita ÷ orçamento, escala log) · entre parênteses, % de filmes abaixo de 1× · "
                   f"gêneros com ao menos {min_filmes} filmes",
                   modo=modo, altura=max(300, min(760, 170 + 36 * len(tab))))
    fig.update_xaxes(title_text="Retorno (receita ÷ orçamento, escala logarítmica)", tickvals=list(np.log10(valores)),
                     ticktext=[fmt_mult(v, 0 if v >= 1 else (1 if v >= 0.1 else 2)) for v in valores],
                     range=[np.log10(0.1), np.log10(2000)])
    fig.update_yaxes(showgrid=False, autorange="reversed", tickfont=dict(family=FONTE_TEXTO, size=12, color=c["texto"]),
                     title_text="Gênero (% abaixo de 1×)")
    fig.update_layout(violingap=0.15, separators=",.")
    return fig


# ==========================================================================================
# Catálogo (ordem de exibição no dashboard)
# ==========================================================================================
GRAFICOS = [
# "secao": título descritivo usado no dashboard enquanto o autor não escreve o título-achado (notas.TITULOS_ACHADO)
    {"id": "p1_exploratorio", "tipo": "exploratório", "autor": "Pessoa 1", "funcao": p1_exploratorio_cobertura,
     "secao": "Cobertura de orçamento e receita por gênero", "curto": "Cobertura por gênero"},
    {"id": "p2_exploratorio", "tipo": "exploratório", "autor": "Pessoa 2", "funcao": p2_exploratorio_dispersao,
     "secao": "Orçamento e retorno de cada filme, em quatro gêneros", "curto": "Dispersão por gênero"},
    {"id": "p3_exploratorio", "tipo": "exploratório", "autor": "Pessoa 3", "funcao": p3_exploratorio_densidade,
     "secao": "Onde os filmes se concentram em orçamento e retorno, em seis gêneros", "curto": "Concentração por gênero"},
    {"id": "p4_exploratorio", "tipo": "exploratório", "autor": "Pessoa 2", "funcao": p4_exploratorio_violino,
     "secao": "Distribuição do retorno por gênero e parcela de filmes abaixo de 1×", "curto": "Violino por gênero"},
    {"id": "p2_explanatorio", "tipo": "explanatório", "autor": "Pessoa 2", "funcao": p2_explanatorio_faixas,
     "secao": "Retorno mediano por faixa de orçamento e gênero", "curto": "Retorno por faixa"},
    {"id": "p3_explanatorio", "tipo": "explanatório", "autor": "Pessoa 3", "funcao": p3_explanatorio_spearman,
     "secao": "Associação entre orçamento e retorno em cada gênero", "curto": "Associação por gênero",
     "titulo_neutro": "Associação entre orçamento e retorno por gênero no recorte"},
]


# ==========================================================================================
# Dashboard — animação década a década (diferencial interativo; não é gráfico de nenhum autor)
# ==========================================================================================
# [IA] Claude Code – animação do percurso das medianas por década, com troca de métrica e de escala
MIN_FILMES_DECADA = 10  # décadas com menos filmes que isso (com retorno) ficam fora da animação

# métrica do eixo vertical: rótulo, nome no resumo (concordância), unidade ("x" ou "usd") e linha de empate
METRICAS = {
    "retorno": {"rotulo": "Retorno (receita ÷ orçamento)", "resumo": "retorno mediano", "unidade": "x",
                "empate": "empate (1×)"},
    "lucro": {"rotulo": "Lucro bruto (receita − orçamento, US$)", "resumo": "lucro bruto mediano", "unidade": "usd",
              "empate": "empate (US$ 0)"},
    "receita": {"rotulo": "Receita (US$)", "resumo": "receita mediana", "unidade": "usd",
                "empate": "empate (receita = orçamento)"},
}
ESCALAS = {"log": "Logarítmica", "linear": "Linear"}


def _rotulo_decada(d: int) -> str:
    return f"anos {d % 100:02d}" if d < 2000 else f"anos {d}"


def _fmt_valor(v: float, unidade: str) -> str:
    if unidade == "x":
        return fmt_mult(v)
    casas = 1 if 1e6 <= abs(v) < 1e7 else 0
    return ("−" if v < 0 else "") + fmt_usd(abs(v), casas)


def _marcas_lineares(lo: float, hi: float, n: int = 5) -> list[float]:
    """~n marcas "redondas" (1, 2, 2,5 ou 5 × 10^k) entre lo e hi."""
    bruto = (hi - lo) / n
    base = 10 ** np.floor(np.log10(bruto))
    passo = next(m * base for m in (1, 2, 2.5, 5, 10) if m * base >= bruto)
    return list(np.arange(np.ceil(lo / passo) * passo, hi + passo / 2, passo))


def _eixo(fig: go.Figure, eixo: str, valores: pd.Series, escala: str, unidade: str, titulo: str) -> str:
    """Intervalo fixo (o mesmo em todos os quadros) e marcas do eixo. Devolve o aviso de corte, se houver."""
    atualizar = fig.update_xaxes if eixo == "x" else fig.update_yaxes
    rot = (lambda v: ("−" if v < 0 else "") + fmt_usd(abs(v)).replace(",0B", "B")) if unidade == "usd" else \
          (lambda v: f"{_br(v, max(0, -int(np.floor(np.log10(v)))) if v > 0 else 0)}×")
    if escala == "log":
        lv = np.log10(valores)
        lo, hi = lv.min() - 0.25, lv.max() + 0.25
        marcas = [10.0 ** k for k in range(int(np.floor(lo)), int(np.ceil(hi)) + 1) if lo <= k <= hi]
        atualizar(type="log", range=[lo, hi], tickvals=marcas, ticktext=[rot(v) for v in marcas],
                  minor=dict(showgrid=False),
                  title_text=f"{titulo[:-1]}, escala logarítmica)" if titulo.endswith(")") else f"{titulo} (escala logarítmica)")
        return ""
    # linear: corta o 1% mais extremo de cada lado (senão poucos outliers achatam todo o resto)
    lo, hi = valores.quantile(0.01), valores.quantile(0.99)
    lo = min(lo, 0)
    folga = (hi - lo) * 0.05
    marcas = [m for m in _marcas_lineares(lo, hi) if lo - folga <= m <= hi + folga]
    atualizar(type="linear", range=[lo - folga, hi + folga], tickvals=marcas, ticktext=[rot(v) for v in marcas],
              title_text=titulo)
    return "corta" if (valores > hi + folga).any() or (valores < lo - folga).any() else ""


def animacao_decadas(df: pd.DataFrame, modo: str | None = None, generos=None,
                     metrica: str = "retorno", escala: str = "log") -> go.Figure:
    """Dispersão orçamento × métrica animada por década, com a mediana da década e o percurso das medianas.

    metrica: "retorno" | "lucro" | "receita" (eixo vertical).  escala: "log" | "linear" (os dois eixos);
    lucro pode ser negativo, então usa sempre escala linear.
    `generos`: None = todos; senão, filmes com algum dos gêneros escolhidos. Levanta SemDados se sobrar < 2 décadas.
    Os mesmos filmes entram em todas as métricas: os com orçamento e receita válidos.
    """
    modo = modo or modo_atual()
    c = CORES[modo]
    m = METRICAS[metrica]
    escala = "linear" if metrica == "lucro" else escala
    base = df.loc[df["retorno"].gt(0) & df["budget"].gt(0) & df["ano"].notna(),
                  ["title", "ano", "budget", "revenue", "retorno", "genres"]]
    if generos is not None:
        base = base.loc[base["genres"].map(lambda gs: bool(set(gs) & set(generos)))]
    base = base.assign(decada=(base["ano"] // 10 * 10).astype(int),
                       y={"retorno": base["retorno"], "lucro": base["revenue"] - base["budget"],
                          "receita": base["revenue"]}[metrica])
    base = base.loc[base["decada"].map(base["decada"].value_counts()).ge(MIN_FILMES_DECADA)]
    decadas = sorted(base["decada"].unique())
    if len(decadas) < 2:
        raise SemDados
    med = base.groupby("decada").agg(orc=("budget", "median"), y=("y", "median"), n=("y", "size")).loc[decadas]

    # linha de empate: 1× no retorno, US$ 0 no lucro, receita = orçamento na receita (diagonal)
    x_lim = np.array([base["budget"].min() / 10, base["budget"].max() * 10])
    y_empate = {"retorno": [1, 1], "lucro": [0, 0], "receita": x_lim}[metrica]
    fmt_y = "%{y:.2f}×" if m["unidade"] == "x" else "US$ %{y:,.0f}"
    fonte_rotulo = dict(family=FONTE_TEXTO, size=11, color=c["texto_suave"])

    def tracos(i: int) -> list[go.Scatter]:
        d = decadas[i]
        filmes = base.loc[base["decada"].eq(d)]
        trilha = med.iloc[: i + 1]
        return [
            go.Scatter(x=x_lim, y=y_empate, mode="lines", name=m["empate"], hoverinfo="skip",
                       line=dict(dash="dash", color=c["texto_suave"], width=1)),
            go.Scatter(x=filmes["budget"], y=filmes["y"], mode="markers", name="filmes da década",
                       # décadas com poucos filmes: pontos mais opacos para não sumirem
                       marker=dict(color=c["destaque"], size=6 if len(filmes) > 200 else 8,
                                   opacity=0.35 if len(filmes) > 200 else 0.7, line=dict(width=0)),
                       customdata=np.column_stack([filmes["title"], filmes["ano"].astype(int)]),
                       hovertemplate="<b>%{customdata[0]}</b> (%{customdata[1]})<br>orçamento: US$ %{x:,.0f}"
                                     f"<br>{m['resumo'].split()[0]}: {fmt_y}<extra></extra>"),
            go.Scatter(x=trilha["orc"], y=trilha["y"], mode="lines+markers+text", name="percurso das medianas",
                       line=dict(color=c["texto"], width=2.5),
                       marker=dict(color=c["texto"], size=8, line=dict(color=c["cartao"], width=1.5)),
                       # rótulo só no início do percurso: as medianas ficam próximas e os rótulos se sobrepõem
                       text=[_rotulo_decada(x) if k == 0 else "" for k, x in enumerate(trilha.index)],
                       customdata=[_rotulo_decada(x) for x in trilha.index],
                       textposition="top left", textfont=fonte_rotulo,
                       hovertemplate=f"%{{customdata}}<br>orçamento mediano: US$ %{{x:,.0f}}<br>{m['resumo']}: {fmt_y}"
                                     "<extra></extra>"),
            go.Scatter(x=[med["orc"].iloc[i]], y=[med["y"].iloc[i]], mode="markers", name="mediana da década",
                       marker=dict(color=c["texto"], size=18, symbol="diamond", line=dict(color=c["cartao"], width=2)),
                       hovertemplate=f"{_rotulo_decada(d)}<br>orçamento mediano: US$ %{{x:,.0f}}"
                                     f"<br>{m['resumo']}: {fmt_y}<extra></extra>"),
        ]

    def legenda_quadro(i: int) -> list[dict]:
        r = med.iloc[i]
        return [dict(text=f"<b>{_rotulo_decada(decadas[i])}</b> · {_br(r['n'], 0)} filmes · {m['resumo']} "
                          f"{_fmt_valor(r['y'], m['unidade'])} · orçamento mediano {_fmt_valor(r['orc'], 'usd')}",
                     xref="paper", yref="paper", x=0, y=1, xanchor="left", yanchor="bottom", yshift=6, showarrow=False,
                     font=dict(family=FONTE_TEXTO, size=13, color=c["texto"]))]

    fig = go.Figure(data=tracos(0),
                    frames=[go.Frame(data=tracos(i), name=str(d), layout=dict(annotations=legenda_quadro(i)))
                            for i, d in enumerate(decadas)])
    # eixos fixos (senão a escala muda a cada quadro)
    corte = _eixo(fig, "x", base["budget"], escala, "usd", "Orçamento (US$ nominais)")
    corte += _eixo(fig, "y", base["y"], escala, m["unidade"], m["rotulo"])
    subtitulo = ("Cada ponto é um filme · ◆ = mediana da década · linha = percurso das medianas · "
                 + ("valores nominais · eixos cortados no 1% mais extremo" if corte
                    else "valores nominais, sem correção pela inflação"))
    aplicar_estilo(fig, f"{m['rotulo'].split(' (')[0]} e orçamento década a década", subtitulo, modo=modo, altura=640)

    quadro = dict(frame=dict(duration=1100, redraw=True), transition=dict(duration=500, easing="cubic-in-out"), mode="immediate")
    fonte_controle = dict(family=FONTE_TEXTO, size=12, color=c["texto"])
    fig.update_layout(
        annotations=[*fig.layout.annotations, *legenda_quadro(0)],
        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="right", x=1),
        margin=dict(b=120),
        updatemenus=[dict(
            type="buttons", direction="left", x=0, y=-0.13, xanchor="left", yanchor="top", showactive=False,
            pad=dict(r=8, t=0), font=fonte_controle, bgcolor=c["cartao"], bordercolor=c["linha"],
            buttons=[dict(label="▶ Animar", method="animate", args=[None, {**quadro, "fromcurrent": True}]),
                     dict(label="⏸ Pausar", method="animate",
                          args=[[None], dict(frame=dict(duration=0, redraw=False), transition=dict(duration=0), mode="immediate")])],
        )],
        sliders=[dict(
            active=0, x=0.2, len=0.8, y=-0.1, xanchor="left", yanchor="top", pad=dict(t=0, b=0),
            currentvalue=dict(visible=False), font=fonte_controle, bgcolor=c["cartao"], bordercolor=c["linha"],
            activebgcolor=c["destaque"], tickcolor=c["texto_suave"],
            transition=dict(duration=500, easing="cubic-in-out"),
            steps=[dict(label=_rotulo_decada(d), method="animate",
                        args=[[str(d)], dict(frame=dict(duration=500, redraw=True), transition=dict(duration=500), mode="immediate")])
                   for d in decadas],
        )],
    )
    return fig
