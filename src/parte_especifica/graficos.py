"""Gráficos da Parte Específica em plotly, com a identidade visual do projeto (estilo.py).

Cada função recebe o DataFrame já tratado (tratar(carregar_filmes())) e devolve uma figura plotly.
O conteúdo (tipo de gráfico, variáveis, gêneros, faixas, títulos) é o que cada autor definiu no
eda.ipynb; aqui só muda o visual (paleta, fontes, formato brasileiro, modo claro/escuro).
As notas analíticas de cada gráfico ficam em notas.py.

Uso no dashboard:
    from graficos import GRAFICOS
    for g in GRAFICOS: grafico(g["funcao"](df)); nota(NOTAS[g["id"]])
"""
# [IA] Claude Code – conversão dos gráficos do eda.ipynb para plotly com o estilo do projeto
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dados import explodir
from estilo import (CORES, GENEROS_PT, LIMITES_FAIXAS, ROTULOS_FAIXAS, FONTE_NUMERO, FONTE_TEXTO, _br,
                    aplicar_estilo, eixo_log, fmt_mult, fmt_pct, fmt_usd, mapa_cores_generos, modo_atual)


def _cor_genero(genero_en: str, modo: str) -> str:
    """Cor fixa do gênero (estilo.py); gêneros fora dos 6 com cor ficam em cinza de contexto."""
    return mapa_cores_generos(modo).get(genero_en, CORES[modo]["contexto"])


def _base_genero(df: pd.DataFrame, generos: list[str], so_positivo: bool = False) -> pd.DataFrame:
    """Uma linha por (filme, gênero) só com filmes de retorno válido e os gêneros pedidos."""
    filtro = df["retorno"].gt(0) if so_positivo else df["retorno"].notna()
    base = explodir(df.loc[filtro, ["id", "budget", "retorno", "genres"]], "genres").dropna(subset=["genres"])
    base = base.drop_duplicates(["id", "genres"])
    return base.loc[base["genres"].isin(generos)].copy()


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


def p1_tabela_cobertura(df: pd.DataFrame) -> pd.DataFrame:
    """Por gênero: nº de filmes e % com orçamento, receita e os dois (= retorno calculável) após tratar().

    Gêneros com menos de MIN_FILMES_GENERO filmes ficam de fora. Ordem: menor cobertura primeiro.
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
    tab = tab.loc[tab["filmes"].ge(MIN_FILMES_GENERO)]
    return tab.sort_values("pct_retorno").reset_index(drop=True)


def p1_exploratorio_cobertura(df: pd.DataFrame, modo: str | None = None) -> go.Figure:
    modo = modo or modo_atual()
    c = CORES[modo]
    tab = p1_tabela_cobertura(df)
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
                       text=f"base toda: {fmt_pct(geral, 0)}", font=dict(family=FONTE_TEXTO, size=11, color=c["texto_suave"]))
    aplicar_estilo(fig, "Quanto de cada gênero tem orçamento e receita informados",
                   f"% de filmes com retorno calculável (orçamento e receita válidos) · gêneros com ao menos {MIN_FILMES_GENERO} filmes",
                   modo=modo, altura=620)
    fig.update_xaxes(title_text="Filmes com retorno calculável (% do gênero)", range=[0, 1.08], tickformat=".0%",
                     tickvals=[0, 0.25, 0.5, 0.75, 1], ticktext=["0%", "25%", "50%", "75%", "100%"])
    fig.update_yaxes(showgrid=False, tickfont=dict(family=FONTE_TEXTO, size=12, color=c["texto"]))
    fig.update_layout(separators=",.", bargap=0.25)
    return fig


# ==========================================================================================
# Pessoa 2 — exploratório: dispersão orçamento × retorno por gênero (cell "Visualizações selecionadas")
# ==========================================================================================
GENEROS_P2 = ["Action", "Comedy", "Drama", "Adventure"]


def p2_exploratorio_dispersao(df: pd.DataFrame, modo: str | None = None) -> go.Figure:
    modo = modo or modo_atual()
    base = _base_genero(df, GENEROS_P2)
    base["orcamento_mi"] = base["budget"] / 1e6
    fig = make_subplots(rows=2, cols=2, shared_xaxes=True, shared_yaxes=True,
                        subplot_titles=[GENEROS_PT[g] for g in GENEROS_P2],
                        horizontal_spacing=0.06, vertical_spacing=0.12)
    for k, genero in enumerate(GENEROS_P2):
        d = base.loc[base["genres"].eq(genero)]
        fig.add_trace(go.Scatter(
            x=d["orcamento_mi"], y=d["retorno"], mode="markers", name=GENEROS_PT[genero], showlegend=False,
            marker=dict(color=_cor_genero(genero, modo), size=5, opacity=0.35),
            hovertemplate="orçamento: US$ %{x:,.1f}M<br>retorno: %{y:.2f}×<extra>" + GENEROS_PT[genero] + "</extra>",
        ), row=k // 2 + 1, col=k % 2 + 1)
    aplicar_estilo(fig, "Orçamento e retorno dos filmes por gênero",
                   "Cada ponto é um filme · eixos em escala logarítmica · mesma escala nos quatro painéis", modo=modo, altura=620)
    eixo_log(fig, "x", sufixo="", valores=[0.01, 0.1, 1, 10, 100])
    eixo_log(fig, "y", valores=[0.01, 0.1, 1, 10, 100, 1000, 10000])
    fig.update_xaxes(ticktext=[fmt_usd(v * 1e6) for v in [0.01, 0.1, 1, 10, 100]])
    _rotulos_titulos_subplot(fig, modo, "Orçamento (US$, escala logarítmica)", "Retorno (receita ÷ orçamento, escala logarítmica)")
    return fig


# ==========================================================================================
# Pessoa 2 — explanatório: mediana do retorno por faixa de orçamento e gênero (barras agrupadas)
# ==========================================================================================
def p2_tabela_faixas(df: pd.DataFrame) -> pd.DataFrame:
    """Mediana do retorno e nº de filmes por faixa e gênero (mesmo cálculo do notebook)."""
    base = _base_genero(df, GENEROS_P2)
    base["faixa"] = pd.cut(base["budget"], bins=[0, *LIMITES_FAIXAS, np.inf], labels=ROTULOS_FAIXAS, right=True)
    return base.groupby(["faixa", "genres"], observed=True)["retorno"].agg(mediana="median", filmes="size").reset_index()


def p2_explanatorio_faixas(df: pd.DataFrame, modo: str | None = None) -> go.Figure:
    modo = modo or modo_atual()
    c = CORES[modo]
    tab = p2_tabela_faixas(df)
    maximos = tab.groupby("faixa", observed=True)["mediana"].transform("max")
    tab["maior"] = tab["mediana"].eq(maximos)
    fig = go.Figure()
    for genero in GENEROS_P2:
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


def p3_exploratorio_densidade(df: pd.DataFrame, modo: str | None = None) -> go.Figure:
    modo = modo or modo_atual()
    c = CORES[modo]
    base = _base_genero(df, GENEROS_P3, so_positivo=True)
    lx, ly = np.log10(base["budget"]), np.log10(base["retorno"])
    x_bins = np.linspace(lx.min(), lx.max(), 34)  # = np.logspace do notebook, em log10
    y_bins = np.linspace(ly.min(), ly.max(), 34)
    xc, yc = (x_bins[:-1] + x_bins[1:]) / 2, (y_bins[:-1] + y_bins[1:]) / 2

    titulos = [f"{GENEROS_PT[g]} · n={_br(base['genres'].eq(g).sum(), 0)}" for g in GENEROS_P3]
    fig = make_subplots(rows=2, cols=3, shared_xaxes=True, shared_yaxes=True, subplot_titles=titulos,
                        horizontal_spacing=0.04, vertical_spacing=0.12)
    for k, genero in enumerate(GENEROS_P3):
        d = base.loc[base["genres"].eq(genero)]
        cont, _, _ = np.histogram2d(np.log10(d["budget"]), np.log10(d["retorno"]), bins=[x_bins, y_bins])
        z = np.where(cont > 0, np.log10(np.maximum(cont, 1)), np.nan).T  # escala de cor log (como LogNorm), bins vazios sem cor
        fig.add_trace(go.Heatmap(
            x=xc, y=yc, z=z, customdata=cont.T, showscale=False, hoverongaps=False,
            colorscale=[[0, c["cartao"]], [1, _cor_genero(genero, modo)]], zmin=0,
            hovertemplate="orçamento ≈ US$ %{x:.2f} (log10)<br>retorno ≈ %{y:.2f} (log10)<br>filmes: %{customdata:.0f}<extra>"
                          + GENEROS_PT[genero] + "</extra>",
        ), row=k // 3 + 1, col=k % 3 + 1)
        fig.add_hline(y=0, line=dict(dash="dash", color=c["texto_suave"], width=1), row=k // 3 + 1, col=k % 3 + 1)

    marcas_x = [1e4, 1e6, 1e8]  # poucas marcas para não encostar entre painéis
    marcas_y = [0.01, 0.1, 1, 10, 100, 1000, 10000]
    fig.update_xaxes(tickvals=np.log10(marcas_x), ticktext=[fmt_usd(v) for v in marcas_x], showgrid=False)
    fig.update_yaxes(tickvals=np.log10(marcas_y), ticktext=[f"{_br(v, 0 if v >= 1 else (1 if v >= 0.1 else 2))}×" for v in marcas_y], showgrid=False)
    aplicar_estilo(fig, "Concentração e dispersão de orçamento e retorno por gênero",
                   "Filmes por região (cor mais intensa = mais filmes; cada painel tem sua própria escala) · linha tracejada = 1×",
                   modo=modo, altura=620)
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


def p3_tabela_spearman(df: pd.DataFrame) -> pd.DataFrame:
    """Mesmo cálculo da célula da Pessoa 3 (mesmas sementes) — os números batem com o storytelling."""
    base = _base_genero(df, GENEROS_P3, so_positivo=True)
    linhas = []
    for i, genero in enumerate(GENEROS_P3):
        d = base.loc[base["genres"].eq(genero)]
        x, y = d["budget"].to_numpy(), d["retorno"].to_numpy()
        rho = pd.Series(x).rank().corr(pd.Series(y).rank())
        inf, sup = _bootstrap_spearman(x, y, seed=42 + i)
        linhas.append({"genero": genero, "gênero": GENEROS_PT[genero], "rho": rho, "ic_inf": inf, "ic_sup": sup, "filmes": len(d)})
    return pd.DataFrame(linhas).sort_values("rho").reset_index(drop=True)


def p3_explanatorio_spearman(df: pd.DataFrame, modo: str | None = None) -> go.Figure:
    modo = modo or modo_atual()
    c = CORES[modo]
    tab = p3_tabela_spearman(df)
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
    fig.add_vline(x=0, line=dict(dash="dash", color=c["texto_suave"], width=1.2))
    aplicar_estilo(fig, "Associação negativa em quatro gêneros; próxima de zero em dois",
                   "Correlação de Spearman entre orçamento e retorno · segmento = IC bootstrap de 95% · tamanho = nº de filmes",
                   modo=modo, altura=440)
    fig.update_xaxes(title_text="Correlação de Spearman entre orçamento e retorno", range=[-1, 1], zeroline=False)
    fig.update_yaxes(showgrid=False, tickfont=dict(family=FONTE_TEXTO, size=13, color=c["texto"]))
    return fig


# ==========================================================================================
# Catálogo (ordem de exibição no dashboard)
# ==========================================================================================
GRAFICOS = [
    {"id": "p1_exploratorio", "tipo": "exploratório", "autor": "Pessoa 1", "funcao": p1_exploratorio_cobertura},
    {"id": "p2_exploratorio", "tipo": "exploratório", "autor": "Pessoa 2", "funcao": p2_exploratorio_dispersao},
    {"id": "p3_exploratorio", "tipo": "exploratório", "autor": "Pessoa 3", "funcao": p3_exploratorio_densidade},
    {"id": "p2_explanatorio", "tipo": "explanatório", "autor": "Pessoa 2", "funcao": p2_explanatorio_faixas},
    {"id": "p3_explanatorio", "tipo": "explanatório", "autor": "Pessoa 3", "funcao": p3_explanatorio_spearman},
]
