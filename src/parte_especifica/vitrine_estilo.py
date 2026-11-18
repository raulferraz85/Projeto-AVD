"""Vitrine do sistema visual — DADOS FICTÍCIOS, só para ver cores, fontes e componentes.

Rodar da raiz do projeto:  streamlit run src/parte_especifica/vitrine_estilo.py
Alternar claro/escuro: menu ⋮ (canto superior direito) → Settings → Theme.
"""
# [IA] Claude Code – página de demonstração do estilo (exemplos de uso para o grupo)
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from estilo import (CORES, GENEROS_COM_COR, aplicar_estilo, card, cor_resultado, cores_destaque, eixo_log,
                    grafico, injetar_css, linha_empate, mapa_cores_faixas, mapa_cores_generos, nota, usar_tema)

st.set_page_config(page_title="Vitrine de estilo", layout="wide")
modo = usar_tema()
injetar_css()
c = CORES[modo]
rng = np.random.default_rng(7)

st.title("Vitrine do sistema visual")
st.caption(f"Modo atual: **{modo}** · todos os números desta página são **fictícios**.")

# ---------------------------------------------------------------- paleta
st.subheader("Paleta")


def amostras(itens):
    cols = st.columns(len(itens))
    for col, (nome, cor) in zip(cols, itens):
        col.markdown(
            f'<div style="height:56px;border-radius:10px;background:{cor};border:1px solid {c["linha"]}"></div>'
            f'<div style="font-size:.8rem;margin-top:.35rem">{nome}<br><code>{cor}</code></div>',
            unsafe_allow_html=True,
        )


amostras([(k, c[k]) for k in ["fundo", "cartao", "texto", "texto_suave", "linha", "contexto"]])
amostras([(k, c[k]) for k in ["destaque", "destaque_texto", "destaque_fundo", "lucro", "prejuizo"]] + [("", c["cartao"])])
amostras(list(zip(GENEROS_COM_COR, c["generos"])))

# ---------------------------------------------------------------- cards de número
st.subheader("Cards de número")
k1, k2, k3, k4 = st.columns(4)
with k1:
    card("Número principal (destaque)", "2,37×", destaque=True)
with k2:
    card("Filmes analisados", "3.210")
with k3:
    card("Com prejuízo", "24,7%")
with k4:
    card("Orçamento mediano", "US$ 23M")

# ---------------------------------------------------------------- gráficos
st.subheader("Gráficos")
g1, g2 = st.columns(2)

with g1:
    cats = ["A", "B", "C", "D", "E", "F", "G"]
    df = pd.DataFrame({"categoria": cats, "valor": rng.uniform(1, 4, len(cats)).round(2)}).sort_values("valor")
    fig = go.Figure(go.Bar(x=df["valor"], y=df["categoria"], orientation="h",
                           marker_color=cores_destaque(df["categoria"], df["categoria"].iloc[-1], modo),
                           hovertemplate="%{y}: %{x:.2f}×<extra></extra>"))
    aplicar_estilo(fig, "Título = a conclusão do gráfico", "Subtítulo = o que está sendo medido (exemplo: destaque + cinza)")
    fig.update_xaxes(title="valor (×)")
    grafico(fig)
    nota("Exemplo de nota analítica: o que mostra, qual o insight, por que esse gráfico.")

with g2:
    df = pd.DataFrame({"genero": GENEROS_COM_COR + ["Outros"], "valor": rng.uniform(1, 4, 7).round(2)})
    fig = px.bar(df, x="genero", y="valor", color="genero", color_discrete_map=mapa_cores_generos(modo))
    aplicar_estilo(fig, "Cores fixas por gênero", "Cada gênero tem sempre a mesma cor; o resto vira “Outros”")
    fig.update_layout(showlegend=False)
    fig.update_xaxes(title=None)
    fig.update_yaxes(title="valor (×)")
    grafico(fig)

g3, g4 = st.columns(2)
with g3:
    df = pd.DataFrame({"filme": [f"F{i}" for i in range(14)], "retorno": np.round(rng.lognormal(0.3, 0.8, 14), 2)}).sort_values("retorno")
    fig = go.Figure(go.Bar(x=df["filme"], y=df["retorno"], marker_color=cor_resultado(df["retorno"], modo),
                           hovertemplate="%{x}: %{y:.2f}×<extra></extra>"))
    aplicar_estilo(fig, "Lucro × prejuízo", "Prejuízo nunca só pela cor: sempre com a linha de empate em 1×")
    eixo_log(fig, "y", valores=[0.1, 0.5, 1, 2, 5, 10])
    linha_empate(fig, modo=modo)
    fig.update_yaxes(title="retorno (×, log)")
    grafico(fig)

with g4:
    n = 160
    orc = rng.lognormal(17, 1, n)
    df = pd.DataFrame({"orcamento": orc, "retorno": rng.lognormal(0.6, 1, n)})
    df["faixa"] = pd.cut(df["orcamento"], [0, 10e6, 40e6, 100e6, np.inf],
                         labels=list(mapa_cores_faixas(modo)), right=False)
    fig = px.scatter(df, x="orcamento", y="retorno", color="faixa", color_discrete_map=mapa_cores_faixas(modo),
                     category_orders={"faixa": list(mapa_cores_faixas(modo))}, log_x=True, log_y=True)
    aplicar_estilo(fig, "Faixas de orçamento em degradê", "Categorias ordenadas usam um tom só, do claro ao escuro")
    eixo_log(fig, "x", sufixo="US$", valores=[1e6, 1e7, 1e8, 1e9])
    eixo_log(fig, "y", valores=[0.1, 0.5, 1, 2, 5, 10, 100])
    linha_empate(fig, modo=modo)
    fig.update_xaxes(title="orçamento (US$, log)")
    fig.update_yaxes(title="retorno (×, log)")
    fig.update_layout(legend_title_text="")
    grafico(fig)

st.subheader("Controles")
a, b, d = st.columns(3)
a.multiselect("Filtro de exemplo", GENEROS_COM_COR, default=GENEROS_COM_COR[:3])
b.slider("Faixa de exemplo", 0, 300, (10, 100))
d.radio("Opção", ["Mediana", "Média"], horizontal=True)
st.button("Botão de exemplo", type="primary")
