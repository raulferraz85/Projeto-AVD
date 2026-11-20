"""Identidade visual do projeto (Parte Específica).

Fundo Cloud Dancer (off-white, Pantone 2026) + destaque azul cobalto elétrico.
Tipografia: Bricolage Grotesque (títulos), Geist (texto), Geist Mono (números).
Modo claro e escuro com cores próprias (o escuro NÃO é o claro invertido).

Uso em qualquer gráfico plotly:

    from estilo import aplicar_estilo
    fig = px.bar(...)
    aplicar_estilo(fig, titulo="Conclusão do gráfico", subtitulo="O que está sendo medido")

Regras de uso
-------------
1. Uma cor forte por gráfico: o destaque vai só na resposta, o resto fica em CONTEXTO (cinza).
2. O título do gráfico é a conclusão; o subtítulo diz o que está sendo medido.
3. A cor segue o gênero, não a posição no ranking. Filtrar nunca troca a cor de ninguém.
4. Prejuízo nunca aparece só pela cor: sempre com a linha de empate em 1× (linha_empate).
5. Nunca dois eixos Y no mesmo gráfico.
6. Marcas finas: grade clara, sem moldura, barras com cantos levemente arredondados.
7. Números em formato brasileiro: "2,37×", "US$ 150M", "24,7%" (fmt_mult, fmt_usd, fmt_pct).
8. Destaque e cores de gênero nunca no mesmo gráfico (o cobalto também é o 1º gênero).
"""
# [IA] Claude Code – implementação em plotly/streamlit do sistema visual definido pelo grupo
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

# ------------------------------------------------------------------------------------------
# 1. Cores
# ------------------------------------------------------------------------------------------
CORES = {
    "claro": {
        "fundo": "#EFEEEA",
        "cartao": "#F8F7F4",
        "texto": "#16171C",
        "texto_suave": "#5E5F66",
        "linha": "#DCDAD4",
        "destaque": "#3047E6",
        "destaque_texto": "#2236C9",
        "destaque_fundo": "#E1E4FB",
        "contexto": "#C3C2BC",
        "lucro": "#0E9483",
        "prejuizo": "#D2602F",
        "generos": ["#3047E6", "#D2602F", "#0E9483", "#A97A00", "#B83A7A", "#7A8A12"],
        "faixas": ["#9DA9F3", "#7584EE", "#4C61E8", "#2236C9"],
    },
    "escuro": {
        "fundo": "#111216",
        "cartao": "#1A1B21",
        "texto": "#ECEBE6",
        "texto_suave": "#9496A0",
        "linha": "#2A2C34",
        "destaque": "#7D8FFF",
        "destaque_texto": "#8C9BFF",
        "destaque_fundo": "#1F2350",
        "contexto": "#3A3C45",
        "lucro": "#1FA593",
        "prejuizo": "#D96A36",
        "generos": ["#6B7FFF", "#D96A36", "#1FA593", "#B48C14", "#D25A95", "#7F9122"],
        "faixas": ["#3B4AA8", "#5266E0", "#7D8FFF", "#B5C0FF"],
    },
}

# ------------------------------------------------------------------------------------------
# 2. Fontes (Google Fonts; carregadas no dashboard pelo .streamlit/config.toml)
# ------------------------------------------------------------------------------------------
FONTE_TITULO = "Bricolage Grotesque, Geist, sans-serif"
FONTE_TEXTO = "Geist, system-ui, sans-serif"
FONTE_NUMERO = "Geist Mono, ui-monospace, monospace"

# ------------------------------------------------------------------------------------------
# 3. Gêneros com cor fixa (os demais viram "Outros", em cinza de contexto)
#    Sugestão inicial: os 6 mais frequentes na base de retorno — o grupo pode trocar.
#    A ORDEM define a cor (1º = cobalto, 2º = terracota, ...).
# ------------------------------------------------------------------------------------------
GENEROS_COM_COR = ["Drama", "Comedy", "Thriller", "Action", "Adventure", "Romance"]
OUTROS = "Outros"

# Nomes em português para exibição (o dado continua em inglês)
GENEROS_PT = {
    "Action": "Ação", "Adventure": "Aventura", "Animation": "Animação", "Comedy": "Comédia",
    "Crime": "Crime", "Documentary": "Documentário", "Drama": "Drama", "Family": "Família",
    "Fantasy": "Fantasia", "Foreign": "Estrangeiro", "History": "História", "Horror": "Terror",
    "Music": "Música", "Mystery": "Mistério", "Romance": "Romance", "Science Fiction": "Ficção científica",
    "TV Movie": "Filme de TV", "Thriller": "Suspense", "War": "Guerra", "Western": "Faroeste",
}

# ------------------------------------------------------------------------------------------
# 4. Faixas de orçamento (degradê de um tom só) — as mesmas do gráfico explanatório da Pessoa 2:
#    até 20M (inclusive), acima de 20 até 50M, acima de 50 até 100M, acima de 100M
# ------------------------------------------------------------------------------------------
LIMITES_FAIXAS = [20e6, 50e6, 100e6]  # US$
ROTULOS_FAIXAS = ["Até US$ 20M", "US$ 20–50M", "US$ 50–100M", "Acima de US$ 100M"]


# ==========================================================================================
# Helpers de cor
# ==========================================================================================
def paleta(modo: str = "claro") -> dict:
    return CORES[modo]


def cores_destaque(categorias, destacar, modo: str = "claro") -> list[str]:
    """Lista de cores: destaque nas categorias em `destacar`, cinza de contexto no resto.

    Ex.: px.bar(...).update_traces(marker_color=cores_destaque(df["genero"], ["Horror"]))
    """
    destacar = {destacar} if isinstance(destacar, str) else set(destacar)
    c = CORES[modo]
    return [c["destaque"] if cat in destacar else c["contexto"] for cat in categorias]


def agrupar_generos(generos: pd.Series) -> pd.Series:
    """Mantém os GENEROS_COM_COR e troca o resto por 'Outros'."""
    return generos.where(generos.isin(GENEROS_COM_COR), OUTROS)


def mapa_cores_generos(modo: str = "claro", em_portugues: bool = False) -> dict:
    """{gênero: cor} fixo — usar em color_discrete_map para a cor nunca mudar com filtro."""
    c = CORES[modo]
    nome = (lambda g: GENEROS_PT.get(g, g)) if em_portugues else (lambda g: g)
    mapa = {nome(g): cor for g, cor in zip(GENEROS_COM_COR, c["generos"])}
    mapa[OUTROS] = c["contexto"]
    return mapa


def faixa_orcamento(budget: pd.Series) -> pd.Series:
    """Classifica o orçamento nas faixas (categoria ordenada, de barato a caro; limite superior incluso)."""
    bins = [0, *LIMITES_FAIXAS, np.inf]
    return pd.cut(budget, bins=bins, labels=ROTULOS_FAIXAS, right=True)


def mapa_cores_faixas(modo: str = "claro") -> dict:
    return dict(zip(ROTULOS_FAIXAS, CORES[modo]["faixas"]))


def cor_resultado(retorno, modo: str = "claro"):
    """Lucro (retorno ≥ 1) ou prejuízo (< 1). Aceita número ou Series."""
    c = CORES[modo]
    if isinstance(retorno, pd.Series):
        return np.where(retorno >= 1, c["lucro"], c["prejuizo"]).tolist()
    return c["lucro"] if retorno >= 1 else c["prejuizo"]


def linha_empate(fig: go.Figure, eixo: str = "y", modo: str | None = None, rotulo: str = "empate (1×)") -> go.Figure:
    """Linha tracejada em retorno = 1× com rótulo. Obrigatória sempre que houver lucro × prejuízo.

    Chamar DEPOIS de definir se o eixo é log (o rótulo em eixo log usa outra coordenada).
    """
    c = CORES[modo or modo_atual()]
    log = (fig.layout.yaxis.type if eixo == "y" else fig.layout.xaxis.type) == "log"
    pos = 0 if log else 1  # em eixo log, anotações usam log10(valor)
    linha = dict(type="line", line=dict(dash="dash", color=c["texto_suave"], width=1.2), layer="above")
    texto = dict(text=rotulo, showarrow=False, font=dict(family=FONTE_TEXTO, size=11, color=c["texto_suave"]))
    if eixo == "y":
        fig.add_shape(**linha, xref="paper", x0=0, x1=1, yref="y", y0=1, y1=1)
        fig.add_annotation(**texto, xref="paper", x=0, xanchor="left", yref="y", y=pos, yanchor="bottom")
    else:
        fig.add_shape(**linha, yref="paper", y0=0, y1=1, xref="x", x0=1, x1=1)
        fig.add_annotation(**texto, yref="paper", y=1, yanchor="top", xref="x", x=pos, xanchor="left", xshift=4)
    return fig


def eixo_log(fig: go.Figure, eixo: str = "y", sufixo: str = "×",
             valores=(0.01, 0.1, 0.5, 1, 2, 5, 10, 100, 1000, 10000)) -> go.Figure:
    """Eixo em escala log com marcas limpas e formato brasileiro (ex.: 0,5× · 1× · 2× · 10×).

    Para dinheiro: eixo_log(fig, "x", sufixo="", valores=[1e6, 1e7, 1e8, 1e9]) e rótulos US$.
    """
    def rot(v):
        if sufixo == "US$":
            return fmt_usd(v)
        return f"{_br(v, 0 if v >= 1 else (1 if v >= 0.1 else 2))}{sufixo}"

    kw = dict(type="log", tickvals=list(valores), ticktext=[rot(v) for v in valores], minor=dict(showgrid=False))
    (fig.update_yaxes if eixo == "y" else fig.update_xaxes)(**kw)
    return fig


# ==========================================================================================
# Formatação brasileira
# ==========================================================================================
def _br(numero: float, casas: int) -> str:
    return f"{numero:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def fmt_mult(x: float, casas: int = 2) -> str:
    """2.37 -> '2,37×'"""
    return f"{_br(x, casas)}×"


def fmt_usd(x: float, casas: int = 0) -> str:
    """150e6 -> 'US$ 150M' · 2.8e9 -> 'US$ 2,8B' · 45000 -> 'US$ 45mil'"""
    if abs(x) >= 1e9:
        return f"US$ {_br(x / 1e9, max(casas, 1))}B"
    if abs(x) >= 1e6:
        return f"US$ {_br(x / 1e6, casas)}M"
    if abs(x) >= 1e3:
        return f"US$ {_br(x / 1e3, casas)}mil"
    return f"US$ {_br(x, casas)}"


def fmt_pct(x: float, casas: int = 1, fracao: bool = True) -> str:
    """0.247 -> '24,7%' (fracao=True) · 24.7 -> '24,7%' (fracao=False)"""
    return f"{_br(x * 100 if fracao else x, casas)}%"


# ==========================================================================================
# Template plotly
# ==========================================================================================
def _template(modo: str) -> go.layout.Template:
    c = CORES[modo]
    eixo = dict(
        showgrid=True, gridcolor=c["linha"], gridwidth=1,
        showline=False, zeroline=False, ticks="",
        tickfont=dict(family=FONTE_NUMERO, size=11, color=c["texto_suave"]),
        title=dict(font=dict(family=FONTE_TEXTO, size=12, color=c["texto_suave"]), standoff=10),
        automargin=True,
    )
    return go.layout.Template(
        layout=dict(
            paper_bgcolor=c["cartao"],
            plot_bgcolor=c["cartao"],
            font=dict(family=FONTE_TEXTO, size=13, color=c["texto"]),
            colorway=c["generos"],
            separators=",.",  # decimal com vírgula, milhar com ponto
            title=dict(
                font=dict(family=FONTE_TITULO, size=22, color=c["texto"], weight=800),
                subtitle=dict(font=dict(family=FONTE_TEXTO, size=13, color=c["texto_suave"])),
                x=0, xanchor="left", xref="paper", y=0.97, yanchor="top", pad=dict(b=12),
            ),
            xaxis=eixo,
            yaxis=eixo,
            legend=dict(
                orientation="h", x=0, xanchor="left", y=1.02, yanchor="bottom",
                font=dict(family=FONTE_TEXTO, size=12, color=c["texto_suave"]),
                title=dict(font=dict(family=FONTE_TEXTO, size=12, color=c["texto_suave"])),
                bgcolor="rgba(0,0,0,0)",
            ),
            hoverlabel=dict(
                bgcolor=c["cartao"], bordercolor=c["linha"],
                font=dict(family=FONTE_NUMERO, size=12, color=c["texto"]),
            ),
            margin=dict(l=16, r=24, t=96, b=16),
            barcornerradius=4,
            bargap=0.28,
            coloraxis=dict(colorbar=dict(outlinewidth=0, tickfont=dict(family=FONTE_NUMERO, color=c["texto_suave"]))),
        ),
        data=dict(
            bar=[go.Bar(marker=dict(line=dict(width=0)))],
            scatter=[go.Scatter(marker=dict(size=7, line=dict(width=0)), line=dict(width=2.2))],
            box=[go.Box(line=dict(width=1.4), marker=dict(size=4))],
            histogram=[go.Histogram(marker=dict(line=dict(width=0)))],
        ),
    )


pio.templates["avd_claro"] = _template("claro")
pio.templates["avd_escuro"] = _template("escuro")
pio.templates.default = "avd_claro"


def usar_tema(modo: str | None = None) -> str:
    """Define o tema padrão do plotly (claro/escuro) para os PRÓXIMOS gráficos criados.

    O plotly.express escolhe as cores na hora em que o gráfico é criado, por isso o dashboard
    chama isto no topo de cada execução (o tema do usuário pode ter mudado). Retorna o modo.
    """
    modo = modo or modo_atual()
    pio.templates.default = f"avd_{modo}"
    return modo


def aplicar_estilo(fig: go.Figure, titulo: str | None = None, subtitulo: str | None = None,
                   modo: str | None = None, altura: int | None = None) -> go.Figure:
    """Aplica o tema do projeto a uma figura plotly.

    titulo: a CONCLUSÃO do gráfico.  subtitulo: o que está sendo medido.
    modo: "claro" | "escuro" (None = detecta o tema atual do Streamlit; fora dele usa "claro").
    """
    modo = modo or modo_atual()
    tem_titulo = bool(titulo or fig.layout.title.text)
    tem_legenda = fig.layout.showlegend is not False and sum(1 for t in fig.data if t.showlegend is not False) > 1
    topo = (96 if tem_titulo else 24) + (28 if tem_legenda else 0)
    # o px grava margem explícita no layout; sobrescreve para o template valer
    fig.update_layout(template=f"avd_{modo}", margin=dict(l=16, r=24, t=topo, b=16))
    if titulo is not None:
        fig.update_layout(title_text=titulo)
    if subtitulo is not None:
        fig.update_layout(title_subtitle_text=subtitulo)
    if altura is not None:
        fig.update_layout(height=altura)
    return fig


# ==========================================================================================
# Streamlit
# ==========================================================================================
def modo_atual() -> str:
    """'claro' ou 'escuro' conforme o tema que está na tela (fora do Streamlit: 'claro').

    Usa o ?tema= que o sincronizar_tema() grava na URL (é o valor mais confiável logo depois
    de uma troca de tema); se não houver, usa st.context.theme.
    """
    try:
        import streamlit as st

        tema = st.query_params.get("tema")
        if tema in ("claro", "escuro"):
            return tema
        return "escuro" if st.context.theme.type == "dark" else "claro"
    except Exception:
        return "claro"


def sincronizar_tema() -> None:
    """Recarrega a página sozinho quando o usuário troca claro/escuro (menu ⋮ → Settings ou o modo do sistema).

    Sem isso, gráficos e cards só pegariam as cores novas depois de um rerun manual (tecla R).
    Funciona com um script invisível que olha a cor de fundo da página: se ela não bate com o
    modo usado para desenhar, grava ?tema=<modo> na URL e pede um rerun.
    Chamar uma vez no topo do dashboard (o injetar_css já chama).
    """
    import streamlit as st
    import streamlit.components.v1 as components

    modo = modo_atual()
    components.html(
        f"""
        <script>
        const pai = window.parent;
        const desenhado = "{modo}";
        let ultimo = 0, tentativas = 0;
        function luminancia(cor) {{
            const m = (cor.match(/\\d+(\\.\\d+)?/g) || [255, 255, 255]).map(Number);
            return 0.299 * m[0] + 0.587 * m[1] + 0.114 * m[2];
        }}
        function verificar() {{
            const app = pai.document.querySelector(".stApp");
            if (!app) return;
            const naTela = luminancia(pai.getComputedStyle(app).backgroundColor) < 128 ? "escuro" : "claro";
            if (naTela === desenhado) {{ tentativas = 0; return; }}
            if (Date.now() - ultimo < 1500) return;
            ultimo = Date.now();
            tentativas += 1;
            const url = new URL(pai.location.href);
            url.searchParams.set("tema", naTela);
            if (tentativas === 1) {{
                // 1ª tentativa: rerun normal (mantém filtros); pode ser ignorado com a janela de Settings aberta
                pai.history.replaceState(null, "", url);
                pai.document.dispatchEvent(new KeyboardEvent("keydown", {{key: "r", code: "KeyR", keyCode: 82, which: 82, bubbles: true}}));
            }} else {{
                // não adiantou: recarrega a página inteira (sempre funciona)
                pai.location.replace(url.toString());
            }}
        }}
        setInterval(verificar, 400);
        </script>
        """,
        height=0,
    )



def injetar_css() -> None:
    """Ajustes finos que o config.toml não cobre (chamar uma vez no topo do dashboard)."""
    import streamlit as st

    c = CORES[modo_atual()]
    st.markdown(
        f"""
        <style>
        /* blocos invisíveis (CSS injetado e o script do sincronizar_tema) não ocupam espaço */
        [data-testid="stElementContainer"]:has(style),
        [data-testid="stElementContainer"]:has(iframe[height="0"]) {{ display: none; }}
        h1, h2, h3 {{ letter-spacing: -0.025em; }}
        [data-testid="stMetricValue"] {{ font-family: {FONTE_NUMERO}; letter-spacing: -0.02em; }}
        [data-testid="stMetric"] {{
            background: {c["cartao"]}; border: 1px solid {c["linha"]};
            border-radius: 0.75rem; padding: 1rem 1.25rem;
        }}
        [class*="st-key-destaque"] [data-testid="stMetric"] {{
            background: {c["destaque_fundo"]}; border-color: transparent;
        }}
        [class*="st-key-destaque"] [data-testid="stMetricValue"] {{ color: {c["destaque_texto"]}; }}
        [data-testid="stPlotlyChart"] {{
            background: {c["cartao"]}; border: 1px solid {c["linha"]};
            border-radius: 0.75rem; padding: 0.5rem; overflow: hidden;
        }}
        .nota-analitica {{
            color: {c["texto_suave"]}; font-size: 0.92rem; line-height: 1.55;
            border-left: 3px solid {c["linha"]}; padding: 0.25rem 0 0.25rem 0.9rem; margin: 0.25rem 0 1.5rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    sincronizar_tema()


def card(rotulo: str, valor: str, destaque: bool = False, ajuda: str | None = None, chave: str | None = None) -> None:
    """Card de número. destaque=True usa o fundo azul claro (só no número principal da página).

    `chave` só é necessária se dois cards na mesma página tiverem o mesmo rótulo.
    """
    import re

    import streamlit as st

    sufixo = chave or re.sub(r"\W+", "_", rotulo.lower()).strip("_")
    with st.container(key=f"{'destaque' if destaque else 'card'}_{sufixo}"):
        st.metric(rotulo, valor, help=ajuda)


def grafico(fig: go.Figure, **kw) -> None:
    """Mostra a figura no Streamlit com o tema do projeto (e não o tema padrão do Streamlit)."""
    import streamlit as st

    st.plotly_chart(fig, theme=None, width="stretch", **kw)


def nota(texto: str) -> None:
    """Nota analítica embaixo do gráfico, no estilo do projeto."""
    import re

    import streamlit as st

    texto = re.sub(r"`([^`]+)`", r"<code>\1</code>", texto)  # `código` do Markdown dentro do HTML
    st.markdown(f'<div class="nota-analitica">{texto}</div>', unsafe_allow_html=True)
