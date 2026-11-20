"""Dashboard do projeto (Streamlit) — Parte Específica (TMDB 5000) + Parte Comum (Wine Quality).

Rodar da raiz do projeto:  streamlit run src/parte_especifica/dashboard.py
Alternar claro/escuro: menu ⋮ (canto superior direito) → Settings → Theme.

Estrutura: uma função por página, registradas no st.navigation lá embaixo.
Gráficos vêm de graficos.py e notas de notas.py; aqui só fica a montagem das páginas.
Os gráficos usam a base INTEIRA (sem filtro) porque as notas e o storytelling descrevem a base inteira.
"""
# [IA] Claude Code – estrutura do dashboard (navegação, páginas, cache e montagem dos gráficos)
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from dados import DECISOES, RAIZ, carregar_filmes, tratar
from estilo import card, fmt_mult, fmt_pct, fmt_usd, _br, grafico, injetar_css, nota, usar_tema
from graficos import GRAFICOS
from notas import NOTAS, RESPOSTA_PERGUNTA, STORYTELLING_P3

PERGUNTA = "Filmes com orçamento maior têm retorno (receita ÷ orçamento) maior, ou isso varia por gênero?"
PUBLICO = "Gestor de estúdio ou produtora"
OUT_COMUM = RAIZ / "out" / "parte_comum"

st.set_page_config(page_title="Orçamento × retorno no cinema", page_icon="🎬", layout="wide")
modo = usar_tema()
injetar_css()


# ==========================================================================================
# Dados e figuras (com cache: a base e o bootstrap da Pessoa 3 só são calculados uma vez)
# ==========================================================================================
@st.cache_data(show_spinner="Carregando os filmes…")
def base_bruta() -> pd.DataFrame:
    return carregar_filmes()


@st.cache_data(show_spinner=False)
def base_tratada() -> pd.DataFrame:
    return tratar(base_bruta())


@st.cache_data(show_spinner="Montando o gráfico…")
def figura(id_grafico: str, modo: str):
    """Figura plotly de um gráfico do catálogo (graficos.GRAFICOS), já no tema claro/escuro."""
    g = next(g for g in GRAFICOS if g["id"] == id_grafico)
    return g["funcao"](base_tratada(), modo=modo)


def mostrar(id_grafico: str) -> None:
    """Gráfico + nota analítica do autor (ou aviso de pendência, se ainda não existir)."""
    if not any(g["id"] == id_grafico for g in GRAFICOS):
        pendente(f"Gráfico `{id_grafico}` ainda não foi feito.")
        return
    grafico(figura(id_grafico, modo))
    if id_grafico in NOTAS:
        nota(NOTAS[id_grafico])
    else:
        pendente("Nota analítica deste gráfico ainda não foi escrita.")


def pendente(texto: str) -> None:
    st.info(f"**Pendente.** {texto}", icon=":material/pending:")


def autoria(texto: str) -> None:
    st.caption(texto)


# ==========================================================================================
# Páginas — Parte Específica
# ==========================================================================================
def pagina_inicio() -> None:
    df = base_tratada()
    ret = df["retorno"].dropna()

    st.title("Orçamento maior dá retorno maior?")
    st.markdown(f"**Pergunta de pesquisa.** {PERGUNTA}  \n**Público-alvo.** {PUBLICO}.")
    st.caption("Base: TMDB 5000 Movie Dataset, após o tratamento descrito na página *Dados e tratamento*. "
               "Retorno = receita ÷ orçamento (não é lucro: não inclui marketing nem a parte dos cinemas).")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        card("Retorno mediano", fmt_mult(ret.median()), destaque=True,
             ajuda="Metade dos filmes com retorno calculável fica acima deste valor.")
    with c2:
        card("Filmes analisados", _br(len(ret), 0),
             ajuda=f"Filmes com retorno calculável, de {_br(len(df), 0)} na base; "
                   "os demais não têm orçamento e/ou receita válidos.")
    with c3:
        card("Retorno abaixo de 1×", fmt_pct((ret < 1).mean()), ajuda="Filmes cuja receita foi menor que o orçamento.")
    with c4:
        card("Orçamento mediano", fmt_usd(df.loc[ret.index, "budget"].median()),
             ajuda="Entre os filmes com retorno calculável.")

    st.subheader("Resposta à pergunta")
    with st.container(border=True):
        st.markdown(RESPOSTA_PERGUNTA.replace("$", "\\$"))  # "$" sozinho vira fórmula no Markdown

    st.subheader("Como ler este dashboard")
    st.markdown(
        "- **Exploração**: gráficos exploratórios, para ver como orçamento e retorno se distribuem em cada gênero.\n"
        "- **Explicação**: gráficos explanatórios e o storytelling, que respondem à pergunta.\n"
        "- **Dados e tratamento**: o que foi feito com faltantes e valores implausíveis antes da análise.\n"
        "- **Parte Comum**: agrupamento K-Means do Wine Quality.\n\n"
        "Passe o mouse sobre os gráficos para ver os valores; arraste para dar zoom e clique duas vezes para voltar."
    )


def pagina_exploracao() -> None:
    st.title("Exploração")
    st.caption("Gráficos exploratórios: como orçamento e retorno se distribuem dentro de cada gênero.")

    st.subheader("Pessoa 1")
    mostrar("p1_exploratorio")

    st.subheader("Pessoa 2")
    autoria("Dispersão de orçamento e retorno em quatro gêneros.")
    mostrar("p2_exploratorio")

    st.subheader("Pessoa 3")
    autoria("Concentração de filmes por região de orçamento e retorno em seis gêneros.")
    mostrar("p3_exploratorio")

    st.subheader("Pessoa 4")
    mostrar("p4_exploratorio")


def pagina_explicacao() -> None:
    st.title("Explicação")
    st.caption("Gráficos explanatórios: o que responde à pergunta de pesquisa.")

    st.subheader("Retorno por faixa de orçamento — Pessoa 2")
    mostrar("p2_explanatorio")

    st.subheader("Associação entre orçamento e retorno — Pessoa 3")
    mostrar("p3_explanatorio")

    st.subheader("Storytelling para decisão")
    with st.container(border=True):
        st.markdown(STORYTELLING_P3)


def pagina_dados() -> None:
    bruto, df = base_bruta(), base_tratada()
    st.title("Dados e tratamento")
    st.caption("Justificativas completas no eda.ipynb (seções 2 a 4). Aqui fica o resumo do que mudou.")

    c1, c2, c3 = st.columns(3)
    with c1:
        card("Filmes no arquivo original", _br(len(bruto), 0))
    with c2:
        card("Filmes após o tratamento", _br(len(df), 0))
    with c3:
        card("Com retorno calculável", _br(df["retorno"].notna().sum(), 0), destaque=True)

    st.subheader("Valores ausentes antes e depois")
    colunas = ["budget", "revenue", "runtime", "vote_average"]
    tabela = pd.DataFrame({
        "variável": colunas,
        "vazios no original": [bruto[c].isna().sum() for c in colunas],
        "zeros no original": [bruto[c].eq(0).sum() for c in colunas],
        "NaN depois do tratamento": [df[c].isna().sum() for c in colunas],
        "válidos depois do tratamento": [df[c].notna().sum() for c in colunas],
    })
    st.dataframe(tabela, hide_index=True, width="stretch")
    st.caption("Zeros em orçamento, receita e duração significam “não informado” e viram NaN; "
               "a nota vira NaN quando o filme não tem nenhum voto. Também viram NaN orçamentos e receitas "
               "positivos abaixo de US$ 1.000 (valores implausíveis).")

    st.subheader("Decisões do tratamento (`DECISOES` em dados.py)")
    st.dataframe(pd.DataFrame({"decisão": list(DECISOES), "valor": [str(v) for v in DECISOES.values()]}),
                 hide_index=True, width="stretch")

    st.subheader("Base tratada")
    colunas_base = ["title", "ano", "genres", "budget", "revenue", "retorno", "runtime", "vote_average"]
    st.dataframe(df[colunas_base], hide_index=True, width="stretch", height=360,
                 column_config={"budget": st.column_config.NumberColumn("budget (US$)", format="%,d"),
                                "revenue": st.column_config.NumberColumn("revenue (US$)", format="%,d"),
                                "retorno": st.column_config.NumberColumn("retorno (×)", format="%.2f"),
                                "ano": st.column_config.NumberColumn(format="%d")})


# ==========================================================================================
# Página — Parte Comum (Wine Quality). Usa as saídas já geradas em out/parte_comum/.
# ==========================================================================================
def _imagem(caminho: Path, legenda: str) -> None:
    if caminho.exists():
        st.image(str(caminho), caption=legenda, width="stretch")
    else:
        pendente(f"`{caminho.relative_to(RAIZ)}` não encontrado. Rode os scripts de src/parte_comum/.")


def pagina_parte_comum() -> None:
    st.title("Parte Comum — Wine Quality")
    st.caption("K-Means sobre o UCI Wine Quality (tinto + branco, 6.497 vinhos, notas 3 a 9). "
               "Discussão completa no relatorio.pdf.")

    arq = OUT_COMUM / "kmeans" / "resumo.json"
    if not arq.exists():
        pendente("out/parte_comum/kmeans/resumo.json não encontrado. Rode src/parte_comum/kmeans.py.")
        return
    resumo = json.loads(arq.read_text(encoding="utf-8"))
    p1, p2 = resumo["passada1"], resumo["passada2"]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        card("Acurácia (passada 2, k = 7)", fmt_pct(p2["acuracia_global"]), destaque=True)
    with c2:
        card("Baseline “sempre 6”", fmt_pct(p2["baseline_classe_majoritaria"]))
    with c3:
        card("ARI (passada 2)", _br(p2["ARI"], 3))
    with c4:
        card("Silhouette (passada 2)", _br(p2["silhouette"], 3))

    st.subheader("Passadas do K-Means")
    linhas = []
    for nome, p in [("Passada 1 (cotovelo)", p1), ("Passada 2 (k = nº de classes)", p2)]:
        linhas.append({"passada": nome, "k": p["k"], "acurácia": p["acuracia_global"],
                       "baseline": p["baseline_classe_majoritaria"], "ARI": p["ARI"], "NMI": p["NMI"],
                       "silhouette": p["silhouette"],
                       "classes nunca previstas": ", ".join(map(str, p["classes_nunca_previstas"]))})
    st.dataframe(pd.DataFrame(linhas), hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.3f")
                                for c in ["acurácia", "baseline", "ARI", "NMI", "silhouette"]})

    # figuras em abas, cada uma na largura toda (lado a lado ficavam pequenas demais para ler)
    figuras = {
        "Cotovelo": ("kmeans/passada1_cotovelo.png", "Escolha de k pela regra do cotovelo"),
        "PCA": ("kmeans/passada2_pca.png", "Clusters da passada 2 projetados nos 2 primeiros componentes"),
        "Classe × cluster": ("kmeans/passada2_matriz_retangular.png", "Classe real × cluster (passada 2)"),
        "Notas vizinhas": ("granularidade/efeito_vizinhas_heatmap.png", "Diferença entre notas vizinhas (d de Cohen)"),
    }
    for aba, (arquivo, legenda) in zip(st.tabs(list(figuras)), figuras.values()):
        with aba:
            _imagem(OUT_COMUM / arquivo, legenda)


# ==========================================================================================
# Navegação
# ==========================================================================================
paginas = {
    "Parte Específica — TMDB 5000": [
        st.Page(pagina_inicio, title="Início", icon=":material/home:", default=True),
        st.Page(pagina_exploracao, title="Exploração", icon=":material/travel_explore:", url_path="exploracao"),
        st.Page(pagina_explicacao, title="Explicação", icon=":material/insights:", url_path="explicacao"),
        st.Page(pagina_dados, title="Dados e tratamento", icon=":material/table_chart:", url_path="dados"),
    ],
    "Parte Comum": [
        st.Page(pagina_parte_comum, title="Wine Quality", icon=":material/wine_bar:", url_path="parte-comum"),
    ],
}
st.navigation(paginas).run()
