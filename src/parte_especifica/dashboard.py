"""Dashboard do projeto (Streamlit) — Parte Específica (TMDB 5000) + Parte Comum (Wine Quality).

Rodar da raiz do projeto:  streamlit run src/parte_especifica/dashboard.py
Alternar claro/escuro: menu ⋮ (canto superior direito) → Settings → Theme (o rodapé da barra lateral lembra isso).

Estrutura: uma função por página, registradas no st.navigation lá embaixo.
Gráficos vêm de graficos.py e notas de notas.py; aqui só fica a montagem das páginas.
Filtros: barra lateral nas páginas Exploração, Explicação e Dados (barra_filtros). Sem filtro, os gráficos
usam a base INTEIRA, que é o que as notas e o storytelling descrevem; com filtro ativo, a página avisa isso.
"""
# [IA] Claude Code – estrutura do dashboard (navegação, páginas, cache e montagem dos gráficos)
# [IA] Claude Code – filtros na barra lateral (gênero, orçamento, período, mínimo de filmes por gênero)
# [IA] Claude Code – cards de KPI que reagem ao recorte (kpis_recorte)
# [IA] Claude Code – página Década a década (animação, troca de métrica e de escala)
# [IA] Claude Code – Início com a resposta primeiro e navegação por cards; títulos-achado; um gráfico por vez na Exploração
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from dados import DECISOES, RAIZ, carregar_filmes, tratar
from estilo import GENEROS_PT, card, fmt_mult, fmt_pct, fmt_usd, _br, grafico, injetar_css, nota, usar_tema
from graficos import ESCALAS, GRAFICOS, METRICAS, MIN_FILMES_DECADA, MIN_FILMES_GENERO, SemDados, animacao_decadas
from notas import DISCUSSAO_PARTE_COMUM, NOTAS, RESPOSTA_CURTA, RESPOSTA_PERGUNTA, STORYTELLING_P3, TITULOS_ACHADO

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


# ==========================================================================================
# Filtros (barra lateral)
# ==========================================================================================
# marcas do filtro de orçamento, em US$ milhões (inclui os limites das faixas da Pessoa 2: 20, 50 e 100)
MARCAS_ORCAMENTO = [0, 1, 5, 10, 20, 50, 100, 200, 400]
CHAVES_FILTRO = ["f_generos", "f_orcamento", "f_anos", "f_min_filmes"]


# O Streamlit apaga o estado de um widget quando a página não o desenha (Início, Wine Quality). Para os filtros
# sobreviverem à troca de página, cada widget guarda o valor numa chave "sombra" (_chave) no on_change e é
# recriado a partir dela. (Reatribuir st.session_state[chave] não serve: exige tirar o value= dos sliders,
# e sem value= em tupla o slider deixa de ser de intervalo.)
def _salvo(chave: str, padrao):
    return st.session_state.get("_" + chave, padrao)


def _salvar(chave: str) -> None:
    st.session_state["_" + chave] = st.session_state[chave]


@dataclass(frozen=True)
class Filtros:
    generos: tuple[str, ...] | None = None      # None = todos os gêneros
    orcamento: tuple[float, float] | None = None  # US$; None = sem filtro (mantém filmes sem orçamento)
    anos: tuple[int, int] | None = None           # None = sem filtro
    min_filmes: int = MIN_FILMES_GENERO

    @property
    def ativo(self) -> bool:
        return self != Filtros()


@st.cache_data(show_spinner=False)
def opcoes_filtro() -> tuple[list[str], int, int]:
    df = base_tratada()
    presentes = set(df["genres"].explode().dropna())
    generos = sorted((g for g in presentes if g in GENEROS_PT), key=lambda g: GENEROS_PT[g])
    return generos, int(df["ano"].min()), int(df["ano"].max())


def _rotulo_orcamento(v: float) -> str:
    if v == MARCAS_ORCAMENTO[-1]:
        return "sem limite"
    return "US$ 0" if v == 0 else fmt_usd(v * 1e6)


def _limpar_filtros() -> None:
    for k in CHAVES_FILTRO:
        st.session_state.pop(k, None)
        st.session_state.pop("_" + k, None)


def barra_filtros() -> Filtros:
    """Desenha os filtros na barra lateral e devolve o recorte escolhido."""
    generos, ano_min, ano_max = opcoes_filtro()
    with st.sidebar:
        st.header("Filtros", divider="gray")
        sel = st.multiselect("Gêneros", generos, key="f_generos", default=_salvo("f_generos", []),
                             on_change=_salvar, args=("f_generos",), format_func=lambda g: GENEROS_PT[g],
                             placeholder="Todos os gêneros",
                             help="Cada gráfico mostra só os gêneros escolhidos pelo autor que também estiverem aqui.")
        orc = st.select_slider("Orçamento", options=MARCAS_ORCAMENTO, key="f_orcamento",
                               value=_salvo("f_orcamento", (0, MARCAS_ORCAMENTO[-1])),
                               on_change=_salvar, args=("f_orcamento",), format_func=_rotulo_orcamento,
                               help="Com este filtro ativo, filmes sem orçamento informado ficam de fora. "
                                    "Não se aplica ao gráfico de cobertura (Pessoa 1).")
        anos = st.slider("Ano de lançamento", ano_min, ano_max, key="f_anos", value=_salvo("f_anos", (ano_min, ano_max)),
                         on_change=_salvar, args=("f_anos",))
        min_filmes = st.slider("Mínimo de filmes por gênero", 10, 100, step=5, key="f_min_filmes",
                               value=_salvo("f_min_filmes", MIN_FILMES_GENERO),
                               on_change=_salvar, args=("f_min_filmes",),
                               help="Gêneros com menos filmes que isso no recorte saem dos gráficos. "
                                    f"O padrão ({MIN_FILMES_GENERO}) é o corte usado na análise.")
        f = Filtros(
            generos=tuple(sel) or None,
            orcamento=None if orc == (0, MARCAS_ORCAMENTO[-1]) else
            (orc[0] * 1e6, np.inf if orc[1] == MARCAS_ORCAMENTO[-1] else orc[1] * 1e6),
            anos=None if anos == (ano_min, ano_max) else anos,
            min_filmes=min_filmes,
        )
        df = filmes_do_recorte(f)
        st.caption(f"**{_br(len(df), 0)}** de {_br(len(base_tratada()), 0)} filmes no recorte · "
                   f"{_br(df['retorno'].notna().sum(), 0)} com retorno calculável")
        st.button("Limpar filtros", icon=":material/filter_alt_off:", on_click=_limpar_filtros,
                  disabled=not f.ativo, width="stretch")
    return f


@st.cache_data(show_spinner=False)
def base_filtrada(orcamento: tuple[float, float] | None, anos: tuple[int, int] | None) -> pd.DataFrame:
    """Base tratada com os filtros de linhas (orçamento e ano); gênero e mínimo de filmes ficam com cada gráfico."""
    df = base_tratada()
    if anos is not None:
        df = df.loc[df["ano"].between(*anos)]
    if orcamento is not None:
        df = df.loc[df["budget"].between(*orcamento)]
    return df


def filmes_do_recorte(f: Filtros) -> pd.DataFrame:
    """Filmes do recorte, com gênero contando como "tem algum dos gêneros escolhidos" (para contagem e tabela)."""
    df = base_filtrada(f.orcamento, f.anos)
    if f.generos is not None:
        df = df.loc[df["genres"].map(lambda gs: bool(set(gs) & set(f.generos)))]
    return df


def aviso_recorte(f: Filtros) -> None:
    if f.ativo:
        st.info("**Recorte ativo.** Os números dos gráficos refletem os filtros da barra lateral, mas as notas "
                "dos autores e o storytelling foram escritos para a base inteira. Use *Limpar filtros* para voltar.",
                icon=":material/filter_alt:")


def _resumo(df: pd.DataFrame) -> dict:
    """Os mesmos 4 números dos cards do Início, para qualquer recorte."""
    ret = df["retorno"].dropna()
    return {"n": len(ret), "mediana": ret.median(), "abaixo": (ret < 1).mean() if len(ret) else np.nan,
            "orcamento": df.loc[ret.index, "budget"].median()}


def _sinal(texto: str, valor: float) -> str:
    return texto if valor < 0 else f"+{texto}"  # o "-" já vem do número; o Streamlit usa o sinal para a seta


def kpis_recorte(f: Filtros) -> None:
    """Cards do recorte atual; com filtro ativo, cada card mostra a diferença em relação à base inteira."""
    base, rec = _resumo(base_tratada()), _resumo(filmes_do_recorte(f))
    vazio = rec["n"] == 0
    desc = "vs. base inteira"
    st.caption("**Recorte atual** (filmes com algum dos gêneros escolhidos)" if f.ativo else "**Base inteira**")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        d = rec["mediana"] - base["mediana"]
        card("Retorno mediano", "—" if vazio else fmt_mult(rec["mediana"]), destaque=True, chave="kpi_mediana",
             delta=None if vazio or not f.ativo else _sinal(fmt_mult(d), d), descricao_delta=desc,
             ajuda=f"Metade dos filmes do recorte fica acima deste valor. Base inteira: {fmt_mult(base['mediana'])}.")
    with c2:
        card("Filmes com retorno", _br(rec["n"], 0), chave="kpi_filmes",
             delta=f"{fmt_pct(rec['n'] / base['n'], 0)} da base" if f.ativo else None, cor_delta="off", seta="off",
             ajuda="Filmes do recorte com orçamento e receita válidos (retorno calculável). "
                   "Medianas com poucos filmes oscilam muito.")
    with c3:
        d = (rec["abaixo"] - base["abaixo"]) * 100
        card("Retorno abaixo de 1×", "—" if vazio else fmt_pct(rec["abaixo"]), chave="kpi_abaixo",
             delta=None if vazio or not f.ativo else _sinal(f"{_br(d, 1)} p.p.", d), cor_delta="inverse",
             descricao_delta=desc,
             ajuda=f"Filmes cuja receita foi menor que o orçamento. Base inteira: {fmt_pct(base['abaixo'])}.")
    with c4:
        d = rec["orcamento"] - base["orcamento"]
        card("Orçamento mediano", "—" if vazio else fmt_usd(rec["orcamento"]), chave="kpi_orcamento",
             delta=None if vazio or not f.ativo else ("-" if d < 0 else "+") + fmt_usd(abs(d), 1 if 1e6 <= abs(d) < 1e7 else 0),
             cor_delta="off", descricao_delta=desc,
             ajuda=f"Entre os filmes do recorte com retorno calculável. Base inteira: {fmt_usd(base['orcamento'])}.")


@st.cache_data(show_spinner="Montando o gráfico…")
def figura(id_grafico: str, modo: str, f: Filtros = Filtros()):
    """Figura plotly de um gráfico do catálogo (graficos.GRAFICOS), já no tema claro/escuro e no recorte `f`."""
    g = next(g for g in GRAFICOS if g["id"] == id_grafico)
    # o gráfico de cobertura (Pessoa 1) mede quem tem orçamento informado: filtrar orçamento o deixaria trivial
    df = base_filtrada(None if id_grafico == "p1_exploratorio" else f.orcamento, f.anos)
    extra = {"titulo": g["titulo_neutro"]} if f.ativo and "titulo_neutro" in g else {}
    return g["funcao"](df, modo=modo, generos=f.generos, min_filmes=f.min_filmes, **extra)


def mostrar(id_grafico: str, f: Filtros = Filtros(), autor_pendente: str = "") -> None:
    """Título + autoria (legenda) + gráfico + nota analítica do autor (ou avisos de pendência).

    Título: o achado escrito pelo autor (notas.TITULOS_ACHADO) ou, enquanto não houver, o descritivo do catálogo.
    """
    g = next((g for g in GRAFICOS if g["id"] == id_grafico), None)
    if g is None:
        st.subheader(TITULOS_ACHADO.get(id_grafico) or "Gráfico exploratório")
        autoria(autor_pendente)
        pendente(f"Gráfico `{id_grafico}` ainda não foi feito.")
        return
    st.subheader(TITULOS_ACHADO.get(id_grafico) or g["secao"])
    autoria(f"{g['tipo'].capitalize()} · {g['autor']}")
    try:
        grafico(figura(id_grafico, modo, f))
    except SemDados:
        st.warning(f"Nenhum gênero deste gráfico tem ao menos {f.min_filmes} filmes no recorte. "
                   "Afrouxe os filtros na barra lateral.", icon=":material/filter_alt_off:")
        return
    if id_grafico == "p1_exploratorio" and f.orcamento is not None:
        st.caption("O filtro de orçamento não se aplica a este gráfico: ele mede justamente quantos filmes "
                   "têm orçamento e receita informados.")
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

    # 1. pergunta
    st.title("Orçamento maior dá retorno maior?")
    st.markdown(f"**Pergunta de pesquisa.** {PERGUNTA}  \n**Público-alvo.** {PUBLICO}.")

    # 2. resposta em destaque (versão curta; a completa fica no fim da Explicação)
    with st.container(border=True, key="resposta"):
        st.markdown("**Resposta**")
        if RESPOSTA_CURTA:
            st.markdown(RESPOSTA_CURTA.replace("$", "\\$"))  # "$" sozinho vira fórmula no Markdown
            st.page_link(PAGINAS["explicacao"], label="Ver a resposta completa e os gráficos que a sustentam",
                         icon=":material/arrow_forward:")
        else:
            st.markdown(RESPOSTA_PERGUNTA.replace("$", "\\$"))
    if not RESPOSTA_CURTA:
        pendente("Versão curta da resposta (2 ou 3 linhas) para este destaque: `RESPOSTA_CURTA` em notas.py. "
                 "Quando ela existir, a resposta completa passa a aparecer só no fim da página Explicação.")

    # 3. os números que sustentam a resposta
    st.subheader("Os números da base")
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
    st.caption("Base: TMDB 5000 Movie Dataset, após o tratamento descrito em *Dados e tratamento*. "
               "Retorno = receita ÷ orçamento (não é lucro: não inclui marketing nem a parte dos cinemas).")

    # 4. para onde ir (a explicação de cada página já é o link para ela)
    st.subheader("Para onde ir")
    destinos = [
        ("explicacao", "Os gráficos que respondem à pergunta, o storytelling para decisão e a resposta completa."),
        ("exploracao", "Como orçamento e retorno se distribuem dentro de cada gênero, com filtros."),
        ("decadas", "Animação década a década, com troca de métrica (retorno, lucro, receita) e de escala."),
        ("dados", "O que foi feito com faltantes e valores implausíveis antes da análise."),
        ("parte-comum", "Parte Comum: agrupamento K-Means do Wine Quality."),
    ]
    for linha in (destinos[:3], destinos[3:]):
        for col, (chave, texto) in zip(st.columns(3), linha):
            with col, st.container(border=True, key=f"ir_{chave.replace('-', '_')}", height="stretch"):
                st.page_link(PAGINAS[chave])
                st.caption(texto)
    st.caption("Nos gráficos: passe o mouse para ver os valores; arraste para dar zoom e clique duas vezes para voltar.")


def pagina_exploracao() -> None:
    f = barra_filtros()
    st.title("Exploração")
    st.caption("Gráficos exploratórios: como orçamento e retorno se distribuem dentro de cada gênero.")
    aviso_recorte(f)
    kpis_recorte(f)

    # um gráfico por vez: a página fica curta e só monta o gráfico escolhido
    opcoes = {g["id"]: g["curto"] for g in GRAFICOS if g["tipo"] == "exploratório"}
    escolhido = st.segmented_control("Gráfico", list(opcoes), format_func=opcoes.get, key="v_exploratorio",
                                     required=True, default=_salvo("v_exploratorio", next(iter(opcoes))),
                                     on_change=_salvar, args=("v_exploratorio",), label_visibility="collapsed")
    mostrar(escolhido, f, autor_pendente="Exploratório · Pessoa 2")


def pagina_explicacao() -> None:
    f = barra_filtros()
    st.title("Explicação")
    st.caption("Gráficos explanatórios: o que responde à pergunta de pesquisa.")
    aviso_recorte(f)
    kpis_recorte(f)

    mostrar("p2_explanatorio", f)
    mostrar("p3_explanatorio", f)

    st.subheader("Storytelling para decisão")
    with st.container(border=True):
        st.markdown(STORYTELLING_P3)

    st.subheader("Resposta completa à pergunta", anchor="resposta")
    st.caption(f"{PERGUNTA} Calculada sobre a base inteira, sem os filtros da barra lateral.")
    with st.container(border=True):
        st.markdown(RESPOSTA_PERGUNTA.replace("$", "\\$"))  # "$" sozinho vira fórmula no Markdown


@st.cache_data(show_spinner="Montando a animação…")
def figura_decadas(modo: str, f: Filtros, metrica: str, escala: str):
    return animacao_decadas(base_filtrada(f.orcamento, f.anos), modo=modo, generos=f.generos,
                            metrica=metrica, escala=escala)


def _metrica_mudou() -> None:
    _salvar("v_metrica")
    # lucro só existe em escala linear: o controle de escala é recriado mostrando isso
    if st.session_state["v_metrica"] == "lucro":
        st.session_state["_v_escala"] = "linear"
        st.session_state.pop("v_escala", None)


def pagina_decadas() -> None:
    f = barra_filtros()
    st.title("Década a década")
    st.caption("Como orçamento e retorno mudaram ao longo do tempo. Aperte **▶ Animar** ou arraste o controle de "
               "décadas; passe o mouse sobre um ponto para ver o filme. Respeita os filtros de gênero, orçamento e ano "
               f"da barra lateral (o mínimo de filmes por gênero não se aplica aqui; décadas com menos de "
               f"{MIN_FILMES_DECADA} filmes ficam de fora).")
    c1, c2, _ = st.columns([3, 2, 3])
    with c1:
        metrica = st.segmented_control("Eixo vertical", list(METRICAS), key="v_metrica", required=True,
                                       default=_salvo("v_metrica", "retorno"),
                                       on_change=_metrica_mudou,
                                       format_func=lambda k: {"retorno": "Retorno (×)", "lucro": "Lucro bruto (US$)",
                                                              "receita": "Receita (US$)"}[k],
                                       help="Lucro bruto = receita − orçamento (não inclui marketing nem a parte "
                                            "dos cinemas). Os mesmos filmes entram nas três métricas.")
    with c2:
        escala = st.segmented_control("Escala dos eixos", list(ESCALAS), key="v_escala", required=True,
                                      default=_salvo("v_escala", "log"), on_change=_salvar, args=("v_escala",),
                                      format_func=ESCALAS.get, disabled=metrica == "lucro",
                                      help="Logarítmica: cada marca vale 10 vezes a anterior, bom para valores que "
                                           "variam muito. Linear: distâncias iguais para diferenças iguais. "
                                           "Lucro pode ser negativo, então usa sempre a linear.")
    if metrica == "lucro":
        st.caption("Lucro pode ser negativo e não existe log de número negativo: este eixo usa escala linear.")
    try:
        grafico(figura_decadas(modo, f, metrica, escala), config={"displayModeBar": False})
    except SemDados:
        st.warning(f"O recorte deixou menos de duas décadas com ao menos {MIN_FILMES_DECADA} filmes. "
                   "Afrouxe os filtros na barra lateral.", icon=":material/filter_alt_off:")
    st.caption("Orçamento e receita estão em dólares da época (o TMDB não corrige pela inflação), então parte da "
               "subida do orçamento ao longo das décadas é inflação.")


def pagina_dados() -> None:
    f = barra_filtros()
    bruto, df = base_bruta(), base_tratada()
    st.title("Dados e tratamento")
    st.caption("Justificativas completas no eda.ipynb (seções 2 a 4). Aqui fica o resumo do que mudou.")

    c1, c2, c3 = st.columns(3)
    with c1:
        card("Filmes no arquivo original", _br(len(bruto), 0))
    with c2:
        removidos = len(bruto) - len(df)
        card("Filmes após o tratamento", _br(len(df), 0), delta=f"-{_br(removidos, 0)} "
             f"{'removido' if removidos == 1 else 'removidos'}", cor_delta="off",
             ajuda="Linhas removidas pelo tratamento (ver DECISOES abaixo). Os demais problemas viraram NaN, "
                   "sem remover o filme.")
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
    recorte = filmes_do_recorte(f)
    if f.ativo:
        st.caption(f"Recorte da barra lateral: {_br(len(recorte), 0)} de {_br(len(df), 0)} filmes "
                   "(com algum dos gêneros escolhidos). O mínimo de filmes por gênero só vale para os gráficos.")
    colunas_base = ["title", "ano", "genres", "budget", "revenue", "retorno", "runtime", "vote_average"]
    st.dataframe(recorte[colunas_base], hide_index=True, width="stretch", height=360,
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

    st.subheader("Discussão")
    nota(DISCUSSAO_PARTE_COMUM)
    st.caption("Discussão crítica completa na seção 6 do relatório (botão no fim da página).")

    # [IA] Claude Code – comparação entre as passadas (uma por vez ou lado a lado) e download do relatório
    st.subheader("Comparar as passadas")
    passadas = {"p1": f"Passada 1 (k = {p1['k']}, cotovelo)", "p2": f"Passada 2 (k = {p2['k']}, nº de classes)",
                "ambas": "Lado a lado"}
    escolha = st.segmented_control("Passada", list(passadas), format_func=passadas.get, key="v_passada",
                                   required=True, default=_salvo("v_passada", "ambas"),
                                   on_change=_salvar, args=("v_passada",), label_visibility="collapsed")
    st.caption("“Lado a lado” facilita a comparação; para ler os detalhes, veja uma passada por vez.")

    # figuras que existem para as duas passadas: (arquivo com {p} = 1 ou 2, legenda)
    por_passada = {
        "PCA": ("kmeans/passada{p}_pca.png", "Clusters projetados nos 2 primeiros componentes"),
        "Classe × cluster": ("kmeans/passada{p}_matriz_retangular.png", "Classe real × cluster"),
        "Matriz de confusão": ("kmeans/passada{p}_matriz_confusao.png", "Classe real × classe mapeada (maioria do cluster)"),
        "Perfil dos centroides": ("kmeans/passada{p}_perfil_centroides.png", "Perfil químico de cada cluster (z-score)"),
    }
    unicas = {
        "Cotovelo": ("kmeans/passada1_cotovelo.png", "Escolha de k pela regra do cotovelo (passada 1)"),
        "Notas vizinhas": ("granularidade/efeito_vizinhas_heatmap.png", "Diferença entre notas vizinhas (d de Cohen)"),
    }
    abas = st.tabs(list(por_passada) + list(unicas))
    for aba, (arquivo, legenda) in zip(abas, por_passada.values()):
        with aba:
            if escolha == "ambas":
                for col, n in zip(st.columns(2), (1, 2)):
                    with col:
                        _imagem(OUT_COMUM / arquivo.format(p=n), f"{legenda} · passada {n}")
            else:
                n = 1 if escolha == "p1" else 2
                _imagem(OUT_COMUM / arquivo.format(p=n), f"{legenda} · passada {n}")
    for aba, (arquivo, legenda) in zip(abas[len(por_passada):], unicas.values()):
        with aba:
            _imagem(OUT_COMUM / arquivo, legenda)

    st.subheader("Relatório completo")
    botao_relatorio()


def botao_relatorio() -> None:
    """Botão para baixar o relatorio.pdf (Parte Comum + Parte Específica) direto do dashboard."""
    pdf = RAIZ / "relatorio.pdf"
    if pdf.exists():
        st.download_button("Baixar relatório (PDF)", pdf.read_bytes(), file_name="relatorio.pdf",
                           mime="application/pdf", icon=":material/download:")
    else:
        pendente("relatorio.pdf não encontrado. Rode src/parte_comum/relatorio.py.")


# ==========================================================================================
# Navegação
# ==========================================================================================
PAGINAS = {  # por url_path; o Início usa estes objetos nos st.page_link
    "inicio": st.Page(pagina_inicio, title="Início", icon=":material/home:", default=True),
    "exploracao": st.Page(pagina_exploracao, title="Exploração", icon=":material/travel_explore:", url_path="exploracao"),
    "explicacao": st.Page(pagina_explicacao, title="Explicação", icon=":material/insights:", url_path="explicacao"),
    "decadas": st.Page(pagina_decadas, title="Década a década", icon=":material/animation:", url_path="decadas"),
    "dados": st.Page(pagina_dados, title="Dados e tratamento", icon=":material/table_chart:", url_path="dados"),
    "parte-comum": st.Page(pagina_parte_comum, title="Wine Quality", icon=":material/wine_bar:", url_path="parte-comum"),
}
navegacao = st.navigation({
    "Parte Específica — TMDB 5000": [PAGINAS[k] for k in ["inicio", "exploracao", "explicacao", "decadas", "dados"]],
    "Parte Comum": [PAGINAS["parte-comum"]],
})
navegacao.run()

# rodapé da barra lateral (depois da página, para ficar abaixo dos filtros). O Streamlit não deixa trocar o
# tema pelo Python (só pelo menu), então em vez de um botão que não funcionaria fica o caminho até ele.
with st.sidebar:
    st.divider()
    botao_relatorio()
    st.caption(f":material/{'dark_mode' if modo == 'escuro' else 'light_mode'}: **Tema {modo}.** "
               "Para trocar: menu **⋮** no canto superior direito → *Settings* → *Theme*.")
