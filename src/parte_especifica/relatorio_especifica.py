"""Seção da Parte Específica (TMDB 5000) no relatorio.pdf.

Não roda sozinho: src/parte_comum/relatorio.py chama montar(...) e gera o PDF inteiro.
    python src/parte_comum/relatorio.py

O que entra aqui:
- textos que o grupo já escreveu, sem alteração: NOTAS, RESPOSTA_PERGUNTA, STORYTELLING_P3 (notas.py) e DECISOES (dados.py);
- números calculados direto da base (dimensões, faltantes antes e depois);
- as figuras dos autores, extraídas das saídas do eda.ipynb (por isso o notebook precisa estar executado);
- marcações [PREENCHER — ...] em destaque onde ainda falta texto do grupo. Para preencher, troque a
  string do PREENCHER correspondente em TEXTOS (abaixo) pelo texto; o destaque some sozinho.
"""

# [IA] Claude Code – estrutura da seção da Parte Específica no relatório (texto analítico é do grupo)
from __future__ import annotations

import base64
import html
import json
import re
from pathlib import Path

import pandas as pd

from dados import DECISOES, RAIZ, carregar_filmes, tratar
from notas import NOTAS, RESPOSTA_CURTA, RESPOSTA_PERGUNTA, STORYTELLING_P3

PASTA = Path(__file__).resolve().parent
FIGURAS = RAIZ / "out" / "parte_especifica" / "figuras"
PERGUNTA = "Filmes com orçamento maior têm retorno (receita ÷ orçamento) maior, ou isso varia por gênero?"
PUBLICO = "gestor de estúdio ou produtora"

# ------------------------------------------------------------------------------------------
# Textos do grupo que ainda faltam. Enquanto começarem com "PREENCHER", saem em destaque no PDF.
# ------------------------------------------------------------------------------------------
TEXTOS = {
    # Pessoa 1
    "motivacao": "Orçamento é a principal decisão que o gestor controla. Antes de aprovar um filme, ele precisa definir "
                 "quanto colocar e em que tipo de projeto. A pergunta testa a crença mais comum da indústria, a de que "
                 "“quem gasta mais ganha mais”, e a resposta mexe em três decisões: quanto investir, quanto risco correr "
                 "e em que gênero apostar.",
    # Pessoa 1 — seção 7.11; {TABELA_RISCO} marca onde entra a tabela (calculada da base em tabela_risco())
    "implicacoes": """**Quanto investir.** O retorno mediano não acompanha o orçamento: é de 2,9 vezes nos filmes de até US$ 20 milhões e de cerca de 2 vezes nas faixas de US$ 20 a 100 milhões.

**Quanto risco correr.** A tabela abaixo mostra, para cada faixa de orçamento, o retorno mediano, a parcela de filmes que não recupera o orçamento e a parcela que fatura mais de 5 vezes o que custou.

{TABELA_RISCO}

O filme barato é uma aposta de alto risco e alto potencial: um em cada quatro não recupera o custo, mas um em cada três fatura mais de 5 vezes o orçamento. O blockbuster é mais previsível: só 8% não recuperam o custo. Em vez de “qual dá mais retorno”, a resposta passa a ser “qual perfil de risco o estúdio quer”. Um estúdio pequeno, que não aguenta perder US$ 150 milhões, pode preferir vários filmes baratos. Um estúdio grande pode usar blockbusters para ter receita estável.

**Em que gênero apostar.** A análise da Pessoa 3 mostra que essa relação muda com o gênero. Em Comédia, Drama, Romance e Suspense, orçamentos maiores vêm associados a retorno menor. Em Ação e Aventura, a associação fica perto de zero. É uma associação, e não uma relação de causa: os dados não mostram que aumentar o orçamento reduz o retorno, mas indicam em quais gêneros orçamentos altos costumam aparecer junto com retornos menores.

**Ressalva.** Retorno aqui é receita bruta ÷ orçamento, e não lucro. O estúdio não fica com toda a bilheteria, e o marketing não está incluído no orçamento. Por isso, na vida real, um filme precisa faturar bem mais que 1 vez o orçamento para dar lucro. Os números servem para comparar faixas e gêneros, não para calcular lucro.""",
    "estrutural": 'A base tem 4.803 filmes e 20 colunas, divididas em quantitativas (como `budget`, `revenue` e `runtime`), qualitativas (como `genres`, `original_language` e `status`), um identificador (`id`) e textos livres que não entram na análise. Cinco colunas vinham como texto JSON (`genres`, `keywords`, `production_companies`, `production_countries` e `spoken_languages`) e foram convertidas em listas. Isso revelou um ponto central para a pergunta: os gêneros não são exclusivos, já que cada filme tem em média 2,5 gêneros. Por isso, na análise por gênero, o mesmo filme é contado em todos os seus gêneros, e os grupos se sobrepõem. Não há linhas nem ids duplicados. Os 3 títulos repetidos (*Batman*, *The Host* e *Out of the Blue*) correspondem a filmes diferentes, com datas distintas, e foram mantidos. Os 8 filmes que ainda não tinham sido lançados ("Rumored" ou "Post Production") ficam fora do cálculo do retorno, porque não podem ter receita real. Por fim, não encontramos anos fora do intervalo esperado, notas fora da escala de 0 a 10 nem valores negativos, o que indica que os problemas da base estão nos dados faltantes e nos valores implausíveis, tratados nas seções 7.3 e 7.4, e não em erros de formato.',
    "faltantes": 'O principal problema de ausência está em `budget` e `revenue`, em que o zero significa "não informado": apenas 67% dos filmes têm os dois valores válidos. A ausência não é aleatória. Os filmes sem dado financeiro têm muito menos votos (mediana de 38, contra 471 nos filmes com dado) e são mais frequentemente de idioma original diferente do inglês (57% de ausência, contra 31% nos filmes em inglês). Como a ausência é explicada por variáveis observadas, classificamos o mecanismo como MAR. Também não descartamos um componente MNAR: filmes que fracassaram podem deixar de divulgar a bilheteria justamente porque ela foi baixa. Decidimos não imputar esses valores, porque estimar orçamento ou receita seria inventar a própria variável que a pergunta de pesquisa quer analisar. Os filmes sem dado financeiro continuam na base, mas ficam fora do cálculo do retorno. A consequência é que a análise trabalha com uma amostra enviesada, em que faltam principalmente os filmes pequenos, pouco conhecidos e não falados em inglês. O gráfico de cobertura financeira por gênero (Figura 19, seção 7.8) mostra o mesmo viés por gênero: Aventura e Ação têm cerca de 80% de cobertura, enquanto Drama tem 62% e Documentário 34%. Por isso, as conclusões sobre retorno valem para filmes conhecidos, majoritariamente de língua inglesa, e não para o cinema como um todo. Se o componente MNAR existir, o retorno observado também pode estar superestimado.',
    "outliers": 'Orçamento e receita são muito assimétricos, com muitos filmes pequenos e poucos gigantes. Por isso, a regra do IQR marca 6,5% dos orçamentos e 8,9% das receitas como outliers. Esses valores fazem parte da natureza das variáveis e não indicam erro, então em cada caso avaliamos se o valor é possível na vida real. Os 44 filmes com orçamento ou receita entre US$ 1 e 999 foram considerados erros, provavelmente valores cadastrados na unidade errada (em milhões em vez de dólares), e esses valores viraram NaN. A única exceção conhecida é *Tarnation*, cujo orçamento de US$ 218 é real; aceitamos excluí-lo junto para manter uma regra simples e reprodutível. O efeito desse tratamento é grande: apenas 18 desses filmes estavam na base de retorno, mas bastavam para levar o retorno máximo a 8,5 milhões de vezes o orçamento e a média a cerca de 2.955. Depois do tratamento, o máximo cai para cerca de 12.900 vezes e a média para 11,1, enquanto a mediana fica em 2,3 nos dois casos. Os blockbusters, como *Avatar*, e os retornos extremos legítimos, como *Atividade Paranormal*, foram mantidos, porque são reais e fazem parte da resposta à pergunta. Como o retorno continua muito assimétrico mesmo depois do tratamento, com a média quase cinco vezes maior que a mediana, a análise usa a **mediana** como medida central e **escala log** nos gráficos de dispersão.',
    "distribucional": (
        "O orçamento mediano é de US$ 23,3 milhões e três quartos dos orçamentos informados ficam em até "
        "US$ 50 milhões. A receita tem mediana de US$ 52,3 milhões e média de US$ 117,8 milhões, puxada pelos "
        "filmes de maior bilheteria. Essa diferença também aparece no retorno: a mediana é de 2,30×, a média "
        "é de 11,14× e metade dos valores fica entre 1,03× e 4,42×. A duração é mais concentrada, com metade "
        "dos filmes entre 94 e 118 minutos. Há orçamento válido para 77,8% dos filmes e receita válida para 69,8%. "
        "Comparar as medianas dessas duas colunas não diz se os filmes mais caros rendem mais por dólar investido. Para responder "
        "à pergunta, comparamos a mediana do retorno por faixa de orçamento e gênero entre os 3.210 filmes "
        "com os dois valores válidos.\n\n"
        "Mais de 72% dos filmes se concentram nas décadas de 2000 e 2010 (3.494 de 4.802). Setembro tem o "
        "maior número de lançamentos (588), e fevereiro e novembro têm 326 cada. A base é majoritariamente "
        "de idioma original inglês (93,8%) e os EUA aparecem como país produtor em 82,4% dos filmes, o que "
        "limita a leitura para outros mercados e épocas. Nas listas, 1.490 filmes têm dois gêneros e 1.525 "
        "têm três, então as comparações por gênero incluem filmes em comum. Em 411 filmes não há palavras-chave "
        "registradas. As sinopses têm mediana de 284 caracteres e faltam em apenas três filmes. Já a frase "
        "promocional falta em 843, e a coluna descartada `homepage` estava vazia em 3.091 dos 4.803 registros "
        "originais. Cada um dos 4.802 IDs aparece uma vez, embora haja títulos repetidos."
    ),
    "relacional": "A análise investiga se o retorno bruto (receita ÷ orçamento) varia com o orçamento e se esse padrão descritivo muda entre gêneros. Considera filmes lançados com orçamento e receita positivos e válidos após o tratamento; como um filme pode pertencer a vários gêneros, ele aparece em cada grupo correspondente e as comparações entre gêneros se sobrepõem. Retorno não é lucro. A exploração conjunta localiza concentrações e dispersões por orçamento e retorno, enquanto a associação de Spearman resume a direção e a intensidade da relação monotônica por gênero, com intervalos bootstrap de 95%. A leitura é descritiva: não demonstra causalidade nem constitui teste formal de diferenças entre gêneros.",
    "relacional_leitura": 'Para ler a tabela, chamamos de fraca a correlação com valor absoluto abaixo de 0,3, de moderada a que fica entre 0,3 e 0,6 e de forte a que fica a partir de 0,6. Os quatro pares mais fortes medem o tamanho e a visibilidade do filme: popularity e vote_count (0,96) quase coincidem e funcionam como uma mesma medida de público, e filmes com orçamento maior também têm receita maior (0,67), o que não diz nada sobre o retorno. As correlações do retorno com a receita (0,57) e com o orçamento (−0,14) vêm em parte da própria conta: como retorno = receita ÷ orçamento, ele tende a subir com a receita e a cair com o orçamento, o que reforça o cuidado com o orçamento no denominador. O retorno tem associação moderada com o número de votos (0,47), a popularidade (0,43) e a nota (0,33), mas a direção não fica clara: votos e popularidade são medidos depois do lançamento, então é mais provável que o sucesso de bilheteria traga público para o filme no TMDB do que o contrário. Já orçamento e receita quase não se relacionam com a nota (−0,08 e 0,12), e a duração não se relaciona com o retorno (0,04). Para um gestor, isso indica que gastar mais não compra uma nota melhor do público e que a duração do filme não pesa no retorno. Os pares com orçamento, receita ou retorno usam entre cerca de 3.200 e 3.700 filmes, enquanto os demais usam cerca de 4.700; como faltam justamente os filmes pequenos e pouco conhecidos (MAR), essas correlações valem para filmes conhecidos e devem ser comparadas com cuidado com as outras. Todas são associações e não indicam causa.',
    "p4_exploratorio": f"**Nota analítica (Pessoa 2).**\n\n{NOTAS['p4_exploratorio']}",
    "dashboard": 'O dashboard permite que o gestor escolha o próprio recorte de gênero e faixa de orçamento e compare o retorno mediano e a chance de não recuperar o orçamento desse recorte com os da base inteira, sem precisar abrir o notebook. Ele também mostra a cobertura financeira de cada gênero, para que o gestor saiba quanto confiar nos números antes de usá-los numa decisão.',
}

# Figuras do eda.ipynb, achadas pela 1ª linha do código da célula (não pelo número da célula, que muda)
FIGURAS_NOTEBOOK = {
    "ausencia_proporcao": "proporção de dados financeiros faltando",
    "ausencia_popularidade": "distribuição de popularidade: com x sem",
    "outliers_boxplots": "boxplots (budget/revenue só > 0)",
    "retorno_distribuicao": "distribuição do retorno em escala linea",
    "p1_exploratorio": "% de filmes com retorno calculável por",
    "p2_exploratorio": "apoio com pandas e implementação da dispersão",
    "p3_exploratorio": "Densidade conjunta de orçamento e retorno",
    "p4_exploratorio": "violino do retorno por gênero",
    "p2_explanatorio": "agregação em pandas e barras agrupadas",
    "p3_explanatorio": "Associação monotônica por gênero",
}


def extrair_figuras() -> dict[str, Path]:
    """Salva em out/parte_especifica/figuras/ a 1ª figura PNG de cada célula listada em FIGURAS_NOTEBOOK."""
    FIGURAS.mkdir(parents=True, exist_ok=True)
    nb = json.loads((PASTA / "eda.ipynb").read_text(encoding="utf-8"))
    achadas = {}
    for celula in nb["cells"]:
        if celula["cell_type"] != "code" or not celula["source"]:
            continue
        primeira = "".join(celula["source"]).splitlines()[0]
        for nome, trecho in FIGURAS_NOTEBOOK.items():
            if nome in achadas or trecho not in primeira:
                continue
            png = next((o["data"]["image/png"] for o in celula.get("outputs", []) if "image/png" in o.get("data", {})), None)
            if png:
                destino = FIGURAS / f"{nome}.png"
                destino.write_bytes(base64.b64decode(png if isinstance(png, str) else "".join(png)))
                achadas[nome] = destino
    return achadas


def _n(x: float, casas: int = 0) -> str:
    """Número no formato brasileiro: 4802 -> '4.802' · 2.53 -> '2,5'."""
    return f"{x:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _inline(texto: str) -> str:
    """Markdown simples (negrito, itálico, código) para a marcação do pdf_simples."""
    t = html.escape(texto, quote=False)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", t)
    return re.sub(r"`([^`]+)`", r"\1", t)


def tabela_risco(df: pd.DataFrame) -> pd.DataFrame:
    """Retorno e risco por faixa de orçamento (seção 7.11), calculados da base tratada."""
    import numpy as np

    from estilo import LIMITES_FAIXAS, ROTULOS_FAIXAS

    r = df.loc[df["retorno"].notna()]
    faixa = pd.cut(r["budget"], [0, *LIMITES_FAIXAS, np.inf], labels=ROTULOS_FAIXAS, right=True)
    t = r.groupby(faixa, observed=True)["retorno"].agg(
        filmes="size", mediana="median", nao_recupera=lambda x: (x < 1).mean(), mais_5x=lambda x: (x > 5).mean())
    return pd.DataFrame({
        "orçamento": t.index.astype(str),
        "filmes": [_n(v) for v in t["filmes"]],
        "retorno mediano": [f"{_n(v, 1)}×" for v in t["mediana"]],
        "não recupera o orçamento": [f"{_n(100 * v)}%" for v in t["nao_recupera"]],
        "fatura mais de 5×": [f"{_n(100 * v)}%" for v in t["mais_5x"]],
    })


def tabela_spearman(df: pd.DataFrame) -> pd.DataFrame:
    """Pares de variáveis quantitativas com o rho de Spearman e o nº de filmes do par (seção 7.7)."""
    q = df[["budget", "revenue", "retorno", "popularity", "runtime", "vote_average", "vote_count"]]
    m = q.corr(method="spearman")
    linhas = []
    for i, a in enumerate(m.columns):
        for b in m.columns[i + 1:]:
            linhas.append({"variável A": a, "variável B": b, "rho": m.loc[a, b], "filmes": q[[a, b]].dropna().shape[0]})
    t = pd.DataFrame(linhas).sort_values("rho", key=abs, ascending=False)
    t["rho"] = [_n(v, 2) for v in t["rho"]]
    t["filmes"] = [_n(v) for v in t["filmes"]]
    return t


def montar(P, H1, H2, CORPO, LEG, img, tabela) -> list:
    """Blocos da seção 7. Recebe os estilos e helpers de src/parte_comum/relatorio.py para o PDF ficar uniforme."""
    destaque = CORPO.com(cor="#B83A7A")
    s: list = []

    def md(texto: str) -> None:
        for bloco in re.split(r"\n\s*\n", texto.strip()):
            linhas = [l.strip() for l in bloco.strip().splitlines()]
            if all(l.startswith("- ") for l in linhas):
                s.extend(P("• " + _inline(l[2:])) for l in linhas)
            else:
                s.append(P(_inline(" ".join(l.lstrip("> ") for l in linhas))))

    def texto(chave: str, tabelas: dict | None = None) -> None:
        t = TEXTOS[chave]
        if t.startswith("PREENCHER"):
            s.append(P(f"<i>[{html.escape(t, quote=False)}]</i>", destaque))
            return
        for pedaco in re.split(r"(\{[A-Z_]+\})", t):  # {TABELA_...} vira a tabela correspondente
            if pedaco.startswith("{") and tabelas and pedaco[1:-1] in tabelas:
                s.append(tabela(tabelas[pedaco[1:-1]], col0=False, tam=8))
            elif pedaco.strip():
                md(pedaco)

    def figura(nome: str, legenda: str, largura: float = 15.5) -> None:
        caminho = figs.get(nome)
        if caminho:
            s.append(img(caminho, largura, legenda))
        else:
            s.append(P(f"<i>[Figura “{nome}” não encontrada no eda.ipynb: executar o notebook e gerar de novo.]</i>", destaque))

    def nota_grafico(gid: str, autor: str) -> None:
        if gid in NOTAS:
            s.append(P(f"<b>Nota analítica ({autor}).</b>"))
            md(NOTAS[gid])
        else:
            s.append(P(f"<i>[PREENCHER — {autor}: nota analítica deste gráfico.]</i>", destaque))

    figs = extrair_figuras()
    bruto = carregar_filmes()
    df = tratar(bruto)
    n_ret = int(df["retorno"].notna().sum())

    # ---------------------------------------------------------------------------------------
    s += [P("7. Parte Específica: TMDB 5000", H1), P("7.1 Dataset, pergunta e público-alvo", H2),
          P(f"<b>Dataset.</b> TMDB 5000 Movie Dataset (Kaggle): {_n(len(bruto))} filmes e {bruto.shape[1] - 1} colunas no arquivo original, "
            "com orçamento, receita, gêneros, datas e notas do público."),
          P(f"<b>Pergunta de pesquisa.</b> {PERGUNTA}"),
          P(f"<b>Público-alvo.</b> {PUBLICO.capitalize()}."),
          P("<b>Variável de resposta.</b> Retorno = receita ÷ orçamento, calculado só quando os dois valores são válidos. "
            "Retorno não é lucro: marketing e a parte dos cinemas ficam de fora.")]
    texto("motivacao")
    if RESPOSTA_CURTA:
        s += [P("<b>Resposta em uma frase.</b> " + _inline(RESPOSTA_CURTA))]

    # ---------------------------------------------------------------------------------------
    s += [P("7.2 Pilar estrutural", H2),
          P(f"Depois do tratamento ficam {_n(len(df))} filmes, dos quais {_n(n_ret)} têm retorno calculável.")]
    texto("estrutural")

    # ---------------------------------------------------------------------------------------
    colunas = ["budget", "revenue", "runtime", "vote_average"]
    faltantes = pd.DataFrame({
        "variável": colunas,
        "vazios (original)": [int(bruto[c].isna().sum()) for c in colunas],
        "zeros (original)": [int(bruto[c].eq(0).sum()) for c in colunas],
        "NaN (tratado)": [int(df[c].isna().sum()) for c in colunas],
        "válidos (tratado)": [int(df[c].notna().sum()) for c in colunas],
    })
    s += [P("7.3 Dados faltantes", H2),
          P("Zeros em orçamento, receita e duração significam “não informado” e viraram NaN; a nota vira NaN quando o "
            "filme não tem nenhum voto. A tabela mostra cada variável antes e depois do tratamento."),
          tabela(faltantes, col0=False, tam=8),
          ]
    figura("ausencia_proporcao", "Figura 15 — Proporção de filmes sem dados financeiros (eda.ipynb, seção 2.3).")
    figura("ausencia_popularidade", "Figura 16 — Popularidade dos filmes com e sem orçamento/receita (eda.ipynb, seção 2.3).")
    texto("faltantes")

    # ---------------------------------------------------------------------------------------
    s += [P("7.4 Outliers e valores implausíveis", H2)]
    figura("outliers_boxplots", "Figura 17 — Boxplots das variáveis quantitativas (orçamento e receita só com valores positivos).")
    figura("retorno_distribuicao", "Figura 18 — Distribuição do retorno em escala linear e logarítmica (eda.ipynb, seção 3.3).")
    texto("outliers")

    # ---------------------------------------------------------------------------------------
    decisoes = pd.DataFrame({"decisão": list(DECISOES), "valor": [str(v) for v in DECISOES.values()]})
    removidas = len(bruto) - len(df)
    s += [P("7.5 Tratamento (função tratar)", H2),
          P(f"Todo o grupo usa <b>tratar(carregar_filmes())</b>, de src/parte_especifica/dados.py. O tratamento removeu "
            f"{removidas} {'linha' if removidas == 1 else 'linhas'}; os demais problemas viraram NaN, sem remover o filme. "
            "As decisões ficam no dicionário DECISOES, reproduzido abaixo, e as justificativas no eda.ipynb (seção 4)."),
          tabela(decisoes, col0=False, tam=7.6, largura=[6.5, 10])]

    # ---------------------------------------------------------------------------------------
    s += [P("7.6 Pilar distribucional", H2)]
    texto("distribucional")
    s += [P("7.7 Pilar relacional", H2)]
    texto("relacional")
    # [IA] Claude Code – tabela de Spearman entre as quantitativas (mesmo cálculo da seção 6.0 do eda.ipynb)
    s.append(P("<b>Correlação de Spearman entre as variáveis quantitativas</b> (pares ordenados pela força; "
               "cada par usa os filmes com os dois valores válidos):", CORPO.com(junto_proximo=True)))
    s.append(tabela(tabela_spearman(df), col0=False, tam=7.6))
    texto("relacional_leitura")

    # ---------------------------------------------------------------------------------------
    s += [P("7.8 Gráficos exploratórios", H2)]
    figura("p1_exploratorio", "Figura 19 — Cobertura financeira por gênero: % de filmes com retorno calculável (Pessoa 1).")
    nota_grafico("p1_exploratorio", "Pessoa 1")
    figura("p2_exploratorio", "Figura 20 — Orçamento × retorno por gênero, escala logarítmica (Pessoa 2).")
    nota_grafico("p2_exploratorio", "Pessoa 2")
    figura("p3_exploratorio", "Figura 21 — Concentração de filmes por orçamento e retorno em seis gêneros (Pessoa 3).")
    nota_grafico("p3_exploratorio", "Pessoa 3")
    figura("p4_exploratorio", "Figura 22 — Distribuição do retorno por gênero e parcela abaixo de 1× (Pessoa 2).")
    texto("p4_exploratorio")

    # ---------------------------------------------------------------------------------------
    s += [P("7.9 Gráficos explanatórios", H2)]
    figura("p2_explanatorio", "Figura 23 — Mediana do retorno por faixa de orçamento e gênero (Pessoa 2).")
    nota_grafico("p2_explanatorio", "Pessoa 2")
    figura("p3_explanatorio", "Figura 24 — Correlação de Spearman entre orçamento e retorno por gênero, IC bootstrap de 95% (Pessoa 3).")
    nota_grafico("p3_explanatorio", "Pessoa 3")

    # ---------------------------------------------------------------------------------------
    s += [P("7.10 Resposta à pergunta de pesquisa", H2), P(f"<b>{PERGUNTA}</b>")]
    md(RESPOSTA_PERGUNTA)
    s += [P("7.11 O que isso significa para o gestor", H2)]
    texto("implicacoes", {"TABELA_RISCO": tabela_risco(df)})
    s += [P("7.12 Storytelling para decisão", H2)]
    md(STORYTELLING_P3)

    # ---------------------------------------------------------------------------------------
    s += [P("7.13 Dashboard", H2),
          P("<b>streamlit run src/parte_especifica/dashboard.py</b>. Páginas: <b>Início</b> (pergunta, resposta em destaque, "
            "números da base e navegação); <b>Exploração</b> (gráficos exploratórios, um por vez, com filtros e indicadores "
            "do recorte); <b>Explicação</b> (gráficos explanatórios, storytelling e resposta completa); <b>Década a década</b> "
            "(animação do percurso das medianas, com troca de métrica e de escala); <b>Dados e tratamento</b>; e a "
            "<b>Parte Comum</b> (Wine Quality). Os filtros da barra lateral (gênero, orçamento, ano e mínimo de filmes por "
            "gênero) valem para os gráficos, e a página avisa quando o recorte difere da base descrita nas notas.")]
    texto("dashboard")
    return s
