"""Gera relatorio.pdf (Parte Comum) a partir das tabelas/figuras já salvas em out/parte_comum/.

Uso (depois de rodar eda.py, kmeans.py e granularidade.py):  python src/parte_comum/relatorio.py
A seção 7 (Parte Específica, TMDB 5000) vem de src/parte_especifica/relatorio_especifica.py.
"""
from __future__ import annotations

import json
import sys

import pandas as pd

from common import OUT_DIR, RAIZ
from pdf_simples import DocumentoPDF, Espaco, Estilo, Imagem, Paragrafo, QuebraPagina, Tabela

sys.path.insert(0, str(RAIZ / "src" / "parte_especifica"))
import relatorio_especifica  # noqa: E402  (seção 7)
from notas import RESPOSTA_CURTA  # noqa: E402

# [IA] Claude Code – troca do reportlab/PIL pelo gerador em pdf_simples.py (só matplotlib)
cm = 1  # larguras passadas em centímetros

CORPO = Estilo(tamanho=9.2, entrelinha=13.2, alinhamento="justify", depois=5)
H1 = Estilo(tamanho=15, entrelinha=18, negrito=True, alinhamento="left", antes=10, depois=6, junto_proximo=True)
H2 = Estilo(tamanho=11.5, entrelinha=14, negrito=True, alinhamento="left", antes=8, depois=4, junto_proximo=True)
CK = CORPO.com(junto_proximo=True)
LEG = Estilo(tamanho=8, entrelinha=10, cor="#444444", alinhamento="left", depois=5)
TIT = Estilo(tamanho=21, entrelinha=26, negrito=True, alinhamento="center", depois=6)
SUB = CORPO.com(tamanho=11, entrelinha=14.5, alinhamento="center")

E, K, G = OUT_DIR / "eda", OUT_DIR / "kmeans", OUT_DIR / "granularidade"


def P(t, s=CORPO):
    return Paragrafo(t, s)


def Spacer(_largura, altura):
    return Espaco(altura)


def PageBreak():
    return QuebraPagina()


def bullets(itens):
    return [P("• " + i) for i in itens]


def img(caminho, largura=16.5, legenda=None):
    return Imagem(caminho, largura, legenda, LEG)


def tabela(df: pd.DataFrame, fmt="{:.3f}", col0=True, tam=7.6, largura=None):
    d = df.reset_index() if col0 else df
    linhas = [[str(c) for c in d.columns]]
    for _, r in d.iterrows():
        linhas.append([("—" if pd.isna(v) else (fmt.format(v) if isinstance(v, float) else str(v))) for v in r])
    return Tabela(linhas, tam, largura)


def csv(caminho, **kw):
    return pd.read_csv(caminho, **kw)


def montar() -> list:
    est = json.loads((E / "estrutural_resumo.json").read_text("utf-8"))
    r = json.loads((K / "resumo.json").read_text("utf-8"))
    p1, p2 = r["passada1"], r["passada2"]
    gran = json.loads((G / "resumo.json").read_text("utf-8"))
    desc = csv(E / "distribucional_descritiva.csv", index_col=0)
    sens = csv(K / "sensibilidade_cenarios.csv")
    s = []

    # ---------------------------------------------------------------- capa
    s += [Spacer(1, 90), P("Análise Exploratória de Dados: agrupamento de vinhos<br/>e orçamento × retorno no cinema", TIT),
          P("Parte Comum (Wine Quality) e Parte Específica (TMDB 5000) — Análise e Visualização de Dados (AVD)", SUB), Spacer(1, 14),
          P("Dataset: UCI Wine Quality (vinho tinto + branco, <i>Vinho Verde</i>) — archive.ics.uci.edu/dataset/186", SUB),
          P("Dataset: TMDB 5000 Movie Dataset (Kaggle) — kaggle.com/datasets/tmdb/tmdb-movie-metadata", SUB),
          P("Integrantes: Raul Vitor Ferraz Silva, Lucas Rodrigues Campos e Julia Maria Santos Teixeira", SUB), Spacer(1, 40),
          P("<b>Resumo.</b> Exploramos os 6 497 vinhos nos quatro pilares da EDA, padronizamos as 11 variáveis físico-químicas e aplicamos "
            "o K-Means em duas passadas (k livre pelo cotovelo e k = 7 classes). Em ambas, a acurácia dos grupos em relação à nota de "
            f"qualidade ({p1['acuracia_global']:.1%} e {p2['acuracia_global']:.1%}) é praticamente igual à de um classificador trivial que "
            f"sempre responde “nota 6” ({p1['baseline_classe_majoritaria']:.1%}). Os clusters descobertos refletem principalmente o tipo de vinho "
            "(tinto × branco) e perfis químicos, e não a nota sensorial: as notas vizinhas não são separáveis pelos dados.", CORPO),
          P("<b>Parte Específica (TMDB 5000).</b> " + relatorio_especifica._inline(RESPOSTA_CURTA), CORPO) if RESPOSTA_CURTA
          else P("<i>[PREENCHER — resumo da Parte Específica]</i>", CORPO.com(cor="#B83A7A")),
          PageBreak()]

    # ---------------------------------------------------------------- 1. dados + EDA
    s += [P("1. Dados e escolhas de projeto", H1),
          P(f"O Wine Quality reúne {est['linhas']} vinhos (1 599 tintos e 4 898 brancos) descritos por 11 medidas físico-químicas e uma nota de "
            "qualidade (mediana de pelo menos três enólogos). Usamos os dois arquivos juntos, acrescentando a coluna auxiliar <i>tipo</i> "
            "(tinto/branco), que <b>não</b> entra no K-Means nem na padronização. A variável de classe é <i>quality</i>, com "
            f"<b>{est['n_classes']} classes</b> observadas (notas 3 a 9). Análises de sensibilidade com apenas tintos (6 classes) e apenas brancos (7 classes) "
            "estão na Seção 6.")]

    s += [P("2. Análise Exploratória de Dados", H1), P("2.1 Pilar estrutural", H2)]
    s += bullets([
        f"{est['linhas']} linhas × {est['colunas']} colunas: 11 variáveis numéricas contínuas, <i>quality</i> (inteira, ordinal) e <i>tipo</i> (categórica).",
        f"Nenhum valor nulo ({est['nulos_total']}). Consistência física verificada: nenhuma amostra com SO₂ livre maior que o total.",
        f"<b>{est['duplicatas']} linhas duplicadas ({est['duplicatas_pct']}%)</b>. Mantidas: com 11 medidas de precisão limitada, é plausível que vinhos "
        "diferentes tenham a mesma ficha, e removê-las eliminaria sobretudo vinhos de nota comum, agravando o desbalanceamento.",
    ])
    s += [P("2.2 Pilar distribucional", H2),
          tabela(desc[["mean", "std", "min", "50%", "max", "assimetria", "curtose"]].round(3), fmt="{:.3f}"),
          img(E / "dist_histogramas.png", 16.5, "Figura 1 — Histogramas com KDE das 11 variáveis."),
          P("Quase todas as variáveis têm cauda longa à direita (cloretos: assimetria 5,4; sulfatos 1,8; acidez fixa 1,7; açúcar residual 1,4) e <b>não seguem uma distribuição normal</b>. "
            "O SO₂ total é simétrico, mas tem dois picos porque mistura tintos (mediana 38) e brancos (mediana 134) (Figura 1). As escalas variam em ordens de grandeza (densidade ≈ 0,99; SO₂ total ≈ 116). "
            "A variável de classe é fortemente desbalanceada: as notas 5 e 6 somam 76,6% dos vinhos, enquanto 3, 8 e 9 têm 30, 193 e 5 vinhos."),
          img(E / "dist_classes.png", 14, "Figura 2 — Distribuição das notas de qualidade (total e por tipo). A nota 9 só existe no vinho branco.")]

    com = csv(E / "relacional_com_classe.csv", index_col=0).round(3)
    pares = csv(E / "relacional_pares.csv").head(5).round(3)[["var_a", "var_b", "pearson", "spearman"]]
    s += [P("2.3 Pilar relacional", H2),
          img(E / "corr_pearson.png", 12.5, "Figura 3 — Correlação de Pearson entre todas as variáveis, incluindo <i>quality</i> (a de Spearman, em out/, é equivalente)."),
          P("<b>Variáveis × classe</b> (correlação de Spearman, adequada à nota ordinal; o teste de Kruskal–Wallis rejeita a igualdade entre notas para todas as variáveis, p &lt; 0,05):", CK),
          tabela(com, fmt="{:.3f}"),
          P("<b>Entre variáveis</b> (maiores correlações): ", CK), tabela(pares, fmt="{:.3f}", col0=False),
          img(E / "rel_boxplots_por_classe.png", 16.5, "Figura 4 — Variáveis por nota de qualidade: as distribuições de notas vizinhas se sobrepõem quase totalmente."),
          P("O <b>álcool</b> é a variável mais associada à qualidade (ρ = +0,45), seguida de densidade (−0,32), cloretos (−0,30) e acidez volátil (−0,26); nenhuma supera |0,45|. "
            "Há redundância entre variáveis (SO₂ livre × total: 0,72; densidade × álcool: −0,69; açúcar × densidade: 0,55). Além disso, as variáveis separam "
            "<b>tinto de branco</b> muito mais do que separam notas (diferença padronizada de até 1,6 desvio-padrão em SO₂ total; Figura em out/), o que antecipa que o K-Means agrupará por tipo."),
          P("O <i>scipy</i> não está entre as bibliotecas permitidas pelo enunciado, então implementamos o teste de Kruskal–Wallis usando apenas <i>numpy</i> e <i>pandas</i>. "
            "Ele foi usado no pilar relacional da EDA para verificar se cada variável muda entre as notas de qualidade dos vinhos. "
            "A implementação segue a mesma fórmula do <i>scipy.stats.kruskal</i>: os valores viram postos (com média nos empates), calcula-se a estatística H com correção de empates, "
            "e o p-valor sai da distribuição qui-quadrado com k − 1 graus de liberdade. Comparamos os resultados com os do <i>scipy</i> e eles foram idênticos. "
            "Escolhemos um teste não paramétrico porque as variáveis não seguem distribuição normal. Essa não normalidade fica evidente na assimetria, na curtose e nos histogramas, "
            "por isso dispensamos um teste formal de normalidade."),
          ]

    ot = csv(E / "outliers_tabela.csv", index_col=0)[["limite_inf_iqr", "limite_sup_iqr", "n_outliers_iqr", "pct_iqr", "n_outliers_z3", "max_|z|"]]
    opc = csv(E / "outliers_por_classe.csv", index_col=0)
    s += [P("2.4 Pilar de outliers", H2),
          tabela(ot, fmt="{:.3f}", tam=7),
          img(E / "outliers_boxplots.png", 16, "Figura 5 — Boxplots e percentual de pontos fora de 1,5·IQR."),
          P("<b>Identificação:</b> regra de Tukey (1,5·IQR) e |z| &gt; 3. Todas as variáveis têm outliers; <b>22,7% dos vinhos (1 473)</b> têm ao menos um. "
            "Os mais extremos são cloretos (até 15,8 desvios), SO₂ livre = 289 mg/L e a amostra de açúcar 65,8 g/L com densidade 1,039."),
          P("<b>Decisão: não remover.</b> (i) Removê-los descartaria quase um quarto da base. (ii) Eles se concentram nas classes raras — "
            "63% dos vinhos de nota 3 e 37% da nota 4 têm algum outlier, contra 13–14% das notas 7–8 — de modo que a exclusão eliminaria justamente as classes de interesse. "
            "(iii) São quimicamente plausíveis (variáveis positivas de cauda longa); só 3 linhas têm três ou mais extremos simultâneos e nenhuma é um erro comprovado. "
            "(iv) Como mitigação, testamos <b>winsorização a 1%–99%</b> como análise de sensibilidade: o resultado do K-Means não muda (Seção 6). "
            "Ressalvamos que o K-Means, sensível a extremos, dedica um cluster inteiro (33 vinhos, cloretos ≈ +8,8 e sulfatos ≈ +3,5 desvios) a esses pontos na Passada 2."),
          PageBreak()]

    # ---------------------------------------------------------------- 3. padronização
    s += [P("3. Padronização", H1),
          P("Aplicamos <b>z-score</b> (<i>StandardScaler</i>: média 0, desvio 1) às 11 variáveis numéricas, ajustado e aplicado antes do K-Means. "
            "A classe <i>quality</i> e o <i>tipo</i> ficam de fora."),
          P("<b>Por que é necessário.</b> O K-Means atribui cada ponto ao centroide mais próximo em <b>distância euclidiana</b> e minimiza a soma dos quadrados dessas distâncias. "
            "Se as variáveis têm unidades e variâncias muito diferentes, as de maior escala dominam a distância e as demais ficam praticamente ignoradas, independentemente de serem informativas. "
            "Aqui o SO₂ total (variância ≈ 3 200) pesa cerca de 10⁸ vezes mais do que a densidade (variância ≈ 9·10⁻⁶). "
            f"Nos dados brutos, <b>{csv(K / 'padronizacao_contribuicao_distancia_bruta.csv', index_col=0)['pct'].iloc[0]:.0f}%</b> da distância entre centroides vem só do SO₂ total, e o ARI contra a nota cai de "
            f"{p2['ARI']:.3f} (padronizado) para {r['sem_padronizar']['ARI']:.4f} (bruto). Com o z-score cada variável contribui na mesma unidade (desvios-padrão), e "
            "o K-Means passa a refletir a estrutura conjunta. Não se usou min–max por ser mais sensível aos outliers que preservamos."),
          img(K / "padronizacao_antes_depois.png", 15.5, "Figura 6 — Boxplots antes (eixo symlog) e depois da padronização."), PageBreak()]

    # ---------------------------------------------------------------- 4. K-means
    vr = csv(K / "passada1_varredura_k.csv").round(3).set_index("k")
    s += [P("4. K-Means em duas passadas", H1), P("4.1 Passada 1 — k livre (Método do Cotovelo)", H2),
          P(f"Rodamos o K-Means para k = 2…10 (20 inicializações, semente 42). O cotovelo foi escolhido de forma objetiva: o ponto da curva de inércia (normalizada) mais distante "
            f"da reta entre k = 2 e k = 10. Resultado: <b>k = {p1['k']}</b>. O silhouette é máximo em k = 2 (0,277) e tem um segundo pico em k = 4 (0,249), consistente com o cotovelo."),
          tabela(vr, fmt="{:.3f}", tam=8),
          img(K / "passada1_cotovelo.png", 14, "Figura 7 — Método do Cotovelo (esquerda) e silhouette (direita)."),
          img(K / "passada1_pca.png", 16.5, "Figura 8 — Passada 1 (k = 4) projetada em PCA (PC1+PC2 = 50% da variância): clusters (esq.) e nota real (dir.). Os grupos se separam, as notas não."),
          img(K / "passada1_perfil_centroides.png", 14, "Figura 9 — Centroides da Passada 1 em z-score."),
          P("4.2 Passada 2 — k fixo = 7", H2),
          P("Mesma configuração, com k = 7 (número de notas observadas).", CK),
          img(K / "passada2_pca.png", 16.5, "Figura 10 — Passada 2 (k = 7) em PCA: clusters × nota real."),
          img(K / "passada2_perfil_centroides.png", 14, "Figura 11 — Centroides da Passada 2 em z-score. O cluster 0 é o de álcool alto (≈ +12,2% vol., densidade baixa); 2 e 5 são tintos ácidos; 3 reúne 33 outliers de cloretos/sulfatos."),
          PageBreak()]

    # ---------------------------------------------------------------- 5. métricas
    s += [P("5. Métricas e matrizes de confusão", H1),
          P("Cada cluster foi mapeado para a classe real <b>majoritária</b> entre seus membros. A matriz <b>retangular</b> mostra classe real × cluster (rotulado "
            "“C<i>i</i>→classe mapeada”); a matriz <b>quadrada</b> agrega os clusters pelo rótulo mapeado (real × mapeado). "
            "Acurácia, precisão e recall por classe saem da matriz quadrada (acurácia por classe = um-contra-todos: (TP+TN)/N). "
            "Precisão “—” significa que a classe nunca foi atribuída a nenhum cluster (TP + FP = 0).")]
    for nome, tit in (("passada1", f"5.1 Passada 1 (k = {p1['k']})"), ("passada2", f"5.2 Passada 2 (k = {p2['k']})")):
        ret = csv(K / f"{nome}_matriz_retangular.csv", index_col=0)
        cmq = csv(K / f"{nome}_matriz_confusao.csv", index_col=0)
        pc = csv(K / f"{nome}_metricas_por_classe.csv", index_col=0)[["suporte", "acuracia_ovr", "precisao", "recall", "f1"]]
        pc["suporte"] = pc["suporte"].astype(int).astype(str)
        res = r[nome]
        s += [P(tit, H2), P("Matriz de confusão retangular (linhas = classe real; colunas = cluster→classe mapeada):", CK),
              tabela(ret, fmt="{:.0f}", tam=8), P("Matriz de confusão quadrada (linhas = real; colunas = rótulo mapeado):", CK),
              tabela(cmq, fmt="{:.0f}", tam=8), P("Métricas por classe:", CK), tabela(pc, fmt="{:.3f}", tam=8),
              P(f"<b>Globais:</b> acurácia = {res['acuracia_global']:.3f} (baseline “sempre nota 6” = {res['baseline_classe_majoritaria']:.3f}); "
                f"precisão macro = {res['precisao_macro']:.3f}; recall macro = {res['recall_macro']:.3f}; silhouette = {res['silhouette']:.3f}; "
                f"ARI = {res['ARI']:.3f}; NMI = {res['NMI']:.3f}; homogeneidade = {res['homogeneidade']:.3f}. "
                f"Classes que nunca foram atribuídas a nenhum cluster: {res['classes_nunca_previstas']}.")]
        s += [img(K / f"{nome}_matriz_retangular.png", 16.5, f"Figura {12 if nome == 'passada1' else 13} — Matriz retangular da {nome.replace('passada', 'Passada ')}: contagem e % da classe real.")]
    s += [P("<b>Nota metodológica.</b> O K-Means não tem acesso aos rótulos durante a execução; a “acurácia” mede apenas o quanto os grupos descobertos coincidem com as notas. "
            "Como o mapeamento por maioria é feito <i>a posteriori</i> e as classes são desbalanceadas, comparamos sempre com o baseline da classe majoritária e com ARI/NMI, "
            "que não dependem do mapeamento."), PageBreak()]

    # ---------------------------------------------------------------- 6. discussão
    cmp_ = pd.DataFrame({
        f"Passada 1 (k={p1['k']})": {"acurácia": p1["acuracia_global"], "precisão macro": p1["precisao_macro"], "recall macro": p1["recall_macro"],
                                      "silhouette": p1["silhouette"], "ARI": p1["ARI"], "NMI": p1["NMI"], "homogeneidade": p1["homogeneidade"]},
        f"Passada 2 (k={p2['k']})": {"acurácia": p2["acuracia_global"], "precisão macro": p2["precisao_macro"], "recall macro": p2["recall_macro"],
                                      "silhouette": p2["silhouette"], "ARI": p2["ARI"], "NMI": p2["NMI"], "homogeneidade": p2["homogeneidade"]},
    })
    cmp_.index.name = "métrica"
    acc_tinto = sens[(sens["cenario"] == "somente tinto") & (sens["passada"] == 2)]["acuracia"].iloc[0]
    sc = sens.copy()
    sc["cenario"] = sc["cenario"].str.replace(" (PRINCIPAL)", "*", regex=False)
    sc = sc[["cenario", "passada", "k", "acuracia", "baseline_maioria", "silhouette", "ARI", "NMI"]]
    sc.columns = ["cenário", "pass.", "k", "acurácia", "baseline", "silh.", "ARI", "NMI"]
    s += [P("6. Discussão crítica", H1), P("6.1 Passada 1 (k livre) × Passada 2 (k fixo)", H2), tabela(cmp_, fmt="{:.3f}", tam=8.5),
          P("<b>Qual gerou grupos mais coerentes?</b> Depende do critério — e em nenhum a resposta é favorável a “recuperar a qualidade”:"),
          *bullets([
              f"<b>Coerência interna (geometria dos dados):</b> vence a <b>Passada 1</b>. Com k = 4 o silhouette é {p1['silhouette']:.3f} contra {p2['silhouette']:.3f} em k = 7: "
              "os grupos são mais compactos e mais separados. Eles correspondem sobretudo a <b>tinto × branco</b> (dois clusters quase só de brancos e dois quase só de tintos), "
              "isto é, a estrutura natural do dado.",
              f"<b>Coerência com as notas:</b> a <b>Passada 2</b> é um pouco melhor em ARI ({p2['ARI']:.3f} vs {p1['ARI']:.3f}), NMI e homogeneidade, mas as diferenças são minúsculas "
              f"e a acurácia é a mesma ({p2['acuracia_global']:.3f} vs {p1['acuracia_global']:.3f}). Ambas ficam <b>apenas ~2 pontos acima do baseline</b> ({p1['baseline_classe_majoritaria']:.3f}).",
              "Em ambas, só as notas 5 e 6 são “mapeadas”; as notas 3, 4, 7, 8 e 9 nunca são majoritárias (recall 0). A acurácia por classe dessas notas parece alta (≥ 0,83) "
              "apenas porque dizer “não é essa classe” quase sempre acerta.",
              "Forçar k = 7 apenas fragmenta os mesmos grupos naturais em mais pedaços, sem que cada pedaço corresponda a uma nota; em k = 7 há ainda um cluster de 33 vinhos dedicado a outliers. "
              "O único sinal real de qualidade é o cluster de <b>álcool alto</b> (≈ 12,2%), que concentra 42% dos vinhos nota 7, 51% da nota 8 e 4 dos 5 da nota 9 (nota média 6,43 contra 5,2–5,9 nos demais) — coerente com ρ(álcool, qualidade) = 0,45.",
          ]),
          P("6.2 A granularidade das classes faz sentido?", H2),
          img(G / "efeito_vizinhas_heatmap.png", 15.5, "Figura 14 — |d de Cohen| entre notas vizinhas por variável: |d| &lt; 0,2 desprezível, 0,2–0,5 pequeno, 0,5–0,8 médio. Só o álcool (5×6, 6×7) e o SO₂ livre (notas raras 3×4, 4×5) chegam ao nível médio."),
          *bullets([
              f"<b>Notas vizinhas são pouco distinguíveis.</b> O tamanho de efeito médio (d de Cohen) entre notas consecutivas é de apenas {min(gran['d_medio_vizinhas_todos'].values()):.2f}–{max(gran['d_medio_vizinhas_todos'].values()):.2f} "
              f"(5×6: {gran['d_medio_vizinhas_todos']['5 vs 6']:.2f}; 6×7: {gran['d_medio_vizinhas_todos']['6 vs 7']:.2f}); mesmo a melhor variável isolada (álcool) chega a só ≈ 0,7 entre 5×6 e 6×7, e 7×8 mal passa de 0,27. "
              "Ou seja, <b>notas 5 e 6 (e 6 e 7) não são realmente distinguíveis pelos dados físico-químicos</b>: as distribuições se sobrepõem quase por completo (Figura 4).",
              "<b>A nota é um rótulo ruidoso e ordinal.</b> É a mediana de três ou mais provadores — subjetiva —, e as classes são desbalanceadas: 3 (30) e 9 (5 vinhos) têm amostra insuficiente para formar um grupo; "
              "5 e 6 concentram 77% dos dados. Um K-Means, que busca grupos esféricos e compactos, não tem como isolar classes extremas tão pequenas.",
              f"<b>Teste com classes agrupadas</b> (baixa ≤ 5, média = 6, alta ≥ 7; k = 3): acurácia {gran['agrupadas_k3']['acuracia_global']:.3f} (baseline {gran['agrupadas_k3']['baseline']:.3f}), "
              f"ARI {gran['agrupadas_k3']['ARI']:.3f}. A fusão é mais defensível do ponto de vista dos dados e do vocabulário de negócio, mas <b>mesmo assim o K-Means não a recupera</b>, "
              "porque a principal estrutura latente é o tipo de vinho, não a nota. Assim, <b>o problema não é só a granularidade</b>: as 11 variáveis, vistas sem supervisão, simplesmente não organizam os vinhos por qualidade.",
              "<b>Recomendação:</b> (a) tratar a qualidade como ordinal e agrupá-la em 3 níveis (ou binarizar em “≥ 7”); (b) analisar tinto e branco separadamente "
              f"(só tintos: acurácia ≈ {acc_tinto:.2f} contra baseline 0,43 — melhor, ainda modesta); "
              "(c) para prever a nota, usar aprendizado supervisionado (ex.: floresta aleatória, regressão ordinal), no qual os rótulos orientam a busca.",
          ]),
          P("6.3 Sensibilidade e limitações", H2),
          tabela(sc, fmt="{:.3f}", col0=False, tam=7.4, largura=[6.3 * cm, 1.1 * cm, 0.9 * cm, 1.7 * cm, 1.7 * cm, 1.5 * cm, 1.5 * cm, 1.5 * cm]),
          P("* Cenário principal. A winsorização a 1%–99% e a separação por tipo de vinho não alteram a conclusão central (acurácia ≈ baseline; ARI &lt; 0,07). "
            "Limitações: resultados dependem da inicialização (usamos n_init = 20 e semente fixa); o K-Means pressupõe grupos esféricos de variância parecida; "
            "o mapeamento por maioria pode associar vários clusters à mesma classe e deixar outras sem representante."),
          PageBreak(),
          *relatorio_especifica.montar(P, H1, H2, CORPO, LEG, img, tabela),
          P("8. Uso de IA e referências", H1),
          P("O uso de ferramentas de IA generativa neste trabalho está declarado em docs/DECLARACAO_IA.pdf (assinada pelos integrantes)."),
          # referências alinhadas à esquerda: a URL longa abre buracos no texto justificado
          P("• P. Cortez, A. Cerdeira, F. Almeida, T. Matos, J. Reis. <i>Modeling wine preferences by data mining from physicochemical properties.</i> Decision Support Systems, 47(4):547–553, 2009.", CORPO.com(alinhamento="left")),
          P("• TMDB 5000 Movie Dataset, Kaggle — kaggle.com/datasets/tmdb/tmdb-movie-metadata (dados de The Movie Database, TMDB).", CORPO.com(alinhamento="left")),
          P("• Software: Python 3, pandas, NumPy, scikit-learn, seaborn, matplotlib, plotly, Streamlit.", CORPO.com(alinhamento="left"))]
    return s


def main() -> None:
    destino = RAIZ / "relatorio.pdf"
    doc = DocumentoPDF(destino, titulo="Análise Exploratória de Dados: agrupamento de vinhos e orçamento × retorno no cinema",
                       rodape="AVD — Wine Quality (Parte Comum) · TMDB 5000 (Parte Específica)")
    doc.construir(montar())
    print("Relatório gerado:", destino)


if __name__ == "__main__":
    main()
