"""Notas analíticas de cada gráfico — texto dos autores, copiado do eda.ipynb sem alteração.

Se a nota mudar no notebook, atualizar aqui também (ou vice-versa).
"""

NOTAS = {
    "p3_exploratorio": (
        'Cada painel mostra contagens de filmes por bin de orçamento e retorno bruto, com ambos os eixos em escala logarítmica; o `hist2d` apresenta contagens por bin, não densidade normalizada. Tons mais intensos indicam mais filmes por bin dentro daquele painel. Cada painel possui sua própria escala de cor, portanto a intensidade não deve ser comparada diretamente entre gêneros; os gêneros também possuem tamanhos de amostra diferentes. As contagens entre painéis não devem ser somadas, pois um filme pode pertencer a mais de um gênero. A linha tracejada marca retorno de 1× (receita igual ao orçamento, sem representar lucro). A comparação ajuda a localizar concentrações e dispersões dentro de cada gênero, sem indicar causalidade.'
    ),
    "p3_explanatorio": (
        'Spearman resume a direção e a intensidade da associação monotônica entre orçamento e retorno em cada gênero; o ponto marca a estimativa, o segmento mostra o intervalo bootstrap de 95% e o tamanho do ponto representa o número de filmes. A linha vertical marca ausência de associação monotônica. Como filmes podem pertencer a vários gêneros, as estimativas compartilham observações e os intervalos não devem ser lidos como comparação independente ou teste formal de diferença. Retorno também contém orçamento no denominador, criando acoplamento matemático que limita a interpretação da correlação.'
    ),
    "p2_exploratorio": (
        'Cada ponto representa um filme e mostra seu orçamento e o retorno obtido. A maior concentração está entre US$ 10 milhões e US$ 100 milhões de orçamento, com retornos entre cerca de 1 e 10 vezes o orçamento. Os filmes de drama aparecem mais espalhados, principalmente entre os orçamentos menores, enquanto ação e aventura se concentram mais nos orçamentos altos. Os quatro painéis usam a mesma escala e os mesmos limites nos eixos, o que facilita comparar os gêneros.'
    ),
    "p2_explanatorio": (
        'As barras agrupadas permitem comparar a mediana do retorno de cada gênero nas diferentes faixas de orçamento. Em aventura, o retorno mediano foi de 3,44 vezes o orçamento nos filmes de até US$ 20 milhões e caiu para cerca de 1,9 vez na faixa acima de US$ 50 milhões até US$ 100 milhões. Na faixa acima de US$ 20 milhões até US$ 50 milhões, comédia apresentou retorno mediano de 2,21 vezes o orçamento. Usei a mediana porque os retornos extremos poderiam distorcer a média.'
    ),
}
