"""Mini-gerador de PDF usando só matplotlib (substitui o reportlab).

Monta um documento A4 a partir de uma lista de blocos:
    Paragrafo(texto, estilo)  – aceita <b>, <i>, <br/> e &lt; &gt; &amp;
    Espaco(altura_pt)
    QuebraPagina()
    Imagem(caminho_png, largura_cm, legenda)
    Tabela(DataFrame/linhas, tamanho_fonte, larguras_cm)

e escreve com matplotlib.backends.backend_pdf.PdfPages. O texto é quebrado em
linhas medindo cada palavra com a própria fonte (DejaVu Sans, que vem com o
matplotlib e cobre acentos, ρ, →, ₂, ⁻¹² etc.).
"""
# [IA] Claude Code – layout de PDF com matplotlib para não depender de reportlab/PIL
from __future__ import annotations

import html
import re
from dataclasses import dataclass, field, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.ft2font import FT2Font  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.transforms import Affine2D  # noqa: E402

CM = 72 / 2.54  # pontos por centímetro
A4 = (595.28, 841.89)

_FONTES = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
_ARQ_FONTE = {
    (False, False): _FONTES / "DejaVuSans.ttf",
    (True, False): _FONTES / "DejaVuSans-Bold.ttf",
    (False, True): _FONTES / "DejaVuSans-Oblique.ttf",
    (True, True): _FONTES / "DejaVuSans-BoldOblique.ttf",
}
_PROPS = {k: FontProperties(fname=str(v)) for k, v in _ARQ_FONTE.items()}
_FT = {k: FT2Font(str(v)) for k, v in _ARQ_FONTE.items()}
_cache_largura: dict = {}


def largura_texto(texto: str, tamanho: float, negrito=False, italico=False) -> float:
    """Largura do texto em pontos."""
    chave = (texto, tamanho, negrito, italico)
    if chave not in _cache_largura:
        ft = _FT[(negrito, italico)]
        ft.set_size(tamanho, 72)
        ft.set_text(texto, 0.0)
        _cache_largura[chave] = ft.get_width_height()[0] / 64.0
    return _cache_largura[chave]


# ------------------------------------------------------------------------- estilos e blocos
@dataclass(frozen=True)
class Estilo:
    tamanho: float = 9.2
    entrelinha: float = 13.2
    negrito: bool = False
    cor: str = "#000000"
    alinhamento: str = "justify"  # left | center | justify
    antes: float = 0.0
    depois: float = 5.0
    junto_proximo: bool = False

    def com(self, **kw) -> "Estilo":
        return replace(self, **kw)


@dataclass
class Paragrafo:
    texto: str
    estilo: Estilo = field(default_factory=Estilo)


@dataclass
class Espaco:
    altura: float


@dataclass
class QuebraPagina:
    pass


@dataclass
class Imagem:
    caminho: Path
    largura_cm: float = 16.5
    legenda: str | None = None
    estilo_legenda: Estilo | None = None


@dataclass
class Tabela:
    linhas: list[list[str]]  # primeira linha = cabeçalho
    tamanho: float = 7.6
    larguras_cm: list[float] | None = None


# ------------------------------------------------------------------------- texto rico
_TOKENS = re.compile(r"(<b>|</b>|<i>|</i>|<br\s*/?>|\s+)")


def _palavras(texto: str):
    """Quebra o texto em itens (palavra, negrito, itálico, espaço_antes) e marcadores de quebra de linha."""
    b = i = False
    espaco = False
    for parte in _TOKENS.split(texto):
        if not parte:
            continue
        if parte == "<b>":
            b = True
        elif parte == "</b>":
            b = False
        elif parte == "<i>":
            i = True
        elif parte == "</i>":
            i = False
        elif parte.startswith("<br"):
            yield "\n"
            espaco = False
        elif parte.isspace():
            espaco = True
        else:
            yield (html.unescape(parte), b, i, espaco)
            espaco = False


def _quebrar_linhas(texto: str, estilo: Estilo, largura: float):
    """Retorna lista de linhas; cada linha = (lista de (x, palavra, b, i), é_ultima_do_paragrafo)."""
    tam = estilo.tamanho
    linhas, atual, x = [], [], 0.0
    esp = largura_texto(" ", tam)

    def fecha(ultima):
        nonlocal atual, x
        linhas.append((atual, ultima))
        atual, x = [], 0.0

    # agrupa pedaços colados (ex.: "<b>texto</b>," vira um grupo só) para não quebrar no meio
    grupos, grupo = [], []
    for item in _palavras(texto):
        if item == "\n" or (item[3] and grupo):
            grupos.append(grupo)
            grupo = []
        if item == "\n":
            grupos.append("\n")
            continue
        palavra, b, it, _ = item
        b = b or estilo.negrito
        grupo.append((palavra, b, it, largura_texto(palavra, tam, b, it)))
    if grupo:
        grupos.append(grupo)

    for g in grupos:
        if g == "\n":
            fecha(True)
            continue
        if not g:
            continue
        w_g = sum(p[3] for p in g)
        gap = esp if atual else 0.0
        if atual and x + gap + w_g > largura:
            fecha(False)
            gap = 0.0
        for k, (palavra, b, it, w) in enumerate(g):
            primeiro = k == 0
            atual.append([x + (gap if primeiro else 0.0), palavra, b, it, w, primeiro])
            x += (gap if primeiro else 0.0) + w
    if atual or not linhas:
        fecha(True)
    return linhas


# ------------------------------------------------------------------------- documento
class DocumentoPDF:
    def __init__(self, destino, titulo="", rodape="", margens_cm=(2.0, 2.0, 1.8, 1.8)):
        self.destino = Path(destino)
        self.titulo = titulo
        self.rodape = rodape
        esq, dir_, topo, base = (m * CM for m in margens_cm)
        self.x0, self.x1 = esq, A4[0] - dir_
        self.y_topo, self.y_base = A4[1] - topo, base
        self.largura = self.x1 - self.x0
        self.fig = None
        self.pagina = 0

    # --- páginas
    def _nova_pagina(self):
        if self.fig is not None:
            self._fecha_pagina()
        self.fig = plt.figure(figsize=(A4[0] / 72, A4[1] / 72))
        self.trans = Affine2D().scale(1 / 72) + self.fig.dpi_scale_trans  # coordenadas em pontos
        self.pagina += 1
        self.y = self.y_topo

    def _fecha_pagina(self):
        if self.rodape:
            self.fig.text(A4[0] / 2, 1.1 * CM, f"{self.rodape} · página {self.pagina}", transform=self.trans,
                          ha="center", va="baseline", fontsize=7.5, color="grey", fontproperties=_PROPS[(False, False)])
        self.pdf.savefig(self.fig)
        plt.close(self.fig)
        self.fig = None

    def _resta(self):
        return self.y - self.y_base

    # --- medidas (para manter título junto do bloco seguinte)
    def _altura(self, bloco) -> float:
        if isinstance(bloco, Paragrafo):
            n = len(_quebrar_linhas(bloco.texto, bloco.estilo, self.largura))
            return bloco.estilo.antes + n * bloco.estilo.entrelinha + bloco.estilo.depois
        if isinstance(bloco, Imagem):
            h = self._dim_imagem(bloco)[1]
            if bloco.legenda:
                h += self._altura(Paragrafo(bloco.legenda, bloco.estilo_legenda or Estilo()))
            return h + 6
        if isinstance(bloco, Tabela):
            return min((len(bloco.linhas)) * self._altura_linha_tab(bloco) + 8, 4 * self._altura_linha_tab(bloco) + 8)
        if isinstance(bloco, Espaco):
            return bloco.altura
        return 0.0

    # --- desenho
    def _paragrafo(self, p: Paragrafo, minimo_proximo: float = 0.0):
        e = p.estilo
        linhas = _quebrar_linhas(p.texto, e, self.largura)
        if self.y < self.y_topo:  # não aplica espaço "antes" no topo da página
            self.y -= e.antes
        if self._resta() < e.entrelinha + minimo_proximo:
            self._nova_pagina()
        for palavras, ultima in linhas:
            if self._resta() < e.entrelinha:
                self._nova_pagina()
            base = self.y - e.tamanho  # linha de base
            fim = (palavras[-1][0] + palavras[-1][4]) if palavras else 0.0
            folga = self.largura - fim
            desloc = folga / 2 if e.alinhamento == "center" else 0.0
            gaps = sum(1 for k, w in enumerate(palavras) if k > 0 and w[5])
            extra = folga / gaps if (e.alinhamento == "justify" and not ultima and gaps) else 0.0
            n_gap = 0
            for k, (x, palavra, b, it, _w, tem_gap) in enumerate(palavras):
                if k > 0 and tem_gap:
                    n_gap += 1
                self.fig.text(self.x0 + desloc + x + n_gap * extra, base, palavra, transform=self.trans,
                              fontproperties=_PROPS[(b, it)], fontsize=e.tamanho, color=e.cor, va="baseline")
            self.y -= e.entrelinha
        self.y -= e.depois

    def _dim_imagem(self, im: Imagem):
        arr = mpimg.imread(str(im.caminho))
        w = min(im.largura_cm * CM, self.largura)
        h = w * arr.shape[0] / arr.shape[1]
        maximo = (self.y_topo - self.y_base) * 0.85
        if h > maximo:
            w, h = w * maximo / h, maximo
        return w, h, arr

    def _imagem(self, im: Imagem):
        w, h, arr = self._dim_imagem(im)
        leg = Paragrafo(im.legenda, im.estilo_legenda or Estilo()) if im.legenda else None
        h_leg = self._altura(leg) if leg else 0
        if self._resta() < h + h_leg:
            self._nova_pagina()
        x = self.x0 + (self.largura - w) / 2
        ax = self.fig.add_axes([x / A4[0], (self.y - h) / A4[1], w / A4[0], h / A4[1]])
        ax.imshow(arr, interpolation="lanczos")
        ax.axis("off")
        self.y -= h + 3
        if leg:
            self._paragrafo(leg)
        self.y -= 6

    def _altura_linha_tab(self, t: Tabela):
        return t.tamanho * 1.25 + 4

    def _larguras_tab(self, t: Tabela):
        if t.larguras_cm:
            ws = [c * CM for c in t.larguras_cm]
        else:
            ncol = len(t.linhas[0])
            ws = [max(largura_texto(lin[c], t.tamanho, negrito=(r == 0)) for r, lin in enumerate(t.linhas)) + 8 for c in range(ncol)]
        total = sum(ws)
        tam = t.tamanho
        if total > self.largura:  # encolhe a fonte até caber
            fator = self.largura / total
            ws = [w * fator for w in ws]
            tam = t.tamanho * fator
        return ws, tam

    def _tabela(self, t: Tabela):
        ws, tam = self._larguras_tab(t)
        hl = tam * 1.25 + 4
        total = sum(ws)
        x_ini = self.x0
        cab, corpo = t.linhas[0], t.linhas[1:]
        if self._resta() < hl * min(len(t.linhas), 4) or (self._resta() < hl * len(t.linhas) and hl * len(t.linhas) < self.y_topo - self.y_base):
            self._nova_pagina()

        def linha(valores, cabecalho, fundo):
            self.fig.add_artist(Rectangle((x_ini, self.y - hl), total, hl, transform=self.trans, facecolor=fundo,
                                          edgecolor="grey", linewidth=0.3))
            x = x_ini
            for c, (v, w) in enumerate(zip(valores, ws)):
                if c > 0:
                    self.fig.add_artist(plt.Line2D([x, x], [self.y - hl, self.y], transform=self.trans, color="grey", linewidth=0.3))
                ha, xt = ("left", x + 3) if c == 0 else ("center", x + w / 2)
                self.fig.text(xt, self.y - hl / 2, v, transform=self.trans, ha=ha, va="center_baseline", fontsize=tam,
                              fontproperties=_PROPS[(cabecalho, False)])
                x += w
            self.y -= hl

        linha(cab, True, "#DDE4F0")
        for k, valores in enumerate(corpo):
            if self._resta() < hl:
                self._nova_pagina()
                linha(cab, True, "#DDE4F0")  # repete o cabeçalho
            linha(valores, False, "#FFFFFF" if k % 2 == 0 else "#F6F7FA")
        self.y -= 8

    # --- API
    def construir(self, blocos: list):
        matplotlib.rcParams["pdf.fonttype"] = 42  # embute a fonte TrueType (texto selecionável)
        with PdfPages(self.destino, metadata={"Title": self.titulo}) as self.pdf:
            self._nova_pagina()
            for k, bloco in enumerate(blocos):
                if isinstance(bloco, QuebraPagina):
                    if self.y < self.y_topo:
                        self._nova_pagina()
                elif isinstance(bloco, Espaco):
                    self.y -= bloco.altura
                elif isinstance(bloco, Paragrafo):
                    prox = blocos[k + 1] if k + 1 < len(blocos) else None
                    minimo = 0.0
                    if bloco.estilo.junto_proximo and prox is not None:
                        # imagem não se divide: o título precisa caber junto com ela inteira
                        h_prox = self._altura(prox) if isinstance(prox, Imagem) else min(self._altura(prox), 60)
                        minimo = self._altura_paragrafo_completo(bloco) - bloco.estilo.entrelinha + h_prox
                    self._paragrafo(bloco, minimo)
                elif isinstance(bloco, Imagem):
                    self._imagem(bloco)
                elif isinstance(bloco, Tabela):
                    self._tabela(bloco)
                else:
                    raise TypeError(f"bloco desconhecido: {bloco!r}")
            self._fecha_pagina()

    def _altura_paragrafo_completo(self, p: Paragrafo):
        return len(_quebrar_linhas(p.texto, p.estilo, self.largura)) * p.estilo.entrelinha + p.estilo.depois
