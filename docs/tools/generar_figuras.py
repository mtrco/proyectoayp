#!/usr/bin/env python3
"""Genera las figuras SVG de docs/README.md.

    python docs/tools/generar_figuras.py [resultados.json]

Sin argumentos dibuja solo los diagramas. Con el JSON de resultados (lo escribe
docs/tools/resumir_resultados.py) dibuja tambien las graficas de tiempos.
Las figuras usan colores de un tema claro y uno oscuro (prefers-color-scheme), asi que
se leen bien en GitHub con cualquiera de los dos temas.
"""
import json
import math
import os
import sys
from html import escape

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "img")
FONT = "system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace"

# Tema: superficie, texto, lineas y los colores de nodo (operador / variable / constante,
# como en el manual), mas dos colores de serie validados (azul y naranja).
CSS = f"""
text {{ font-family: {FONT}; }}
.mono {{ font-family: {MONO}; }}
.bg {{ fill: #fcfcfb; }}
.card {{ fill: #f4f3ef; stroke: #e2e1dc; }}
.t {{ fill: #0b0b0b; }}
.t2 {{ fill: #52514e; }}
.t3 {{ fill: #7a7974; }}
.edge {{ stroke: #9b9a94; stroke-width: 2; fill: none; }}
.edge2 {{ stroke: #c9c8c2; stroke-width: 2; fill: none; }}
.grid {{ stroke: #e6e5e0; stroke-width: 1; }}
.axis {{ stroke: #b9b8b2; stroke-width: 1; }}
.op {{ fill: #6366f1; }} .opt {{ fill: #ffffff; }}
.var {{ fill: #3fb874; }} .vart {{ fill: #08321b; }}
.cst {{ fill: #d3d4de; }} .cstt {{ fill: #23242a; }}
.fold {{ fill: #f6c445; }} .foldt {{ fill: #3a2a00; }}
.s1 {{ fill: #2a78d6; }} .s2 {{ fill: #eb6834; }} .s3 {{ fill: #1baf7a; }}
.s1s {{ stroke: #2a78d6; }} .s2s {{ stroke: #eb6834; }} .s3s {{ stroke: #1baf7a; }}
.o1 {{ fill: #86b6ef; }} .o2 {{ fill: #2a78d6; }} .o3 {{ fill: #184f95; }}
.ring {{ fill: none; stroke-width: 3; }}
.dash {{ fill: none; stroke-width: 2; stroke-dasharray: 6 5; }}
.hl {{ fill: #eb6834; fill-opacity: 0.10; stroke: #eb6834; stroke-width: 2; stroke-dasharray: 6 5; }}
.hlb {{ fill: #2a78d6; fill-opacity: 0.08; stroke: #2a78d6; stroke-width: 2; stroke-dasharray: 6 5; }}
.good {{ fill: #0ca30c; }} .bad {{ fill: #d03b3b; }}
.goods {{ stroke: #0ca30c; }} .bads {{ stroke: #d03b3b; }}
.cell {{ fill: #ffffff; stroke: #cfcec8; }}
.arrow {{ stroke: #7a7974; stroke-width: 2; fill: none; marker-end: url(#ah); }}
.ah {{ fill: #7a7974; }}
@media (prefers-color-scheme: dark) {{
  .bg {{ fill: #1a1a19; }}
  .card {{ fill: #242423; stroke: #3a3a37; }}
  .t {{ fill: #f0efec; }}
  .t2 {{ fill: #c3c2b7; }}
  .t3 {{ fill: #9d9c95; }}
  .edge {{ stroke: #8f8e88; }}
  .edge2 {{ stroke: #4a4a46; }}
  .grid {{ stroke: #2f2f2d; }}
  .axis {{ stroke: #55554f; }}
  .op {{ fill: #7c7ff5; }}
  .var {{ fill: #2f9e63; }} .vart {{ fill: #ecfff3; }}
  .cst {{ fill: #55565f; }} .cstt {{ fill: #f2f2f6; }}
  .fold {{ fill: #c99a1c; }} .foldt {{ fill: #1d1500; }}
  .s1 {{ fill: #3987e5; }} .s2 {{ fill: #d95926; }} .s3 {{ fill: #199e70; }}
  .s1s {{ stroke: #3987e5; }} .s2s {{ stroke: #d95926; }} .s3s {{ stroke: #199e70; }}
  .o1 {{ fill: #184f95; }} .o2 {{ fill: #3987e5; }} .o3 {{ fill: #86b6ef; }}
  .hl {{ fill: #d95926; stroke: #d95926; }}
  .hlb {{ fill: #3987e5; stroke: #3987e5; }}
  .cell {{ fill: #20201f; stroke: #45453f; }}
  .arrow {{ stroke: #9d9c95; }}
  .ah {{ fill: #9d9c95; }}
}}
"""


class Svg:
    def __init__(self, w, h, title):
        self.w, self.h = w, h
        self.parts = []
        self.title = title

    def add(self, s):
        self.parts.append(s)

    def text(self, x, y, s, cls="t", size=14, anchor="start", weight="normal", extra=""):
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" font-size="{size}" '
                 f'text-anchor="{anchor}" font-weight="{weight}" {extra}>{escape(str(s))}</text>')

    def lines(self, x, y, rows, cls="t2", size=13, gap=18, anchor="start", weight="normal"):
        for k, row in enumerate(rows):
            self.text(x, y + k * gap, row, cls, size, anchor, weight)

    def rect(self, x, y, w, h, cls="card", rx=10, extra=""):
        self.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" class="{cls}" {extra}/>')

    def line(self, x1, y1, x2, y2, cls="edge", extra=""):
        self.add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" class="{cls}" {extra}/>')

    def path(self, d, cls="edge", extra=""):
        self.add(f'<path d="{d}" class="{cls}" {extra}/>')

    def arrow(self, x1, y1, x2, y2, cls="arrow", curve=0.0):
        if curve:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            nx, ny = -(y2 - y1), (x2 - x1)
            L = math.hypot(nx, ny) or 1
            cx, cy = mx + curve * nx / L, my + curve * ny / L
            self.path(f"M{x1:.1f},{y1:.1f} Q{cx:.1f},{cy:.1f} {x2:.1f},{y2:.1f}", cls)
        else:
            self.line(x1, y1, x2, y2, cls)

    def save(self, name):
        os.makedirs(OUT, exist_ok=True)
        body = "\n".join(self.parts)
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
               f'viewBox="0 0 {self.w} {self.h}" role="img" aria-label="{escape(self.title)}">\n'
               f'<title>{escape(self.title)}</title>\n<style>{CSS}</style>\n'
               f'<defs><marker id="ah" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" '
               f'markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="ah"/></marker></defs>\n'
               f'<rect width="{self.w}" height="{self.h}" rx="14" class="bg"/>\n{body}\n</svg>\n')
        with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
            f.write(svg)


# ---------------------------------------------------------------------------
# Arboles
# ---------------------------------------------------------------------------
SYMBOL = {"+": "+", "-": "−", "*": "×", "/": "÷", "sqrt": "√"}


def kind_of(spec):
    if isinstance(spec, tuple):
        return "op"
    if isinstance(spec, str) and spec.startswith("x"):
        return "var"
    if isinstance(spec, str) and spec.startswith("="):
        return "fold"
    return "cst"


def layout(spec):
    """Posiciones (columna, profundidad) con columnas por hoja, en preorden."""
    nodes = []
    leaf = [0]

    def rec(s, depth, path):
        k = kind_of(s)
        idx = len(nodes)
        nodes.append(None)
        if k == "op":
            kids = [rec(c, depth + 1, path + ("L" if j == 0 else "R")) for j, c in enumerate(s[1:])]
            col = sum(nodes[c]["col"] for c in kids) / len(kids)
            label = SYMBOL[s[0]]
        else:
            kids = []
            col = leaf[0]
            leaf[0] += 1
            label = s[1:] if k == "fold" else (s if k == "var" else fmt_const(s))
        nodes[idx] = dict(col=col, depth=depth, kids=kids, kind=k, label=label, path=path)
        return idx

    rec(spec, 0, "")
    return nodes, leaf[0]


def fmt_const(v):
    if isinstance(v, str):
        return v
    s = f"{v:g}"
    return s.replace("-", "−")


def draw_tree(svg, spec, x0, y0, dx=46, dy=62, r=17, styles=None, label_size=None,
              depth_labels=False, edge_cls="edge", hide=(), boxes=()):
    """Dibuja un arbol. `styles`: {ruta: 'new'|'dim'}; `boxes`: [(ruta, clase, margen)]
    recuadros detras de un subarbol. Devuelve ({ruta: (x, y)}, nodos, hojas)."""
    styles = styles or {}
    nodes, nleaf = layout(spec)
    pos = {}
    for n in nodes:
        pos[n["path"]] = (x0 + n["col"] * dx, y0 + n["depth"] * dy)
    for path, cls, pad in boxes:
        subtree_box(svg, pos, nodes, path, cls, pad)
    for n in nodes:
        if n["path"] in hide:
            continue
        x, y = pos[n["path"]]
        for c in n["kids"]:
            cp = nodes[c]["path"]
            if cp in hide:
                continue
            cx, cy = pos[cp]
            cls = "edge2" if styles.get(cp) == "dim" else edge_cls
            svg.line(x, y, cx, cy, cls)
    for n in nodes:
        if n["path"] in hide:
            continue
        x, y = pos[n["path"]]
        k = n["kind"]
        fill, tcls = {"op": ("op", "opt"), "var": ("var", "vart"), "cst": ("cst", "cstt"),
                      "fold": ("fold", "foldt")}[k]
        st = styles.get(n["path"])
        op = ' opacity="0.35"' if st == "dim" else ""
        svg.add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" class="{fill}"{op}/>')
        if st == "new":
            svg.add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r + 4}" class="ring s2s"/>')
        size = label_size or (16 if k == "op" else 13)
        if k == "cst" and len(n["label"]) > 4:
            size = 11
        svg.text(x, y + size * 0.36, n["label"], tcls, size, "middle", "600" if k != "cst" else "500",
                 extra=op)
    if depth_labels:
        maxd = max(n["depth"] for n in nodes)
        for d in range(maxd + 1):
            svg.text(x0 - 40, y0 + d * dy + 4, f"prof. {d}", "t3", 11, "end")
    return pos, nodes, nleaf


def subtree_box(svg, pos, nodes, path, cls="hl", pad=24):
    pts = [pos[n["path"]] for n in nodes if n["path"].startswith(path)]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    svg.rect(min(xs) - pad, min(ys) - pad, max(xs) - min(xs) + 2 * pad, max(ys) - min(ys) + 2 * pad,
             cls, rx=pad)


def legend_nodes(svg, x, y, items=("op", "var", "cst"), size=12):
    names = {"op": ("Operador", "op"), "var": ("Variable", "var"), "cst": ("Constante", "cst"),
             "fold": ("Subárbol plegado", "fold")}
    cx = x
    for it in items:
        label, cls = names[it]
        svg.add(f'<circle cx="{cx + 7}" cy="{y - 4}" r="7" class="{cls}"/>')
        svg.text(cx + 20, y, label, "t2", size)
        cx += 34 + len(label) * 7.2


def title(svg, s, sub=None, x=28, y=38):
    svg.text(x, y, s, "t", 19, weight="700")
    if sub:
        svg.text(x, y + 22, sub, "t2", 13)


# ---------------------------------------------------------------------------
# Diagramas
# ---------------------------------------------------------------------------
EXAMPLE = ("*", ("+", "x2", ("sqrt", ("-", "x1", 0.1))), ("/", "x3", 0.3))


def fig_arbol():
    s = Svg(880, 460, "Un individuo es un árbol")
    title(s, "Un individuo es un árbol", "Los nodos internos son operaciones; las hojas, variables o constantes.")
    pos, nodes, _ = draw_tree(s, EXAMPLE, 150, 120, dx=62, dy=66, r=20, depth_labels=True)
    legend_nodes(s, 60, 440)
    s.rect(480, 100, 372, 268)
    s.text(500, 132, "Se lee de abajo hacia arriba", "t", 15, weight="700")
    s.lines(500, 160, [
        "1. Cada hoja da un número: una variable",
        "   (x1, x2, x3 vienen de la fila de datos)",
        "   o una constante (0.1, 0.3).",
        "2. Cada operador combina el valor de sus",
        "   hijos: + − × ÷ con dos hijos, √ con uno.",
        "3. La raíz da la predicción del árbol.",
    ], "t2", 13, 20)
    s.text(500, 300, "Expresión que genera expression():", "t2", 12)
    s.text(500, 324, "((x2 + sqrt((x1 - 0.1))) * (x3 / 0.3))", "t mono", 12.5)
    s.text(500, 350, "Profundidad del árbol: 4 (la raíz está en 0).", "t2", 12)
    s.save("01_arbol.svg")


def fig_eval_mse():
    s = Svg(880, 400, "Fitness: error cuadrático medio, fila por fila")
    title(s, "Fitness = MSE, evaluando fila por fila",
          "Ejemplo del manual: el árbol (x1 + x2 × 0.5) contra 3 filas de datos.")
    tree = ("+", "x1", ("*", "x2", 0.5))
    draw_tree(s, tree, 70, 120, dx=46, dy=60, r=17)
    cols = ["x1", "x2", "x3", "y", "ŷ = árbol(fila)", "(y − ŷ)²"]
    rows = [["1.0", "2.0", "0.0", "2.5", "1 + 2×0.5 = 2.0", "0.25"],
            ["0.0", "1.0", "1.0", "1.0", "0 + 1×0.5 = 0.5", "0.25"],
            ["2.0", "0.5", "1.0", "3.5", "2 + 0.5×0.5 = 2.25", "1.5625"]]
    widths = [52, 52, 52, 52, 170, 92]
    x0, y0 = 250, 100
    xs = [x0]
    for w in widths[:-1]:
        xs.append(xs[-1] + w)
    for j, c in enumerate(cols):
        s.text(xs[j] + widths[j] / 2, y0, c, "t2", 12.5, "middle", "600")
    for i, row in enumerate(rows):
        y = y0 + 16 + i * 40
        for j, v in enumerate(row):
            s.rect(xs[j] + 2, y, widths[j] - 4, 32, "cell", rx=6)
            cls = "t mono" if j < 4 else ("t" if j == 4 else "t mono")
            s.text(xs[j] + widths[j] / 2, y + 21, v, cls, 12.5, "middle")
    s.arrow(200, 160, 245, 160)
    s.text(250, 268, "MSE = (0.25 + 0.25 + 1.5625) / 3 = 0.6875", "t", 15, weight="700")
    s.lines(250, 296, [
        "Menor es mejor: 0 significa que el árbol reproduce los datos exactamente.",
        "Si algún resultado no es finito (desborde), el MSE vale +∞ y el árbol pierde.",
        "Se evalúa una fila a la vez (está prohibido evaluar por lotes).",
    ], "t2", 13, 20)
    s.save("02_mse.svg")


def fig_grow():
    s = Svg(880, 420, "grow_full y grow_random")
    title(s, "Crear árboles al azar: grow_full y grow_random")
    full = ("-", ("+", "x1", "x2"), ("*", 0.4, ("sqrt", "x2")))
    full = ("-", ("+", "x1", "x2"), ("+", 0.4, "x1"))
    rnd = ("*", "x2", ("-", ("sqrt", "x1"), 0.7))
    s.text(70, 92, "grow_full(2): todas las hojas a profundidad 2", "t", 14, weight="600")
    draw_tree(s, full, 120, 140, dx=50, dy=64, r=17, depth_labels=True)
    s.text(500, 92, "grow_random(3): profundidad de A LO MÁS 3", "t", 14, weight="600")
    draw_tree(s, rnd, 560, 140, dx=56, dy=58, r=17)
    s.lines(70, 378, [
        "Reglas comunes de A, B y C: operador 1/5 cada uno (+ − × ÷ √); una hoja es variable o constante con 1/2;",
        "las constantes son uniformes en [−1, 1); grow_random corta antes de tiempo con probabilidad 1/4.",
    ], "t2", 12.5, 19)
    s.save("03_grow.svg")


A_TREE = ("+", "x1", ("*", "x2", 0.4))
B_TREE = ("-", ("sqrt", "x3"), "x4")


def fig_crossover():
    s = Svg(880, 610, "Crossover convencional contra copia de camino")
    title(s, "Crossover: copiar todo (A y B) o compartir (C)",
          "Se intercambia el subárbol marcado de cada padre. Los padres nunca cambian.")
    # Panel izquierdo: convencional
    s.rect(20, 82, 410, 510)
    s.text(40, 110, "Convencional: clonar y luego intercambiar", "t", 14, weight="700")
    pa, na, _ = draw_tree(s, A_TREE, 80, 160, dx=44, dy=56, r=16, boxes=[("R", "hl", 24)])
    pb, nb, _ = draw_tree(s, B_TREE, 270, 160, dx=44, dy=56, r=16, boxes=[("L", "hlb", 24)])
    s.text(124, 318, "padre A", "t3", 12, "middle")
    s.text(292, 318, "padre B", "t3", 12, "middle")
    s.arrow(210, 330, 210, 356)
    c1 = ("+", "x1", ("sqrt", "x3"))
    c2 = ("-", ("*", "x2", 0.4), "x4")
    allnew = {p: "new" for p in ["", "L", "R", "RL", "RR", "LL", "LR"]}
    draw_tree(s, c1, 80, 390, dx=44, dy=50, r=15, styles=allnew)
    draw_tree(s, c2, 270, 390, dx=44, dy=50, r=15, styles=allnew)
    s.text(124, 532, "hijo 1", "t3", 12, "middle")
    s.text(292, 532, "hijo 2", "t3", 12, "middle")
    s.lines(40, 560, ["Todos los nodos de los hijos son nuevos (anillo naranja):",
                      "clonar cuesta tanto como el tamaño de los padres."], "t2", 12, 17)
    # Panel derecho: copia de camino
    s.rect(450, 82, 410, 510)
    s.text(470, 110, "C: copia de camino (se comparten nodos)", "t", 14, weight="700")
    pa2, na2, _ = draw_tree(s, A_TREE, 500, 170, dx=46, dy=58, r=16, boxes=[("R", "hl", 24)])
    pb2, nb2, _ = draw_tree(s, B_TREE, 700, 170, dx=46, dy=58, r=16, boxes=[("L", "hlb", 24)])
    s.text(546, 330, "padre A", "t3", 12, "middle")
    s.text(722, 330, "padre B", "t3", 12, "middle")
    # hijo 1: nueva raiz + que apunta a x1 de A y al sqrt de B
    hx, hy = 620, 430
    s.add(f'<circle cx="{hx}" cy="{hy}" r="17" class="op"/>')
    s.add(f'<circle cx="{hx}" cy="{hy}" r="21" class="ring s2s"/>')
    s.text(hx, hy + 6, "+", "opt", 16, "middle", "600")
    s.arrow(hx - 10, hy - 14, pa2["L"][0] + 6, pa2["L"][1] + 16, curve=-18)
    s.arrow(hx + 12, hy - 13, pb2["L"][0] - 8, pb2["L"][1] + 15, curve=20)
    s.text(hx, hy + 44, "hijo 1 = 1 nodo nuevo", "t2", 12, "middle")
    s.lines(470, 560, ["El hijo reutiliza x1 de A y √x3 de B tal cual: solo se crean",
                       "los nodos del camino de la raíz al corte (≤ 1 por nivel)."], "t2", 12, 17)
    s.save("04_crossover.svg")


def fig_path_copy_depth():
    s = Svg(880, 300, "Nodos creados por crossover según la profundidad")
    title(s, "¿Cuántos nodos crea un crossover?", "Árboles completos (grow_full), 2 hijos por cruce.")
    rows = [(7, 136), (9, 446), (10, 800), (12, 2600), (15, 15200)]
    x0, y0, bw = 170, 100, 560
    maxv = math.log10(2 * 15200)
    for i, (d, n) in enumerate(rows):
        y = y0 + i * 36
        s.text(x0 - 14, y + 15, f"profundidad {d}", "t2", 12.5, "end")
        conv = 2 * n
        path = 2 * d
        w1 = bw * math.log10(conv) / maxv
        w2 = bw * math.log10(path) / maxv
        s.rect(x0, y, w1, 10, "s2", rx=3)
        s.rect(x0, y + 13, w2, 10, "s1", rx=3)
        s.text(x0 + w1 + 8, y + 9, f"{conv:,} copiados (A, B)".replace(",", ","), "t2", 11)
        s.text(x0 + w2 + 8, y + 22, f"≤ {path} nuevos (C)", "t2", 11)
    s.text(x0, 292, "Escala logarítmica. En profundidad 15, C crea ~1,000 veces menos nodos que A o B.", "t3", 12)
    s.save("05_nodos_creados.svg")


def fig_protected():
    s = Svg(880, 430, "Crossover y mutación protegidos por profundidad")
    title(s, "Versiones protegidas: el hijo nunca supera max_depth")
    s.rect(20, 66, 410, 346)
    s.text(40, 94, "Crossover protegido (max_depth = 3)", "t", 14, weight="700")
    t = ("+", ("*", "x1", 0.5), ("-", "x2", ("sqrt", "x1")))
    pos, nodes, _ = draw_tree(s, t, 110, 140, dx=42, dy=52, r=15, depth_labels=True,
                              boxes=[("RR", "hl", 20)])
    s.lines(40, 344, [
        "Cortar en i y poner el subárbol j del otro padre deja",
        "al hijo con profundidad prof(i) + altura(j).",
        "Se acepta si prof_A(i) + altura_B(j) ≤ max y al revés.",
        "Hasta 128 intentos; si ninguno sirve, se intercambian raíces.",
    ], "t2", 12.5, 18)
    s.text(300, 160, "corte i: prof(i) = 2", "t", 13, weight="600")
    s.text(300, 182, "admite altura(j) ≤ 1", "t2", 12)
    s.rect(450, 66, 410, 346)
    s.text(470, 94, "Mutación protegida (max_depth = 3)", "t", 14, weight="700")
    t2 = ("+", "x1", ("*", "x2", 0.4))
    pos2, nodes2, _ = draw_tree(s, t2, 540, 150, dx=50, dy=60, r=16, depth_labels=True,
                                boxes=[("R", "hl", 22)])
    s.text(690, 150, "presupuesto =", "t2", 12)
    s.text(690, 170, "max − prof(i) = 3 − 1 = 2", "t", 13, weight="600")
    s.arrow(690, 196, 690, 226)
    rep = ("-", "x3", -0.7)
    draw_tree(s, rep, 690, 250, dx=44, dy=48, r=15, styles={"": "new", "L": "new", "R": "new"})
    s.lines(470, 344, [
        "El subárbol nuevo sale de grow_random con una",
        "profundidad al azar entre 0 y el presupuesto.",
        "Cerca de la mitad de los nodos son hojas al fondo:",
        "ahí el presupuesto es 0 y la hoja cambia por otra hoja.",
    ], "t2", 12.5, 18)
    s.save("06_protegidos.svg")


def fig_gp_loop():
    s = Svg(880, 360, "Una generación del GP completo")
    title(s, "Una generación del GP completo (población N = 5,000)")
    boxes = [
        ("Población", "N árboles con su MSE"),
        ("Torneo binario", "N veces: 2 al azar,\ngana el de menor MSE"),
        ("Variación 50/50", "N/4 cruces → N/2 hijos\nN/2 mutaciones → N/2 hijos"),
        ("Evaluar hijos", "MSE de los N hijos"),
        ("Reemplazo 50/50", "mejor 50% de la población\n+ mejor 50% de los hijos"),
    ]
    x, y, w, h, gap = 26, 92, 152, 112, 18
    for k, (t, sub) in enumerate(boxes):
        bx = x + k * (w + gap)
        s.rect(bx, y, w, h)
        s.text(bx + w / 2, y + 30, t, "t", 14, "middle", "700")
        for j, row in enumerate(sub.split("\n")):
            s.text(bx + w / 2, y + 58 + j * 18, row, "t2", 11.5, "middle")
        if k < len(boxes) - 1:
            s.arrow(bx + w + 2, y + h / 2, bx + w + gap - 2, y + h / 2)
    s.path(f"M{x + 4 * (w + gap) + w / 2:.1f},{y + h + 4} C{x + 4 * (w + gap) + w / 2:.1f},{y + h + 60} "
           f"{x + w / 2:.1f},{y + h + 60} {x + w / 2:.1f},{y + h + 6}", "arrow")
    s.text(440, y + h + 74, "se repite G generaciones (el benchmark usa 10)", "t3", 12, "middle")
    s.lines(26, 318, [
        "Inicio: 100% grow_full. Elitismo: el mejor de la población está siempre en su mejor 50%, así que el",
        "mejor MSE nunca empeora. Orden estable en A, B y C: en un empate gana el que llegó primero.",
    ], "t2", 12.5, 19)
    s.save("07_gp.svg")


def fig_branch():
    s = Svg(880, 380, "Evaluación con saltos contra sin saltos")
    title(s, "Evaluación sin saltos: calcular de más para no adivinar",
          "El procesador adivina hacia dónde va cada if/switch. Si se equivoca, tira ~15–20 ciclos.")
    s.rect(20, 86, 410, 270)
    s.text(40, 114, "Con salto (antes): switch por tipo de nodo", "t", 14, weight="700")
    ops = ["+", "−", "×", "÷"]
    s.text(225, 150, "¿qué operador es?", "t2", 12.5, "middle")
    s.rect(165, 160, 120, 34, "cell", rx=8)
    s.text(225, 182, "switch (kind)", "t mono", 12.5, "middle")
    for j, o in enumerate(ops):
        bx = 70 + j * 88
        s.arrow(225, 196, bx + 26, 236)
        s.rect(bx, 238, 52, 34, "cell", rx=8)
        s.text(bx + 26, 260, "a" + o + "b", "t mono", 13, "middle")
    s.lines(40, 300, ["En un árbol de miles de nodos el orden de los operadores",
                      "es impredecible: el procesador falla seguido."], "t2", 12.5, 18)
    s.rect(450, 86, 410, 270)
    s.text(470, 114, "Sin salto (ahora): calcular las 4 y elegir", "t", 14, weight="700")
    for j, o in enumerate(ops):
        bx = 480 + j * 92
        s.rect(bx, 140, 70, 34, "cell", rx=8)
        s.text(bx + 35, 162, f"r[{j}]=a{o}b", "t mono", 11.5, "middle")
        s.arrow(bx + 35, 176, 655, 222)
    s.rect(565, 224, 180, 34, "cell", rx=8)
    s.text(655, 246, "return r[kind − Add]", "t mono", 12.5, "middle")
    s.lines(470, 300, ["Las mismas operaciones IEEE: resultado idéntico bit a bit.",
                       "En profundidad 12 la evaluación de B pasa de 5.6 a 2.9 ns",
                       "por nodo y fila (g++) y a 2.0 ns con clang."], "t2", 12.5, 18)
    s.save("08_sin_saltos.svg")


def fig_fold():
    s = Svg(880, 390, "Plegado de constantes")
    title(s, "Plegado de constantes: calcular una vez lo que no depende de la fila")
    t = ("*", ("+", ("sqrt", 0.25), 0.1), ("-", "x1", ("/", 1, 0)))
    pos, nodes, _ = draw_tree(s, t, 80, 110, dx=50, dy=56, r=16,
                              boxes=[("L", "hl", 22), ("RR", "hl", 20)])
    s.arrow(400, 200, 470, 200)
    t2 = ("*", "=0.6", ("-", "x1", "=1e12"))
    draw_tree(s, t2, 560, 160, dx=60, dy=60, r=19)
    legend_nodes(s, 520, 350, items=("op", "var", "fold"))
    s.lines(40, 350, ["Sin variables, el subárbol vale lo mismo en todas las filas.",
                      "Se calcula una vez con las mismas operaciones protegidas."], "t2", 12.5, 18)
    s.text(500, 100, "La evaluación lee 0.6 y 1e12 sin recorrer nada.", "t2", 12.5)
    s.text(500, 120, "El árbol (expresión, tamaño, cortes) no cambia.", "t2", 12.5)
    s.save("09_plegado.svg")


def fig_cut():
    s = Svg(880, 430, "Corte temprano exacto")
    title(s, "Corte temprano exacto: dejar de evaluar a quien ya perdió",
          "MSE parcial (suma de errores hasta la fila / n) de tres hijos. Curvas ilustrativas.")
    x0, y0, w, h = 90, 100, 600, 220
    rows, vmax, T = 49, 0.8, 0.42
    s.line(x0, y0 + h, x0 + w, y0 + h, "axis")
    for k in range(0, 5):
        v = k * 0.2
        y = y0 + h - v / vmax * h
        s.line(x0, y, x0 + w, y, "grid")
        s.text(x0 - 8, y + 4, f"{v:.1f}", "t3", 11, "end")
    for r in (1, 10, 20, 30, 40, 49):
        s.text(x0 + (r - 1) / (rows - 1) * w, y0 + h + 18, str(r), "t3", 11, "middle")
    s.text(x0 + w / 2, y0 + h + 36, "fila del dataset", "t2", 12, "middle")

    def cumulative(errors):
        acc, out = 0.0, []
        for e in errors:
            acc += e
            out.append(acc / rows)
        return out
    a = cumulative([0.25 + 0.15 * math.sin(r) for r in range(rows)])          # termina en ~0.25
    b = cumulative([0.70 + 0.30 * math.sin(1.7 * r) for r in range(rows)])    # rebasa T a la mitad
    c = cumulative([7.0, 6.0, 9.0] + [5.0] * (rows - 3))                         # rebasa T muy pronto
    ty = y0 + h - T / vmax * h
    s.line(x0, ty, x0 + w, ty, "edge", extra='stroke-dasharray="7 5"')
    s.text(x0 + w + 8, ty + 4, "umbral T", "t", 12, weight="600")
    for name, pts, cls, dy in (("hijo A: sobrevive", a, "s1s", -12), ("hijo B: se corta en la fila", b, "s2s", -12),
                               ("hijo C: se corta en la fila", c, "s3s", -10)):
        cut = next((r for r, v in enumerate(pts) if v > T), None)
        end = cut if cut is not None else rows - 1
        d = " ".join(f"{'M' if r == 0 else 'L'}{x0 + r / (rows - 1) * w:.1f},{y0 + h - pts[r] / vmax * h:.1f}"
                     for r in range(end + 1))
        s.path(d, cls, extra='stroke-width="2.5" fill="none" stroke-linecap="round" stroke-linejoin="round"')
        ex, ey = x0 + end / (rows - 1) * w, y0 + h - pts[end] / vmax * h
        if cut is not None:
            s.add(f'<circle cx="{ex:.1f}" cy="{ey:.1f}" r="5" class="bad"/>')
            s.text(ex + 10, ey + dy, f"{name} {cut + 1}", "t", 12)
        else:
            s.add(f'<circle cx="{ex:.1f}" cy="{ey:.1f}" r="5" class="good"/>')
            s.text(ex - 8, ey - 12, f"{name} (MSE exacto)", "t", 12, "end")
    s.lines(90, 392, ["T = el (N/2)-ésimo mejor MSE entre los hijos ya evaluados; solo puede bajar. La suma solo crece, así que",
                      "quien rebasa T terminaría peor que N/2 hijos y nunca sobreviviría: se deja de evaluar y recibe +∞."],
            "t2", 12, 17)
    s.save("10_corte_temprano.svg")


def fig_memoria():
    s = Svg(880, 360, "Postorden contra preorden en memoria")
    title(s, "Árboles en preorden: recorrer la memoria en orden",
          "Mismo árbol; el número es el orden en que se reservó cada nodo en memoria.")
    t = ("+", ("*", "x1", 0.5), ("-", "x2", 0.3))
    post = {"": 7, "L": 3, "LL": 1, "LR": 2, "R": 6, "RL": 4, "RR": 5}
    pre = {"": 1, "L": 2, "LL": 3, "LR": 4, "R": 5, "RL": 6, "RR": 7}
    for k, (name, order, x0) in enumerate((("Antes: hijos primero (postorden)", post, 70),
                                           ("Ahora: padre primero (preorden)", pre, 510))):
        s.text(x0 - 30, 102, name, "t", 14, weight="700")
        pos, nodes, _ = draw_tree(s, t, x0 + 20, 150, dx=54, dy=58, r=16)
        for p, (x, y) in pos.items():
            s.add(f'<circle cx="{x + 15:.1f}" cy="{y - 15:.1f}" r="9" class="s1" />')
            s.text(x + 15, y - 11, str(order[p]), "opt", 10.5, "middle", "700")
        # cinta de memoria en orden de recorrido (preorden)
        seq = ["", "L", "LL", "LR", "R", "RL", "RR"]
        s.text(x0 - 30, 300, "posiciones que lee el recorrido:", "t2", 12)
        cx = x0 - 30
        for p in seq:
            s.rect(cx, 310, 34, 26, "cell", rx=5)
            s.text(cx + 17, 328, str(order[p]), "t mono", 12, "middle")
            cx += 38
    s.text(70, 352, "Saltos en memoria: el procesador no puede adelantar lecturas.", "t3", 12)
    s.text(510, 352, "1, 2, 3...: el procesador adelanta la siguiente lectura.", "t3", 12)
    s.save("11_preorden.svg")


def fig_nodos_memoria():
    s = Svg(880, 330, "Cómo se guarda un nodo en B y en C")
    title(s, "Un nodo en memoria: B (32 bytes) y C (24 bytes)", "Cada casilla es un byte; los campos de 1 byte van abreviados.")
    scale = 22

    def row(y, header, fields, total):
        s.text(28, y - 12, header, "t", 13.5, weight="600")
        x = 28
        for b, name in fields:
            s.rect(x, y, b * scale - 2, 42, "cell", rx=6)
            s.text(x + (b * scale - 2) / 2, y + 26, name, "t2" if name == "relleno" else "t", 12, "middle")
            x += b * scale
        s.text(x + 10, y + 26, total, "t", 13, weight="700")
    row(112, "B · struct Node: objeto independiente enlazado con unique_ptr",
        [(1, "k"), (1, "d"), (6, "relleno"), (8, "value | var"), (8, "left →"), (8, "right →")], "32 bytes")
    s.text(28, 172, "k = kind (tipo), d = direct (se lee directo). value guarda la constante, el valor plegado o el índice de variable.",
           "t3", 11.5)
    row(222, "C · NodeRec: registro dentro de un Vector (sin punteros: isbits)",
        [(4, "left"), (4, "right"), (4, "size"), (2, "height"), (1, "k"), (1, "f"), (8, "value")], "24 bytes")
    s.text(28, 282, "left y right son POSICIONES en el arreglo (Int32), no direcciones. f = flags (plegado). value: constante,",
           "t3", 11.5)
    s.text(28, 300, "valor plegado o mayor índice de variable del subárbol. Sin punteros, el recolector de basura no revisa nada.",
           "t3", 11.5)
    s.save("12_nodo_memoria.svg")


def fig_grafo():
    s = Svg(880, 440, "Toda la población en un solo grafo")
    title(s, "C: toda la población en un solo grafo (en un arreglo)",
          "Tres individuos comparten subárboles; cada nodo es una fila de un Vector{NodeRec}.")
    # Nodos dibujados a mano para mostrar el DAG
    N = {
        "a": (110, 130, "op", "+"), "b": (250, 130, "op", "+"), "c": (390, 130, "op", "×"),
        "d": (70, 220, "var", "x1"), "e": (180, 220, "op", "×"), "f": (330, 220, "op", "√"),
        "g": (140, 310, "var", "x2"), "h": (220, 310, "cst", "0.4"), "i": (330, 310, "var", "x3"),
    }
    E = [("a", "d"), ("a", "e"), ("b", "d"), ("b", "f"), ("c", "e"), ("c", "f"), ("e", "g"), ("e", "h"), ("f", "i")]
    for u, v in E:
        s.line(N[u][0], N[u][1], N[v][0], N[v][1], "edge")
    for k, (x, y, kind, lab) in N.items():
        cls, tc = {"op": ("op", "opt"), "var": ("var", "vart"), "cst": ("cst", "cstt")}[kind]
        s.add(f'<circle cx="{x}" cy="{y}" r="18" class="{cls}"/>')
        s.text(x, y + 5, lab, tc, 14 if kind == "op" else 12.5, "middle", "600")
    for k, name in (("a", "individuo 1"), ("b", "individuo 2"), ("c", "individuo 3")):
        s.text(N[k][0], N[k][1] - 28, name, "t2", 12, "middle", "600")
    s.text(70, 372, "Las raíces son índices; todo lo demás se comparte.", "t2", 12.5)
    # tabla de registros
    tx, ty = 480, 92
    s.text(tx, ty, "nodes::Vector{NodeRec}", "t mono", 13, weight="600")
    head = ["#", "kind", "left", "right", "size"]
    ws = [34, 60, 56, 56, 52]
    rows = [("1", "var", "x1", "—", "1"), ("2", "var", "x2", "—", "1"), ("3", "const", "—", "—", "1"),
            ("4", "×", "2", "3", "3"), ("5", "+", "1", "4", "5"), ("6", "var", "x3", "—", "1"),
            ("7", "√", "6", "—", "2"), ("8", "+", "1", "7", "4"), ("9", "×", "4", "7", "6")]
    xs = [tx]
    for w in ws[:-1]:
        xs.append(xs[-1] + w)
    for j, hname in enumerate(head):
        s.text(xs[j] + ws[j] / 2, ty + 24, hname, "t2", 11.5, "middle", "600")
    for i, row in enumerate(rows):
        y = ty + 32 + i * 30
        for j, v in enumerate(row):
            s.rect(xs[j] + 1, y, ws[j] - 2, 26, "cell", rx=4)
            s.text(xs[j] + ws[j] / 2, y + 17, v, "t mono", 11.5, "middle")
    s.text(tx + 268, ty + 32 + 4 * 30 + 17, "← raíz 1", "t3", 11)
    s.text(tx + 268, ty + 32 + 7 * 30 + 17, "← raíz 2", "t3", 11)
    s.text(tx + 268, ty + 32 + 8 * 30 + 17, "← raíz 3", "t3", 11)
    s.text(70, 400, "Ventajas: sin un objeto por nodo (casi nada de recolección de basura), registros de 24 bytes", "t3", 12)
    s.text(70, 418, "contiguos, e hijos de crossover que solo agregan unas cuantas filas al final del arreglo.", "t3", 12)
    s.save("13_grafo.svg")


def fig_cache():
    s = Svg(880, 400, "Cache de resultados por nodo")
    title(s, "Cache de resultados: el hijo reutiliza lo que ya calcularon sus padres",
          "Los nodos con tamaño ≥ S guardan su valor en cada una de las n filas.")
    pa = ("+", ("*", ("+", "x1", 0.3), ("-", "x2", "x1")), ("sqrt", ("*", "x2", 0.5)))
    pos, nodes, _ = draw_tree(s, pa, 70, 120, dx=46, dy=56, r=15, boxes=[("L", "hlb", 20)])
    lx, ly = pos["L"]
    s.text(36, 368, "subárbol con cache (tamaño ≥ S)", "t2", 12)
    # vector de cache
    cx, cy = 420, 110
    s.text(cx, cy, "cache del subárbol azul, una entrada por fila:", "t2", 12.5)
    vals = ["0.91", "−2.4", "0.07", "1.33", "…", "0.52"]
    for j, v in enumerate(vals):
        s.rect(cx + j * 58, cy + 12, 54, 30, "cell", rx=5)
        s.text(cx + j * 58 + 27, cy + 32, v, "t mono", 12, "middle")
    for j, lab in enumerate(["fila 1", "fila 2", "fila 3", "fila 4", "", "fila n"]):
        s.text(cx + j * 58 + 27, cy + 58, lab, "t3", 11, "middle")
    s.text(cx, 220, "Hijo de una mutación en la rama derecha:", "t", 13.5, weight="600")
    s.lines(cx, 244, [
        "• el subárbol izquierdo es EL MISMO nodo del padre: en cada fila",
        "  se lee su valor del cache (1 lectura en lugar de 7 nodos);",
        "• solo se calculan la raíz nueva y la rama que cambió;",
        "• debajo de S se evalúa directo (no vale la pena guardar).",
        "El cache se llena en la primera evaluación, se marca completo al",
        "terminar y se descarta si el MSE resulta +∞.",
    ], "t2", 12.5, 19)
    s.save("14_cache.svg")


def fig_pool():
    s = Svg(880, 300, "Asignador de nodos de B")
    title(s, "B: asignador propio (NodePool) en lugar del heap del sistema")
    x0, y0 = 40, 100
    s.text(x0, y0 - 10, "bloque de 2 MB = 65,536 casillas de 32 bytes", "t2", 12.5)
    for j in range(16):
        cls = "s1" if j < 9 else "cell"
        s.rect(x0 + j * 40, y0, 36, 30, cls, rx=4)
    s.text(x0 + 9 * 40 + 4, y0 + 52, "↑ siguiente casilla libre (next_)", "t2", 12)
    s.text(x0, 200, "Lista libre (LIFO): lo último liberado se reutiliza primero (aún está en cache).", "t2", 12.5)
    xs = [40, 170, 300]
    for k, x in enumerate(xs):
        s.rect(x, 214, 90, 30, "s2", rx=4)
        s.text(x + 45, 234, "libre", "opt", 12, "middle", "600")
        if k < len(xs) - 1:
            s.arrow(x + 92, 229, xs[k + 1] - 4, 229)
    s.text(40, 280, "Pedir un nodo = tomar la cabeza de la lista o avanzar next_. Liberarlo = ponerlo al frente.", "t3", 12)
    s.save("15_pool.svg")


# ---------------------------------------------------------------------------
# Graficas de resultados
# ---------------------------------------------------------------------------
def bar_groups(name, title_s, subtitle, groups, series, values, unit_fmt, note=None, logscale=False,
               x0=170):
    """Barras horizontales agrupadas. values[g][s] = numero (o None). Con una sola serie no
    hay leyenda (el titulo la nombra)."""
    gh = 26 * len(series) + 22
    h = 110 + gh * len(groups) + (44 if note else 20)
    s = Svg(880, h, title_s)
    title(s, title_s, subtitle)
    y0, bw = 96, 730 - x0
    vmax = max(v for row in values for v in row if v is not None)
    def scale(v):
        if logscale:
            return bw * math.log10(1 + v) / math.log10(1 + vmax)
        return bw * v / vmax
    # leyenda
    lx = x0
    for k, (sname, cls) in enumerate(series if len(series) > 1 else []):
        s.rect(lx, y0 - 22, 12, 12, cls, rx=2)
        s.text(lx + 18, y0 - 12, sname, "t2", 12.5)
        lx += 40 + len(sname) * 7.4
    for gi, g in enumerate(groups):
        gy = y0 + 8 + gi * gh
        s.text(x0 - 14, gy + 12 * len(series) + 6, g, "t2", 12.5, "end")
        for si, (sname, cls) in enumerate(series):
            v = values[gi][si]
            if v is None:
                continue
            y = gy + si * 26
            w = max(scale(v), 2)
            # extremo redondeado de 4 px y base recta
            s.path(f"M{x0},{y} H{x0 + w - 4:.1f} Q{x0 + w:.1f},{y} {x0 + w:.1f},{y + 4} V{y + 16} "
                   f"Q{x0 + w:.1f},{y + 20} {x0 + w - 4:.1f},{y + 20} H{x0} Z", cls)
            s.text(x0 + w + 8, y + 15, unit_fmt(v), "t", 12)
    s.line(x0, y0, x0, y0 + 8 + gh * len(groups) - 22, "axis")
    if note:
        s.text(28, h - 22, note, "t3", 12)
    s.save(name)


def charts(res):
    gp = res["gp"]
    depths = ["7", "9", "12"]
    def sp(old, new, d):
        return gp[old][d] / gp[new][d]
    impls = [("B · g++", "b_old_gcc", "b_new_gcc"), ("B · clang", "b_old_clang", "b_new_clang"),
             ("C · Julia", "c_old", "c_new")]
    bar_groups("20_gp_aceleracion.svg", "GP completo: cuántas veces más rápido",
               "Mediana de 3 semillas; población 5,000; 10 generaciones; mismo resultado antes y ahora.",
               [name for name, _, _ in impls],
               [(f"profundidad {d}", c) for d, c in zip(depths, ("o1", "o2", "o3"))],
               [[sp(old, new, d) for d in depths] for _, old, new in impls],
               lambda v: f"{v:.2f}×",
               note="Antes = versión del commit 4561dac, compilada y medida en el mismo equipo (contenedor Linux, un hilo).")
    xo = res["crossover"]
    xd = ["7", "9", "10", "12", "15"]
    bar_groups("21_crossover_tiempo.svg", "Benchmark de crossover: microsegundos por cruce",
               "10,000 padres en memoria a la vez; 5,000 cruces; escala logarítmica.",
               [f"profundidad {d}" for d in xd],
               [("B antes", "s2"), ("B ahora", "s1")],
               [[xo["b_old_gcc"][d], xo["b_new_gcc"][d]] for d in xd],
               lambda v: f"{v:,.1f} µs", logscale=True)
    bar_groups("22_crossover_c.svg", "Benchmark de crossover de C: microsegundos por cruce",
               "Copia de camino: el costo casi no crece con la profundidad.",
               [f"profundidad {d}" for d in xd],
               [("C antes", "s2"), ("C ahora", "s1")],
               [[xo["c_old"][d], xo["c_new"][d]] for d in xd],
               lambda v: f"{v:.2f} µs")
    br = res["breakdown_b"]
    bar_groups("23_desglose_b.svg", "B antes, profundidad 12: ¿en qué se iba el tiempo?",
               "Porcentaje del tiempo del GP completo (g++, semilla 1).",
               [k for k, _ in br], [("parte del tiempo", "s1")], [[v] for _, v in br],
               lambda v: f"{v:.0f} %", x0=230)
    bc = res["breakdown_c"]
    bar_groups("24_desglose_c.svg", "C, profundidad 12: población inicial, segundos",
               "Generar y evaluar los 5,000 árboles iniciales (no comparten nada).",
               [k for k, _ in bc[0]], [("antes (objetos)", "s2"), ("ahora (grafo en arreglos)", "s1")],
               [[a, b] for (_, a), (_, b) in zip(bc[0], bc[1])],
               lambda v: f"{v:.2f} s")


def main():
    fig_arbol()
    fig_eval_mse()
    fig_grow()
    fig_crossover()
    fig_path_copy_depth()
    fig_protected()
    fig_gp_loop()
    fig_branch()
    fig_fold()
    fig_cut()
    fig_memoria()
    fig_nodos_memoria()
    fig_grafo()
    fig_cache()
    fig_pool()
    if len(sys.argv) > 1:
        with open(sys.argv[1], encoding="utf-8") as f:
            charts(json.load(f))


if __name__ == "__main__":
    main()
