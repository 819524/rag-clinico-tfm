#!/usr/bin/env python3
"""
estilo_tfm.py — Estilo gráfico común para todas las figuras de la memoria.

Principios:
  * Las figuras se generan YA al tamaño final de impresión (ANCHO_TEXTO) y se
    insertan en LaTeX SIN escalar:  \includegraphics{figures/nombre.pdf}
    Así los tamaños de fuente de la figura son reales (9 pt = 9 pt).
  * Tipografía Latin Modern (la misma que LaTeX) si está instalada en el
    sistema; si no, cae a una serif equivalente. Matemáticas en Computer Modern.
  * Paleta Paul Tol "muted": segura para daltonismo y distinguible en B/N
    si se combina con marcadores (TYPE_MARKER).
  * Sin títulos incrustados en la imagen: el título va en el \caption de LaTeX.

Uso:
    import estilo_tfm as st
    st.aplicar_estilo()
    fig, ax = plt.subplots(figsize=(st.ANCHO_TEXTO, 4.0))
    ...
    st.save(fig, "nombre_figura")
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

# ── Geometría del documento ───────────────────────────────────────────────────
# Ancho de \textwidth en pulgadas. Para conocer el valor exacto de tu memoria,
# añade temporalmente en el .tex:   \showthe\textwidth   (valor en pt / 72.27).
# 6.30 in ≈ 16 cm ≈ habitual en A4 con márgenes de 2.5 cm.
ANCHO_TEXTO = 6.30

OUT_DIR = Path(__file__).resolve().parents[1] / "docs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Tipografía ────────────────────────────────────────────────────────────────
def _serif_disponible():
    candidatas = ["Latin Modern Roman", "CMU Serif", "TeX Gyre Termes",
                  "Times New Roman", "DejaVu Serif"]
    instaladas = {f.name for f in fm.fontManager.ttflist}
    return [c for c in candidatas if c in instaladas] or ["DejaVu Serif"]


def aplicar_estilo():
    plt.rcParams.update({
        # Tipografía — coherente con LaTeX (Latin Modern / Computer Modern)
        "font.family":         "serif",
        "font.serif":          _serif_disponible(),
        "mathtext.fontset":    "cm",
        "axes.unicode_minus":  False,

        # Tamaños pensados para insertar la figura SIN escalar en LaTeX
        "font.size":           9,
        "axes.labelsize":      9,
        "xtick.labelsize":     8,
        "ytick.labelsize":     8,
        "legend.fontsize":     7.5,
        "legend.title_fontsize": 8,

        # Ejes y rejilla sobrios
        "figure.facecolor":    "white",
        "axes.facecolor":      "white",
        "axes.edgecolor":      "#999999",
        "axes.linewidth":      0.7,
        "axes.spines.top":     False,
        "axes.spines.right":   False,
        "axes.grid":           True,
        "grid.color":          "#dddddd",
        "grid.linewidth":      0.5,
        "axes.axisbelow":      True,
        "xtick.direction":     "out",
        "ytick.direction":     "out",
        "xtick.major.size":    3,
        "ytick.major.size":    3,
        "xtick.color":         "#555555",
        "ytick.color":         "#555555",

        # Leyendas
        "legend.framealpha":   0.92,
        "legend.edgecolor":    "#cccccc",
        "legend.fancybox":     False,

        # Exportación
        "savefig.dpi":         300,
        "savefig.bbox":        "tight",
        "savefig.pad_inches":  0.02,
        "pdf.fonttype":        42,
        "ps.fonttype":         42,
        "figure.constrained_layout.use": True,
    })


# ── Paleta por tipo de documento (Paul Tol "muted", apta para daltonismo) ─────
TYPE_COLOR = {
    "GPC": "#332288",
    "PA":  "#117733",
    "PC":  "#CC6677",
    "PE":  "#AA4499",
    "PO":  "#999933",
    "PTP": "#44AA99",
    "RCP": "#88CCEE",
    "RPC": "#882255",
}

TYPE_MARKER = {
    "GPC": "o", "PA": "s", "PC": "^", "PE": "D",
    "PO": "v", "PTP": "P", "RCP": "X", "RPC": "*",
}

TYPE_LABEL = {
    "GPC": "Guía de Práctica Clínica",
    "PA":  "Procedimiento Asistencial",
    "PC":  "Procedimiento Clínico",
    "PE":  "Plan Específico / de Contingencia",
    "PO":  "Procedimiento Operativo",
    "PTP": "Protocolo / Manual Técnico",
    "RCP": "Registro Clínico de Proceso",
    "RPC": "Registro / Protocolo Clínico",
}

# ── Paleta por categoría de bloque (Okabe–Ito, apta para daltonismo) ──────────
BLOCK_CFG = [
    ("paragraph", "Párrafo",    "#0072B2"),
    ("heading",   "Encabezado", "#009E73"),
    ("list",      "Lista",      "#E69F00"),
    ("table",     "Tabla",      "#CC79A7"),
    ("image",     "Imagen",     "#56B4E9"),
    ("other",     "Otros",      "#999999"),
]

# ── Paleta para 22 documentos individuales ────────────────────────────────────
DOC_PALETTE = [
    "#332288", "#88CCEE", "#44AA99", "#117733", "#999933", "#DDCC77",
    "#CC6677", "#882255", "#AA4499", "#0072B2", "#E69F00", "#009E73",
    "#56B4E9", "#D55E00", "#CC79A7", "#661100", "#6699CC", "#888888",
    "#225555", "#997700", "#604E97", "#B2474E",
]
DOC_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "p", "h"]


def fmt_miles(v) -> str:
    """1234 → '1 234' (separador de miles con espacio, estilo europeo)."""
    return f"{int(v):,}".replace(",", " ")


def save(fig, name: str, png: bool = True):
    """Guarda PDF (vectorial, para LaTeX) y opcionalmente PNG de apoyo."""
    fig.savefig(OUT_DIR / f"{name}.pdf")
    if png:
        fig.savefig(OUT_DIR / f"{name}.png")
    plt.close(fig)
    print(f"  ✓  {name}.pdf" + (" / .png" if png else ""))
