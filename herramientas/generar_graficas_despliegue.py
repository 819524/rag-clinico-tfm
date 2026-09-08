#!/usr/bin/env python3
"""
generar_graficas_despliegue.py — Figuras del capítulo "Estudio de despliegue".

Produce las dos figuras que faltaban en memoria/Imagenes/:

  1. despliegue_motores.pdf     Latencia y throughput frente a la concurrencia
                                para los tres motores (gemma-4 26B-A4B, embed
                                0.6B, voz05). Mediana de 10 repeticiones con
                                banda intercuartílica.
  2. despliegue_prefixcache.pdf A/B de prefix caching en vLLM hasta 200
                                usuarios concurrentes (tiempo muro y
                                throughput global).

Los datos están embebidos aquí y proceden de los informes ya generados:
  · eval/prueba_carga_usuarios/INFORME_CONFIANZA.html  (matriz x10, objeto P)
  · eval/prueba_carga_usuarios/MEMORIA_TFM.html        (bloque P.pc)
que a su vez se calculan de los sweep.json de confianza/ y prefixcache_AB/.

Ejecutar desde la raíz del proyecto:
    tfm/bin/python Scripts/generar_graficas_despliegue.py
"""

import glob
import json
import os
import shutil
import statistics
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import estilo_tfm as st

# Copia adicional de las figuras junto al resto de imágenes de la memoria.
DIR_MEMORIA = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "memoria", "Imagenes"
)

# ── Motores: color Okabe–Ito + marcador, coherente con las figuras del 5090 ──
MOTORES = [
    ("vLLM", "#0072B2", "o"),
    ("llama.cpp (NP=8)", "#009E73", "s"),
    ("Ollama (NP=8)", "#E69F00", "^"),
]

# ── Figura 1 — matriz de confianza (26B-A4B, embed 0.6B, 10 repeticiones) ────
NIVELES = [1, 5, 10, 15, 20, 25]

LATENCIA = {  # mediana, p25, p75 del tiempo muro (s)
    "vLLM":             ([3.1, 5.6, 7.0, 8.1, 9.6, 10.8],
                         [1.6, 4.9, 6.8, 7.7, 9.1, 10.1],
                         [4.9, 6.4, 8.0, 8.8, 10.2, 11.8]),
    "llama.cpp (NP=8)": ([3.1, 9.8, 14.0, 21.3, 24.8, 31.5],
                         [2.5, 9.2, 13.3, 20.9, 23.7, 29.9],
                         [4.0, 10.3, 15.0, 21.8, 26.4, 33.8]),
    "Ollama (NP=8)":    ([4.4, 13.7, 19.1, 30.4, 42.7, 51.6],
                         [3.1, 12.4, 18.6, 28.7, 39.6, 49.4],
                         [6.3, 16.3, 20.8, 32.8, 44.3, 152.3]),
}

THROUGHPUT = {  # mediana, p25, p75 de los tokens/s globales
    "vLLM":             ([74.9, 269.9, 409.3, 548.5, 642.4, 709.4],
                         [47.2, 250.8, 363.8, 527.0, 615.4, 657.8],
                         [111.9, 296.2, 456.9, 584.3, 663.9, 751.5]),
    "llama.cpp (NP=8)": ([41.7, 141.6, 208.3, 204.8, 242.2, 234.0],
                         [23.5, 127.8, 196.2, 198.6, 227.2, 226.9],
                         [82.5, 147.0, 228.9, 234.0, 256.6, 250.8]),
    "Ollama (NP=8)":    ([47.6, 106.4, 147.0, 149.4, 141.2, 142.4],
                         [19.8, 96.0, 138.2, 123.9, 129.1, 64.5],
                         [78.6, 119.6, 159.0, 159.8, 155.8, 153.1]),
}

# Techo del eje de latencia: la banda de Ollama a 25u llega a 152 s (cola
# bimodal por el atasco de la sliding-window attention) y aplastaría el resto
# de las curvas. Se recorta y se anota explícitamente.
TOPE_LATENCIA = 62
IQR_OLLAMA_25U = 152.3


def figura_motores():
    fig, (ax_lat, ax_thr) = plt.subplots(
        1, 2, figsize=(st.ANCHO_TEXTO, 3.1)
    )

    for etiqueta, color, marcador in MOTORES:
        med, lo, hi = LATENCIA[etiqueta]
        ax_lat.fill_between(NIVELES, lo, hi, color=color, alpha=0.15, linewidth=0)
        ax_lat.plot(NIVELES, med, color=color, marker=marcador, markersize=4,
                    markerfacecolor="white", markeredgewidth=1.1, linewidth=1.4,
                    label=etiqueta)

        med, lo, hi = THROUGHPUT[etiqueta]
        ax_thr.fill_between(NIVELES, lo, hi, color=color, alpha=0.15, linewidth=0)
        ax_thr.plot(NIVELES, med, color=color, marker=marcador, markersize=4,
                    markerfacecolor="white", markeredgewidth=1.1, linewidth=1.4,
                    label=etiqueta)

    ax_lat.set_ylim(0, TOPE_LATENCIA)
    ax_lat.set_ylabel("Tiempo muro (s)")
    ax_lat.annotate(
        f"p75 = {IQR_OLLAMA_25U:.0f} s".replace(".", ","),
        xy=(25, TOPE_LATENCIA), xytext=(23.4, TOPE_LATENCIA - 9),
        color="#E69F00", fontsize=7, ha="right",
        arrowprops=dict(arrowstyle="-|>", color="#E69F00", linewidth=0.8,
                        shrinkA=1, shrinkB=1),
    )

    ax_thr.set_ylim(0, None)
    ax_thr.set_ylabel("Throughput global (tokens/s)")

    for ax in (ax_lat, ax_thr):
        ax.set_xlabel("Usuarios concurrentes")
        ax.set_xticks(NIVELES)
        ax.set_xlim(0, 26)

    handles, labels = ax_lat.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.06))

    guardar(fig, "despliegue_motores")


# ── Figura 2 — A/B de prefix caching en vLLM (26B-A4B), hasta 200 usuarios ──
NIVELES_PC = [1, 25, 50, 100, 150, 200]

PREFIX_AB = {
    # etiqueta: (tiempo muro s, throughput global tok/s, color, marcador, estilo)
    "Con prefix caching":  ([5.4, 12.8, 17.4, 25.1, 35.4, 45.0],
                            [133.7, 975.6, 1537.3, 2014.8, 2205.2, 2339.0],
                            "#0072B2", "o", "-"),
    "Sin prefix caching":  ([5.5, 21.6, 34.1, 59.0, 87.1, 113.6],
                            [131.6, 604.7, 759.3, 889.5, 880.6, 883.7],
                            "#CC79A7", "s", "--"),
}


def figura_prefixcache():
    fig, (ax_wall, ax_thr) = plt.subplots(1, 2, figsize=(st.ANCHO_TEXTO, 3.1))

    for etiqueta, (wall, thr, color, marcador, estilo) in PREFIX_AB.items():
        ax_wall.plot(NIVELES_PC, wall, color=color, marker=marcador, markersize=4,
                     markerfacecolor="white", markeredgewidth=1.1, linewidth=1.4,
                     linestyle=estilo, label=etiqueta)
        ax_thr.plot(NIVELES_PC, thr, color=color, marker=marcador, markersize=4,
                    markerfacecolor="white", markeredgewidth=1.1, linewidth=1.4,
                    linestyle=estilo, label=etiqueta)

    ax_wall.set_ylabel("Tiempo muro (s)")
    ax_thr.set_ylabel("Throughput global (tokens/s)")

    for ax in (ax_wall, ax_thr):
        ax.set_xlabel("Usuarios concurrentes")
        ax.set_xticks(NIVELES_PC)
        ax.set_xlim(0, 208)
        ax.set_ylim(0, None)

    handles, labels = ax_wall.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.06))

    guardar(fig, "despliegue_prefixcache")


def guardar(fig, nombre: str):
    st.save(fig, nombre)
    os.makedirs(DIR_MEMORIA, exist_ok=True)
    for ext in ("pdf", "png"):
        shutil.copy2(st.OUT_DIR / f"{nombre}.{ext}",
                     os.path.join(DIR_MEMORIA, f"{nombre}.{ext}"))


# ── Figura 3 — motores x hardware: latencia y throughput en una sola figura ─
# Esta se calcula de los datos crudos: eval/prueba_carga_usuarios/
#   confianza/<motor>/26b_<emb>[_np8]/run_*/sweep.json      (RTX PRO 6000)
#   confianza_5090/<motor>/26b_<emb>[_np8]/run_*/sweep.json (RTX 5090)
# De cada barrido se toman avg_total (latencia media por petición del nivel) y
# throughput (tokens/s globales), y se resumen por la mediana de las 20 medidas
# = 10 repeticiones x 2 modelos de embedding.
RAIZ_DATOS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "eval", "prueba_carga_usuarios",
)

# El color codifica el MOTOR y el trazo la MÁQUINA: continuo = RTX 5090,
# discontinuo = RTX PRO 6000 (mismo criterio que la figura de saturación).
MAQUINAS = [
    ("confianza_5090", "RTX 5090 (32 GB)",     "-"),
    ("confianza",      "RTX PRO 6000 (96 GB)", "--"),
]

# (subdirectorio del motor, sufijo NP, etiqueta, color, marcador)
MOTORES_HW = [
    ("vllm",     "",     "vLLM",             "#0072B2", "o"),
    ("llamacpp", "_np8", "llama.cpp (NP=8)", "#009E73", "s"),
    ("ollama",   "_np8", "Ollama (NP=8)",    "#E69F00", "^"),
]


def _serie_cruda(maquina, motor, sufijo_np):
    """Mediana por nivel de la latencia por petición (s) y del throughput."""
    lat = {n: [] for n in NIVELES}
    thr = {n: [] for n in NIVELES}
    for emb in ("06b", "4b"):
        celda = os.path.join(RAIZ_DATOS, maquina, motor, f"26b_{emb}{sufijo_np}")
        for ruta in sorted(glob.glob(os.path.join(celda, "run_*", "sweep.json"))):
            with open(ruta, encoding="utf-8") as fh:
                barrido = json.load(fh)
            for nivel in barrido["sweep"]:
                n = nivel["level"]
                if n not in lat:
                    continue
                if nivel.get("avg_total"):
                    lat[n].append(nivel["avg_total"] / 1000.0)
                if nivel.get("throughput"):
                    thr[n].append(nivel["throughput"])

    def mediana(d):
        salida = []
        for n in NIVELES:
            if not d[n]:
                raise SystemExit(f"sin datos: {maquina}/{motor}{sufijo_np} nivel {n}")
            salida.append(statistics.median(d[n]))
        return salida

    return mediana(lat), mediana(thr)


def figura_hardware():
    from matplotlib.lines import Line2D

    fig, (ax_lat, ax_thr) = plt.subplots(1, 2, figsize=(st.ANCHO_TEXTO, 3.0))

    for maquina, _, trazo in MAQUINAS:
        for motor, sufijo, _, color, marcador in MOTORES_HW:
            lat, thr = _serie_cruda(maquina, motor, sufijo)
            comun = dict(color=color, marker=marcador, markersize=3.6,
                         markerfacecolor="white", markeredgewidth=1.0,
                         linewidth=1.3, linestyle=trazo)
            ax_lat.plot(NIVELES, lat, **comun)
            ax_thr.plot(NIVELES, thr, **comun)

    ax_lat.set_ylabel("Latencia media por petición (s)")
    ax_thr.set_ylabel("Throughput global (tokens/s)")
    for ax in (ax_lat, ax_thr):
        ax.set_xlabel("Usuarios concurrentes")
        ax.set_xticks(NIVELES)
        ax.set_xlim(0, 26)
        ax.set_ylim(0, None)

    # Dos leyendas: el color identifica el motor, el trazo la máquina.
    leyenda_motor = [
        Line2D([], [], color=c, marker=m, markersize=3.6, markerfacecolor="white",
               markeredgewidth=1.0, linewidth=1.3, label=e)
        for _, _, e, c, m in MOTORES_HW
    ]
    leyenda_maquina = [
        Line2D([], [], color="#555555", linewidth=1.3, linestyle=t, label=n)
        for _, n, t in MAQUINAS
    ]
    fig.legend(handles=leyenda_motor, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.11))
    fig.legend(handles=leyenda_maquina, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.20))

    guardar(fig, "despliegue_hardware")


# ── Figura 4 — saturación: hasta dónde llega cada máquina ───────────────────
# RTX 5090  → confianza_5090/saturacion/sweep_saturacion.json (barrido 10-200 u.)
# RTX PRO 6000 → 2_vllm_experimentos/prefixcache_AB, rama con prefix caching
#   activo (barrido 1-200 u.), que es la configuración de producción y el
#   único barrido de esa máquina que llega a 200 usuarios.
# Ambos con prefix caching activo y 0 peticiones fallidas en todo el rango.
#
# Los niveles se igualan a los comunes a las dos máquinas. El punto de 1 usuario
# NO existe en los barridos de saturación de la 5090 (empiezan en 10), y el del
# servidor está inflado por el régimen sostenido del propio barrido (133,7 tok/s
# frente a 101,7 en la matriz), así que para ese nivel se toma en AMBAS máquinas
# la mediana de la matriz de concurrencia. De este modo las dos series comparten
# protocolo en cada punto y son comparables entre sí.
NIVELES_SAT = [1, 25, 50, 100, 150, 200]


def _throughput_1_usuario(maquina):
    """Mediana del throughput a 1 usuario en la matriz de concurrencia."""
    v = []
    for emb in ("06b", "4b"):
        celda = os.path.join(RAIZ_DATOS, maquina, "vllm", f"26b_{emb}")
        for ruta in sorted(glob.glob(os.path.join(celda, "run_*", "sweep.json"))):
            with open(ruta, encoding="utf-8") as fh:
                barrido = json.load(fh)
            for nivel in barrido["sweep"]:
                if nivel["level"] == 1 and nivel.get("throughput"):
                    v.append(nivel["throughput"])
    return statistics.median(v)


def _saturacion_5090():
    ruta = os.path.join(RAIZ_DATOS, "confianza_5090", "saturacion",
                        "sweep_saturacion.json")
    with open(ruta, encoding="utf-8") as fh:
        barrido = json.load(fh)
    por_nivel = {n["level"]: n["throughput"] for n in barrido["sweep"]}
    por_nivel[1] = _throughput_1_usuario("confianza_5090")
    return [por_nivel[n] for n in NIVELES_SAT]


def _saturacion_pro():
    por_nivel = dict(zip(NIVELES_PC, PREFIX_AB["Con prefix caching"][1]))
    por_nivel[1] = _throughput_1_usuario("confianza")
    return [por_nivel[n] for n in NIVELES_SAT]


def _saturacion_voz04(carpeta, celda="26b_06b"):
    """Medianas por nivel de un barrido de saturación de voz04 (5 iteraciones).

    Devuelve (niveles, throughput, ttft_s). Se lee de los ficheros en vez de
    llevar los números a mano, para que la figura se pueda regenerar sin
    transcribir nada.
    """
    base = os.path.join("eval", "prueba_carga_usuarios", "voz04", carpeta, celda)
    por_nivel = {}
    for ruta in sorted(glob.glob(os.path.join(base, "run_*", "sweep.json"))):
        with open(ruta, encoding="utf-8") as fh:
            barrido = json.load(fh)
        for nivel in barrido["sweep"]:
            e = por_nivel.setdefault(nivel["level"], {"thr": [], "ttft": []})
            e["thr"].append(nivel["throughput"])
            e["ttft"].append(nivel["avg_ttft"] / 1000.0)
    if not por_nivel:
        raise SystemExit(f"sin datos de saturación en {base}")
    niveles = sorted(por_nivel)
    return (niveles,
            [statistics.median(por_nivel[n]["thr"]) for n in niveles],
            [statistics.median(por_nivel[n]["ttft"]) for n in niveles])


def _nivel1_5090():
    """Nivel de 1 usuario de la RTX 5090, tomado de la matriz de concurrencia.

    Los barridos de saturación de la 5090 arrancan en 10 usuarios, así que este
    punto se presta de la matriz (celda 26b_4b, la que comparte colección con
    ellos). Se usa la MEDIANA de las diez iteraciones, no una medida suelta: con
    una única petición por iteración la dispersión es enorme (46-157 tokens/s).

    La matriz se midió con prefix caching activo y no existe medida equivalente
    sin él, así que el mismo punto sirve para las dos ramas. A un solo usuario la
    diferencia entre ellas es pequeña y queda por debajo del ruido del nivel.
    """
    thr, ttft = [], []
    patron = os.path.join("eval", "prueba_carga_usuarios", "confianza_5090",
                          "vllm", "26b_4b", "run_*", "sweep.json")
    for ruta in sorted(glob.glob(patron)):
        with open(ruta, encoding="utf-8") as fh:
            barrido = json.load(fh)
        for nivel in barrido["sweep"]:
            if nivel["level"] == 1:
                thr.append(nivel["throughput"])
                ttft.append(nivel["avg_ttft"] / 1000.0)
    if not thr:
        return None
    return statistics.median(thr), statistics.median(ttft)


def _saturacion_5090(carpeta):
    """Barrido de saturación de la RTX 5090 (una sola pasada, niveles 10-200),
    con el nivel de 1 usuario prestado de la matriz (ver `_nivel1_5090`)."""
    ruta = os.path.join("eval", "prueba_carga_usuarios", "confianza_5090",
                        carpeta, "sweep_saturacion.json")
    with open(ruta, encoding="utf-8") as fh:
        barrido = json.load(fh)
    niveles = [n["level"] for n in barrido["sweep"]]
    thr     = [n["throughput"] for n in barrido["sweep"]]
    ttft    = [n["avg_ttft"] / 1000.0 for n in barrido["sweep"]]
    uno = _nivel1_5090()
    if uno and niveles[0] != 1:
        niveles = [1] + niveles
        thr     = [uno[0]] + thr
        ttft    = [uno[1]] + ttft
    return niveles, thr, ttft


# Incluir o no la RTX 5090 en la figura de saturación. Sus barridos se midieron
# con vLLM 0.23.0 y la colección del embedder 4b, mientras que los de la RTX PRO
# 6000 son con vLLM 0.28.0 y el embedder 0.6b, así que la comparación entre
# máquinas arrastra dos variables además del hardware.
INCLUIR_5090 = True


def figura_saturacion():
    """Saturación con y sin prefix caching.

    La RTX PRO 6000 procede del A/B emparejado de voz04 (5 iteraciones, 110
    preguntas distintas, vLLM 0.28.0). La RTX 5090, de sus dos barridos de
    saturación en gtc2pc4 (una sola pasada, niveles 10-200, vLLM 0.23.0).
    """
    fig, (ax_ttft, ax) = plt.subplots(1, 2, figsize=(st.ANCHO_TEXTO, 3.1))

    series = []
    if INCLUIR_5090:
        for carpeta, con_pc in (("saturacion", True), ("saturacion_noPC", False)):
            x, thr, ttft = _saturacion_5090(carpeta)
            series.append((con_pc, x, thr, ttft, "#0072B2", "o"))
    for carpeta, con_pc in (("saturacion", True), ("saturacion_noPC", False)):
        x, thr, ttft = _saturacion_voz04(carpeta)
        series.append((con_pc, x, thr, ttft, "#D55E00", "s"))

    for con_pc, x, thr, ttft, color, marcador in series:
        comun = dict(color=color, marker=marcador, markersize=3.8,
                     markerfacecolor="white" if con_pc else color,
                     markeredgewidth=1.0, linewidth=1.4,
                     linestyle="-" if con_pc else "--")
        ax.plot(x, thr, **comun)
        ax_ttft.plot(x, ttft, **comun)

    ax.set_ylabel("Throughput global (tokens/s)")
    ax_ttft.set_ylabel("Tiempo hasta el primer token (s)")
    for a in (ax, ax_ttft):
        a.set_xlabel("Usuarios concurrentes")
        a.set_xticks([1, 25, 50, 100, 150, 200])
        a.set_xlim(0, 212)
        a.set_ylim(0, None)

    from matplotlib.lines import Line2D
    maquinas = [("RTX PRO 6000", "#D55E00", "s")]
    if INCLUIR_5090:
        maquinas.insert(0, ("RTX 5090", "#0072B2", "o"))
    leyenda_maquina = [
        Line2D([], [], color=c, marker=m, markersize=3.8,
               markerfacecolor="white", markeredgewidth=1.0, linewidth=1.4,
               label=n)
        for n, c, m in maquinas
    ]
    leyenda_pc = [
        Line2D([], [], color="#555555", linewidth=1.4, linestyle=e, label=t)
        for e, t in (("-", "Con prefix caching"), ("--", "Sin prefix caching"))
    ]
    ax_ttft.legend(handles=leyenda_pc, loc="upper left", frameon=False,
                   handlelength=2.4, borderaxespad=0.6)
    fig.legend(handles=leyenda_maquina, loc="lower center", ncol=len(maquinas),
               frameon=False, bbox_to_anchor=(0.5, -0.10))

    guardar(fig, "despliegue_saturacion")


# ── Figura 5 — mezcla de expertos frente a modelo denso ─────────────────────
# Latencia media por petición a 25 usuarios, para las 6 combinaciones de motor
# y máquina, con los dos modelos generadores. Mediana de las 20 medidas de cada
# celda (10 repeticiones x 2 modelos de embedding), de los sweep.json crudos.
MODELOS = [
    ("26b", "gemma-4 26B-A4B (MoE)", "#0072B2"),
    ("12b", "gemma-4 12B (denso)",   "#E69F00"),
]


def _latencia_25u(maquina, motor, modelo, sufijo_np):
    v = []
    for emb in ("06b", "4b"):
        celda = os.path.join(RAIZ_DATOS, maquina, motor,
                             f"{modelo}_{emb}{sufijo_np}")
        for ruta in sorted(glob.glob(os.path.join(celda, "run_*", "sweep.json"))):
            with open(ruta, encoding="utf-8") as fh:
                barrido = json.load(fh)
            for nivel in barrido["sweep"]:
                if nivel["level"] == 25 and nivel.get("avg_total"):
                    v.append(nivel["avg_total"] / 1000.0)
    if not v:
        raise SystemExit(f"sin datos: {maquina}/{motor}/{modelo}{sufijo_np}")
    return statistics.median(v)


def figura_moe():
    fig, ejes = plt.subplots(1, 2, figsize=(st.ANCHO_TEXTO, 3.0), sharey=True)
    ancho = 0.34
    posiciones = range(len(MOTORES_HW))

    for ax, (maquina, titulo, _trazo) in zip(ejes, MAQUINAS):
        for i, (modelo, etiqueta, color) in enumerate(MODELOS):
            valores = [_latencia_25u(maquina, motor, modelo, sufijo)
                       for motor, sufijo, _, _, _ in MOTORES_HW]
            x = [p + (i - 0.5) * ancho for p in posiciones]
            barras = ax.bar(x, valores, ancho, color=color, label=etiqueta,
                            edgecolor="white", linewidth=0.6)
            ax.bar_label(barras, fmt="%.0f", fontsize=6.5, padding=1.5,
                         color="#444444")

        ax.set_title(titulo, fontsize=9, fontweight="bold", pad=6)
        ax.set_xticks(list(posiciones))
        ax.set_xticklabels(["vLLM", "llama.cpp", "Ollama"])
        ax.grid(axis="x", visible=False)

    ejes[0].set_ylabel("Latencia media por petición (s)\ncon 25 usuarios concurrentes")
    ejes[0].set_ylim(0, 100)

    handles, labels = ejes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.08))

    guardar(fig, "despliegue_moe")


# ── Figura 6 — evolución de la presión sobre la KV Cache ────────────────────
# Métricas internas del propio vLLM (endpoint /metrics) muestreadas durante los
# barridos de saturación. Para situar cada muestra en su nivel de concurrencia
# se usa el contador acumulado de peticiones completadas: el barrido ejecuta los
# niveles en orden, así que las primeras N1 peticiones pertenecen al nivel 1,
# las N1+N2 siguientes al 2, y así sucesivamente.
TOPE_CLIENTE = 100   # httpx.AsyncClient: max_connections por defecto

CORRIDAS_KV = {
    # etiqueta: (ruta relativa, niveles del barrido, color)
    "RTX 5090": ("confianza_5090/saturacion/vllm_metrics.csv",
                 [10, 25, 50, 75, 100, 150, 200], "#0072B2", "o", "-"),
    "RTX PRO 6000": ("2_vllm_experimentos/prefixcache_AB/metrics_ab_pcON.csv",
                     [1, 25, 50, 100, 150, 200], "#D55E00", "s", "--"),
}


def _series_por_nivel(ruta_rel, niveles):
    """Agrupa las muestras de /metrics por nivel de concurrencia."""
    import csv
    with open(os.path.join(RAIZ_DATOS, ruta_rel), encoding="utf-8") as fh:
        filas = list(csv.DictReader(fh))

    def col(*subs):
        return next(c for c in filas[0] if all(s in c for s in subs))

    c_kv, c_run, c_cola = col("kv"), col("running"), col("waiting")
    # Contador acumulado de peticiones terminadas (el nombre cambia de versión).
    c_fin = next((c for c in filas[0] if "succ" in c or
                  ("generation_tokens_count" in c and "iteration" not in c)), None)

    corte, acumulado = [], 0
    for n in niveles:
        acumulado += n
        corte.append(acumulado)
    base = float(filas[0][c_fin])

    datos = {n: {"kv": [], "run": [], "cola": []} for n in niveles}
    for f in filas:
        hechas = float(f[c_fin]) - base
        for nivel, limite in zip(niveles, corte):
            if hechas < limite:
                datos[nivel]["kv"].append(float(f[c_kv]) * 100.0)
                datos[nivel]["run"].append(float(f[c_run]))
                datos[nivel]["cola"].append(float(f[c_cola]))
                break

    usados = [n for n in niveles if datos[n]["kv"]]
    # Se resume por la MEDIANA, no por la media: cada nivel incluye su arranque
    # y su vaciado con la caché casi libre, transitorios que hunden la media y
    # disimulan que el pool opera en su techo durante el servicio efectivo.
    return (usados,
            [statistics.median(datos[n]["kv"]) for n in usados],
            [max(datos[n]["kv"]) for n in usados],
            [statistics.median(datos[n]["run"]) for n in usados],
            [statistics.median(datos[n]["cola"]) for n in usados],
            [max(datos[n]["cola"]) for n in usados])


def figura_kvcache():
    """Ocupación de la KV Cache y cola resultante, las dos máquinas juntas.

    En ambos paneles la línea es el PICO de cada nivel y la banda baja hasta la
    mediana. Se usa el pico porque la ocupación oscila continuamente —la caché
    se llena, las secuencias que terminan liberan bloques y entran otras— y es
    el pico el que determina si el motor puede o no ampliar el lote.
    """
    fig, (ax_kv, ax_cola) = plt.subplots(1, 2, figsize=(st.ANCHO_TEXTO, 3.2))

    for etiqueta, (ruta, niveles, color, marcador, trazo) in CORRIDAS_KV.items():
        x, kv_med, kv_max, _lote, cola_med, cola_max = _series_por_nivel(
            ruta, niveles)
        comun = dict(color=color, marker=marcador, markersize=4,
                     markerfacecolor="white", markeredgewidth=1.1,
                     linewidth=1.4, linestyle=trazo)
        ax_kv.fill_between(x, kv_med, kv_max, color=color, alpha=0.15,
                           linewidth=0)
        ax_kv.plot(x, kv_max, label=etiqueta, **comun)
        ax_cola.fill_between(x, cola_med, cola_max, color=color, alpha=0.15,
                             linewidth=0)
        ax_cola.plot(x, cola_max, label=etiqueta, **comun)

    ax_kv.axhline(100, color="#B03030", linewidth=0.9, linestyle=":")
    ax_kv.text(206, 101.5, "pool agotado", fontsize=6.5, color="#B03030",
               ha="right", va="bottom")
    ax_kv.set_ylabel("Ocupación del pool\nde KV Cache (%)")
    ax_kv.set_ylim(0, 116)

    ax_cola.set_ylabel("Peticiones en cola")
    ax_cola.set_ylim(0, None)

    # El punto donde la caché se agota y, a partir de ahí, empieza la cola.
    for ax, y, texto in ((ax_kv, 108, "se agota"),
                         (ax_cola, None, "empieza la cola")):
        ax.axvline(50, color="#777777", linewidth=0.8, linestyle="-.",
                   alpha=0.8)
    ax_kv.text(53, 60, "se agota\na 50 usuarios", fontsize=6.5,
               color="#777777", va="center")
    ax_cola.text(53, 38, "la cola arranca\nen ese mismo punto", fontsize=6.5,
                 color="#777777", va="center")

    for ax in (ax_kv, ax_cola):
        ax.set_xlabel("Usuarios concurrentes")
        ax.set_xticks([1, 25, 50, 100, 150, 200])
        ax.set_xlim(0, 208)

    handles, labels = ax_kv.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.07))

    guardar(fig, "despliegue_kvcache")


def figura_saturacion_modelos():
    """Saturación de los dos modelos en la RTX PRO 6000, con y sin prefix caching.

    Misma máquina, mismo motor (vLLM 0.28.0), mismas preguntas y mismos niveles
    para las cuatro curvas: lo único que cambia es el modelo y si la caché de
    prefijo está activa. Es la comparación limpia entre la mezcla de expertos
    (26B-A4B, ~4B activos) y el modelo denso de 12B bajo carga.
    """
    fig, (ax_ttft, ax) = plt.subplots(1, 2, figsize=(st.ANCHO_TEXTO, 3.1))

    modelos = [
        ("gemma-4 26B-A4B (MoE)", "26b_06b", "saturacion",     "saturacion_noPC",     "#D55E00", "s"),
        ("gemma-4 12B (denso)",   "12b_06b", "saturacion_12b", "saturacion_12b_noPC", "#0072B2", "o"),
    ]

    for _etiqueta, celda, con, sin, color, marcador in modelos:
        for carpeta, con_pc in ((con, True), (sin, False)):
            x, thr, ttft = _saturacion_voz04(carpeta, celda)
            comun = dict(color=color, marker=marcador, markersize=3.8,
                         markerfacecolor="white" if con_pc else color,
                         markeredgewidth=1.0, linewidth=1.4,
                         linestyle="-" if con_pc else "--")
            ax.plot(x, thr, **comun)
            ax_ttft.plot(x, ttft, **comun)

    ax.set_ylabel("Throughput global (tokens/s)")
    ax_ttft.set_ylabel("Tiempo hasta el primer token (s)")
    for a in (ax, ax_ttft):
        a.set_xlabel("Usuarios concurrentes")
        a.set_xticks([1, 25, 50, 100, 150, 200])
        a.set_xlim(0, 212)
        a.set_ylim(0, None)

    from matplotlib.lines import Line2D
    leyenda_modelo = [
        Line2D([], [], color=c, marker=m, markersize=3.8,
               markerfacecolor="white", markeredgewidth=1.0, linewidth=1.4,
               label=etiqueta)
        for etiqueta, _celda, _c1, _c2, c, m in modelos
    ]
    leyenda_pc = [
        Line2D([], [], color="#555555", linewidth=1.4, linestyle=e, label=t)
        for e, t in (("-", "Con prefix caching"), ("--", "Sin prefix caching"))
    ]
    ax_ttft.legend(handles=leyenda_pc, loc="upper left", frameon=False,
                   handlelength=2.4, borderaxespad=0.6)
    fig.legend(handles=leyenda_modelo, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.10))

    guardar(fig, "despliegue_saturacion_modelos")


def main():
    st.aplicar_estilo()
    print("Figuras del capítulo de despliegue:")
    figura_motores()
    figura_prefixcache()
    figura_hardware()
    figura_saturacion()
    figura_saturacion_modelos()
    figura_kvcache()
    figura_moe()
    print(f"  → copiadas también a {DIR_MEMORIA}")


if __name__ == "__main__":
    main()
