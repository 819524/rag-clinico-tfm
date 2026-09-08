#!/usr/bin/env python3
"""
generar_grafica_utilizacion.py — Utilización y potencia de la GPU a lo largo
del tiempo para los tres motores de inferencia sobre una misma configuración.

Se elige una única celda del barrido de confianza para que la comparación sea
justa: mismo modelo (gemma4:26b MoE), mismo servidor de embeddings (0,6B),
misma repetición (run_01) y por tanto la misma secuencia de preguntas sembrada.
Para llama.cpp y Ollama se usa NP=4, el valor con el que ambos motores rinden
mejor en esta celda en las dos máquinas.

La ventana de telemetría coincide con el barrido (±1 s), de modo que los
tiempos acumulados de cada nivel del sweep marcan directamente las fronteras
entre los seis niveles de concurrencia.

Dos avisos sobre lo que mide cada figura:
  * En voz05 el servidor de embeddings vive en la GPU 1, así que la GPU 0 que
    se dibuja es exclusivamente el generador.
  * La gtc2pc4 tiene una sola tarjeta: generador y embeddings comparten GPU,
    y la curva agrega el trabajo de ambos.

Salida, para cada métrica {utilizacion, potencia_tiempo}:
        telemetria_<m>.pdf/.png            (voz05)
        telemetria_<m>_5090.pdf/.png       (gtc2pc4)
        telemetria_<m>_maquinas.pdf/.png   (las dos superpuestas)
"""

import csv
import datetime as dt
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
import estilo_tfm as est

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_MEMORIA = os.path.join(RAIZ, "memoria", "Imagenes")
BASE = os.path.join(RAIZ, "eval", "prueba_carga_usuarios")

GPU_DIBUJADA = "0"
RUN = "run_01"

MOTORES = [
    ("vLLM",      "#0072B2", "vllm/26b_06b"),
    ("llama.cpp", "#009E73", "llamacpp/26b_06b_np4"),
    ("Ollama",    "#E69F00", "ollama/26b_06b_np4"),
]

MAQUINAS = [
    ("confianza",      "voz05 · 2× RTX PRO 6000", ""),
    ("confianza_5090", "gtc2pc4 · RTX 5090",      "_5090"),
]

# nombre → (columna del CSV, rótulo del eje, ticks, formato del resumen)
METRICAS = {
    "utilizacion": {
        "columna": 2,
        "eje": "utilización de la GPU (%)",
        "ticks": [0, 25, 50, 75, 100],
        "tope": 100.0,
        "salida": "telemetria_utilizacion",
    },
    "potencia": {
        "columna": 4,
        "eje": "potencia de la GPU (W)",
        "ticks": [0, 150, 300, 450, 600],
        "tope": None,          # se calcula del propio conjunto de datos
        "salida": "telemetria_potencia_tiempo",
    },
}


def leer_telemetria(ruta, columna):
    """Devuelve (segundos desde el inicio, valor) de la GPU dibujada."""
    t, v = [], []
    with open(ruta, newline="") as fh:
        for fila in csv.reader(fh):
            if not fila or fila[1].strip() != GPU_DIBUJADA:
                continue
            marca = dt.datetime.strptime(fila[0].strip(), "%Y/%m/%d %H:%M:%S.%f")
            t.append(marca)
            v.append(float(fila[columna]))
    t0 = t[0]
    return [(x - t0).total_seconds() for x in t], v


def leer_niveles(ruta):
    """Fronteras acumuladas (s) y etiqueta de usuarios de cada nivel."""
    datos = json.load(open(ruta))
    cortes, acumulado = [], 0.0
    for nivel in datos["sweep"]:
        inicio = acumulado
        acumulado += nivel["wall_ms"] / 1000
        cortes.append((inicio, acumulado, nivel["level"]))
    return cortes


def energia_wh(t, w):
    """Integral trapezoidal de la potencia sobre la ventana, en Wh."""
    julios = sum((w[i] + w[i + 1]) / 2 * (t[i + 1] - t[i])
                 for i in range(len(t) - 1))
    return julios / 3600


def media_movil(t, v, ventana=5.0):
    """Media móvil centrada de `ventana` segundos sobre un muestreo a 1 Hz."""
    radio = max(1, int(ventana / 2))
    suave = []
    for i in range(len(v)):
        ini, fin = max(0, i - radio), min(len(v), i + radio + 1)
        suave.append(sum(v[ini:fin]) / (fin - ini))
    return t, suave


def cargar(metrica):
    """{campaña: [(etiqueta, color, t, valores, cortes), ...]} para una métrica."""
    col = METRICAS[metrica]["columna"]
    datos = {}
    for campana, _, _ in MAQUINAS:
        serie = []
        for etiqueta, color, celda in MOTORES:
            carpeta = os.path.join(BASE, campana, celda, RUN)
            t, v = leer_telemetria(os.path.join(carpeta, "telemetria_gpu.csv"), col)
            cortes = leer_niveles(os.path.join(carpeta, "sweep.json"))
            serie.append((etiqueta, color, t, v, cortes))
        datos[campana] = serie
    return datos


def resumen(metrica, t, v):
    """Rótulo compacto de una serie: duración, nivel medio y, si procede, Wh."""
    media = sum(v) / len(v)
    if metrica == "potencia":
        return f"{t[-1]:.0f} s, {media:.0f} W, {energia_wh(t, v):.1f} Wh"
    return f"{t[-1]:.0f} s, {media:.0f} % de media"


def guardar(fig, nombre):
    est.save(fig, nombre)
    os.makedirs(DIR_MEMORIA, exist_ok=True)
    for ext in ("pdf", "png"):
        shutil.copy2(est.OUT_DIR / f"{nombre}.{ext}",
                     os.path.join(DIR_MEMORIA, f"{nombre}.{ext}"))
    plt.close(fig)
    print(f"    ✓ {nombre}.pdf / .png")


def figura_maquina(metrica, datos, campana, sufijo):
    """Un panel por motor, una sola máquina, con las bandas de concurrencia."""
    cfg = METRICAS[metrica]
    series = datos[campana]
    fig, ejes = plt.subplots(3, 1, sharex=True, figsize=(est.ANCHO_TEXTO, 5.2))

    limite = max(t[-1] for _, _, t, _, _ in series) * 1.02
    tope = cfg["tope"] or max(max(v) for serie in datos.values()
                              for _, _, _, v, _ in serie)
    alto = tope * 1.14

    for ax, (etiqueta, color, t, v, cortes) in zip(ejes, series):
        for i, (ini, fin, _) in enumerate(cortes):
            if i % 2:
                ax.axvspan(ini, fin, color="#000000", alpha=0.04, lw=0)
        for _, fin, _ in cortes[:-1]:
            ax.axvline(fin, color="#999999", lw=0.6, ls=":", zorder=1)

        ax.fill_between(t, v, step="post", color=color, alpha=0.22, lw=0)
        ax.step(t, v, where="post", color=color, lw=1.1, zorder=3)

        for ini, fin, nivel in cortes:
            ax.text((ini + fin) / 2, tope * 1.06, str(nivel), ha="center",
                    va="center", fontsize=6.5, color="#555555")

        ax.text(0.0, 1.03, f"{etiqueta} — {resumen(metrica, t, v)}",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=8.5,
                color=color)

        ax.set_ylim(0, alto)
        ax.set_yticks(cfg["ticks"])
        ax.set_xlim(0, limite)
        ax.grid(axis="y", lw=0.4, alpha=0.4)
        ax.set_axisbelow(True)

    ejes[1].set_ylabel(cfg["eje"])
    ejes[-1].set_xlabel("tiempo transcurrido dentro del barrido (s)")
    ejes[0].text(1.0, 1.03, "cifras superiores: usuarios concurrentes",
                 transform=ejes[0].transAxes, fontsize=7, color="#555555",
                 ha="right", va="bottom")

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.30)
    guardar(fig, cfg["salida"] + sufijo)


def figura_superpuesta(metrica, datos):
    """Las dos máquinas sobre los mismos ejes: voz05 continua, 5090 discontinua."""
    cfg = METRICAS[metrica]
    trazo = {"confianza": ((1, 0), "voz05 · 2× RTX PRO 6000"),
             "confianza_5090": ((4, 2), "gtc2pc4 · RTX 5090")}

    fig, ejes = plt.subplots(3, 1, sharex=True, figsize=(est.ANCHO_TEXTO, 5.2))

    limite = max(t[-1] for serie in datos.values()
                 for _, _, t, _, _ in serie) * 1.02
    tope = cfg["tope"] or max(max(v) for serie in datos.values()
                              for _, _, _, v, _ in serie)

    for i, ax in enumerate(ejes):
        etiqueta = color = None
        textos = []
        for campana, (guiones, _) in trazo.items():
            etiqueta, color, t, v, _ = datos[campana][i]
            # Traza cruda al fondo y media móvil de 5 s en primer plano
            ax.plot(t, v, color=color, lw=0.6, alpha=0.28, dashes=guiones,
                    drawstyle="steps-post", zorder=2)
            ts, vs = media_movil(t, v)
            ax.plot(ts, vs, color=color, lw=1.5, dashes=guiones, zorder=3)
            textos.append(resumen(metrica, t, v).replace(" de media", ""))

        ax.text(0.0, 1.03, f"{etiqueta} — {'  vs  '.join(textos)}",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=8.5,
                color=color)

        ax.set_ylim(0, tope * 1.08)
        ax.set_yticks(cfg["ticks"])
        ax.set_xlim(0, limite)
        ax.grid(axis="y", lw=0.4, alpha=0.4)
        ax.set_axisbelow(True)

    ejes[1].set_ylabel(cfg["eje"])
    ejes[-1].set_xlabel("tiempo transcurrido dentro del barrido (s)")

    from matplotlib.lines import Line2D
    manijas = [Line2D([], [], color="#555555", lw=1.5, dashes=g, label=r)
               for g, r in trazo.values()]
    fig.legend(handles=manijas, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.005))

    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.subplots_adjust(hspace=0.30)
    guardar(fig, cfg["salida"] + "_maquinas")


def main():
    est.aplicar_estilo()
    print("  Celda: 26B MoE + embeddings 0,6B, NP=4, run_01\n")

    for metrica in METRICAS:
        print(f"  ── {metrica} ──")
        datos = cargar(metrica)
        for campana, _, sufijo in MAQUINAS:
            figura_maquina(metrica, datos, campana, sufijo)
        figura_superpuesta(metrica, datos)

        for campana, rotulo, _ in MAQUINAS:
            print(f"    {rotulo}")
            for etiqueta, _, t, v, _ in datos[campana]:
                extra = ""
                if metrica == "potencia":
                    extra = (f"   pico {max(v):5.0f} W   mínimo {min(v):5.0f} W"
                             f"   {energia_wh(t, v):5.2f} Wh")
                print(f"      {etiqueta:<10} {t[-1]:6.0f} s   "
                      f"media {sum(v) / len(v):6.1f}{extra}")
        print()


if __name__ == "__main__":
    main()
