#!/usr/bin/env python3
"""
analisis_costes_nube.py — Evaluación económica: coste real de los experimentos en
local, coste equivalente en la nube y amortización del hardware de un servicio.
================================================================================
Consume el inventario de `inventario_tokens_energia.py` y responde a tres
preguntas:

  (A) ¿Qué han costado los experimentos ejecutados en local?
      → energía medida por telemetría + extrapolación de la parte no
        instrumentada, a precio de electricidad, más la amortización imputable.

  (B) ¿Qué habrían costado facturados a una API comercial?
      → los mismos tokens contra el catálogo de Anthropic, Google y OpenAI,
        con especial atención a los modelos de gama de entrada de cada una.

  (C) ¿Compensa adquirir la máquina, y en cuánto tiempo?
      → coste acumulado a 60 meses para cada volumen mensual, punto de
        equilibrio frente a cada alternativa y sensibilidad a los parámetros.

Salidas (en eval/costes/):
  costes_experimentos.csv   coste local vs cada tarifa, de lo ya ejecutado
  costes_servicio.csv       coste acumulado mes a mes por escenario y volumen
  breakeven.csv             punto de equilibrio por combinación
  sensibilidad.csv          barrido de €/kWh y de CAPEX
  analisis_costes.json      todo lo anterior, para el informe y la memoria
  ../../docs/figures/coste_*.pdf|png   figuras vectoriales para la memoria

Uso:
    tfm/bin/python Scripts/analisis_costes_nube.py [--precio-kwh 0.145]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COSTES = os.path.join(BASE, "eval", "costes")
FIGS = os.path.join(BASE, "docs", "figures")
IMAGENES = os.path.join(BASE, "memoria", "Imagenes")

# ── Parámetros energéticos ──────────────────────────────────────────────────
# Banda de precio eléctrico acordada para el análisis: consumidor no doméstico
# en España. Eurostat NRG_PC_205 sitúa la media en ~0,12 €/kWh sin impuestos
# deducibles, pero esa cifra no incluye peajes de acceso ni cargos, de modo que
# el valor central se fija en 0,17 €/kWh, representativo de una factura 6.1TD
# real con todos los conceptos repercutidos. La banda mantiene ±0,025 €/kWh.
PRECIO_KWH_DEF = 0.17
PRECIO_KWH_BANDA = (0.145, 0.195)
# Factor de conversión de energía de GPU a energía en la toma de corriente:
# cubre CPU, RAM, ventilación y pérdidas de la fuente del mismo nodo.
PUE_DEF = 1.35

# ── Catálogo de tarifas de API ──────────────────────────────────────────────
# Precios consultados el 2026-08-26 en las páginas oficiales de cada proveedor:
#   Anthropic → platform.claude.com/docs/en/about-claude/pricing
#   Google    → ai.google.dev/gemini-api/docs/pricing
#   OpenAI    → developers.openai.com/api/docs/pricing
# USD por millón de tokens. `gama` clasifica el modelo dentro del catálogo de su
# proveedor; `entrada` marca el modelo de gama de entrada vigente de cada casa.
TARIFAS = [
    # (id, etiqueta, proveedor, $/M entrada, $/M salida, $/M entrada en caché, gama, es_entrada)
    ("opus5",      "Claude Opus 5",         "Anthropic", 5.00, 25.00, 0.50, "frontera",   False),
    ("sonnet5",    "Claude Sonnet 5",       "Anthropic", 2.00, 10.00, 0.20, "intermedia", False),
    ("haiku45",    "Claude Haiku 4.5",      "Anthropic", 1.00,  5.00, 0.10, "entrada",    True),

    ("g36flash",   "Gemini 3.6 Flash",      "Google",    0.75,  3.75, 0.075, "intermedia", False),
    ("g35lite",    "Gemini 3.5 Flash-Lite", "Google",    0.30,  2.50, 0.03,  "entrada",    False),
    ("g31lite",    "Gemini 3.1 Flash-Lite", "Google",    0.25,  1.50, 0.025, "entrada",    True),
    ("g25lite",    "Gemini 2.5 Flash-Lite", "Google",    0.10,  0.40, 0.01,  "entrada",    False),

    ("gpt54mini",  "GPT-5.4 mini",          "OpenAI",    0.75,  4.50, 0.075, "intermedia", False),
    ("gpt56luna",  "GPT-5.6 luna",          "OpenAI",    0.20,  1.20, 0.02,  "entrada",    True),
    ("gpt5mini",   "GPT-5 mini",            "OpenAI",    0.25,  2.00, 0.025, "entrada",    False),
    ("gpt5nano",   "GPT-5 nano",            "OpenAI",    0.05,  0.40, 0.005, "entrada",    False),
]
# Tipo de cambio aplicado en todo el análisis. Referencia del 31 de agosto de
# 2026: 1 EUR = 1,1609 USD (Reserva Federal, tabla H.10), de donde 1 USD =
# 0,861 EUR. Al ser el parámetro que caduca más deprisa de todo el capítulo, se
# fija con fecha explícita en lugar de tomarse como constante.
USD_EUR = 0.861
DESCUENTO_LOTE = 0.50         # Batch API: 50 % en Anthropic y OpenAI

# Fracción del prompt que es prefijo estable y, por tanto, cacheable. Medida
# sobre el pipeline: el SYSTEM_PROMPT del generador son ~400 tokens de los
# ~4.700 de entrada; el resto son fragmentos recuperados, distintos en cada
# consulta y por tanto no cacheables.
FRACCION_CACHEABLE = 400 / 4688

# ── Escenarios de adquisición ───────────────────────────────────────────────
MAQUINAS = {
    "5090": {
        "etiqueta": "Estación RTX 5090",
        "detalle": "1× RTX 5090 (32 GB)",
        "capex_gpu": 2000, "capex_resto": 5500,   # CAPEX total 7.500 € (precio real)
        "pot_carga_w": 415,          # medido: vLLM, gemma-4 26B-A4B, bajo barrido
        "pot_reposo_w": 60,          # nodo encendido con el modelo residente
        "consultas_h_pico": 900,     # capacidad sostenida observada en saturación
    },
    "pro6000": {
        "etiqueta": "Servidor 1× PRO 6000",
        "detalle": "1× RTX PRO 6000 (96 GB)",
        "capex_gpu": 8500, "capex_resto": 4000,
        "pot_carga_w": 550,
        "pot_reposo_w": 90,
        "consultas_h_pico": 1400,
    },
}
# Máquina con la que se ejecutó el estudio; solo se usa para imputar la
# amortización de los experimentos, no se evalúa como opción de adquisición.
MAQUINA_ESTUDIO = {"capex": 22000, "pot_carga_w": 900}

VIDA_UTIL_MESES = 60
MANTENIMIENTO_ANUAL_PCT = 0.05     # repuestos, SAI, soporte, sobre el CAPEX

# ── Alquiler de GPU (IaaS) ──────────────────────────────────────────────────
# Índices de precio de GPU en nube (agosto 2026): la mediana de una L40S ronda
# 1,54 $/h y una A100-80 GB 1,86 $/h, con proveedores de bajo coste desde
# ~0,50 $/h para tarjetas de gama consumo. Se modelan los dos extremos.
IAAS = {
    "iaas_bajo": {"etiqueta": "GPU alquilada (bajo coste)", "eur_h": 0.55},
    "iaas_alto": {"etiqueta": "GPU alquilada (proveedor mayor)", "eur_h": 1.45},
}

# ── Perfil de consulta medido (60.503 consultas RAG del libro mayor) ────────
TOKENS_IN_CONSULTA = 4688
TOKENS_OUT_CONSULTA = 357

VOLUMENES = [1_000, 5_000, 20_000, 100_000]
HORIZONTE_MESES = 60


def eur(usd_por_millon: float, n_tokens: float) -> float:
    return n_tokens / 1e6 * usd_por_millon * USD_EUR


def coste_api(pin: float, pout: float, tin: float, tout: float) -> float:
    return eur(pin, tin) + eur(pout, tout)


def coste_mensual_local(m: dict, vol: int, precio_kwh: float, pue: float) -> float:
    """Electricidad (reposo 24/7 + carga proporcional) + mantenimiento."""
    horas_carga = vol / m["consultas_h_pico"]
    kwh = ((m["pot_reposo_w"] * (730 - horas_carga)
            + m["pot_carga_w"] * horas_carga) / 1000) * pue
    capex = m["capex_gpu"] + m["capex_resto"]
    return kwh * precio_kwh + capex * MANTENIMIENTO_ANUAL_PCT / 12


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--precio-kwh", type=float, default=PRECIO_KWH_DEF)
    ap.add_argument("--pue", type=float, default=PUE_DEF)
    ap.add_argument("--sin-figuras", action="store_true")
    args = ap.parse_args()

    res = json.load(open(os.path.join(COSTES, "resumen_global.json"), encoding="utf-8"))
    tok_in = res["tokens"]["entrada_total"]
    tok_out = res["tokens"]["salida_total"]
    n_consultas = res["consultas"]["total"]
    kwh_medida = res["energia_medida_kwh_total"]
    n_telemetria = res["cobertura_telemetria"]["consultas_bajo_telemetria"]
    cobertura = n_telemetria / n_consultas

    # ── (A) Coste local de lo ya ejecutado ──────────────────────────────────
    wh_por_consulta = kwh_medida * 1000 / n_telemetria
    kwh_extrapolada = (n_consultas - n_telemetria) * wh_por_consulta / 1000
    kwh_gpu = kwh_medida + kwh_extrapolada
    kwh_pared = kwh_gpu * args.pue
    coste_electrico = kwh_pared * args.precio_kwh
    banda_electrico = tuple(kwh_pared * p for p in PRECIO_KWH_BANDA)

    horas_experimento = res["horas_telemetria"] / cobertura
    amortizacion = (MAQUINA_ESTUDIO["capex"] * horas_experimento
                    / (VIDA_UTIL_MESES * 30 * 24))

    # ── (B) Coste en la nube de lo ya ejecutado ─────────────────────────────
    filas_exp = []
    for tid, etiq, prov, pin, pout, pcache, gama, es_ent in TARIFAS:
        c_in, c_out = eur(pin, tok_in), eur(pout, tok_out)
        total = c_in + c_out
        # Con caché de prefijo: la fracción estable del prompt se factura a
        # precio de lectura de caché; el resto, a precio normal.
        c_in_cache = (eur(pin, tok_in * (1 - FRACCION_CACHEABLE))
                      + eur(pcache, tok_in * FRACCION_CACHEABLE))
        filas_exp.append({
            "id": tid, "opcion": etiq, "proveedor": prov, "gama": gama,
            "gama_entrada": es_ent,
            "precio_entrada_usd_mtok": pin,
            "precio_salida_usd_mtok": pout,
            "coste_entrada_eur": round(c_in, 2),
            "coste_salida_eur": round(c_out, 2),
            "coste_total_eur": round(total, 2),
            "coste_con_cache_eur": round(c_in_cache + c_out, 2),
            "coste_por_lote_eur": round(total * DESCUENTO_LOTE, 2),
            "eur_por_1000_consultas": round(total / n_consultas * 1000, 2),
            "pct_coste_entrada": round(100 * c_in / total, 1),
            "veces_vs_local": round(total / coste_electrico, 1),
        })
    filas_exp.sort(key=lambda r: -r["coste_total_eur"])

    # ── (C) Coste de un servicio y punto de equilibrio ──────────────────────
    escenarios = {}
    for vol in VOLUMENES:
        tin, tout = vol * TOKENS_IN_CONSULTA, vol * TOKENS_OUT_CONSULTA
        for tid, etiq, prov, pin, pout, pcache, gama, es_ent in TARIFAS:
            escenarios[(vol, f"api::{tid}")] = {
                "capex": 0.0, "mensual": coste_api(pin, pout, tin, tout),
                "tipo": "api", "etiqueta": etiq, "proveedor": prov, "gama": gama}
        for iid, i in IAAS.items():
            escenarios[(vol, f"iaas::{iid}")] = {
                "capex": 0.0, "mensual": i["eur_h"] * 24 * 30,
                "tipo": "iaas", "etiqueta": i["etiqueta"], "proveedor": "IaaS",
                "gama": "alquiler"}
        for mid, m in MAQUINAS.items():
            escenarios[(vol, f"local::{mid}")] = {
                "capex": m["capex_gpu"] + m["capex_resto"],
                "mensual": coste_mensual_local(m, vol, args.precio_kwh, args.pue),
                "tipo": "local", "etiqueta": m["etiqueta"], "proveedor": "local",
                "gama": "compra"}

    filas_serv = [
        {"preguntas_mes": vol, "escenario": key, "etiqueta": e["etiqueta"],
         "tipo": e["tipo"], "mes": mes,
         "coste_acumulado_eur": round(e["capex"] + e["mensual"] * mes, 2)}
        for (vol, key), e in sorted(escenarios.items())
        for mes in range(0, HORIZONTE_MESES + 1)
    ]

    filas_be = []
    for vol in VOLUMENES:
        locales = [v for (v_, k), v in escenarios.items()
                   if v_ == vol and v["tipo"] == "local"]
        otras = [v for (v_, k), v in escenarios.items()
                 if v_ == vol and v["tipo"] != "local"]
        for lv in locales:
            for ov in otras:
                d = ov["mensual"] - lv["mensual"]
                mes_be = lv["capex"] / d if d > 0 else None
                filas_be.append({
                    "preguntas_mes": vol,
                    "opcion_local": lv["etiqueta"],
                    "alternativa": ov["etiqueta"],
                    "proveedor": ov["proveedor"],
                    "gama": ov["gama"],
                    "capex_local_eur": lv["capex"],
                    "coste_mensual_local_eur": round(lv["mensual"], 2),
                    "coste_mensual_alternativa_eur": round(ov["mensual"], 2),
                    "breakeven_meses": round(mes_be, 1) if mes_be else "",
                    "ahorro_a_60_meses_eur": round(
                        ov["mensual"] * 60 - (lv["capex"] + lv["mensual"] * 60), 2),
                    "nota": "" if mes_be else "la alternativa cuesta menos al mes",
                })

    # ── Sensibilidad ────────────────────────────────────────────────────────
    filas_sens = []
    for precio in [0.10, 0.12, 0.145, 0.17, 0.20, 0.25]:
        for mid, m in MAQUINAS.items():
            for vol in VOLUMENES:
                mens = coste_mensual_local(m, vol, precio, args.pue)
                fila = {"precio_kwh": precio, "maquina": m["etiqueta"],
                        "preguntas_mes": vol,
                        "coste_mensual_local_eur": round(mens, 2)}
                for tid, etiq, prov, pin, pout, _pc, gama, es_ent in TARIFAS:
                    if not es_ent:
                        continue
                    mens_api = coste_api(pin, pout, vol * TOKENS_IN_CONSULTA,
                                         vol * TOKENS_OUT_CONSULTA)
                    d = mens_api - mens
                    capex = m["capex_gpu"] + m["capex_resto"]
                    fila[f"be_{tid}"] = round(capex / d, 1) if d > 0 else ""
                filas_sens.append(fila)

    # ── Escritura ───────────────────────────────────────────────────────────
    os.makedirs(COSTES, exist_ok=True)

    def dump(name, rows):
        with open(os.path.join(COSTES, name), "w", newline="",
                  encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"  → eval/costes/{name} ({len(rows)} filas)")

    dump("costes_experimentos.csv", filas_exp)
    dump("costes_servicio.csv", filas_serv)
    dump("breakeven.csv", filas_be)
    dump("sensibilidad.csv", filas_sens)

    salida = {
        "consultado": "2026-08-26",
        "parametros": {
            "precio_kwh_eur": args.precio_kwh,
            "precio_kwh_banda": list(PRECIO_KWH_BANDA),
            "pue": args.pue, "usd_eur": USD_EUR,
            "vida_util_meses": VIDA_UTIL_MESES,
            "mantenimiento_anual_pct": MANTENIMIENTO_ANUAL_PCT,
            "descuento_lote": DESCUENTO_LOTE,
            "fraccion_cacheable": round(FRACCION_CACHEABLE, 4),
            "iaas": IAAS, "maquinas": MAQUINAS,
            "tokens_por_consulta": {"entrada": TOKENS_IN_CONSULTA,
                                    "salida": TOKENS_OUT_CONSULTA},
        },
        "tarifas": [
            {"id": t[0], "modelo": t[1], "proveedor": t[2],
             "usd_entrada_mtok": t[3], "usd_salida_mtok": t[4],
             "usd_cache_mtok": t[5], "gama": t[6], "gama_entrada": t[7]}
            for t in TARIFAS
        ],
        "experimentos": {
            "consultas": n_consultas,
            "tokens_entrada": tok_in, "tokens_salida": tok_out,
            "energia_gpu_kwh_medida": round(kwh_medida, 2),
            "energia_gpu_kwh_extrapolada": round(kwh_extrapolada, 2),
            "energia_gpu_kwh_total": round(kwh_gpu, 2),
            "energia_pared_kwh": round(kwh_pared, 2),
            "wh_por_consulta": round(wh_por_consulta, 3),
            "coste_electrico_eur": round(coste_electrico, 2),
            "coste_electrico_banda_eur": [round(x, 2) for x in banda_electrico],
            "horas_maquina": round(horas_experimento, 1),
            "amortizacion_imputable_eur": round(amortizacion, 2),
            "coste_total_local_eur": round(coste_electrico + amortizacion, 2),
            "comparativa": filas_exp,
        },
        "servicio": {
            "volumenes": VOLUMENES,
            "coste_mensual": {
                str(vol): sorted(
                    [{"etiqueta": e["etiqueta"], "proveedor": e["proveedor"],
                      "tipo": e["tipo"], "gama": e["gama"],
                      "capex_eur": e["capex"], "mensual_eur": round(e["mensual"], 2)}
                     for (v_, k), e in escenarios.items() if v_ == vol],
                    key=lambda r: r["mensual_eur"])
                for vol in VOLUMENES
            },
        },
        "breakeven": filas_be,
    }
    with open(os.path.join(COSTES, "analisis_costes.json"), "w",
              encoding="utf-8") as fh:
        json.dump(salida, fh, ensure_ascii=False, indent=2)
    print("  → eval/costes/analisis_costes.json")

    if not args.sin_figuras:
        figuras(escenarios, filas_exp, salida, args)

    # ── Consola ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("(A) COSTE REAL DE LOS EXPERIMENTOS EN LOCAL")
    print(f"    {n_consultas:,} consultas · {tok_in/1e6:.1f} M entrada · "
          f"{tok_out/1e6:.1f} M salida")
    print(f"    Energía: {kwh_gpu:.1f} kWh GPU → {kwh_pared:.1f} kWh en la toma")
    print(f"    Electricidad: {coste_electrico:.2f} € "
          f"(banda {banda_electrico[0]:.2f}–{banda_electrico[1]:.2f} €)")
    print(f"    + amortización imputable: {amortizacion:.2f} €  "
          f"→ TOTAL {coste_electrico + amortizacion:.2f} €")
    print("\n(B) LOS MISMOS TOKENS EN LA NUBE")
    print(f"    {'Modelo':<24}{'Proveedor':<11}{'Gama':<12}"
          f"{'Total':>10}  {'€/1k cons.':>10}  {'vs local':>9}")
    for r in filas_exp:
        ent = " ◂ entrada" if r["gama_entrada"] else ""
        print(f"    {r['opcion']:<24}{r['proveedor']:<11}{r['gama']:<12}"
              f"{r['coste_total_eur']:>9,.0f} €  {r['eur_por_1000_consultas']:>9.2f} €"
              f"  ×{r['veces_vs_local']:>8,.0f}{ent}")
    _cap = MAQUINAS["5090"]["capex_gpu"] + MAQUINAS["5090"]["capex_resto"]
    print(f"\n(C) PUNTO DE EQUILIBRIO DE LA ESTACIÓN RTX 5090 ({_cap:,} €)"
          .replace(",", "."))
    ent = [t for t in TARIFAS if t[7]]
    print(f"    {'preguntas/mes':>14}" + "".join(f"{t[1]:>22}" for t in ent))
    for vol in VOLUMENES:
        celdas = []
        for t in ent:
            r = next(x for x in filas_be
                     if x["preguntas_mes"] == vol and x["alternativa"] == t[1]
                     and x["opcion_local"] == "Estación RTX 5090")
            celdas.append(f"{r['breakeven_meses']} meses" if r["breakeven_meses"]
                          else "nunca")
        print(f"    {vol:>14,}" + "".join(f"{c:>22}" for c in celdas))
    print("=" * 80)


def figuras(escenarios, filas_exp, salida, args):
    import estilo_tfm as st
    import matplotlib.pyplot as plt
    import numpy as np

    from matplotlib.ticker import FuncFormatter

    st.aplicar_estilo()
    meses = np.arange(0, HORIZONTE_MESES + 1)
    p = salida["parametros"]

    def coma(dec=0):
        """Eje con coma decimal y punto de millar, como en el resto de la memoria."""
        return FuncFormatter(
            lambda v, _p: f"{v:,.{dec}f}".replace(",", "\x00").replace(".", ",")
                                          .replace("\x00", "."))

    # IaaS va en azul oscuro (#004488, azul de la paleta de alto contraste de
    # Tol). El azul original (#0072B2) se confundía con el celeste de OpenAI
    # (#88CCEE) por ser de luminosidad parecida; este, mucho más oscuro, se
    # separa de él sin salirse de la gama azul.
    PROV_COLOR = {"Anthropic": "#CC6677", "Google": "#DDCC77",
                  "OpenAI": "#88CCEE", "IaaS": "#004488", "local": "#117733"}
    MAQ_COLOR = {"Estación RTX 5090": "#117733", "Servidor 1× PRO 6000": "#44AA99"}

    # ── Fig 1 · Coste del estudio por proveedor y gama ───────────────────────
    fig, ax = plt.subplots(figsize=(st.ANCHO_TEXTO, 3.9))
    filas = sorted(filas_exp, key=lambda r: r["coste_total_eur"])
    y = np.arange(len(filas))
    ax.barh(y, [r["coste_total_eur"] for r in filas],
            color=[PROV_COLOR[r["proveedor"]] for r in filas],
            edgecolor=["#222222" if r["gama_entrada"] else "none" for r in filas],
            linewidth=[0.9 if r["gama_entrada"] else 0 for r in filas], height=0.68)
    loc = salida["experimentos"]["coste_electrico_eur"]
    ax.axvline(loc, color="#117733", lw=1.4)
    ax.text(loc * 0.88, len(filas) / 2,
            f"electricidad local · {loc:.2f} €".replace(".", ","),
            fontsize=7.5, color="#117733", rotation=90, va="center", ha="center")
    ax.set_yticks(y)
    ax.set_yticklabels([r["opcion"] for r in filas])
    ax.set_xscale("log")
    ax.set_xlim(left=loc * 0.45,
                right=max(r["coste_total_eur"] for r in filas) * 3.4)
    for yi, r in zip(y, filas):
        ax.text(r["coste_total_eur"] * 1.12, yi,
                f"{r['coste_total_eur']:,.0f} €".replace(",", "."),
                va="center", fontsize=7.2)
    ax.set_xlabel("coste del estudio completo (€, escala logarítmica)")
    manijas = [plt.Line2D([], [], marker="s", ls="", ms=7, color=c, label=k)
               for k, c in PROV_COLOR.items() if k in ("Anthropic", "Google", "OpenAI")]
    manijas.append(plt.Line2D([], [], marker="s", ls="", ms=7, mfc="white",
                              mec="#222222", mew=0.9, label="gama de entrada"))
    ax.legend(handles=manijas, loc="lower right", ncol=2, fontsize=7)
    st.save(fig, "coste_estudio_proveedores")

    # ── Fig 1b · Versión simplificada: gama de entrada, local y alquiler ─────
    # Mismo conjunto de opciones que la tabla comparativa reducida.
    e = salida["experimentos"]
    horas = e["horas_maquina"]
    entrada = {r["opcion"]: r for r in filas_exp if r["gama_entrada"]}
    opciones = [
        # El coste local comparable con el precio íntegro de una API no es el
        # recibo de la luz, sino este más la amortización imputable a las horas
        # de máquina del estudio (ver "amortizacion_imputable_eur").
        ("Local (luz + amortización)", e["coste_total_local_eur"], PROV_COLOR["local"]),
        ("GPU alquilada (%.0f h)" % horas,
         IAAS["iaas_bajo"]["eur_h"] * horas, PROV_COLOR["IaaS"]),
    ] + [(r["opcion"], r["coste_total_eur"], PROV_COLOR[r["proveedor"]])
         for r in sorted(entrada.values(), key=lambda r: r["coste_total_eur"])]
    opciones.sort(key=lambda o: o[1])

    fig, ax = plt.subplots(figsize=(st.ANCHO_TEXTO, 2.9))
    y = np.arange(len(opciones))
    ax.barh(y, [o[1] for o in opciones], color=[o[2] for o in opciones],
            height=0.66, edgecolor="#333333", linewidth=0.5)
    # Separación constante en unidades del eje: con escala lineal, un desfase
    # multiplicativo alejaría la etiqueta de las barras largas y pegaría la de
    # las cortas.
    tope = max(o[1] for o in opciones)
    hueco = tope * 0.012
    for yi, o in zip(y, opciones):
        txt = (f"{o[1]:,.2f}".replace(",", "\x00").replace(".", ",")
                             .replace("\x00", ".") if o[1] < 10
               else f"{o[1]:,.0f}".replace(",", "."))
        ax.text(o[1] + hueco, yi, txt + " €", va="center", fontsize=7.5)
    ax.set_yticks(y)
    ax.set_yticklabels([o[0] for o in opciones])
    ax.set_xlim(0, max(o[1] for o in opciones) * 1.18)
    ax.xaxis.set_major_formatter(coma())
    ax.set_xlabel("costes de los experimentos (€)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    st.save(fig, "coste_estudio_simple")

    # ── Fig 2 · Coste acumulado del servicio, cuatro volúmenes ───────────────
    fig, axes = plt.subplots(2, 2, figsize=(st.ANCHO_TEXTO, 5.4), sharex=True)
    MOSTRAR = {"haiku45", "g31lite", "gpt56luna"}
    ref = "Estación RTX 5090"
    OCULTAR = {"local::pro6000"}     # el servidor se retira de esta figura
    for ax, vol in zip(axes.flat, VOLUMENES):
        sel = {k: v for (v_, k), v in escenarios.items()
               if v_ == vol and k not in OCULTAR}
        e_ref = next(v for v in sel.values() if v["etiqueta"] == ref)
        for k, e in sorted(sel.items(), key=lambda kv: kv[1]["etiqueta"]):
            if e["tipo"] == "api" and k.split("::")[1] not in MOSTRAR:
                continue
            if k == "iaas::iaas_alto":
                continue
            color = (MAQ_COLOR[e["etiqueta"]] if e["tipo"] == "local"
                     else PROV_COLOR[e["proveedor"]])
            # Trazo continuo para las dos opciones de coste fijo (equipo propio
            # y GPU alquilada); discontinuo solo para las APIs por token.
            ls = "--" if e["tipo"] == "api" else "-"
            # En esta figura solo se dibuja el proveedor de bajo coste, así que
            # el matiz sobra en la leyenda; los CSV conservan el nombre completo
            # porque ahí sí conviven las dos tarifas de IaaS.
            etq = e["etiqueta"].replace(" (bajo coste)", "")
            ax.plot(meses, e["capex"] + e["mensual"] * meses, ls, lw=1.25,
                    color=color, label=etq if vol == VOLUMENES[0] else None)
            if e["tipo"] != "local":
                d = e["mensual"] - e_ref["mensual"]
                if d > 0 and 0 < e_ref["capex"] / d <= HORIZONTE_MESES:
                    t = e_ref["capex"] / d
                    ax.plot(t, e_ref["capex"] + e_ref["mensual"] * t, "o", ms=3.8,
                            mfc="white", mew=1.1, mec=color, zorder=5)
        ax.set_ylim(0, (e_ref["capex"] + e_ref["mensual"] * HORIZONTE_MESES) * 3.0)
        ax.set_title(f"{vol:,} preguntas/mes".replace(",", "."), fontsize=8.5)
        ax.set_ylabel("coste acumulado (€)")
        ax.yaxis.set_major_formatter(coma())
        # Los cuatro paneles llevan rótulo y números en el eje x: comparten
        # escala, pero con dos filas el lector pierde la referencia si solo la
        # tiene abajo.
        ax.tick_params(labelbottom=True)
        ax.set_xlabel("meses de servicio")
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.10))
    st.save(fig, "coste_acumulado_servicio")

    # ── Fig 2b · Coste acumulado frente al volumen mensual ───────────────────
    # Traspone la figura anterior: en lugar de fijar el volumen y recorrer los
    # meses, fija el horizonte y recorre el volumen. Es la vista que responde a
    # "¿a partir de cuántas preguntas al mes compensa la compra?".
    MOSTRAR_V = {"haiku45", "g31lite", "gpt56luna"}
    HORIZONTES = [(12, "al cabo de 1 año"), (60, "al cabo de 5 años")]
    vols_x = np.logspace(np.log10(500), np.log10(200_000), 200)
    maq = MAQUINAS["5090"]
    capex_loc = maq["capex_gpu"] + maq["capex_resto"]

    fig, axes = plt.subplots(1, 2, figsize=(st.ANCHO_TEXTO, 3.3), sharey=True)
    for ax, (T, titulo) in zip(axes, HORIZONTES):
        loc = np.array([capex_loc + coste_mensual_local(maq, v, args.precio_kwh,
                                                        args.pue) * T
                        for v in vols_x])
        ax.plot(vols_x, loc, "-", lw=1.5, color=MAQ_COLOR["Estación RTX 5090"],
                label="Estación RTX 5090" if T == HORIZONTES[0][0] else None,
                zorder=4)
        ax.plot(vols_x, np.full_like(vols_x, IAAS["iaas_bajo"]["eur_h"] * 24 * 30 * T),
                ":", lw=1.4, color=PROV_COLOR["IaaS"],
                label="GPU alquilada (bajo coste)" if T == HORIZONTES[0][0] else None)
        for tid, etiq, prov, pin, pout, _pc, _g, _e in TARIFAS:
            if tid not in MOSTRAR_V:
                continue
            api = np.array([coste_api(pin, pout, v * TOKENS_IN_CONSULTA,
                                      v * TOKENS_OUT_CONSULTA) * T for v in vols_x])
            ax.plot(vols_x, api, "--", lw=1.4, color=PROV_COLOR[prov],
                    label=etiq if T == HORIZONTES[0][0] else None)
            # Volumen en que la compra pasa a ser más barata que la API.
            cruce = np.where(api > loc)[0]
            if len(cruce):
                i = cruce[0]
                ax.plot(vols_x[i], loc[i], "o", ms=4.2, mfc="white", mew=1.2,
                        mec=PROV_COLOR[prov], zorder=6)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("preguntas al mes")
        ax.set_title(titulo, fontsize=8.5)
        ax.xaxis.set_major_formatter(coma())
        ax.yaxis.set_major_formatter(coma())
    axes[0].set_ylabel("coste acumulado (€)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.16))
    st.save(fig, "coste_volumen_acumulado")

    # ── Fig 3 · Meses hasta amortizar, frente al volumen ─────────────────────
    fig, ax = plt.subplots(figsize=(st.ANCHO_TEXTO, 3.7))
    vols = np.logspace(2.7, 5.7, 220)
    for mid, m in MAQUINAS.items():
        capex = m["capex_gpu"] + m["capex_resto"]
        for t in TARIFAS:
            if not t[7]:
                continue
            tid, etiq, prov, pin, pout = t[0], t[1], t[2], t[3], t[4]
            be = [(capex / d if (d := coste_api(pin, pout, v * TOKENS_IN_CONSULTA,
                                                v * TOKENS_OUT_CONSULTA)
                                 - coste_mensual_local(m, v, args.precio_kwh,
                                                       args.pue)) > 0 else np.nan)
                  for v in vols]
            ax.plot(vols, be, lw=1.25, color=PROV_COLOR[prov],
                    ls="-" if mid == "5090" else "--",
                    label=f"{m['etiqueta'].replace('Estación ', '').replace('Servidor ', '')}"
                          f" vs {etiq}")
    ax.axhspan(VIDA_UTIL_MESES, 400, color="#bbbbbb", alpha=0.25, lw=0)
    ax.axhline(VIDA_UTIL_MESES, color="#555555", lw=0.9, ls=":")
    ax.axhline(12, color="#555555", lw=0.7, ls="-.")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(vols[0], vols[-1]); ax.set_ylim(0.3, 400)
    ax.text(vols[-1] * 0.96, VIDA_UTIL_MESES * 1.3,
            "no se amortiza dentro de la vida útil", fontsize=7,
            color="#555555", ha="right")
    ax.text(vols[0] * 1.1, 12 * 1.16, "un año", fontsize=7, color="#555555")
    ax.set_xlabel("preguntas al mes")
    ax.set_ylabel("meses hasta amortizar la compra")
    ax.xaxis.set_major_formatter(coma())
    ax.legend(loc="lower center", ncol=2, fontsize=6.8, frameon=False,
              bbox_to_anchor=(0.5, -0.46))
    st.save(fig, "coste_breakeven_volumen")

    # ── Fig 4 · Sensibilidad al precio de la electricidad ────────────────────
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(st.ANCHO_TEXTO, 3.0))
    precios = np.linspace(0.08, 0.28, 120)
    for mid, m in MAQUINAS.items():
        a1.plot(precios, [coste_mensual_local(m, 20000, pk, args.pue)
                          for pk in precios],
                lw=1.4, color=MAQ_COLOR[m["etiqueta"]], label=m["etiqueta"])
    a1.axvspan(*PRECIO_KWH_BANDA, color="#117733", alpha=0.12, lw=0)
    a1.axvline(args.precio_kwh, color="#117733", lw=0.9, ls=":")
    a1.set_xlabel("precio de la electricidad (€/kWh)")
    a1.set_ylabel("coste mensual local (€)")
    a1.set_ylim(bottom=0)
    a1.xaxis.set_major_formatter(coma(2))
    a1.legend(loc="upper left", fontsize=7)
    a1.set_title("Coste mensual a 20.000 preguntas/mes", fontsize=8.5)

    # Panel derecho: volumen mensual mínimo que amortiza la compra en dos años.
    # Es la magnitud accionable y, a diferencia del tiempo de amortización, no
    # diverge. Se resuelve en forma cerrada: con coste mensual local A + B·v y
    # coste de API k·v, exigir capex/(k·v − A − B·v) ≤ T da
    #     v ≥ (capex/T + A) / (k − B).
    OBJETIVO_MESES = 24
    m = MAQUINAS["5090"]
    capex = m["capex_gpu"] + m["capex_resto"]
    for t in TARIFAS:
        if not t[7]:
            continue
        k = coste_api(t[3], t[4], TOKENS_IN_CONSULTA, TOKENS_OUT_CONSULTA)
        umbral = []
        for pk in precios:
            A = (m["pot_reposo_w"] * 730 / 1000) * args.pue * pk \
                + capex * MANTENIMIENTO_ANUAL_PCT / 12
            B = ((m["pot_carga_w"] - m["pot_reposo_w"]) / m["consultas_h_pico"]
                 / 1000) * args.pue * pk
            umbral.append((capex / OBJETIVO_MESES + A) / (k - B) if k > B else np.nan)
        a2.plot(precios, umbral, lw=1.4, color=PROV_COLOR[t[2]], label=t[1])
    a2.axvspan(*PRECIO_KWH_BANDA, color="#117733", alpha=0.12, lw=0)
    a2.set_ylim(0, 2.2e5)
    a2.set_xlabel("precio de la electricidad (€/kWh)")
    a2.set_ylabel("preguntas/mes (amortización ≤ 2 años)")
    a2.xaxis.set_major_formatter(coma(2))
    a2.yaxis.set_major_formatter(coma())
    a2.legend(loc="center left", fontsize=7)
    a2.set_title("Volumen mínimo que justifica la compra", fontsize=8.5)

    st.save(fig, "coste_sensibilidad_kwh")

    # ── Copia a la carpeta de imágenes de la memoria ────────────────────────
    os.makedirs(IMAGENES, exist_ok=True)
    for n in ("coste_estudio_proveedores", "coste_estudio_simple",
              "coste_acumulado_servicio", "coste_volumen_acumulado",
              "coste_breakeven_volumen", "coste_sensibilidad_kwh"):
        for ext in ("pdf", "png"):
            shutil.copy2(os.path.join(FIGS, f"{n}.{ext}"),
                         os.path.join(IMAGENES, f"{n}.{ext}"))
    print(f"  → memoria/Imagenes/coste_*.pdf|png")


if __name__ == "__main__":
    sys.exit(main())
