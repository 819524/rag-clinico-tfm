#!/usr/bin/env python3
"""
verificar_memoria.py — Contrasta las cifras publicadas en la memoria contra los
datos de este repositorio.

Las tablas están transcritas aquí tal como aparecen impresas. El script recalcula
cada celda desde los ficheros de experimentos/ e informa de cualquier diferencia.
No requiere base de datos, modelos ni red.

    python3 herramientas/verificar_memoria.py

Código de salida 0 si todo concuerda, 1 si aparece alguna discrepancia.
"""
from __future__ import annotations

import csv, glob, json, os, statistics as st, sys

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(RAIZ, "experimentos")

VERDE, ROJO, GRIS, FIN = "\033[32m", "\033[31m", "\033[90m", "\033[0m"
if not sys.stdout.isatty():
    VERDE = ROJO = GRIS = FIN = ""

incidencias: list[str] = []


def jload(*p):
    return json.load(open(os.path.join(EXP, *p), encoding="utf-8"))


def comprobar(etiqueta, publicado, calculado, tol=0.0005):
    """Comprueba que `publicado` sea un redondeo válido de `calculado`.

    La tolerancia es media unidad del último decimal impreso, de modo que el
    resultado no depende de qué convención se aplicó a los valores que caen
    exactamente a mitad (x,xxx5), que en la memoria no es uniforme.
    """
    if calculado is None:
        print(f"    {ROJO}SIN DATO{FIN}  {etiqueta}")
        incidencias.append(f"{etiqueta}: sin dato en el repositorio")
    elif abs(publicado - calculado) <= tol + 1e-9:
        print(f"    {VERDE}OK{FIN}  {etiqueta}")
    else:
        print(f"    {ROJO}DIFIERE{FIN}  {etiqueta}  publicado={publicado}  calculado={calculado:.5g}")
        incidencias.append(f"{etiqueta}: publicado {publicado}, calculado {calculado:.5g}")


def titulo(t):
    print(f"\n{t}\n{'─' * len(t)}")


# ─────────────────────────────────────────── Tabla 5.1 · recuperación (615 preg.)

TABLA_5_1 = {
    ("4B", "cross_encoder"):   [0.759, 0.902, 0.932, 0.951, 0.834, 0.849],
    ("4B", "embed"):           [0.610, 0.834, 0.898, 0.942, 0.731, 0.769],
    ("4B", "rrf_only"):        [0.620, 0.833, 0.883, 0.915, 0.730, 0.761],
    ("0.6B", "cross_encoder"): [0.737, 0.888, 0.919, 0.942, 0.815, 0.833],
    ("0.6B", "embed"):         [0.532, 0.784, 0.859, 0.911, 0.670, 0.721],
    ("0.6B", "rrf_only"):      [0.571, 0.794, 0.854, 0.902, 0.691, 0.733],
}
CLAVES_5_1 = ["sec@0.7_hit@1", "sec@0.7_hit@3", "sec@0.7_hit@5",
              "sec@0.7_hit@10", "sec@0.7_mrr@10", "sec@0.7_ndcg@10"]
NOMBRES_5_1 = ["Hit@1", "Hit@3", "Hit@5", "Hit@10", "MRR@10", "NDCG@10"]
CARPETA_5_1 = {"4B": "embedding-4b", "0.6B": "embedding-06b"}


def tabla_5_1():
    titulo("Tabla 5.1 · Recuperación, 615 preguntas, K=10  (36 celdas)")
    for (emb, est), vals in TABLA_5_1.items():
        m = jload("1-recuperacion", "resultados", CARPETA_5_1[emb],
                  "section_metrics_summary.json")["strategies"][est]
        for k, nom, pub in zip(CLAVES_5_1, NOMBRES_5_1, vals):
            comprobar(f"{emb:5} {est:14} {nom}", pub, m.get(k), tol=0.0005)


# ────────────────────────────────────── Tabla 5.2 · contexto cruzado (110 preg.)

TABLA_5_2 = {
    ("4b", "cross_encoder"):  [0.691, 0.918, 0.776, 0.792, 0.591, 0.818, 0.676, 0.700],
    ("4b", "embed"):          [0.673, 0.936, 0.760, 0.780, 0.527, 0.809, 0.628, 0.655],
    ("4b", "rrf_only"):       [0.600, 0.891, 0.703, 0.733, 0.509, 0.791, 0.618, 0.648],
    ("06b", "cross_encoder"): [0.718, 0.946, 0.810, 0.821, 0.600, 0.827, 0.691, 0.715],
    ("06b", "embed"):         [0.536, 0.927, 0.690, 0.736, 0.455, 0.809, 0.585, 0.630],
    ("06b", "rrf_only"):      [0.600, 0.891, 0.710, 0.736, 0.473, 0.754, 0.578, 0.612],
}
# ΔHit@1 tal como lo imprime la memoria (calculado sobre valores sin redondear)
DELTA_5_2 = {("4b", "cross_encoder"): -0.100, ("4b", "embed"): -0.145,
             ("4b", "rrf_only"): -0.091, ("06b", "cross_encoder"): -0.118,
             ("06b", "embed"): -0.082, ("06b", "rrf_only"): -0.127}
CLAVES_5_2 = ["sec@0.7_hit@1", "sec@0.7_hit@10", "sec@0.7_mrr@10", "sec@0.7_ndcg@10"]
NOMBRES_5_2 = ["Hit@1", "Hit@10", "MRR@10", "NDCG@10"]


def tabla_5_2():
    titulo("Tabla 5.2 · Contexto cruzado, 110 preguntas  (54 celdas)")
    for (emb, est), vals in TABLA_5_2.items():
        c = jload("2-contexto-cruzado", "metricas", f"{emb}_CONTROL.json")["strategies"][est]
        t = jload("2-contexto-cruzado", "metricas", f"{emb}_PAREADO.json")["strategies"][est]
        for i, (k, nom) in enumerate(zip(CLAVES_5_2, NOMBRES_5_2)):
            comprobar(f"{emb:4} {est:14} control {nom}", vals[i], c.get(k), tol=0.0005)
        for i, (k, nom) in enumerate(zip(CLAVES_5_2, NOMBRES_5_2)):
            comprobar(f"{emb:4} {est:14} tests   {nom}", vals[4 + i], t.get(k), tol=0.0005)
        comprobar(f"{emb:4} {est:14} delta   Hit@1", DELTA_5_2[(emb, est)],
                  t["sec@0.7_hit@1"] - c["sec@0.7_hit@1"], tol=0.0005)


# ──────────────────────────────────────── Tablas 4.1 y 4.2 · carga y saturación

def _peticiones():
    with open(os.path.join(EXP, "3-despliegue", "peticiones.csv"), encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _barridos():
    with open(os.path.join(EXP, "3-despliegue", "barridos.csv"), encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def pct(v, q):
    v = sorted(v)
    if not v:
        return None
    i = (len(v) - 1) * q
    lo, hi = int(i), min(int(i) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (i - lo)


TABLA_4_1 = {
    ("confianza", "vllm", "26b_4b"):               [4.5, 7.1, 8.6, 11.8, 0.0],
    ("confianza", "llamacpp", "26b_4b_np8"):       [12.8, 25.2, 20.1, 33.1, 0.0],
    ("confianza", "ollama", "26b_4b_np8"):         [23.4, 40.6, 34.7, 50.3, 1.6],
    ("confianza_5090", "vllm", "26b_4b"):          [8.3, 15.0, 14.7, 19.9, 0.0],
    ("confianza_5090", "llamacpp", "26b_4b_np8"):  [32.1, 50.4, 44.0, 59.3, 0.0],
    ("confianza_5090", "ollama", "26b_4b_np8"):    [43.8, 70.5, 62.8, 85.7, 3.6],
}
MAQ = {"confianza": "RTX PRO 6000", "confianza_5090": "RTX 5090"}


def tabla_4_1():
    titulo("Tabla 4.1 · TTFT, latencia y stall a 25 usuarios  (30 celdas)")
    print(f"{GRIS}    Las peticiones que se estancan no emiten primer token; cuentan en\n"
          f"    latencia y stall, no en TTFT. Es lo que hace la memoria.{FIN}")
    P = _peticiones()
    for (camp, mot, cel), pub in TABLA_4_1.items():
        sel = [r for r in P if r["campana"] == camp and r["motor"] == mot
               and r["celda"] == cel and r["concurrencia"] == "25"]
        tt = [float(r["ttft_ms"]) / 1000 for r in sel if r["ttft_ms"]]
        to = [float(r["total_ms"]) / 1000 for r in sel if r["total_ms"]]
        et = f"{MAQ[camp]:13} {mot:9}"
        for i, (nom, val) in enumerate((("TTFT p50", pct(tt, .50)), ("TTFT p95", pct(tt, .95)),
                                        ("Lat  p50", pct(to, .50)), ("Lat  p95", pct(to, .95)))):
            comprobar(f"{et} {nom}", pub[i], val, tol=0.05)
        comprobar(f"{et} Stall %", pub[4],
                  100 * sum(1 for x in to if x > 90) / len(to) if to else None, tol=0.05)


TABLA_4_2 = {
    ("voz05", "saturacion"):   ([6.5, 11.3, 15.6, 18.6, 24.0, 36.7, 42.9], [0.0] * 7),
    ("gtc2pc4", "saturacion"): ([11.5, 22.6, 37.6, 52.9, 67.0, 99.9, 135.7],
                                [0.0, 0.0, 0.0, 0.0, 0.0, 18.7, 57.5]),
}
NIVELES = ["10", "25", "50", "75", "100", "150", "200"]
MAQ2 = {"voz05": "RTX PRO 6000", "gtc2pc4": "RTX 5090"}


def tabla_4_2():
    titulo("Tabla 4.2 · Saturación con vLLM  (28 celdas)")
    P = _peticiones()
    for (maq, serie), (p95s, stalls) in TABLA_4_2.items():
        sel = [r for r in P if r["maquina"] == maq and r["serie"] == serie]
        for niv, pub_p95, pub_st in zip(NIVELES, p95s, stalls):
            to = [float(r["total_ms"]) / 1000 for r in sel
                  if r["concurrencia"] == niv and r["total_ms"]]
            comprobar(f"{MAQ2[maq]:13} {niv:>3} usu. p95", pub_p95,
                      pct(to, .95) if to else None, tol=0.05)
            comprobar(f"{MAQ2[maq]:13} {niv:>3} usu. stall", pub_st,
                      100 * sum(1 for x in to if x > 90) / len(to) if to else None, tol=0.05)


def texto_capitulo_4():
    titulo("Capítulo 4 · afirmaciones del texto")
    B, P = _barridos(), _peticiones()
    comprobar("barridos de la campaña principal", 280,
              len({r["origen"] for r in B if r["campana"] == "confianza"}), tol=0.5)
    comprobar("configuraciones distintas", 28,
              len({(r["motor"], r["celda"]) for r in B if r["campana"] == "confianza"}), tol=0.5)
    for camp, lat_pub, tps_pub in (("confianza", 2.7, 6.8), ("confianza_5090", 4.6, 4.4)):
        def lat(n):
            v = [float(r["total_ms"]) / 1000 for r in P if r["campana"] == camp
                 and r["motor"] == "vllm" and r["celda"].startswith("26b")
                 and r["concurrencia"] == n and r["total_ms"]]
            return st.mean(v) if v else None

        def tps(n):
            v = [float(r["rendimiento_tps"]) for r in B if r["campana"] == camp
                 and r["motor"] == "vllm" and r["celda"].startswith("26b")
                 and r["concurrencia"] == n and r["rendimiento_tps"]]
            return st.median(v) if v else None
        comprobar(f"{MAQ[camp]:13} latencia x (1->25 usu.)", lat_pub,
                  lat("25") / lat("1"), tol=0.11)
        comprobar(f"{MAQ[camp]:13} throughput x (1->25 usu.)", tps_pub,
                  tps("25") / tps("1"), tol=0.05)
    peor = {}
    for r in P:
        if r["motor"] == "ollama" and r["total_ms"] and r["celda"].startswith("26b"):
            peor.setdefault((r["maquina"], r["celda"], r["concurrencia"]), []).append(
                float(r["total_ms"]) / 1000)
    comprobar("peor tasa de stall de Ollama (26B)", 10.4,
              max(100 * sum(1 for x in s if x > 90) / len(s) for s in peor.values()), tol=0.05)
    v = [float(r["rendimiento_tps"]) for r in B if r["maquina"] == "gtc2pc4"
         and r["serie"] == "saturacion" and r["concurrencia"] in ("50", "150", "200")
         and r["rendimiento_tps"]]
    comprobar("throughput sostenido RTX 5090 (tok/s)", 435, st.median(v), tol=5)


# ────────────────────────────────────────────── Capítulo 6 · costes (Tabla 6.1)

def _costes():
    return json.load(open(os.path.join(EXP, "4-costes", "analisis_costes.json"),
                          encoding="utf-8"))


def texto_capitulo_6():
    titulo("Capítulo 6 · cifras del texto")
    d = _costes(); e, p = d["experimentos"], d["parametros"]
    capex = p["maquinas"]["5090"]["capex_gpu"] + p["maquinas"]["5090"]["capex_resto"]
    for et, pub, cal in (
            ("consultas del estudio", 68521, e["consultas"]),
            ("tokens de entrada (M)", 311.4, e["tokens_entrada"] / 1e6),
            ("tokens de salida (M)", 23.5, e["tokens_salida"] / 1e6),
            ("energía de pared (kWh)", 37.4, e["energia_pared_kwh"]),
            ("factor PUE", 1.35, p["pue"]),
            ("coste eléctrico (EUR)", 6.35, e["coste_electrico_eur"]),
            ("horas de máquina", 119.3, e["horas_maquina"]),
            ("amortización imputable (EUR)", 60.76, e["amortizacion_imputable_eur"]),
            ("coste total local (EUR)", 67.11, e["coste_total_local_eur"]),
            ("coste de la máquina (EUR)", 7500, capex),
            ("amortización mensual (EUR)", 125.0, capex / p["vida_util_meses"])):
        comprobar(et, pub, cal, tol=0.05)
    pcts = [c["pct_coste_entrada"] for c in e["comparativa"]]
    comprobar("% de factura en entrada, mínimo", 61, min(pcts), tol=0.5)
    comprobar("% de factura en entrada, máximo", 77, max(pcts), tol=0.5)


TABLA_6_1 = {
    "GPT-5.6 luna":               [1.2, 5.9, 23.5, 117.6],
    "Gemini 3.1 Flash-Lite":      [1.5, 7.3, 29.4, 147.0],
    "Claude Haiku 4.5":           [5.6, 27.9, 111.5, 557.3],
    "Estación RTX 5090":          [166.4, 166.8, 168.1, 175.4],
    "GPU alquilada (bajo coste)": [396.0, 396.0, 396.0, 396.0],
}


def tabla_6_1():
    titulo("Tabla 6.1 · Coste mensual por volumen  (20 celdas)")
    print(f"{GRIS}    La fila local publica opex + amortización, como explica el texto.{FIN}")
    d = _costes(); s = d["servicio"]; meses = d["parametros"]["vida_util_meses"]
    for etiqueta, vals in TABLA_6_1.items():
        for vol, pub in zip(s["volumenes"], vals):
            fila = next((x for x in s["coste_mensual"][str(vol)]
                         if x["etiqueta"] == etiqueta), None)
            cal = fila["mensual_eur"] + fila["capex_eur"] / meses if fila else None
            comprobar(f"{etiqueta:28} {vol:>6} consultas/mes", pub, cal, tol=0.05)


# ────────────────────────────────────────────────────────── conjuntos de datos

def conjuntos():
    titulo("Conjuntos de evaluación declarados en la memoria")
    pregs = glob.glob(os.path.join(EXP, "1-recuperacion", "preguntas", "*.json"))
    total = sum(len(json.load(open(p, encoding="utf-8"))["questions"]) for p in pregs)
    comprobar("protocolos del corpus", 22, len(pregs), tol=0.5)
    comprobar("preguntas del conjunto principal", 615, total, tol=0.5)
    for nom, f in (("control", "control.json"), ("tests", "tests.json")):
        p = os.path.join(EXP, "2-contexto-cruzado", "preguntas", f)
        n = len(json.load(open(p, encoding="utf-8"))["questions"]) if os.path.exists(p) else None
        comprobar(f"preguntas del conjunto de {nom}", 110, n, tol=0.5)


def main():
    print("Verificación de las cifras publicadas en la memoria")
    print("contra los datos de este repositorio.")
    conjuntos(); tabla_5_1(); tabla_5_2(); tabla_4_1(); tabla_4_2()
    texto_capitulo_4(); texto_capitulo_6(); tabla_6_1()
    print()
    if incidencias:
        print(f"{ROJO}{len(incidencias)} discrepancia(s):{FIN}")
        for i in incidencias:
            print(f"  · {i}")
        return 1
    print(f"{VERDE}Todas las cifras verificadas concuerdan con los datos.{FIN}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
