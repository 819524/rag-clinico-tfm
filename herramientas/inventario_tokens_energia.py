#!/usr/bin/env python3
"""
inventario_tokens_energia.py — Inventario unificado de tokens y energía de todos
los experimentos del TFM.
================================================================================
Cruza cuatro fuentes de verdad y produce un inventario por experimento:

  1) logs/rag_queries_pilot.jsonl  → tokens REALES (prompt_eval_count/eval_count
     de Ollama, usage.* de la API OpenAI de vLLM/llama.cpp). Es el libro mayor
     unificado: por él pasan las evaluaciones, las pruebas de carga y el piloto.
  2) eval/**/results_*.json        → tokens reales de las evaluaciones (incluidas
     las anteriores al 2026-06-15, que no están en el libro mayor).
  3) eval/**/sweep.json            → estructura de las pruebas de carga
     (motor, modelo, NP, nivel de concurrencia, pared).
  4) telemetria_gpu*.csv / gpu_*.csv → potencia instantánea por GPU
     (nvidia-smi) → energía por integración trapezoidal.

Salidas (en eval/costes/):
  inventario_experimentos.csv   una fila por experimento (carpeta de run)
  inventario_telemetria.csv     una fila por fichero de telemetría
  inventario_evaluaciones.csv   una fila por results_*.json
  resumen_global.json           agregados listos para el modelo de costes

Uso:
    tfm/bin/python Scripts/inventario_tokens_energia.py
"""
from __future__ import annotations

import csv
import glob
import io
import json
import os
import re
import sys
from bisect import bisect_left
from datetime import datetime, timedelta

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE, "eval", "costes")

PILOT_LOG = os.path.join(BASE, "logs", "rag_queries_pilot.jsonl")
V2_LOG = os.path.join(BASE, "logs", "rag_queries_v2.jsonl")

# Ventana de gracia al cerrar un experimento: el log se escribe al terminar la
# consulta, la telemetría se corta unos segundos antes.
SLACK_END_S = 15.0

# ── Máquinas ────────────────────────────────────────────────────────────────
# voz05    : 2x RTX PRO 6000 Blackwell (96 GB)  — servidor profesional
# gtc2pc4  : 1x RTX 5090 (32 GB)                — GPU de consumo
MACHINES = {
    "voz05":   {"gpus": "2x RTX PRO 6000 Blackwell", "n_gpu": 2, "tdp_w": 600},
    "gtc2pc4": {"gpus": "1x RTX 5090",               "n_gpu": 1, "tdp_w": 575},
    "gtc2pc9": {"gpus": "1x RTX 4090 (baseline)",    "n_gpu": 1, "tdp_w": 450},
}


def machine_of(path: str) -> str:
    p = path.replace(os.sep, "/")
    if "confianza_5090" in p or "_5090" in p:
        return "gtc2pc4"
    if "gtc2pc9" in p:
        return "gtc2pc9"
    return "voz05"


# ── Telemetría → energía ────────────────────────────────────────────────────
def parse_telemetry(path: str) -> dict | None:
    """Integra potencia por GPU. Devuelve energía (Wh), potencia media, ventana."""
    try:
        raw = open(path, encoding="utf-8", errors="replace").read().replace("\x00", "")
        rows = list(csv.reader(io.StringIO(raw)))
    except OSError:
        return None
    if len(rows) < 3:
        return None
    hdr = [h.strip().lower() for h in rows[0]]

    def col(*names):
        for n in names:
            for i, h in enumerate(hdr):
                if h.startswith(n):
                    return i
        return None

    i_ts, i_idx = col("timestamp"), col("index")
    i_pw, i_util = col("power.draw"), col("utilization.gpu")
    i_mem = col("memory.used")
    if None in (i_ts, i_idx, i_pw):
        return None

    per_gpu: dict[int, list[tuple[datetime, float, float, float]]] = {}
    for r in rows[1:]:
        if len(r) <= max(i_ts, i_idx, i_pw):
            continue
        try:
            ts = datetime.strptime(r[i_ts].strip(), "%Y/%m/%d %H:%M:%S.%f")
            idx = int(r[i_idx])
            pw = float(r[i_pw])
            util = float(r[i_util]) if i_util is not None else 0.0
            mem = float(r[i_mem]) if i_mem is not None else 0.0
        except (ValueError, IndexError):
            continue
        per_gpu.setdefault(idx, []).append((ts, pw, util, mem))
    if not per_gpu:
        return None

    total_wh = 0.0
    idle_wh = 0.0
    gpus = {}
    t_min = t_max = None
    for idx, samples in sorted(per_gpu.items()):
        samples.sort(key=lambda s: s[0])
        # potencia de reposo: percentil 5 de la serie (GPU cargada pero ociosa)
        pw_sorted = sorted(s[1] for s in samples)
        p05 = pw_sorted[max(0, int(0.05 * len(pw_sorted)) - 1)]
        j = 0.0
        j_idle = 0.0
        for a, b in zip(samples, samples[1:]):
            dt = (b[0] - a[0]).total_seconds()
            if dt <= 0 or dt > 30:          # hueco: no interpolar
                continue
            j += (a[1] + b[1]) / 2.0 * dt   # julios
            j_idle += p05 * dt
        wh = j / 3600.0
        total_wh += wh
        idle_wh += j_idle / 3600.0
        gpus[idx] = {
            "wh": round(wh, 3),
            "avg_power_w": round(sum(s[1] for s in samples) / len(samples), 1),
            "peak_power_w": round(max(s[1] for s in samples), 1),
            "idle_power_w": round(p05, 1),
            "avg_util_pct": round(sum(s[2] for s in samples) / len(samples), 1),
            "peak_vram_gb": round(max(s[3] for s in samples) / 1024.0, 2),
        }
        t0, t1 = samples[0][0], samples[-1][0]
        t_min = t0 if t_min is None or t0 < t_min else t_min
        t_max = t1 if t_max is None or t1 > t_max else t_max

    return {
        "file": os.path.relpath(path, BASE),
        "t_start": t_min,
        "t_end": t_max,
        "duracion_s": round((t_max - t_min).total_seconds(), 1),
        "energia_wh": round(total_wh, 3),
        "energia_marginal_wh": round(total_wh - idle_wh, 3),
        "n_gpus": len(gpus),
        "gpus": gpus,
    }


# ── Libro mayor de consultas ────────────────────────────────────────────────
def load_ledger() -> list[dict]:
    """Carga rag_queries_pilot.jsonl (+v2) ordenado por timestamp."""
    out = []
    for path, fuente in ((PILOT_LOG, "pilot"), (V2_LOG, "v2")):
        if not os.path.exists(path):
            continue
        for line in open(path, encoding="utf-8", errors="replace"):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = d.get("timestamp")
            if not ts:
                continue
            try:
                t = datetime.fromisoformat(ts)
            except ValueError:
                continue
            g = d.get("generation") or {}
            out.append({
                "t": t,
                "fuente": fuente,
                "model": g.get("model"),
                "pt": int(g.get("prompt_tokens") or 0),
                "ct": int(g.get("completion_tokens") or 0),
                "chars": int(g.get("response_chars") or 0),
                "gen_s": float((d.get("timings") or {}).get("Generador") or 0),
                "total_s": float(d.get("total_time_s") or 0),
                "error": d.get("error"),
            })
    out.sort(key=lambda r: r["t"])
    return out


def ledger_slice(ledger: list[dict], times: list[datetime],
                 t0: datetime, t1: datetime) -> list[dict]:
    lo = bisect_left(times, t0)
    hi = bisect_left(times, t1)
    return ledger[lo:hi]


# ── Pruebas de carga (sweep.json + telemetría hermana) ──────────────────────
CELL_RE = re.compile(r"(\d+b)_(\d*\.?\d*b|06b|4b)(?:_np(\d+))?")


def parse_sweep(path: str) -> dict:
    d = json.load(open(path, encoding="utf-8"))
    reqs = [r for lvl in d.get("sweep", []) for r in lvl.get("requests", [])]
    ok = [r for r in reqs if not r.get("error")]
    return {
        "timestamp": d.get("timestamp"),
        "model": d.get("model"),
        "rerank": d.get("rerank"),
        "collection": d.get("collection"),
        "levels": d.get("levels"),
        "n_req": len(reqs),
        "n_ok": len(ok),
        "wall_s": round(sum(l.get("wall_ms", 0) for l in d.get("sweep", [])) / 1000.0, 1),
        "tokens_palabras": sum(r.get("tokens", 0) for r in ok),
    }


def engine_of(path: str) -> str:
    p = path.replace(os.sep, "/")
    for eng in ("vllm", "llamacpp", "ollama"):
        if f"/{eng}/" in p or f"_{eng}" in p:
            return eng
    return "?"


def campaign_of(path: str) -> str:
    p = os.path.relpath(path, os.path.join(BASE, "eval", "prueba_carga_usuarios"))
    return p.split(os.sep)[0]


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    print("· Cargando libro mayor de consultas…", flush=True)
    ledger = load_ledger()
    times = [r["t"] for r in ledger]
    print(f"  {len(ledger):,} consultas registradas "
          f"({ledger[0]['t'].date()} → {ledger[-1]['t'].date()})")

    # ── 1. Telemetría ───────────────────────────────────────────────────────
    print("· Integrando telemetría GPU…", flush=True)
    tele_paths = sorted(set(
        glob.glob(os.path.join(BASE, "eval", "**", "telemetria_gpu*.csv"), recursive=True) +
        glob.glob(os.path.join(BASE, "eval", "**", "gpu_*.csv"), recursive=True)
    ))
    telemetrias = []
    for p in tele_paths:
        t = parse_telemetry(p)
        if t:
            t["maquina"] = machine_of(p)
            t["campana"] = ("prueba_carga_usuarios/" + campaign_of(p)
                            if "prueba_carga_usuarios" in p else "otros")
            t["motor"] = engine_of(p)
            telemetrias.append(t)
    print(f"  {len(telemetrias)} ficheros · "
          f"{sum(t['energia_wh'] for t in telemetrias) / 1000:.2f} kWh integrados")

    # ── 2. Pruebas de carga ─────────────────────────────────────────────────
    print("· Cruzando sweeps con telemetría y tokens…", flush=True)
    experimentos = []
    consumidas = 0
    for sw_path in sorted(glob.glob(
            os.path.join(BASE, "eval", "prueba_carga_usuarios", "**", "sweep.json"),
            recursive=True)):
        run_dir = os.path.dirname(sw_path)
        sw = parse_sweep(sw_path)
        tele = parse_telemetry(os.path.join(run_dir, "telemetria_gpu.csv"))

        pt = ct = n_q = 0
        if tele:
            win = ledger_slice(ledger, times, tele["t_start"],
                               tele["t_end"] + timedelta(seconds=SLACK_END_S))
            n_q = len(win)
            pt = sum(r["pt"] for r in win)
            ct = sum(r["ct"] for r in win)
            consumidas += n_q

        rel = os.path.relpath(run_dir, BASE)
        experimentos.append({
            "experimento": rel,
            "campana": campaign_of(run_dir),
            "maquina": machine_of(run_dir),
            "motor": engine_of(run_dir),
            "celda": os.path.basename(os.path.dirname(run_dir)),
            "run": os.path.basename(run_dir),
            "fecha": sw["timestamp"],
            "modelo_gen": sw["model"],
            "rerank": sw["rerank"],
            "coleccion": sw["collection"],
            "niveles": "/".join(str(x) for x in (sw["levels"] or [])),
            "n_requests": sw["n_req"],
            "n_ok": sw["n_ok"],
            "wall_s": sw["wall_s"],
            "duracion_telemetria_s": tele["duracion_s"] if tele else "",
            "n_consultas_log": n_q,
            "tokens_entrada": pt,
            "tokens_salida": ct,
            "tokens_salida_estim_palabras": sw["tokens_palabras"],
            "energia_wh": tele["energia_wh"] if tele else "",
            "energia_marginal_wh": tele["energia_marginal_wh"] if tele else "",
            "pot_media_w": (round(sum(g["avg_power_w"] for g in tele["gpus"].values()), 1)
                            if tele else ""),
            "n_gpus": tele["n_gpus"] if tele else "",
        })
    print(f"  {len(experimentos)} runs · {consumidas:,} consultas atribuidas")

    # ── 3. Evaluaciones (results_*.json) ────────────────────────────────────
    print("· Recorriendo evaluaciones…", flush=True)
    evals = []
    for p in sorted(glob.glob(os.path.join(BASE, "eval", "**", "results_*.json"),
                              recursive=True)):
        try:
            d = json.load(open(p, encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        pq = d.get("per_question") or []
        if not pq:
            continue
        pt = ct = 0
        gen_s = 0.0
        tot_s = 0.0
        for q in pq:
            ts = (q.get("generation") or {}).get("token_stats") or {}
            pt += int(ts.get("prompt_tokens") or 0)
            ct += int(ts.get("completion_tokens") or 0)
            tm = q.get("timings") or {}
            gen_s += float(tm.get("Generador") or 0)
            tot_s += sum(float(v or 0) for v in tm.values())
        ejec = d.get("executed_at") or ""
        # ¿Está esta evaluación ya contabilizada en el libro mayor? Se comprueba
        # por densidad: cuántas consultas registró el log en su ventana real.
        cubierta = False
        n_log_ventana = 0
        if ejec:
            try:
                t0 = datetime.fromisoformat(ejec.replace("/", "-"))
            except ValueError:
                t0 = None
            if t0 is not None:
                t1 = t0 + timedelta(seconds=tot_s + 300)
                n_log_ventana = len(ledger_slice(ledger, times, t0, t1))
                cubierta = n_log_ventana >= 0.5 * len(pq)
        evals.append({
            "fichero": os.path.relpath(p, BASE),
            "carpeta": os.path.relpath(os.path.dirname(p), BASE),
            "fecha": ejec,
            "ground_truth": d.get("ground_truth"),
            "modelo_gen": d.get("model"),
            "coleccion": d.get("collection"),
            "rerank": d.get("rerank_strategy"),
            "top_k": d.get("top_k"),
            "n_preguntas": len(pq),
            "tokens_entrada": pt,
            "tokens_salida": ct,
            "seg_generacion": round(gen_s, 1),
            "seg_total": round(tot_s, 1),
            "en_libro_mayor": cubierta,
            "n_consultas_log_ventana": n_log_ventana,
        })
    # Deduplicación: `rerun_topk10.py` produce carpetas `*_topk10` reejecutando
    # SOLO el retriever y reutilizando la generación original; los backups
    # (`_baseline_backup`) son copias literales. En ambos casos las estadísticas
    # de tokens son las mismas del run original y contarlas duplicaría el gasto.
    # Se detectan por firma de generación idéntica (ignorando top_k), no por ruta.
    vistos: dict[tuple, str] = {}
    for e in sorted(evals, key=lambda x: (x["fecha"], x["fichero"])):
        firma = (e["ground_truth"], e["modelo_gen"], e["rerank"], e["coleccion"],
                 e["n_preguntas"], e["tokens_entrada"], e["tokens_salida"])
        if e["tokens_entrada"] == 0:
            e["derivado_de"] = ""
            continue
        if firma in vistos:
            e["derivado_de"] = vistos[firma]
        else:
            vistos[firma] = e["fichero"]
            e["derivado_de"] = ""
    n_der = sum(1 for e in evals if e["derivado_de"])
    print(f"  {len(evals)} evaluaciones · "
          f"{sum(e['n_preguntas'] for e in evals):,} preguntas "
          f"({n_der} derivadas/duplicadas, excluidas del total)")

    # ── 4. Escritura ────────────────────────────────────────────────────────
    def dump_csv(name, rows, fields):
        path = os.path.join(OUT_DIR, name)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"  → {os.path.relpath(path, BASE)}  ({len(rows)} filas)")

    dump_csv("inventario_experimentos.csv", experimentos, list(experimentos[0].keys()))
    dump_csv("inventario_evaluaciones.csv", evals, list(evals[0].keys()))
    dump_csv("inventario_telemetria.csv", [{
        "fichero": t["file"], "campana": t["campana"], "maquina": t["maquina"],
        "motor": t["motor"], "inicio": t["t_start"].isoformat(),
        "fin": t["t_end"].isoformat(), "duracion_s": t["duracion_s"],
        "n_gpus": t["n_gpus"], "energia_wh": t["energia_wh"],
        "energia_marginal_wh": t["energia_marginal_wh"],
        "pot_media_w": round(sum(g["avg_power_w"] for g in t["gpus"].values()), 1),
        "pot_pico_w": round(sum(g["peak_power_w"] for g in t["gpus"].values()), 1),
        "vram_pico_gb": round(sum(g["peak_vram_gb"] for g in t["gpus"].values()), 1),
    } for t in telemetrias], ["fichero", "campana", "maquina", "motor", "inicio",
                              "fin", "duracion_s", "n_gpus", "energia_wh",
                              "energia_marginal_wh", "pot_media_w", "pot_pico_w",
                              "vram_pico_gb"])

    # ── 5. Resumen global (sin doble conteo) ────────────────────────────────
    led_pt = sum(r["pt"] for r in ledger)
    led_ct = sum(r["ct"] for r in ledger)
    ev_fuera = [e for e in evals
                if not e["en_libro_mayor"] and not e["derivado_de"]]
    tot_pt = led_pt + sum(e["tokens_entrada"] for e in ev_fuera)
    tot_ct = led_ct + sum(e["tokens_salida"] for e in ev_fuera)

    por_modelo = {}
    for r in ledger:
        m = por_modelo.setdefault(r["model"] or "?",
                                  {"n": 0, "pt": 0, "ct": 0, "gen_s": 0.0})
        m["n"] += 1
        m["pt"] += r["pt"]
        m["ct"] += r["ct"]
        m["gen_s"] += r["gen_s"]
    for e in ev_fuera:
        m = por_modelo.setdefault(e["modelo_gen"] or "?",
                                  {"n": 0, "pt": 0, "ct": 0, "gen_s": 0.0})
        m["n"] += e["n_preguntas"]
        m["pt"] += e["tokens_entrada"]
        m["ct"] += e["tokens_salida"]
        m["gen_s"] += e["seg_generacion"]

    energia_por_maquina = {}
    for t in telemetrias:
        energia_por_maquina.setdefault(t["maquina"], 0.0)
        energia_por_maquina[t["maquina"]] += t["energia_wh"]

    resumen = {
        "generado": datetime.now().isoformat(timespec="seconds"),
        "consultas": {
            "libro_mayor": len(ledger),
            "evaluaciones_previas_al_libro_mayor": sum(e["n_preguntas"] for e in ev_fuera),
            "evaluaciones_derivadas_excluidas": sum(
                e["n_preguntas"] for e in evals if e["derivado_de"]),
            "total": len(ledger) + sum(e["n_preguntas"] for e in ev_fuera),
            "rango": [ledger[0]["t"].isoformat(), ledger[-1]["t"].isoformat()],
        },
        "tokens": {
            "entrada_total": tot_pt,
            "salida_total": tot_ct,
            "entrada_libro_mayor": led_pt,
            "salida_libro_mayor": led_ct,
            "entrada_evals_previas": tot_pt - led_pt,
            "salida_evals_previas": tot_ct - led_ct,
        },
        "por_modelo": {k: {**v, "gen_s": round(v["gen_s"], 1)}
                       for k, v in sorted(por_modelo.items())},
        "energia_medida_wh": {k: round(v, 1) for k, v in energia_por_maquina.items()},
        "energia_medida_kwh_total": round(sum(energia_por_maquina.values()) / 1000, 3),
        "horas_telemetria": round(sum(t["duracion_s"] for t in telemetrias) / 3600, 1),
        "horas_telemetria_por_maquina": {
            k: round(sum(t["duracion_s"] for t in telemetrias if t["maquina"] == k) / 3600, 1)
            for k in energia_por_maquina},
        "cobertura_telemetria": {
            "consultas_bajo_telemetria": consumidas,
            "pct": round(100 * consumidas / max(1, len(ledger)), 1),
        },
        "maquinas": MACHINES,
    }
    with open(os.path.join(OUT_DIR, "resumen_global.json"), "w", encoding="utf-8") as fh:
        json.dump(resumen, fh, ensure_ascii=False, indent=2)
    print(f"  → eval/costes/resumen_global.json")

    print("\n" + "=" * 70)
    print(f"TOKENS DE ENTRADA : {tot_pt:>15,}")
    print(f"TOKENS DE SALIDA  : {tot_ct:>15,}")
    print(f"CONSULTAS         : {resumen['consultas']['total']:>15,}")
    print(f"ENERGÍA MEDIDA    : {resumen['energia_medida_kwh_total']:>15,.2f} kWh "
          f"(cobertura {resumen['cobertura_telemetria']['pct']}% de las consultas)")
    print("=" * 70)


if __name__ == "__main__":
    sys.exit(main())
