#!/usr/bin/env python3
"""eval_cortafuegos.py — Evaluacion del cortafuegos (clasificacion de intencion) y de la
abstencion ante consultas clinicas no cubiertas por el corpus.

EXPERIMENTO A — clasificacion de intencion
  Recorre eval/ROUTER_intent_set.json (140 mensajes etiquetados) invocando UNICAMENTE al
  router, y contrasta la intencion predicha con la esperada. El router distingue cuatro
  intenciones (fase2_router.py): clinica | conversacion | lista_protocolos |
  fuera_de_alcance. Se registra ademas si la regex rapida (detect_chitchat) resolvio el
  mensaje sin llegar a llamar al LLM.

EXPERIMENTO B — abstencion fuera de cobertura
  Recorre eval/ROUTER_fuera_corpus_set.json (40 preguntas clinicas legitimas cuya respuesta
  NO esta en los 22 protocolos) por el pipeline completo, registrando la intencion, la
  puntuacion del mejor fragmento tras el reordenador, si el recuperador marco baja
  relevancia y el texto de la respuesta generada.

  Por la "regla de oro" del prompt el router DEBE clasificarlas como "clinica": filtrarlas
  corresponde al umbral de relevancia, no al cortafuegos.

Uso:  tfm/bin/python Scripts/eval_cortafuegos.py [--solo A|B]
Salida: eval/cortafuegos/resultados_{intent,fuera_corpus}.json
"""
from __future__ import annotations
import argparse, asyncio, json, time
from pathlib import Path

from Scripts.rag_pipeline2.config import (CRATE_COLLECTION, DEFAULT_MODEL,
                                          DEFAULT_RERANK_STRATEGY, TOP_K_RERANK)
from Scripts.rag_pipeline2.fase2_router import run_router, detect_chitchat
from Scripts.rag_pipeline2.fase3_retriever import run_retriever
from Scripts.rag_pipeline2.fase4_generator import stream_generator

OUT = Path("eval/cortafuegos"); OUT.mkdir(parents=True, exist_ok=True)
MODELO = "gemma4:26b"
ESTRATEGIA = "cross_encoder"          # la de produccion
COLECCION = "docs_rag_qwen3"

def intencion(r) -> str:
    """Traduce los flags del RouterOutput a la etiqueta de intencion."""
    if r.is_chitchat:     return "conversacion"
    if r.out_of_scope:    return "fuera_de_alcance"
    if r.is_list_request: return "lista_protocolos"
    return "clinica"

# ── EXPERIMENTO A ─────────────────────────────────────────────────────────
def experimento_a(ruta="eval/ROUTER_intent_set.json", sufijo=""):
    qs = json.loads(Path(ruta).read_text(encoding="utf-8"))["questions"]
    print(f"\n[A] Clasificacion de intencion — {len(qs)} mensajes, solo router\n")
    filas = []
    for i, q in enumerate(qs, 1):
        rapida = detect_chitchat(q["query"])
        t0 = time.time()
        try:
            r = run_router(q["query"], model=MODELO, history=[], previous_doc=None)
            pred, err = intencion(r), None
        except Exception as e:
            pred, err = "ERROR", str(e)[:120]
        ok = pred == q["intent_esperado"]
        filas.append(dict(id=q["id"], query=q["query"], esperado=q["intent_esperado"],
                          predicho=pred, acierto=ok, via_regex=rapida,
                          segundos=round(time.time()-t0, 2), error=err))
        print(f"  {i:3}/{len(qs)}  {'OK ' if ok else '>><'}  {q['intent_esperado']:17}"
              f"-> {pred:17} {'[regex]' if rapida else '':8} {q['query'][:46]}")
    (OUT/f"resultados_intent{sufijo}.json").write_text(json.dumps(filas, ensure_ascii=False, indent=1), encoding="utf-8")
    return filas

# ── EXPERIMENTO B ─────────────────────────────────────────────────────────
def experimento_b(ruta="eval/ROUTER_fuera_corpus_set.json", sufijo=""):
    qs = json.loads(Path(ruta).read_text(encoding="utf-8"))["questions"]
    print(f"\n[B] Abstencion fuera de cobertura — {len(qs)} preguntas, pipeline completo\n")
    loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
    filas = []
    for i, q in enumerate(qs, 1):
        t0 = time.time()
        r = run_router(q["query"], model=MODELO, history=[], previous_doc=None)
        pred = intencion(r)
        score1 = None; low = None; resp = ""
        if pred == "clinica":
            res = loop.run_until_complete(run_retriever(
                r, rerank_strategy=ESTRATEGIA, top_k=TOP_K_RERANK, collection=COLECCION))
            low = bool(res.low_relevance)
            if res.docs:
                score1 = float(res.docs[0].rerank_score)
                for chunk, final in stream_generator(q["query"], res.docs, model=MODELO):
                    if final is not None: resp = final.response_text
        filas.append(dict(id=q["id"], query=q["query"], intent=pred, top1_score=score1,
                          low_relevance=low, respuesta=resp, segundos=round(time.time()-t0, 2)))
        marca = "abstiene" if low else (f"score={score1:.3f}" if score1 is not None else "sin docs")
        print(f"  {i:3}/{len(qs)}  intent={pred:15} {marca:18} {q['query'][:44]}")
    (OUT/f"resultados_fuera_corpus{sufijo}.json").write_text(json.dumps(filas, ensure_ascii=False, indent=1), encoding="utf-8")
    return filas

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--solo", choices=["A","B"], default=None)
    ap.add_argument("--set-intent", default="eval/ROUTER_intent_set.json",
                    help="conjunto de intenciones para el experimento A")
    ap.add_argument("--set", default="eval/ROUTER_fuera_corpus_set.json",
                    help="conjunto para el experimento B")
    ap.add_argument("--sufijo", default="", help="sufijo del fichero de salida")
    a = ap.parse_args()
    if a.solo in (None,"A"): experimento_a(a.set_intent, a.sufijo)
    if a.solo in (None,"B"): experimento_b(a.set, a.sufijo)
    print(f"\nResultados en {OUT}/")
