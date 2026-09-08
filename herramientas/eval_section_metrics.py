#!/usr/bin/env python3
"""
eval_section_metrics.py — Métricas de recuperación a NIVEL DE PASAJE/SECCIÓN
============================================================================

Las métricas IR del evaluador (eval_retriever.py) etiquetan la relevancia a nivel
de DOCUMENTO: un chunk recuperado es relevante si pertenece al documento correcto.
Esto mide "¿recupera el documento adecuado?", no "¿recupera el pasaje que contiene
la respuesta?".

Este script calcula un segundo nivel de relevancia, de PASAJE, de forma totalmente
offline (sin reejecutar el pipeline):

  1. Cada `results_*.json` guarda, por chunk recuperado, (doc_reference, h1, h2).
  2. Con esa clave se recupera el TEXTO del chunk desde CrateDB.
  3. Un chunk es relevante a nivel de sección si el `expected_quote` del ground
     truth está suficientemente presente en su texto: recall de palabras de
     contenido (sin stopwords) >= umbral.
  4. Se recalculan Hit@k / MRR@k / NDCG@k con estas etiquetas de pasaje y se
     comparan con las de documento.

Uso:
    python Scripts/eval_section_metrics.py \\
        --input-dir eval/2026-05-19 eval/2026-05-31_06b \\
        --threshold 0.7 --crate http://localhost:4201

Salida:
    section_metrics_summary.json  (en cada carpeta de entrada)
    + tabla comparativa por consola.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import unicodedata
from collections import defaultdict

import httpx

from section_overrides import load_overrides, apply_override

# ── Mapa colección lógica → tabla CrateDB (el texto es idéntico entre tablas) ──
COLLECTION_TABLE = {
    "docs_rag_qwen3":     "rag_chunks_qwen3_v2",
    "docs_rag_qwen3_06b": "rag_chunks_qwen3_06b_v2",
    "docs_rag_v2":        "rag_chunks_v2",
}
DEFAULT_TABLE = "rag_chunks_qwen3_v2"

_STOP = set(
    "de la el en y a los las un una que se con por para del al es su sus o e como "
    "mas más este esta estos estas lo le les ha han hay si no ni u un al".split()
)

KS = [1, 3, 5, 10]


def _toks(s: str) -> list[str]:
    s = unicodedata.normalize("NFD", s or "").encode("ascii", "ignore").decode().lower()
    return [t for t in re.findall(r"[a-z0-9]+", s) if t not in _STOP and len(t) > 2]


def content_recall(quote: str, text: str) -> float | None:
    q = set(_toks(quote))
    if not q:
        return None
    t = set(_toks(text))
    return len(q & t) / len(q)


# ── IR helpers ────────────────────────────────────────────────────────────────
def hit_at_k(labels, k):  return 1.0 if any(labels[:k]) else 0.0
def mrr_at_k(labels, k):
    for i, l in enumerate(labels[:k], 1):
        if l:
            return 1.0 / i
    return 0.0
def ndcg_at_k(labels, k):
    dcg = sum(1.0 / math.log2(i + 2) for i, l in enumerate(labels[:k]) if l)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(sum(labels[:k]), k)))
    return dcg / idcg if idcg > 0 else 0.0


class SectionTextCache:
    """Recupera (y cachea) el texto concatenado de una sección desde CrateDB."""
    def __init__(self, crate_url: str):
        self.url = crate_url.rstrip("/")
        self.sess = httpx.Client(timeout=60)
        self.cache: dict = {}
        self.misses = 0

    def get(self, table: str, doc_id: str, h1: str, h2: str) -> str:
        key = (table, doc_id, h1, h2)
        if key in self.cache:
            return self.cache[key]
        r = self.sess.post(
            f"{self.url}/_sql",
            json={"stmt": f"SELECT text FROM {table} WHERE doc_id=? AND h1=? AND h2=?",
                  "args": [doc_id, h1, h2]},
        ).json()
        rows = r.get("rows", [])
        if not rows:
            self.misses += 1
        txt = " ".join(row[0] for row in rows)
        self.cache[key] = txt
        return txt


def strat_of(fname: str) -> str:
    m = re.search(r"(embed|cross_encoder|rrf_only)", fname)
    return m.group(1) if m else "?"


def process_folder(folder: str, cache: SectionTextCache, thresholds: list[float]) -> dict:
    files = sorted(glob.glob(os.path.join(folder, "results_*.json")))
    overrides = load_overrides(os.path.join(folder, "section_overrides.json"))
    n_overridden = 0
    # acumuladores: por estrategia → lista de (doc_labels, sec_labels@th)
    per_strat = defaultdict(lambda: {"doc": [], **{f"sec@{th}": [] for th in thresholds}})
    # mismo acumulador pero segmentado por (estrategia, documento)
    per_doc = defaultdict(lambda: defaultdict(lambda: {"doc": [], **{f"sec@{th}": [] for th in thresholds}}))
    n_q = defaultdict(int)

    def _canon(gt):
        return gt.replace("grounf", "ground").replace("_ground_truth.json", "").replace("_ground_truth", "")

    for f in files:
        d = json.loads(open(f, encoding="utf-8").read())
        strat = strat_of(os.path.basename(f))
        docid = _canon(d.get("ground_truth", ""))
        table = COLLECTION_TABLE.get(d.get("collection", ""), DEFAULT_TABLE)

        for q in d.get("per_question", []):
            quote = q.get("expected_quote") or ""
            docs = q.get("retrieved_docs") or []
            if not quote or not docs:
                continue
            n_q[strat] += 1

            # etiquetas de documento (ya calculadas en el eval)
            doc_labels = [int(doc.get("relevance_label", 0)) for doc in docs]
            per_strat[strat]["doc"].append(doc_labels)
            per_doc[strat][docid]["doc"].append(doc_labels)

            # recall del quote en cada chunk recuperado
            recalls = []
            for doc in docs:
                txt = cache.get(table, doc["doc_reference"], doc.get("h1", ""), doc.get("h2", ""))
                cr = content_recall(quote, txt)
                recalls.append(cr if cr is not None else 0.0)

            qid = q.get("id", "")
            for th in thresholds:
                sec_labels = [1 if r >= th else 0 for r in recalls]
                n_overridden += apply_override(docid, qid, strat, docs, sec_labels, overrides)
                per_strat[strat][f"sec@{th}"].append(sec_labels)
                per_doc[strat][docid][f"sec@{th}"].append(sec_labels)

    def _agg(lab: dict) -> dict:
        entry = {}
        for level, all_labels in lab.items():
            if not all_labels:
                continue
            n = len(all_labels)
            for k in KS:
                entry[f"{level}_hit@{k}"]  = round(sum(hit_at_k(l, k) for l in all_labels) / n, 4)
                entry[f"{level}_mrr@{k}"]  = round(sum(mrr_at_k(l, k) for l in all_labels) / n, 4)
                entry[f"{level}_ndcg@{k}"] = round(sum(ndcg_at_k(l, k) for l in all_labels) / n, 4)
        return entry

    # agregación micro (sobre todas las preguntas) + por documento
    out = {"folder": folder, "n_questions": dict(n_q), "n_overridden": n_overridden,
           "strategies": {}, "by_doc": {}}
    for strat, lab in per_strat.items():
        out["strategies"][strat] = _agg(lab)
    for strat, docs in per_doc.items():
        out["by_doc"][strat] = {docid: _agg(lab) for docid, lab in docs.items()}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", nargs="+", required=True)
    ap.add_argument("--threshold", type=float, default=0.7, help="Umbral principal de recall del quote")
    ap.add_argument("--extra-thresholds", type=float, nargs="*", default=[0.5, 0.9],
                    help="Umbrales adicionales para análisis de sensibilidad")
    ap.add_argument("--crate", default="http://localhost:4201")
    args = ap.parse_args()

    thresholds = sorted({args.threshold, *args.extra_thresholds})
    cache = SectionTextCache(args.crate)

    for folder in args.input_dir:
        print(f"\n{'='*72}\n  {folder}\n{'='*72}")
        summary = process_folder(folder, cache, thresholds)
        out_path = os.path.join(folder, "section_metrics_summary.json")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, ensure_ascii=False)

        th = args.threshold
        print(f"  preguntas: {summary['n_questions']}")
        if summary.get("n_overridden"):
            print(f"  overrides manuales aplicados (chunks): {summary['n_overridden']}")
        print(f"\n  {'estrategia':14s} | {'nivel':18s} | Hit@1  Hit@3  Hit@5  MRR@5  NDCG@5")
        print("  " + "-"*70)
        for strat in ["cross_encoder", "embed", "rrf_only"]:
            e = summary["strategies"].get(strat, {})
            if not e:
                continue
            for level, tag in [("doc", "documento"), (f"sec@{th}", f"sección(≥{th})")]:
                print(f"  {strat:14s} | {tag:18s} | "
                      f"{e.get(level+'_hit@1',0):.3f}  {e.get(level+'_hit@3',0):.3f}  "
                      f"{e.get(level+'_hit@5',0):.3f}  {e.get(level+'_mrr@5',0):.3f}  "
                      f"{e.get(level+'_ndcg@5',0):.3f}")
            print()
        print(f"  ✓ {out_path}")
    if cache.misses:
        print(f"\n  [aviso] {cache.misses} secciones sin texto en CrateDB (revisar claves)")


if __name__ == "__main__":
    main()
