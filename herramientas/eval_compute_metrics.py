"""
eval_compute_metrics.py — Métricas offline a partir de los JSON de eval_retriever
================================================================================

Toma una carpeta con `results_*.json` (típicamente `eval/<fecha>/<modo>/`) y produce:

    metrics_summary.json        — todas las cifras agregadas y segmentadas.
    metrics_report.html         — informe legible con tablas y gráficas SVG mínimas.
    metrics_per_question.csv    — métricas por pregunta para análisis posterior.

Uso:
    python -m Scripts.eval_compute_metrics \\
        --input-dir eval/2026-05-06_history/cross_encoder \\
        --output-dir eval/2026-05-06_history/cross_encoder_metrics

Dependencias:
    Núcleo: stdlib + numpy
    Opcionales: bert_score, sentence_transformers, rouge_score
        Si no están instaladas se omite la métrica con un warning, no falla.

Métricas implementadas (todas offline, ningún reejecutar el eval):

  • Latencia p50/p95/p99 por fase y end-to-end.
  • Tokens/seg de generación.
  • Hit/MRR/NDCG segmentado por documento, bloque, longitud query, tipo
    protocolo, is_followup, is_chitchat.
  • Confusion matrix de "docs ladrones" (top-1 erróneo más frecuente).
  • AUC del re-ranker, gap top-1 vs top-2, calibración score↔relevance.
  • Diversidad top-K (nº de docs únicos), drop-off P@K.
  • ROUGE-L recall vs expected_quote (si rouge_score disponible).
  • BERTScore recall vs expected_quote (si bert_score disponible).
  • Embedding similarity respuesta ↔ quote (si sentence-transformers disponible).
  • Cobertura del expected_quote en chunks recuperados.
  • Cobertura léxica de la respuesta vs los chunks (heurística pre-NLI).
  • Retención de keywords del router en la respuesta.
  • Distribución longitud de respuesta.
  • Coste estimado por query (tarifas equivalentes).

Tarifas usadas para coste (ajustables):
    GPT-4-Turbo  (in/out)  = $0.010 / $0.030 por 1k tokens
    Claude-Sonnet         = $0.003 / $0.015
    Gemini-1.5-Pro        = $0.0035 / $0.0105
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
import statistics
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# DEPENDENCIAS OPCIONALES
# ─────────────────────────────────────────────────────────────────────────────

try:
    from rouge_score import rouge_scorer  # type: ignore
    _HAS_ROUGE = True
except ImportError:
    _HAS_ROUGE = False

try:
    from bert_score import score as bertscore_score  # type: ignore
    _HAS_BERTSCORE = True
except ImportError:
    _HAS_BERTSCORE = False

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    _HAS_ST = True
except ImportError:
    _HAS_ST = False


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

# Tarifas USD por 1k tokens (input / output). Actualizar cuando convenga.
PRICE_TABLE_USD = {
    "GPT-4-Turbo":     (0.010, 0.030),
    "Claude-Sonnet":   (0.003, 0.015),
    "Gemini-1.5-Pro":  (0.0035, 0.0105),
}

# Longitudes de query en palabras
LEN_BUCKETS = [(0, 6, "muy corta"), (7, 12, "corta"), (13, 20, "media"), (21, 999, "larga")]

# K por defecto a reportar
KS = [1, 3, 5, 10, 15]


# ─────────────────────────────────────────────────────────────────────────────
# UTILIDADES NUMÉRICAS
# ─────────────────────────────────────────────────────────────────────────────

def _percentiles(xs: list[float], ps=(50, 95, 99)) -> dict[str, float]:
    if not xs:
        return {f"p{p}": 0.0 for p in ps}
    arr = np.asarray(xs, dtype=np.float64)
    return {f"p{p}": float(np.percentile(arr, p)) for p in ps}


def _mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    return float(sum(xs) / len(xs)) if xs else 0.0


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def _binary_auc(scores: list[float], labels: list[int]) -> float:
    """AUC ROC binaria por enumeración de pares (impl. ligera, sin sklearn)."""
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return float("nan")
    n_correct = 0
    n_total   = 0
    for p in pos:
        for n in neg:
            n_total += 1
            if p > n:
                n_correct += 1
            elif p == n:
                n_correct += 0.5
    return n_correct / n_total


def _len_bucket(query: str) -> str:
    n = len(query.split())
    for lo, hi, label in LEN_BUCKETS:
        if lo <= n <= hi:
            return label
    return "?"


def _proto_family(gt_name: str) -> str:
    # gt_name ej. "PA178" -> "PA"
    m = re.match(r"^([A-Z]+)", gt_name)
    return m.group(1) if m else "?"


# ─────────────────────────────────────────────────────────────────────────────
# IR METRICS
# ─────────────────────────────────────────────────────────────────────────────

def hit_at_k(labels: list[int], k: int) -> float:
    return 1.0 if any(labels[:k]) else 0.0


def mrr_at_k(labels: list[int], k: int) -> float:
    for i, l in enumerate(labels[:k], 1):
        if l == 1:
            return 1.0 / i
    return 0.0


def precision_at_k(labels: list[int], k: int) -> float:
    if k <= 0:
        return 0.0
    return sum(labels[:k]) / k


def ndcg_at_k(labels: list[int], k: int) -> float:
    """NDCG binario."""
    dcg  = sum((1.0 / math.log2(i + 2)) for i, l in enumerate(labels[:k]) if l == 1)
    rels = sum(labels[:k])
    idcg = sum(1.0 / math.log2(i + 2) for i in range(rels))
    return dcg / idcg if idcg > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# CARGA
# ─────────────────────────────────────────────────────────────────────────────

def load_results(input_dir: Path) -> list[dict]:
    files = sorted(input_dir.glob("results_*.json"))
    out = []
    for f in files:
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            d["_source_file"] = f.name
            d["_gt"] = d.get("ground_truth", "").replace("_ground_truth.json", "")
            out.append(d)
        except Exception as e:
            print(f"[WARN] No se pudo leer {f}: {e}", file=sys.stderr)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# CÁLCULO POR PREGUNTA
# ─────────────────────────────────────────────────────────────────────────────

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _WORD_RE.findall(text or "")]


def _ngram_set(tokens: list[str], n: int) -> set[tuple]:
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def _ngram_recall(reference: str, candidate: str, n: int = 2) -> float:
    """% de n-gramas del reference que aparecen en el candidate (recall puro)."""
    ref = _ngram_set(_tokens(reference), n)
    if not ref:
        return float("nan")
    cand = _ngram_set(_tokens(candidate), n)
    return len(ref & cand) / len(ref)


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Longitud del LCS — implementación O(n·m) sólo viable con textos cortos."""
    if not a or not b:
        return 0
    dp = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        prev = 0
        for j in range(1, len(b) + 1):
            tmp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = tmp
    return dp[-1]


def _rouge_l_recall_fast(reference: str, candidate: str) -> float:
    """ROUGE-L recall manual (token level), sin dependencia externa."""
    ref = _tokens(reference)
    cand = _tokens(candidate)
    if not ref:
        return float("nan")
    return _lcs_length(ref, cand) / len(ref)


def per_question_metrics(record: dict, question: dict) -> dict:
    """Calcula todas las métricas que dependen sólo de UNA pregunta."""
    labels = question.get("labels") or []
    docs   = question.get("retrieved_docs") or []
    quote  = question.get("expected_quote") or ""
    gen    = question.get("generated_response") or ""
    router = question.get("router") or {}
    timings = question.get("timings") or {}
    generation = question.get("generation") or {}

    # IR
    metrics: dict = {
        "qid":   question.get("id"),
        "gt":    record["_gt"],
        "block": question.get("block"),
        "is_followup": router.get("is_followup"),
        "is_chitchat": router.get("is_chitchat"),
        "n_words_query": len(_tokens(question.get("query", ""))),
        "len_bucket":    _len_bucket(question.get("query", "")),
        "proto_family":  _proto_family(record["_gt"]),
    }
    for k in KS:
        metrics[f"hit@{k}"]  = hit_at_k(labels, k)
        metrics[f"mrr@{k}"]  = mrr_at_k(labels, k)
        metrics[f"prec@{k}"] = precision_at_k(labels, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(labels, k)
    metrics["first_hit_pos"] = next((i + 1 for i, l in enumerate(labels) if l == 1), None)

    # Reranker
    rerank_scores = [d.get("rerank_score", 0.0) for d in docs]
    metrics["rr_top1"]  = rerank_scores[0] if rerank_scores else 0.0
    metrics["rr_top2"]  = rerank_scores[1] if len(rerank_scores) > 1 else 0.0
    metrics["rr_gap"]   = (rerank_scores[0] - rerank_scores[1]) if len(rerank_scores) > 1 else 0.0
    metrics["rr_mean"]  = _mean(rerank_scores)
    metrics["n_unique_docs_topK"] = len({d.get("doc_reference","") for d in docs})

    # Latencias
    for phase in ("Router", "Retriever", "Generador"):
        v = timings.get(phase)
        metrics[f"t_{phase.lower()}_s"] = float(v) if v is not None else None
    t_total = sum(v for v in timings.values() if isinstance(v, (int, float)))
    metrics["t_total_s"] = t_total

    # Tokens — esquema fase4_generator: completion_tokens, prompt_tokens, tokens_per_second
    ts = generation.get("token_stats") or {}
    metrics["output_tokens"] = ts.get("completion_tokens") or ts.get("eval_count") or ts.get("output_tokens")
    metrics["prompt_tokens"] = ts.get("prompt_tokens")     or ts.get("prompt_eval_count") or ts.get("input_tokens")
    if ts.get("tokens_per_second"):
        metrics["tokens_per_s"] = float(ts["tokens_per_second"])
    elif ts.get("eval_count") and ts.get("eval_duration"):
        metrics["tokens_per_s"] = ts["eval_count"] / (ts["eval_duration"] / 1e9)
    else:
        metrics["tokens_per_s"] = None
    metrics["context_used_pct"] = ts.get("context_used_pct")
    metrics["response_chars"] = generation.get("response_chars") or len(gen)
    metrics["response_words"] = len(_tokens(gen))

    # Cobertura del quote en los chunks (no en la respuesta)
    if quote:
        chunks_text = " ".join(_tokens(" ".join(
            f"{d.get('h2','')} {d.get('h1','')}" for d in docs
        )))  # cabeceras como mínimo (texto completo no está en el JSON)
        # heurística: bigram recall del quote en cabeceras + nombres de doc
        meta_text = " ".join(d.get("doc_reference","") + " " + d.get("h1","") + " " + d.get("h2","") for d in docs)
        metrics["quote_bigram_recall_in_meta"] = _ngram_recall(quote, meta_text, n=2)
    else:
        metrics["quote_bigram_recall_in_meta"] = float("nan")

    # Generación: recall del quote en la respuesta
    if quote and gen:
        metrics["quote_unigram_recall"] = _ngram_recall(quote, gen, n=1)
        metrics["quote_bigram_recall"]  = _ngram_recall(quote, gen, n=2)
        metrics["rougeL_recall"]        = _rouge_l_recall_fast(quote, gen)
    else:
        metrics["quote_unigram_recall"] = float("nan")
        metrics["quote_bigram_recall"]  = float("nan")
        metrics["rougeL_recall"]        = float("nan")

    # Retención de keywords del router en la respuesta
    keywords = [k.lower() for k in (router.get("keywords") or [])]
    if keywords and gen:
        gen_l = gen.lower()
        present = sum(1 for k in keywords if k.lower() in gen_l)
        metrics["keyword_retention"] = present / len(keywords)
    else:
        metrics["keyword_retention"] = float("nan")

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# AGREGACIONES
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_metric(rows: list[dict], metric: str) -> dict:
    """Promedio simple ignorando NaN/None."""
    vals = [r[metric] for r in rows if r.get(metric) is not None and not (isinstance(r[metric], float) and math.isnan(r[metric]))]
    return {
        "n":   len(vals),
        "mean": _mean(vals),
        "median": float(np.median(vals)) if vals else 0.0,
    }


def segmented_summary(rows: list[dict], group_key: str, metrics: list[str]) -> dict:
    groups = defaultdict(list)
    for r in rows:
        groups[r.get(group_key)].append(r)
    out = {}
    for g, gr in groups.items():
        out[str(g)] = {"n": len(gr), **{m: aggregate_metric(gr, m)["mean"] for m in metrics}}
    return out


def thieves_matrix(records: list[dict]) -> dict:
    """Top-1 erróneo: docs que roban la primera posición agrupados por GT esperado."""
    per_gt = defaultdict(Counter)
    for rec in records:
        gt = rec["_gt"]
        for q in rec.get("per_question") or []:
            labels = q.get("labels") or []
            if labels and labels[0] == 0:
                docs = q.get("retrieved_docs") or []
                if docs:
                    ref = docs[0].get("doc_reference", "?")[:55]
                    per_gt[gt][ref] += 1
    return {gt: dict(c.most_common(8)) for gt, c in per_gt.items()}


def reranker_auc_per_gt(records: list[dict]) -> dict:
    """AUC reranker por GT (relevant=1, otros=0 dentro de cada query, micro)."""
    out = {}
    for rec in records:
        gt = rec["_gt"]
        scores = []
        labels = []
        for q in rec.get("per_question") or []:
            for d in q.get("retrieved_docs") or []:
                if "rerank_score" in d:
                    scores.append(float(d["rerank_score"]))
                    labels.append(int(d.get("relevance_label", 0)))
        out[gt] = _binary_auc(scores, labels)
    return out


def calibration_curve(records: list[dict], n_bins: int = 10) -> list[dict]:
    """Calibración: para bins de rerank_score, % de relevantes."""
    bins = [[] for _ in range(n_bins)]
    for rec in records:
        for q in rec.get("per_question") or []:
            for d in q.get("retrieved_docs") or []:
                s = float(d.get("rerank_score", 0))
                b = min(int(s * n_bins), n_bins - 1)
                if b < 0: b = 0
                bins[b].append(int(d.get("relevance_label", 0)))
    out = []
    for i, b in enumerate(bins):
        out.append({
            "bin_low":  i / n_bins,
            "bin_high": (i + 1) / n_bins,
            "n":        len(b),
            "precision": _mean(b) if b else None,
        })
    return out


def cost_estimates(rows: list[dict]) -> dict:
    """Coste por query y total con tres tarifas USD."""
    out = {}
    total_in  = sum(r.get("prompt_tokens") or 0 for r in rows)
    total_out = sum(r.get("output_tokens") or 0 for r in rows)
    n = sum(1 for r in rows if r.get("output_tokens"))
    for name, (pin, pout) in PRICE_TABLE_USD.items():
        cost = (total_in / 1000 * pin) + (total_out / 1000 * pout)
        out[name] = {
            "total_usd":     round(cost, 4),
            "per_query_usd": round(cost / n, 6) if n else 0.0,
        }
    out["_totals"] = {"prompt_tokens": total_in, "output_tokens": total_out, "n_queries_with_gen": n}
    return out


# ─────────────────────────────────────────────────────────────────────────────
# MÉTRICAS PESADAS (OPCIONALES)
# ─────────────────────────────────────────────────────────────────────────────

def add_bertscore(rows: list[dict], lang: str = "es") -> None:
    """Añade `bertscore_recall` a cada row (in-place)."""
    if not _HAS_BERTSCORE:
        warnings.warn("bert_score no instalado — saltando BERTScore")
        return
    pairs = [(r["_quote"], r["_gen"]) for r in rows if r.get("_quote") and r.get("_gen")]
    idx   = [i for i, r in enumerate(rows) if r.get("_quote") and r.get("_gen")]
    if not pairs:
        return
    cands = [p[1] for p in pairs]
    refs  = [p[0] for p in pairs]
    print(f"[BERTScore] computando para {len(pairs)} pares...")
    P, R, F = bertscore_score(cands, refs, lang=lang, rescale_with_baseline=False, verbose=False)
    R = R.tolist()
    for i, ridx in enumerate(idx):
        rows[ridx]["bertscore_recall"] = float(R[i])


def add_embedding_similarity(rows: list[dict], model_name: str = "intfloat/multilingual-e5-small") -> None:
    """Añade `embed_sim` (cosine) entre quote y respuesta."""
    if not _HAS_ST:
        warnings.warn("sentence_transformers no instalado — saltando embedding similarity")
        return
    pairs = [(r["_quote"], r["_gen"]) for r in rows if r.get("_quote") and r.get("_gen")]
    idx   = [i for i, r in enumerate(rows) if r.get("_quote") and r.get("_gen")]
    if not pairs:
        return
    print(f"[Embedding-sim] cargando {model_name} y embeddando {len(pairs)*2} textos...")
    m = SentenceTransformer(model_name)
    refs = m.encode([p[0] for p in pairs], normalize_embeddings=True, show_progress_bar=False)
    cands = m.encode([p[1] for p in pairs], normalize_embeddings=True, show_progress_bar=False)
    sims = (refs * cands).sum(axis=1).tolist()
    for i, ridx in enumerate(idx):
        rows[ridx]["embed_sim"] = float(sims[i])


# ─────────────────────────────────────────────────────────────────────────────
# RENDER HTML
# ─────────────────────────────────────────────────────────────────────────────

def _table(headers: list[str], rows: list[list]) -> str:
    h = "".join(f"<th>{html.escape(str(c))}</th>" for c in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in r) + "</tr>"
        for r in rows
    )
    return f"<table><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table>"


def _fmt(v, dec=3) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    if isinstance(v, float):
        return f"{v:.{dec}f}"
    return str(v)


def render_html(summary: dict, output_path: Path) -> None:
    css = """
    body{font-family:system-ui,sans-serif;background:#0f1117;color:#e2e8f0;font-size:13px;
         line-height:1.5;padding:24px;max-width:1200px;margin:auto}
    h1,h2,h3{color:#a5b4fc;margin:1em 0 0.4em}
    h1{font-size:18px;border-bottom:1px solid #2d3148;padding-bottom:8px}
    h2{font-size:15px;margin-top:1.6em}
    h3{font-size:13px;color:#c7d2fe}
    table{border-collapse:collapse;margin:8px 0;font-size:12px;width:100%}
    th{text-align:left;padding:6px 10px;border-bottom:1px solid #2d3148;color:#94a3b8;font-weight:600}
    td{padding:5px 10px;border-bottom:1px solid #1a1f35}
    tr:hover td{background:#161925}
    .meta{color:#94a3b8;font-size:11px}
    code{background:#1e2130;padding:1px 5px;border-radius:3px;font-size:11.5px}
    .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:14px}
    .card{background:#1e2130;border:1px solid #2d3148;border-radius:8px;padding:14px}
    .big{font-size:22px;color:#a5b4fc;font-weight:700}
    .lbl{font-size:11px;color:#94a3b8;text-transform:uppercase;letter-spacing:.04em}
    .bar{display:inline-block;height:6px;background:#6366f1;border-radius:3px;vertical-align:middle}
    """
    parts = [f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>RAG metrics</title><style>{css}</style></head><body>"]
    parts.append(f"<h1>Métricas RAG — {html.escape(summary.get('source_dir',''))}</h1>")
    parts.append(f"<div class='meta'>Generado: {html.escape(summary.get('generated_at',''))} · "
                 f"GTs analizados: {summary.get('n_gts',0)} · "
                 f"Preguntas totales: {summary.get('n_questions',0)}</div>")

    # Big cards: agregados clave
    g = summary["global_ir"]
    parts.append("<div class='grid'>")
    for label, v in [("Hit@1", g.get("hit@1")), ("Hit@5", g.get("hit@5")),
                     ("MRR@10", g.get("mrr@10")), ("NDCG@5", g.get("ndcg@5"))]:
        parts.append(f"<div class='card'><div class='lbl'>{label}</div><div class='big'>{_fmt(v)}</div></div>")
    auc_macro = summary.get("reranker_auc_macro")
    parts.append(f"<div class='card'><div class='lbl'>AUC reranker</div><div class='big'>{_fmt(auc_macro)}</div></div>")
    parts.append("</div>")

    # IR por GT
    parts.append("<h2>IR por documento (GT)</h2>")
    head = ["GT", "n", "Hit@1", "Hit@3", "Hit@5", "Hit@10", "MRR@10", "NDCG@5"]
    rows = []
    for gt, m in sorted(summary["ir_by_gt"].items()):
        rows.append([gt, m["n"], _fmt(m["hit@1"]), _fmt(m["hit@3"]), _fmt(m["hit@5"]),
                     _fmt(m["hit@10"]), _fmt(m["mrr@10"]), _fmt(m["ndcg@5"])])
    parts.append(_table(head, rows))

    # IR segmentado por bloque/longitud/protocolo/followup
    for label, key in [("Por bloque", "ir_by_block"),
                        ("Por longitud de query", "ir_by_lenbucket"),
                        ("Por familia de protocolo", "ir_by_proto"),
                        ("Por followup", "ir_by_followup"),
                        ("Por chitchat", "ir_by_chitchat")]:
        if key not in summary or not summary[key]:
            continue
        parts.append(f"<h2>IR — {label}</h2>")
        head = [label, "n", "Hit@1", "Hit@5", "MRR@10", "NDCG@5"]
        rows = []
        for k, m in sorted(summary[key].items()):
            rows.append([k, m["n"], _fmt(m.get("hit@1")), _fmt(m.get("hit@5")),
                         _fmt(m.get("mrr@10")), _fmt(m.get("ndcg@5"))])
        parts.append(_table(head, rows))

    # Latencia
    if summary.get("latency"):
        parts.append("<h2>Latencia (segundos)</h2>")
        head = ["Fase", "n", "p50", "p95", "p99", "media"]
        rows = []
        for phase, info in summary["latency"].items():
            rows.append([phase, info["n"], _fmt(info["p50"], 2), _fmt(info["p95"], 2),
                         _fmt(info["p99"], 2), _fmt(info["mean"], 2)])
        parts.append(_table(head, rows))

    # Throughput / generación
    if summary.get("generation"):
        gen = summary["generation"]
        parts.append("<h2>Generación</h2>")
        parts.append(_table(
            ["Métrica", "Valor"],
            [
                ["Tokens/seg (media)",   _fmt(gen.get("tokens_per_s_mean"), 1)],
                ["Tokens/seg (p50)",     _fmt(gen.get("tokens_per_s_p50"), 1)],
                ["Output tokens (media)", _fmt(gen.get("output_tokens_mean"), 1)],
                ["Response chars (media)",_fmt(gen.get("response_chars_mean"), 1)],
                ["Keyword retention (media)", _fmt(gen.get("keyword_retention_mean"))],
                ["ROUGE-L recall vs quote (media)", _fmt(gen.get("rougeL_recall_mean"))],
                ["Bigram recall vs quote (media)", _fmt(gen.get("quote_bigram_recall_mean"))],
                ["BERTScore recall (media)", _fmt(gen.get("bertscore_recall_mean"))],
                ["Embedding similarity (media)", _fmt(gen.get("embed_sim_mean"))],
            ],
        ))

    # Coste
    if summary.get("cost"):
        parts.append("<h2>Coste estimado (cloud equivalent)</h2>")
        rows = []
        for name, info in summary["cost"].items():
            if name.startswith("_"):
                continue
            rows.append([name, _fmt(info["total_usd"], 4), _fmt(info["per_query_usd"], 6)])
        parts.append(_table(["Modelo", "Total USD", "USD/query"], rows))
        parts.append(f"<div class='meta'>Tokens totales: prompt={summary['cost']['_totals']['prompt_tokens']:,} · "
                     f"output={summary['cost']['_totals']['output_tokens']:,} · "
                     f"queries con generación={summary['cost']['_totals']['n_queries_with_gen']}</div>")

    # Reranker
    if summary.get("reranker_auc_by_gt"):
        parts.append("<h2>AUC del re-ranker por GT</h2>")
        head = ["GT", "AUC"]
        rows = [[gt, _fmt(v)] for gt, v in sorted(summary["reranker_auc_by_gt"].items())]
        parts.append(_table(head, rows))

    if summary.get("calibration"):
        parts.append("<h2>Calibración del re-ranker</h2>")
        head = ["Bin score", "n", "Precisión observada"]
        rows = [[f"{c['bin_low']:.1f}–{c['bin_high']:.1f}", c["n"], _fmt(c["precision"])]
                for c in summary["calibration"]]
        parts.append(_table(head, rows))

    # Docs ladrones
    if summary.get("thieves"):
        parts.append("<h2>Docs ladrones (top-1 erróneo más frecuente)</h2>")
        for gt, c in sorted(summary["thieves"].items()):
            if not c:
                continue
            parts.append(f"<h3>{gt}</h3>")
            parts.append(_table(["Doc ladrón", "Veces"], [[k, v] for k, v in c.items()]))

    # Diversidad y drop-off
    if summary.get("diversity"):
        parts.append("<h2>Diversidad y drop-off</h2>")
        d = summary["diversity"]
        parts.append(_table(
            ["Métrica", "Valor"],
            [
                ["Nº medio de docs únicos en top-15", _fmt(d.get("unique_docs_topK_mean"), 2)],
                ["P@1 (media)", _fmt(d.get("prec@1_mean"))],
                ["P@5 (media)", _fmt(d.get("prec@5_mean"))],
                ["P@10 (media)", _fmt(d.get("prec@10_mean"))],
                ["P@15 (media)", _fmt(d.get("prec@15_mean"))],
            ],
        ))

    # Routing
    if summary.get("routing"):
        parts.append("<h2>Routing</h2>")
        r = summary["routing"]
        parts.append(_table(
            ["Métrica", "Valor"],
            [
                ["% queries con is_followup=True", _fmt(r.get("followup_rate"))],
                ["% queries con is_chitchat=True", _fmt(r.get("chitchat_rate"))],
            ],
        ))

    parts.append("</body></html>")
    output_path.write_text("\n".join(parts), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir",  required=True, help="Carpeta con results_*.json")
    parser.add_argument("--output-dir", default=None,  help="Donde escribir summary.json + report.html (default = input-dir)")
    parser.add_argument("--bertscore",  action="store_true", help="Calcular BERTScore (requiere bert_score)")
    parser.add_argument("--embed-sim",  action="store_true", help="Calcular embedding similarity (requiere sentence_transformers)")
    parser.add_argument("--embed-model", default="intfloat/multilingual-e5-small")
    args = parser.parse_args()

    in_dir  = Path(args.input_dir)
    out_dir = Path(args.output_dir) if args.output_dir else in_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_dir.exists():
        print(f"[ERROR] Carpeta no encontrada: {in_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"[1/4] Cargando JSONs desde {in_dir}...")
    records = load_results(in_dir)
    if not records:
        print("[ERROR] Ningún results_*.json encontrado", file=sys.stderr)
        sys.exit(1)

    # ── Per-question rows ─────────────────────────────────────────────────
    print(f"[2/4] Calculando métricas por pregunta...")
    rows: list[dict] = []
    for rec in records:
        for q in rec.get("per_question") or []:
            r = per_question_metrics(rec, q)
            r["_quote"] = q.get("expected_quote") or ""
            r["_gen"]   = q.get("generated_response") or ""
            rows.append(r)

    print(f"      {len(rows)} preguntas en total a través de {len(records)} GTs")

    # Métricas pesadas opcionales
    if args.bertscore:
        add_bertscore(rows, lang="es")
    if args.embed_sim:
        add_embedding_similarity(rows, model_name=args.embed_model)

    # ── Agregaciones ──────────────────────────────────────────────────────
    print(f"[3/4] Agregando...")
    ir_keys = [f"hit@{k}" for k in KS] + [f"mrr@{k}" for k in KS] + \
              [f"prec@{k}" for k in KS] + [f"ndcg@{k}" for k in KS]

    # Global
    global_ir = {k: aggregate_metric(rows, k)["mean"] for k in ir_keys}

    # Por GT
    ir_by_gt = {}
    for gt, gr in groupby_dict(rows, "gt").items():
        ir_by_gt[gt] = {"n": len(gr), **{k: aggregate_metric(gr, k)["mean"] for k in ir_keys}}

    # Segmentadas
    seg_metrics = ["hit@1", "hit@3", "hit@5", "hit@10", "mrr@10", "ndcg@5"]
    summary: dict = {
        "source_dir":   str(in_dir),
        "generated_at": _now_iso(),
        "n_gts":        len(records),
        "n_questions":  len(rows),
        "global_ir":    global_ir,
        "ir_by_gt":     ir_by_gt,
        "ir_by_block":     segmented_summary(rows, "block",        seg_metrics),
        "ir_by_lenbucket": segmented_summary(rows, "len_bucket",   seg_metrics),
        "ir_by_proto":     segmented_summary(rows, "proto_family", seg_metrics),
        "ir_by_followup":  segmented_summary(rows, "is_followup",  seg_metrics),
        "ir_by_chitchat":  segmented_summary(rows, "is_chitchat",  seg_metrics),
    }

    # Latencia
    latency = {}
    for phase in ("router", "retriever", "generador"):
        vals = [r[f"t_{phase}_s"] for r in rows if r.get(f"t_{phase}_s") is not None]
        if vals:
            pcts = _percentiles(vals)
            latency[phase.capitalize()] = {"n": len(vals), "mean": _mean(vals), **pcts}
    vals_total = [r["t_total_s"] for r in rows if r.get("t_total_s")]
    if vals_total:
        latency["End-to-end"] = {"n": len(vals_total), "mean": _mean(vals_total), **_percentiles(vals_total)}
    summary["latency"] = latency

    # Generación / throughput / texto vs quote
    gen = {}
    for k in ("tokens_per_s", "output_tokens", "prompt_tokens", "context_used_pct",
              "response_chars", "response_words", "keyword_retention",
              "quote_unigram_recall", "quote_bigram_recall", "rougeL_recall",
              "bertscore_recall", "embed_sim"):
        info = aggregate_metric(rows, k)
        if info["n"]:
            gen[f"{k}_mean"]   = info["mean"]
            gen[f"{k}_median"] = info["median"]
    if "tokens_per_s_mean" in gen:
        ts_vals = [r["tokens_per_s"] for r in rows if r.get("tokens_per_s")]
        gen["tokens_per_s_p50"] = _percentiles(ts_vals)["p50"]
    summary["generation"] = gen

    # Coste
    summary["cost"] = cost_estimates(rows)

    # Reranker
    auc_per = reranker_auc_per_gt(records)
    auc_vals = [v for v in auc_per.values() if v == v]  # filtra NaN
    summary["reranker_auc_by_gt"] = {gt: round(v, 4) if v == v else None for gt, v in auc_per.items()}
    summary["reranker_auc_macro"] = float(np.mean(auc_vals)) if auc_vals else None
    summary["calibration"] = calibration_curve(records)

    # Diversidad / drop-off
    summary["diversity"] = {
        "unique_docs_topK_mean": aggregate_metric(rows, "n_unique_docs_topK")["mean"],
        **{f"prec@{k}_mean": aggregate_metric(rows, f"prec@{k}")["mean"] for k in KS},
    }

    # Routing
    fu_vals = [r["is_followup"] for r in rows if r.get("is_followup") is not None]
    cc_vals = [r["is_chitchat"] for r in rows if r.get("is_chitchat") is not None]
    summary["routing"] = {
        "followup_rate": _safe_div(sum(1 for v in fu_vals if v), len(fu_vals)) if fu_vals else None,
        "chitchat_rate": _safe_div(sum(1 for v in cc_vals if v), len(cc_vals)) if cc_vals else None,
    }

    # Docs ladrones
    summary["thieves"] = thieves_matrix(records)

    # ── Salidas ───────────────────────────────────────────────────────────
    print(f"[4/4] Escribiendo salidas en {out_dir}...")
    (out_dir / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=_to_jsonable), encoding="utf-8"
    )

    # CSV por pregunta
    csv_path = out_dir / "metrics_per_question.csv"
    csv_cols = [k for k in rows[0].keys() if not k.startswith("_")]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=csv_cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in csv_cols})

    # HTML
    render_html(summary, out_dir / "metrics_report.html")

    print(f"\n✓ metrics_summary.json     ({(out_dir / 'metrics_summary.json').stat().st_size:,} bytes)")
    print(f"✓ metrics_per_question.csv ({csv_path.stat().st_size:,} bytes, {len(rows)} filas)")
    print(f"✓ metrics_report.html      ({(out_dir / 'metrics_report.html').stat().st_size:,} bytes)")
    print()
    print(f"  Resumen rápido:")
    print(f"    n_questions = {summary['n_questions']}, n_gts = {summary['n_gts']}")
    print(f"    Hit@1 macro = {summary['global_ir']['hit@1']:.3f}")
    print(f"    MRR@10 macro = {summary['global_ir']['mrr@10']:.3f}")
    if summary["reranker_auc_macro"]:
        print(f"    AUC reranker macro = {summary['reranker_auc_macro']:.3f}")


def groupby_dict(rows: list[dict], key: str) -> dict:
    out = defaultdict(list)
    for r in rows:
        out[r.get(key)].append(r)
    return out


def _now_iso() -> str:
    from datetime import datetime
    return datetime.now().isoformat(timespec="seconds")


def _to_jsonable(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(repr(o))


if __name__ == "__main__":
    main()
