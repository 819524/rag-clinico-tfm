"""eval_gen_quality.py — Métricas de calidad de la respuesta generada
=====================================================================
Desglosa por (modelo de embedding, estrategia de reordenación).
Reutiliza _tokens / _ngram_recall / _rouge_l_recall_fast de Scripts/eval_compute_metrics.py.
100% offline sobre los results_*.json existentes: NO reejecuta el pipeline.
Reutiliza _tokens / _ngram_recall / _rouge_l_recall_fast de eval_compute_metrics.py.

Uso:  tfm/bin/python Scripts/eval_gen_quality.py

Salvedades conocidas (documentadas en memoria/TRAZABILIDAD_cap5.md):
  · 14/1845 preguntas del set de 615 no tienen `generated_response` (fallo de
    generacion en la ejecucion original); se excluyen del calculo.
  · Algunos registros traen tokens_per_second = 1e6, centinela de division por
    cero cuando Ollama reporta eval_duration = 0; se excluyen y se usa MEDIANA.
"""
import sys, json, glob, os, re, statistics as st
sys.path.insert(0, "Scripts")
from eval_compute_metrics import _tokens, _ngram_recall, _rouge_l_recall_fast

CITA = re.compile(r"\[(\d{1,2})\]")
STRATS = ["cross_encoder", "embed", "rrf_only"]
NAME = {"cross_encoder": "Cross-encoder", "embed": "Híbrida", "rrf_only": "RRF puro"}
CONF = {"4B (615 preg.)":   "eval/2026-05-19_topk10",
        "0.6B (615 preg.)": "eval/2026-05-31_06b_topk10",
        "4B (110 multi)":   "eval/global_topk10",
        "0.6B (110 multi)": "eval/global_06b_topk10"}

def strat_of(f):
    m = re.search(r"(embed|cross_encoder|rrf_only)", f); return m.group(1) if m else "?"

def m(xs): return st.mean(xs) if xs else float("nan")
def md(xs): return st.median(xs) if xs else float("nan")
SENTINEL=1e6

for etiqueta, folder in CONF.items():
    print(f"\n{'='*104}\n{etiqueta}   ({folder})\n{'='*104}")
    print(f"{'Estrategia':14}{'n':>5}{'ROUGE-L':>9}{'RecCita':>9}{'RetKW':>8}"
          f"{'%c/cita':>9}{'citas':>7}{'%válidas':>10}{'palabras':>10}{'tok/s*':>8}{'t_gen':>7}{'t_total':>8}")
    for s in STRATS:
        R = {k: [] for k in "rouge quote kw pal toks tgen ttot ncit sent".split()}
        con_cita = 0; cit_tot = 0; cit_ok = 0; n = 0
        for f in sorted(glob.glob(os.path.join(folder, "results_*.json"))):
            if strat_of(os.path.basename(f)) != s: continue
            d = json.load(open(f, encoding="utf-8"))
            topk = d.get("top_k", 10)
            for q in d.get("per_question", []):
                gen = q.get("generated_response") or ""; quote = q.get("expected_quote") or ""
                if not gen or not quote: continue
                n += 1
                R["rouge"].append(_rouge_l_recall_fast(quote, gen))
                tq = set(_tokens(quote)); tg = set(_tokens(gen))
                R["quote"].append(len(tq & tg) / len(tq) if tq else 0.0)
                kws = [k.lower() for k in (q.get("router", {}).get("keywords") or [])]
                if kws:
                    gl = gen.lower(); R["kw"].append(sum(1 for k in kws if k in gl) / len(kws))
                cits = [int(x) for x in CITA.findall(gen)]
                if cits: con_cita += 1
                cit_tot += len(cits); cit_ok += sum(1 for c in cits if 1 <= c <= topk)
                R["ncit"].append(len(cits))
                g = q.get("generation", {}) or {}; ts = g.get("token_stats", {}) or {}
                v=g.get("tokens_per_second") or ts.get("tokens_per_second")
                if v and v < SENTINEL: R["toks"].append(v)
                elif v: R.setdefault("sent",[]).append(1)
                R["pal"].append(len(gen.split()))
                t = q.get("timings", {}) or {}
                if t.get("Generador"): R["tgen"].append(t["Generador"])
                if t: R["ttot"].append(sum(v for v in t.values() if isinstance(v, (int, float))))
        if not n: continue
        R2 = R["toks"]
        nsent=len(R.get("sent",[]))
        if nsent: print(f"    [aviso] {nsent} registros con tokens_per_second=1e6 excluidos")
        print(f"{NAME[s]:14}{n:>5}{m(R['rouge']):>9.3f}{m(R['quote']):>9.3f}{m(R['kw']):>8.3f}"
              f"{con_cita/n*100:>8.1f}%{m(R['ncit']):>7.1f}{(cit_ok/cit_tot*100 if cit_tot else 0):>9.1f}%"
              f"{m(R['pal']):>10.0f}{md(R2):>8.1f}{m(R['tgen']):>7.1f}{m(R['ttot']):>8.1f}")
print("""
LEYENDA
  ROUGE-L  : recall de la subsecuencia común más larga cita↔respuesta (_rouge_l_recall_fast)
  RecCita  : fracción de palabras de contenido de la cita presentes en la respuesta
  RetKW    : fracción de keywords del router presentes en la respuesta
  %c/cita  : % de respuestas con al menos una marca [n]
  citas    : nº medio de marcas [n] por respuesta
  %válidas : % de marcas [n] con n dentro del rango de fuentes entregadas (1..top_k)
  tok/s*   : MEDIANA; se excluyen los registros con el centinela 1e6 (eval_duration=0 en Ollama)
""")
