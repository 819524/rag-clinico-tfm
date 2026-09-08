"""
generar_gt_global.py — Construye un ground truth GLOBAL para prueba de estrés
=============================================================================
Muestrea N preguntas de cada ground truth individual (eval/*_ground_truth.json),
les asigna un patrón de relevancia ÚNICO por documento (calculado por pregunta,
no por fichero) y las mezcla en orden aleatorio en un único fichero.

Motivación
----------
El evaluador estándar usa un único `relevant_doc_patterns` por fichero. En un
fichero global con preguntas de muchos documentos eso infla las métricas: una
pregunta de PA-02 que recupere un doc de PA-06 contaría como acierto. Por eso
aquí cada pregunta lleva su propio `relevant_doc_patterns`, y eval_retriever.py
los usa con prioridad sobre el patrón de fichero.

Las preguntas comparten el mismo `block` para que el historial conversacional y
el doc-anchor NO se reseteen entre ellas (memoria deslizante a lo largo de toda
la sesión): es un test de robustez del router ante cambios de tema con contexto
previo no relacionado.

Patrón único por documento
--------------------------
Se usa el CÓDIGO de documento (no slugs de tema, que colisionan entre docs sobre
el mismo asunto). Se desambiguan colisiones de substring (p. ej. `PA-17` ⊂
`PA-178`) con un guion final, y casos dudosos (`PTP`, `LPP`) con un prefijo
seguro y único.

Uso:
    python Scripts/generar_gt_global.py
    python Scripts/generar_gt_global.py --per-doc 5 --seed 42 --out eval/GLOBAL_ground_truth.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Patrón de relevancia ÚNICO por documento. Clave = nombre de fichero (stem sin
# _ground_truth). Si un documento no está aquí, se usa su primer patrón de
# fichero (válido cuando no hay colisiones de substring con otros códigos).
UNIQUE_PATTERN_OVERRIDE = {
    "PA17":   "PA-17-",   # evita colisión con PA-178
    "PTP2":   "PTP",      # el doc_reference real es PTP-02 (el fichero usa PTP-2)
    "LPP2013": "LPP",     # el doc_reference real es "2013_..._LPP"
    "RPC120": "RCP-120",  # ¡el doc_reference real está transpuesto: RCP-120, no RPC-120!
}


def unique_pattern_for(stem: str, file_patterns: list[str]) -> str:
    """Devuelve el patrón único y desambiguado para el documento."""
    if stem in UNIQUE_PATTERN_OVERRIDE:
        return UNIQUE_PATTERN_OVERRIDE[stem]
    # Por defecto, el primer patrón del fichero (suele ser el código del doc).
    return file_patterns[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-doc", type=int, default=5,
                    help="Preguntas a muestrear por documento (default: 5)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Semilla para muestreo y mezcla (default: 42)")
    ap.add_argument("--out", default="eval/GLOBAL_ground_truth.json",
                    help="Fichero de salida (default: eval/GLOBAL_ground_truth.json)")
    ap.add_argument("--block", type=int, default=1,
                    help="Block común para todas las preguntas (memoria deslizante). default: 1")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    gt_files = sorted(glob.glob(str(BASE_DIR / "eval" / "*_ground_truth.json")))
    # Excluir cualquier global previo para no auto-incluirse.
    gt_files = [f for f in gt_files if "GLOBAL" not in os.path.basename(f).upper()]

    pooled: list[dict] = []
    all_patterns: list[str] = []
    summary: list[tuple[str, int, str]] = []

    for f in gt_files:
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        stem = os.path.basename(f).replace("_ground_truth.json", "")
        file_patterns = d.get("relevant_doc_patterns", [])
        pat = unique_pattern_for(stem, file_patterns)
        all_patterns.append(pat)

        questions = list(d.get("questions", []))
        n = min(args.per_doc, len(questions))
        chosen = rng.sample(questions, n)
        summary.append((stem, n, pat))

        for q in chosen:
            entry = {
                "source_doc": stem,
                "orig_id": q.get("id"),
                "block": args.block,
                "query": q["query"],
                "expected_section": q.get("expected_section"),
                "relevant_doc_patterns": [pat],
            }
            # Conservar el campo de respuesta de referencia que tuviera.
            if "expected_quote" in q:
                entry["expected_quote"] = q["expected_quote"]
            if "expected_answer" in q:
                entry["expected_answer"] = q["expected_answer"]
            pooled.append(entry)

    # Mezcla global e ids secuenciales tras mezclar.
    rng.shuffle(pooled)
    for i, q in enumerate(pooled, 1):
        q_ordered = {"id": f"g{i:03d}"}
        q_ordered.update(q)
        pooled[i - 1] = q_ordered

    out = {
        "name": "GLOBAL — Prueba de estrés multi-documento del sistema RAG",
        "description": (
            f"Ground truth global construido muestreando {args.per_doc} preguntas de cada uno "
            f"de los {len(gt_files)} ground truth individuales del corpus del HCU Lozano Blesa "
            f"(semilla {args.seed}). Las preguntas se mezclan en orden aleatorio e interleaved "
            f"entre documentos, compartiendo un mismo block para mantener la memoria "
            f"conversacional deslizante activa durante toda la sesión. Cada pregunta lleva su "
            f"propio relevant_doc_patterns (patrón único por documento) para que las métricas de "
            f"recuperación se calculen por pregunta. Mide la capacidad del router/retriever de "
            f"enrutar a la sección correcta entre todo el corpus ante cambios de tema con "
            f"contexto previo no relacionado."
        ),
        # Unión de patrones a nivel de fichero (solo informativo / fallback).
        "relevant_doc_patterns": sorted(set(all_patterns)),
        "questions": pooled,
    }

    out_path = BASE_DIR / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Fichero global: {out_path}")
    print(f"Documentos: {len(gt_files)}  |  Preguntas totales: {len(pooled)}")
    print(f"{'documento':<26}{'n':>3}  patrón único")
    print("-" * 60)
    for stem, n, pat in summary:
        print(f"{stem:<26}{n:>3}  {pat}")


if __name__ == "__main__":
    main()
