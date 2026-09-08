"""
section_overrides.py — Overrides manuales de relevancia a nivel SECCIÓN
======================================================================
El etiquetado automático de sección marca un chunk como relevante si el
expected_quote tiene recall léxico >= umbral en su texto. Cuando la cita está
reformulada respecto al texto original, puede quedar por debajo del umbral aunque
la sección SÍ contenga la respuesta. Tras revisión humana, este módulo permite
forzar section_label=1 para combinaciones concretas (pregunta, estrategia).

Formato del fichero JSON, SCOPED POR DOCUMENTO (claves "_..." se ignoran):
    {
      "GLOBAL": {
        "g108": {"section": "3. DEFINICIONES", "doc": "PA-195",
                 "strategies": ["embed", "rrf_only", "cross_encoder"]}
      },
      "PE27": {
        "q02": {"sections": [{"section": "...", "doc": "PE-27"}, ...],
                "strategies": ["embed", "rrf_only", "cross_encoder"]}
      }
    }
La clave de primer nivel es el documento (protocolo) tal como lo identifica el
informe: "GLOBAL" para el ground truth global, o el código del protocolo (p.ej.
"PE27") en evaluaciones por documento. Se fuerza a 1 la etiqueta del chunk cuya
h2 (o h1 si no hay h2) coincide exactamente con "section" (normalizada) y cuyo
doc_reference contiene "doc" (si se indica), solo para las estrategias listadas.
"""

from __future__ import annotations

import json
from pathlib import Path


def _norm(s: str) -> str:
    return " ".join((s or "").strip().split()).casefold()


def load_overrides(path) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if not k.startswith("_")}


def _targets(ov: dict) -> list[tuple[str, str]]:
    """Lista de (section_normalizada, doc_filter_lower) del override.
    Acepta tanto el formato simple (section/doc) como una lista en 'sections'."""
    out = []
    if ov.get("sections"):
        for item in ov["sections"]:
            sec = _norm(item.get("section", ""))
            if sec:
                out.append((sec, (item.get("doc") or "").lower()))
    else:
        sec = _norm(ov.get("section", ""))
        if sec:
            out.append((sec, (ov.get("doc") or "").lower()))
    return out


def apply_override(doc: str, qid: str, strat: str, retrieved_docs: list[dict],
                   sec_labels: list[int], overrides: dict) -> int:
    """Fuerza sec_labels[i]=1 en los chunks cuya (h2, doc) coinciden con alguna de
    las secciones revisadas manualmente, para (documento, pregunta, estrategia).
    Modifica sec_labels in-place. Devuelve el nº de etiquetas cambiadas."""
    ov = overrides.get(doc, {}).get(qid)
    if not ov or strat not in ov.get("strategies", []):
        return 0
    targets = _targets(ov)
    if not targets:
        return 0
    changed = 0
    for i, d in enumerate(retrieved_docs):
        if i >= len(sec_labels) or sec_labels[i] == 1:
            continue
        h = _norm(d.get("h2", "") or d.get("h1", ""))
        ref = (d.get("doc_reference", "") or "").lower()
        for sec, doc_filter in targets:
            if h == sec and (not doc_filter or doc_filter in ref):
                sec_labels[i] = 1
                changed += 1
                break
    return changed
