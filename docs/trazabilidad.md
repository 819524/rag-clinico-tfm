# Trazabilidad de las cifras del Capítulo 5

Documento de control interno. Para cada número que aparece en el capítulo, de dónde sale
y con qué comando se reproduce. **No forma parte de la memoria.**

---

## 1. Cadena de datos

```
Scripts/eval_retriever.py            → ejecución original del pipeline (NO se ha reejecutado)
   └── eval/<carpeta>/results_<DOC>_ground_truth_..._<estrategia>_....json
          per_question[]: {id, query, expected_quote, expected_section,
                           router{}, retrieved_docs[10]{doc_reference,h1,h2,relevance_label},
                           timings{}, generated_response, generation{}}
   ├── Scripts/eval_section_metrics.py  → section_metrics_summary.json   (métricas de RECUPERACIÓN)
   └── Scripts/eval_gen_quality.py      → stdout                          (métricas de GENERACIÓN)
```

| Carpeta | Contenido | n |
|---|---|---|
| `eval/2026-05-19_topk10`     | 22 protocolos por separado, embedding 4B   | 615 |
| `eval/2026-05-31_06b_topk10` | 22 protocolos por separado, embedding 0.6B | 615 |
| `eval/global_topk10`         | sesión multi-documento, embedding 4B       | 110 |
| `eval/global_06b_topk10`     | sesión multi-documento, embedding 0.6B     | 110 |

---

## 2. Tablas de recuperación (tab:generador, tab:generador-multidoc)

**Origen exacto:** claves `sec@0.7_*` de `section_metrics_summary.json` en cada carpeta.

Reproducir:
```bash
tfm/bin/python -c "
import json
d=json.load(open('eval/2026-05-19_topk10/section_metrics_summary.json'))
print({k:v for k,v in d['strategies']['cross_encoder'].items() if k.startswith('sec@0.7')})"
```

Verificado: las 36 celdas de las dos tablas del capítulo coinciden al tercer decimal.

**Aviso:** las claves `doc_*` del mismo fichero son relevancia a nivel de DOCUMENTO
(Hit@1 = 0,886 para 4B+cross-encoder) y NO son las que se publican. No confundirlas.

---

## 3. Definición de relevancia (por qué "a nivel de sección")

`Scripts/eval_section_metrics.py`, método `SectionTextCache.get()`:

```sql
SELECT text FROM rag_chunks_qwen3_v2 WHERE doc_id=? AND h1=? AND h2=?
```
seguido de `" ".join(row[0] for row in rows)` — **concatena todos los fragmentos de la
sección**. Por tanto un fragmento recuperado se etiqueta como relevante si SU SECCIÓN
contiene la respuesta, no si el fragmento concreto la contiene.

Esto es coherente con el diseño del sistema: los enlaces de continuidad reconstruyen la
sección completa a partir de cualquiera de sus fragmentos (Cap. 3, §Recuperador híbrido).
De ahí que el caption correcto sea "a nivel de sección" y no "a nivel de chunk".

---

## 4. Etiquetado: filtro automático + revisión humana

**Paso 1 — filtro automático.** `content_recall(quote, texto_seccion) >= 0.7`, donde
`content_recall` = |palabras_contenido(cita) ∩ palabras_contenido(sección)| / |palabras_contenido(cita)|,
tras normalizar a ASCII minúsculas, tokenizar a alfanuméricos de >2 caracteres y quitar stopwords.

**Paso 2 — revisión humana** de la totalidad de los pares pregunta-respuesta. Cuando el
filtro no había identificado la sección correcta, se anotó cuál era y su posición, y la
métrica se recalculó sobre esa posición (`section_overrides.json` + `apply_override()`).
La revisión no identificó falsos positivos del filtro.

**Alcance medido de la corrección (umbral 0,7):**

| Carpeta | Etiquetas de chunk forzadas | Pares (pregunta, estrategia) | Preguntas distintas |
|---|---|---|---|
| 615 · 4B   | 141 | 98 / 1.845 (5,3 %) | 34 / 615 |
| 615 · 0.6B | 128 | 97 / 1.845 (5,3 %) | 35 / 615 |
| 110 · 4B   |  28 | 28 / 330 (8,5 %)   | 10 / 110 |
| 110 · 0.6B |  23 | 23 / 330 (7,0 %)   | 10 / 110 |

**Efecto sobre las métricas publicadas (sin corregir → corregido):**

| Config (615 · 4B) | Hit@1 | Hit@10 | MRR@10 |
|---|---|---|---|
| Cross-encoder | 0,717 → **0,759** (+4,2) | 0,896 → **0,951** (+5,5) | 0,787 → **0,834** |
| Híbrida       | 0,582 → **0,610** (+2,8) | 0,889 → **0,941** (+5,2) | 0,694 → **0,731** |
| RRF puro      | 0,589 → **0,620** (+3,1) | 0,863 → **0,915** (+5,2) | 0,690 → **0,730** |

El efecto es homogéneo entre estrategias, por lo que no altera su ordenación relativa.
Este es el argumento que respalda la frase del capítulo.

Reproducir: `scratchpad/overrides_impacto.py` (adjunto en la conversación).

**NOTA sobre `n_overridden` del summary:** vale 320 / 293 / 67 / 56, pero acumula sobre los
TRES umbrales (0,5 / 0,7 / 0,9) y cuenta etiquetas de chunk, no pares. No usar esa cifra.

---

## 5. Sensibilidad al umbral (párrafo restaurado)

Claves `sec@0.5_hit@1`, `sec@0.7_hit@1`, `sec@0.9_hit@1` de `2026-05-19_topk10`,
estrategia `cross_encoder`: **0,828 → 0,759 → 0,698**.

La afirmación válida es sobre la **robustez de la ordenación** entre configuraciones, no
sobre los valores absolutos: a 0,9 hay preguntas que a 0,7 acertaban y pasan a fallar sin
override que las rescate (no fueron revisadas como fallos), de modo que los valores fuera
de 0,7 son una cota inferior.

---

## 6. Métricas de calidad de la respuesta generada (NUEVO)

**Script:** `Scripts/eval_gen_quality.py`. Ejecutar con `tfm/bin/python`.
100 % offline sobre el campo `generated_response` ya almacenado. No reejecuta nada.

**Definiciones exactas:**

| Métrica | Definición | Implementación |
|---|---|---|
| ROUGE-L (recall) | longitud de la subsecuencia común más larga entre cita de referencia y respuesta, dividida por la longitud de la cita | `_rouge_l_recall_fast()` de `eval_compute_metrics.py` (LCS en stdlib, sin dependencia de `rouge_score`) |
| Recall de la cita | \|palabras_contenido(cita) ∩ palabras_contenido(respuesta)\| / \|palabras_contenido(cita)\| | `_tokens()` de `eval_compute_metrics.py` |
| Retención de keywords | fracción de las keywords emitidas por el router (`per_question[].router.keywords`) que reaparecen en la respuesta | subcadena en minúsculas |
| Respuestas citadas | % de respuestas con al menos una marca `[n]` | regex `\[(\d{1,2})\]` |
| Marcas en rango | % de marcas `[n]` con 1 ≤ n ≤ `top_k` del registro | — |

**Limitaciones que hay que declarar en el capítulo:**

1. Ninguna de estas métricas juzga la **corrección clínica**: miden solapamiento léxico con
   el texto de referencia. Una respuesta correcta con vocabulario distinto puntúa bajo.
   Se interpretan en términos **relativos** (comparando configuraciones), no absolutos.
2. "Marcas en rango" verifica que el generador no inventa números de fuente fuera del
   rango entregado. **No** verifica que la fuente citada sustente la afirmación concreta.
   Es una cota superior de la trazabilidad real.
3. **14 de 1.845** preguntas del set de 615 (0,8 %) y **2 de 330** del multi-documento no
   tienen `generated_response` (fallo de generación en la ejecución original) y se excluyen.
   De ahí que n sea 610-612 en vez de 615.
4. `tokens_per_second` trae el centinela `1e6` en 13 registros del set 0.6B multi-documento
   (división por cero cuando Ollama reporta `eval_duration = 0`). Se excluyen y se reporta
   la **mediana**, no la media.

**Resultados (615 preguntas por configuración, generador gemma4:26b):**

| Emb. | Estrategia | n | ROUGE-L | Recall cita | Ret. KW | % citadas | citas/resp | % en rango | t_total (s) |
|---|---|---|---|---|---|---|---|---|---|
| 4B | Cross-encoder | 610 | 0,716 | 0,831 | 0,662 | 97,2 | 7,4 | 100,0 | 29,8 |
| 4B | Híbrida       | 611 | 0,703 | 0,817 | 0,659 | 97,5 | 7,1 | 100,0 | 17,8 |
| 4B | RRF puro      | 610 | 0,687 | 0,802 | 0,671 | 97,4 | 7,1 | 100,0 | 16,8 |
| 0.6B | Cross-encoder | 612 | 0,708 | 0,822 | 0,663 | 96,6 | 7,1 | 100,0 | 29,3 |
| 0.6B | Híbrida       | 609 | 0,674 | 0,788 | 0,653 | 96,1 | 7,0 | 100,0 | 16,8 |
| 0.6B | RRF puro      | 612 | 0,678 | 0,788 | 0,656 | 97,1 | 6,8 | 100,0 | 16,3 |

Multi-documento (110 preguntas):

| Emb. | Estrategia | n | ROUGE-L | Recall cita | Ret. KW | % citadas | t_total (s) |
|---|---|---|---|---|---|---|---|
| 4B | Cross-encoder | 109 | 0,679 | 0,786 | 0,683 | 94,5 | 30,5 |
| 4B | Híbrida       | 110 | 0,631 | 0,741 | 0,697 | 92,7 | 17,6 |
| 4B | RRF puro      | 109 | 0,638 | 0,740 | 0,685 | 94,5 | 16,2 |
| 0.6B | Cross-encoder | 110 | 0,588 | 0,687 | 0,634 | 83,6 | 39,3 |
| 0.6B | Híbrida       | 110 | 0,623 | 0,730 | 0,701 | 92,7 | 21,9 |
| 0.6B | RRF puro      | 110 | 0,631 | 0,737 | 0,688 | 92,7 | 21,1 |

**Lectura:** el orden de las métricas de generación reproduce el de las de recuperación
(cross-encoder > RRF ≈ híbrida; 4B > 0.6B; aislado > multi-documento). Eso es una
**validación cruzada**: mejor recuperación produce respuestas más fieles a la fuente.
La única inversión está en 0.6B multi-documento, donde el cross-encoder cae por debajo
de las otras dos (0,588 vs 0,623/0,631) — merece una frase, no esconderla.

---

## 7. Recuento de palabras

Estimación de la conversación: ~10.500-11.000 palabras sin conclusiones, sobre un
límite de 10.000. **Verificar con `texcount -inc -total main.tex`** antes de decidir recortes.
