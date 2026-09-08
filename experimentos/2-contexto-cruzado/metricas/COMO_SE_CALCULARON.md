# Métricas a nivel de sección — experimento CONTROL/PAREADO

Recalculadas **offline** desde los `results_*.json` de `../{4b,06b}/*/rep_01/*/`,
sin reejecutar el pipeline:

    cd Scripts
    ../tfm/bin/python eval_section_metrics.py \
      --input-dir <carpeta con los 3 results_*.json de un (embedding, brazo)> \
      --threshold 0.7 --crate http://localhost:4201

`section_overrides_usados.json` reproduce los 10 overrides manuales de
`eval/global_topk10/section_overrides.json` (clave `GLOBAL`) bajo las claves
`CONTROL` y `PAREADO`. Es legítimo: las 110 preguntas son subconjunto de las 615 y
conservan sus ids `gNNN`; los 10 overrides caen dentro del subconjunto.

Con esto la tabla del Cap. 5 es **directamente comparable** con `tab:generador`
(mismo nivel de relevancia, mismo umbral 0,7, mismos overrides).

`rep_rep_02_*` y `rep_rep_03_*` son las repeticiones 2 y 3 de 4B cross-encoder.
