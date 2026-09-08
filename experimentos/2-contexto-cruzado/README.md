<!-- Generado por _build/generar_textos.py — no editar a mano -->

# Experimento 2 · Contexto cruzado

**Respalda la Tabla 5.2 de la memoria** («Experimento de contexto cruzado. CONTROL =
historial del mismo documento; TEST = historial de otro documento, K = 10»).

> Las cifras de esta página son las **impresas en la memoria**. Para comprobar
> que salen de los datos de este repositorio:
>
> ```bash
> python3 herramientas/verificar_memoria.py
> ```

## Qué se midió

El experimento 1 es benévolo: todas las preguntas de una sesión se refieren al mismo
protocolo y la memoria conversacional refuerza esa coherencia. En uso real un profesional
cambia de tema. Este experimento aísla ese efecto comparando dos brazos con las **mismas
110 preguntas**:

- **control** — las preguntas llegan agrupadas de cinco en cinco por protocolo, de modo que
  el historial ayuda.
- **tests** — las mismas preguntas entrelazadas, de modo que cada consulta llega con un
  historial referido a un documento distinto y el sistema debe discriminar entre los 22
  protocolos con el contexto contaminado.

La diferencia entre ambos brazos es la penalización atribuible al contexto, no al azar de
qué preguntas tocaron: son idénticas y en ambos brazos.

## Tabla 5.2

| Embedding | Estrategia | ctrl Hit@1 | ctrl Hit@10 | ctrl MRR | ctrl NDCG | test Hit@1 | test Hit@10 | test MRR | test NDCG | ΔHit@1 |
|---|---|---|---|---|---|---|---|---|---|---|
| 4B | Cross-encoder | 0,691 | 0,918 | 0,776 | 0,792 | 0,591 | 0,818 | 0,676 | 0,700 | -0,100 |
| 4B | Híbrida | 0,673 | 0,936 | 0,760 | 0,780 | 0,527 | 0,809 | 0,628 | 0,655 | -0,145 |
| 4B | RRF puro | 0,600 | 0,891 | 0,703 | 0,733 | 0,509 | 0,791 | 0,618 | 0,648 | -0,091 |
| 0.6B | Cross-encoder | 0,718 | 0,946 | 0,810 | 0,821 | 0,600 | 0,827 | 0,691 | 0,715 | -0,118 |
| 0.6B | Híbrida | 0,536 | 0,927 | 0,690 | 0,736 | 0,455 | 0,809 | 0,585 | 0,630 | -0,082 |
| 0.6B | RRF puro | 0,600 | 0,891 | 0,710 | 0,736 | 0,473 | 0,754 | 0,578 | 0,612 | -0,127 |

Con *cross-encoder*, el modelo de 0.6B iguala e incluso supera al de 4B en primera
posición, lo que refuerza la conclusión del experimento 1: un reordenador potente compensa
la menor capacidad del recuperador base, también bajo contexto contaminado.

## La comparación del texto

El capítulo compara además el escenario por documento con la ejecución multi-documento
completa. Esos números salen de `resultados-global/`:

| Métrica (4B, cross-encoder) | 615 preguntas | multi-documento | caída |
|---|---|---|---|
| Hit@1 | 0,759 | 0,636 | 12,3 puntos |
| Hit@10 | 0,951 | 0,891 | 6,0 puntos |
| MRR@10 | 0,834 | 0,734 | 10,1 puntos |

> **Aviso sobre el texto de la memoria.** El párrafo que sigue a la Tabla 5.2 dice que «el
> acierto en primera posición cae de 0,836 a 0,700». Esos dos valores no corresponden al
> umbral 0,7 que usa el resto del capítulo: 0,836 procede de la ejecución con K = 5 y umbral
> 0,5, y 0,700 de la multi-documento con umbral 0,5. Con el criterio del capítulo los
> valores son **0,759 → 0,636**. La magnitud que
> afirma el texto («doce puntos») es correcta —la caída real es de
> 12,3 puntos— y las otras dos cifras del
> mismo párrafo (MRR@10 de 0,834 a 0,734, y Hit@10
> cediendo seis puntos) son exactas.

## Contenido

| Ruta | Qué es |
|---|---|
| `preguntas/control.json` · `tests.json` | los dos brazos, 110 preguntas cada uno |
| `preguntas/global.json` | el conjunto multi-documento de la comparación del texto |
| `metricas/` | **de aquí sale la Tabla 5.2** |
| `metricas/COMO_SE_CALCULARON.md` | procedimiento exacto de recálculo |
| `resultados/` | 12 ejecuciones, por modelo de embedding, estrategia y brazo |
| `resultados-global/` | ejecución multi-documento de la comparación |

## Reproducir

```bash
python3 -c "
import json
c = json.load(open('experimentos/2-contexto-cruzado/metricas/4b_CONTROL.json'))
t = json.load(open('experimentos/2-contexto-cruzado/metricas/4b_PAREADO.json'))
for e in ('cross_encoder','embed','rrf_only'):
    print(e, round(c['strategies'][e]['sec@0.7_hit@1'],3),
             round(t['strategies'][e]['sec@0.7_hit@1'],3))"
```
