# Asistente clínico RAG sobre protocolos hospitalarios

Material de evaluación de un sistema de recuperación aumentada con generación (RAG)
construido sobre los procedimientos y protocolos asistenciales del Hospital Clínico
Universitario Lozano Blesa. Trabajo Fin de Máster, Universidad de Zaragoza.

**[→ Página del proyecto](https://819524.github.io/rag-clinico-tfm/)**

Este repositorio existe para que las afirmaciones de la memoria puedan contrastarse contra
los datos que las sostienen. Está organizado por experimentos, no por la estructura de
carpetas con la que se trabajó: cada carpeta de `experimentos/` es autocontenida y se abre
explicando qué se midió, cómo y con qué resultado.

## Los experimentos

| | Experimento | Qué mide | Resultado de cabecera |
|---|---|---|---|
| 1 | [Recuperación por documento](experimentos/1-recuperacion-por-documento/) | si recupera la sección correcta con la consulta dirigida a un protocolo | 0,932 de acierto entre los 5 primeros |
| 2 | [Recuperación multi-documento](experimentos/2-recuperacion-multidocumento/) | lo mismo, saltando entre protocolos en una conversación | 0,855 en las mismas condiciones |
| 3 | [Cortafuegos](experimentos/3-cortafuegos/) | si reconoce lo que cae fuera de su corpus | 97,4 % de acierto de intención |
| 4 | [Calidad de la respuesta](experimentos/4-calidad-de-respuesta/) | si la respuesta se apoya en el texto recuperado y cita bien | reproducible sin instalar nada |
| 5 | [Carga y motores](experimentos/5-carga-y-motores/) | cuántos usuarios simultáneos aguanta y con qué motor | 3.716 mediciones de concurrencia |
| 6 | [Coste y energía](experimentos/6-costes-y-energia/) | cuánto cuesta sostenerlo en infraestructura propia | 20,28 kWh medidos |

Cada enlace lleva a una carpeta que GitHub abre mostrando su explicación completa. Son las
direcciones que se citan desde la memoria.

## El material

- **615 preguntas de referencia** sobre 22 protocolos, cada una con la
  frase literal del documento que la responde.
- **138 ejecuciones** de evaluación con la traza completa por pregunta.
- **3.716 mediciones** de carga y 68.521
  consultas en el libro mayor de costes.
- **`herramientas/`**: los scripts que producen las métricas publicadas.

## Verificar una cifra sin instalar nada

```bash
git clone https://github.com/819524/rag-clinico-tfm.git
cd rag-clinico-tfm

python3 -c "
import json
m = json.load(open('experimentos/1-recuperacion-por-documento/resultados/embedding-4b/section_metrics_summary.json'))
print({k: v for k, v in m['strategies']['cross_encoder'].items() if k.startswith('sec@0.7')})"
```

El experimento 4 se reproduce entero con `python3 herramientas/eval_gen_quality.py`.

## Qué no está publicado, y qué implica

Los documentos originales del hospital y el índice vectorial que los contiene **no se
redistribuyen**: su titularidad es del centro, no del autor. Tampoco las conversaciones del
piloto ni las credenciales de la infraestructura.

Esto tiene una consecuencia que conviene declarar: **las métricas de recuperación de los
experimentos 1 y 2 no pueden recalcularse desde cero con sólo este repositorio**, porque la
etiqueta de relevancia se decide comparando la cita de referencia contra el texto completo
de la sección, que vive en la base de datos. Lo que sí puede verificarse es que las tablas
de la memoria coinciden con las métricas publicadas, que los documentos recuperados y su
orden son los declarados, y que los conjuntos de preguntas contienen lo que dicen contener.

Los conjuntos de referencia sí incluyen la cita literal del protocolo que responde a cada
pregunta: 615 citas breves con atribución al documento de origen. Sin ellas no
sería comprobable que la respuesta esperada es razonable y no está ajustada a posteriori.

## Licencia

Código bajo Apache-2.0 (`LICENSE`). Resultados, métricas y figuras bajo CC BY 4.0
(`LICENSE-DATOS.md`). Documentos del hospital: no incluidos, todos los derechos de sus
titulares.

## Cómo citar

```
Repositorio de experimentos del TFM «Asistente clínico RAG sobre protocolos
hospitalarios». Universidad de Zaragoza.
https://github.com/819524/rag-clinico-tfm (revisión <hash-del-commit>)
```

Metadatos para gestores bibliográficos en `CITATION.cff`.
