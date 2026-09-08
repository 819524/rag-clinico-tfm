# Asistente clínico RAG sobre protocolos hospitalarios

Material de consulta del Trabajo Fin de Máster del mismo nombre, Universidad de Zaragoza.
El sistema responde preguntas de profesionales sanitarios citando los procedimientos
asistenciales del Hospital Clínico Universitario Lozano Blesa.

**[→ Página del proyecto](https://819524.github.io/rag-clinico-tfm/)**

Aquí están **las preguntas empleadas para evaluar el sistema y los resultados obtenidos**.
Cada carpeta de `experimentos/` corresponde a una tabla de la memoria y contiene los
ficheros de los que salen sus cifras.

## Los experimentos

| | Experimento | Tabla | Contenido |
|---|---|---|---|
| 1 | [Recuperación](experimentos/1-recuperacion/) | 5.1 | 615 preguntas sobre 22 protocolos y sus resultados |
| 2 | [Contexto cruzado](experimentos/2-contexto-cruzado/) | 5.2 | 110 preguntas en dos disposiciones, agrupadas y entrelazadas |
| 3 | [Despliegue](experimentos/3-despliegue/) | 4.1 y 4.2 | 58.152 peticiones medidas bajo carga concurrente |
| 4 | [Coste y energía](experimentos/4-costes/) | 6.1 | 68.521 consultas registradas, 37,4 kWh |

Cada carpeta se abre explicando qué se midió, la tabla tal como aparece en la memoria y qué
fichero contiene sus cifras.

## El material

- **615 preguntas de referencia** sobre 22 protocolos, cada una con la
  sección esperada y la frase literal del documento que la responde.
- **147 ficheros de resultados**, con la traza completa por pregunta: consulta,
  fragmentos recuperados en orden, etiquetas de relevancia, tiempos y respuesta generada.
- **58.152 peticiones** de carga con su TTFT y tiempos por fase, y la telemetría de GPU.
- Las **figuras** que aparecen en la memoria.

## Qué no está publicado

Los documentos originales del hospital y el índice vectorial que los contiene no se
redistribuyen: su titularidad es del centro. Tampoco las conversaciones del piloto ni las
credenciales de la infraestructura.

Los conjuntos de referencia sí incluyen la cita literal del protocolo que responde a cada
pregunta —615 citas breves con atribución al documento de origen—, porque sin ellas
no podría valorarse si la respuesta esperada es razonable.

## Licencia

Preguntas, resultados, métricas y figuras bajo CC BY 4.0 (`LICENSE-DATOS.md`). Documentos del hospital:
no incluidos, todos los derechos de sus titulares.

## Cómo citar

```
Repositorio de experimentos del TFM «Asistente clínico RAG sobre protocolos
hospitalarios». Universidad de Zaragoza.
https://github.com/819524/rag-clinico-tfm (revisión <hash-del-commit>)
```
