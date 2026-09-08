# Asistente clínico RAG sobre protocolos hospitalarios

Material de evaluación del Trabajo Fin de Máster del mismo nombre, Universidad de Zaragoza.
El sistema responde preguntas de profesionales sanitarios citando los procedimientos
asistenciales del Hospital Clínico Universitario Lozano Blesa.

**[→ Página del proyecto](https://819524.github.io/rag-clinico-tfm/)**

Este repositorio contiene **lo que respalda las cifras publicadas en la memoria, y nada
más**. Cada carpeta de `experimentos/` corresponde a una tabla del documento.

## Comprobar que las cifras concuerdan

```bash
git clone https://github.com/819524/rag-clinico-tfm.git
cd rag-clinico-tfm
python3 herramientas/verificar_memoria.py
```

El script lleva transcritas las tablas **tal como están impresas en la memoria**, recalcula
cada celda desde los ficheros de este repositorio y señala cualquier diferencia. Son
**193 comprobaciones** y no necesita base de datos, modelos ni red.

## Los experimentos

| | Experimento | Respalda | Cifra de cabecera |
|---|---|---|---|
| 1 | [Recuperación](experimentos/1-recuperacion/) | Tabla 5.1 | Hit@5 = 0,932 con *cross-encoder* y embedding 4B |
| 2 | [Contexto cruzado](experimentos/2-contexto-cruzado/) | Tabla 5.2 | la penalización por cambiar de tema en la conversación |
| 3 | [Despliegue](experimentos/3-despliegue/) | Tablas 4.1 y 4.2 | 58.152 peticiones medidas bajo carga |
| 4 | [Coste y energía](experimentos/4-costes/) | Tabla 6.1 | 68.521 consultas, 37,4 kWh |

Cada enlace abre una carpeta que GitHub muestra con su explicación completa: qué se midió,
cómo, la tabla reproducida y qué fichero la sostiene. Son las direcciones citables desde la
memoria.

## El material

- **615 preguntas de referencia** sobre 22 protocolos, cada una con la
  frase literal del documento que la responde.
- **Resultados crudos** de cada ejecución, con la traza completa por pregunta.
- **58.152 peticiones** de carga con su TTFT y tiempos por fase, y la telemetría de GPU.
- **`herramientas/`**: los scripts que producen las cifras publicadas, y el verificador.

## Qué no está publicado, y qué implica

Los documentos originales del hospital y el índice vectorial que los contiene **no se
redistribuyen**: su titularidad es del centro. Tampoco las conversaciones del piloto ni las
credenciales de la infraestructura.

Esto tiene una consecuencia que conviene declarar: **las métricas de recuperación de las
Tablas 5.1 y 5.2 no pueden recalcularse desde cero** con sólo este repositorio, porque la
etiqueta de relevancia se decide comparando la cita de referencia contra el texto completo
de la sección, que vive en la base de datos. Lo que sí se verifica —y es lo que hace el
script— es que las tablas impresas coinciden con las métricas publicadas, que los
documentos recuperados y su orden son los declarados, y que los conjuntos de preguntas
contienen lo que dicen contener.

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
