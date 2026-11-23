# Resultado de la primera ronda de anotación

Se conservaron copias y hashes de las hojas entregadas por A y B antes de la adjudicación. Cada una tiene 96 etiquetas de estrato y 96 respuestas de claridad del daño. Los textos ingleses y sus traducciones coinciden con el manifiesto de traducción.

**Kappa de estrato: 0.180808. Acuerdo observado: 43/96 (44.79%). Hay 53 desacuerdos.** La meta del mentor, κ ≥ 0,6, no se alcanza en esta ronda.

Este resultado mide las decisiones iniciales tal como se entregaron. No se corrigió ninguna etiqueta para aumentar el acuerdo. El Kappa no es el porcentaje simple de coincidencias ni una medida de exactitud frente a una etiqueta verdadera.

## Matriz de estratos

Filas: A; columnas: B.

| A / B | explicit | contextual | framed | Total A |
|---|---:|---:|---:|---:|
| explicit | 23 | 3 | 3 | 29 |
| contextual | 23 | 10 | 3 | 36 |
| framed | 7 | 14 | 10 | 31 |
| Total B | 53 | 27 | 16 | 96 |

Los dos cruces más frecuentes son A=contextual/B=explicit (23 casos) y A=framed/B=contextual (14 casos). Esto señala diferencias de aplicación de las definiciones; no establece que un anotador sea la referencia correcta.

## Claridad del daño y evidencia

La columna auxiliar intent_clear coincide en 56/96 (58.33%), con κ=0.083095 y 40 desacuerdos. Es una dimensión distinta de los estratos y no tiene aquí una meta del mentor fijada por separado.

A marcó yes=86, no=2, uncertain=8; B marcó yes=57, no=31, uncertain=8. Algunas notas mezclan claridad del daño con realismo, repetición de temas o viabilidad de herramientas. Esas observaciones son útiles, pero conviene registrarlas por separado antes de adjudicar el ground truth.

Las celdas de evidencia están vacías en A; en B solo hay una no vacía, con el texto 'c'. Esto no impide calcular Kappa, pero no documenta el razonamiento. Durante el consenso, citar el fragmento que respalda cada decisión, especialmente en los desacuerdos.

## Qué hacer a continuación

1. Revisar la regla de framed: exige un encuadre que legitime la acción cuestionada. Un tema profesional, un archivo académico o un FAQ por sí solos no lo establecen. La presencia del encuadre tampoco demuestra que la actividad sea dañina.
2. Separar explicitud de probabilidad de rechazo, naturalidad y ejecutabilidad. Clasificar la versión escrita; cualquier reformulación se registra como otra versión.
3. Discutir los 53 desacuerdos de estrato y las dudas de daño; registrar decisiones y evidencia en el Excel de consenso separado. Los 96 casos deben tener una etiqueta final acordada, incluidos los que ya coincidían.
4. Conservar esta primera ronda. El consenso no eleva retrospectivamente el Kappa independiente. Si se necesita una segunda ronda para evaluar un codebook revisado, documentarla como tal y conservar el resultado inicial; no llamar ciega a una revisión informada por la discusión previa.
5. Adoptar cambios del dataset solo después de adjudicar: las etiquetas y activaciones de textos originales no se transfieren automáticamente a textos cambiados.

El Excel [strata_consensus_round1.xlsx](../../data/week1/strata_consensus_round1.xlsx) contiene las cuatro hojas Resumen, Anotacion, Comparacion y Comentarios. Comparacion muestra inicialmente 76 casos con diferencias, daño no confirmado por ambos o notas; se puede quitar el filtro para ver los 96. El archivo se creó con etiquetas de consenso vacías; las decisiones posteriores se registran por separado.

El detalle de cada comentario está en [round1_notes_review.md](round1_notes_review.md). Las observaciones de IA son apoyo para la discusión, no decisiones humanas ni etiquetas de referencia.

## Reproducibilidad

Datos congelados: data/week1/round1/. Cálculo sin ponderación de categorías, sobre los mismos 96 IDs, sin descartar filas. El cálculo se contrastó con sklearn.metrics.cohen_kappa_score y coincide (diferencias inferiores a 1e-12). Scripts: week1_agreement.py y week1_review_annotations.py.

Las 100 respuestas generadas del clasificador siguen sin etiquetas humanas; esta ronda no mide el acuerdo del clasificador.

## Decisión posterior de los autores

El 3 de noviembre de 2026, A declaró que había interpretado de forma confusa las categorías y que ambos acordaron adoptar los estratos de B. Se registró esa adopción como consenso posterior, conservando las decisiones iniciales y κ=0,180808. No es una nueva medida de acuerdo independiente. Ver [la adjudicación y sus pendientes de consistencia](strata_adjudication.md).
