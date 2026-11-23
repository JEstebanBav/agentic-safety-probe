# Consenso declarado sobre los estratos — 3 de noviembre de 2026

A (Juan) informó que había interpretado de forma confusa las categorías y que, después de revisarlo con B, ambos acordaron adoptar las clasificaciones de B. Se registró esa declaración como **consenso humano posterior por adopción de las 96 etiquetas de B**. No se atribuye a la IA una adjudicación ni se inventan justificaciones caso por caso.

| Estrato adoptado | Pares |
|---|---:|
| `explicit` | 53 |
| `contextual` | 27 |
| `framed` | 16 |
| Total | 96 |

Se completó la columna Estrato del [Excel de consenso](../../data/week1/strata_consensus_round1.xlsx) y se actualizó [strata_consensus.csv](../../data/week1/strata_consensus.csv) con la misma decisión. La procedencia se documenta en las notas de cada fila y en [strata_adjudication.json](../../data/week1/strata_adjudication.json). Se conservaron las hojas A/B y sus snapshots originales sin modificar. La copia del consenso vacío anterior también se conserva en `data/week1/round1/`.

Las etiquetas adoptadas cambian 47 estratos respecto al dataset experimental original. Los detalles se calculan en `data/week1/stratum_changes.csv`. El dataset original y los resultados publicados se mantienen como evidencia de las corridas realizadas; esta decisión no aplica automáticamente modificaciones al corpus ni a las tablas.

## Qué significa para Kappa

El resultado inicial sigue siendo **κ=0,180808, con 43/96 coincidencias y 53 desacuerdos**. La explicación de A es relevante para interpretar la ronda y debe registrarse junto al resultado. Adoptar las etiquetas de B resuelve qué etiquetas quieren usar como consenso, pero no convierte la ronda inicial en una validación con κ≥0,6.

No se reemplazan las etiquetas iniciales de A ni se calcula un supuesto Kappa nuevo entre B y una copia de B. Si se quiere evaluar acuerdo independiente con reglas aclaradas, debe documentarse otra evaluación y conservar la primera, indicando la exposición previa a las etiquetas y a la discusión.

## Revisión de consistencia que sigue pendiente

La declaración de consenso establece la decisión de los autores; no acredita por sí sola que todas las etiquetas apliquen la definición actual de la guía. Se conservó la decisión sin corregirla automáticamente y se preparó [consensus_codebook_checks.csv](../../data/week1/consensus_codebook_checks.csv) con 17 comprobaciones concretas para revisar.

Ejemplos prioritarios:

- **H009/H091:** B marcó el primer caso como `explicit` y el segundo como `framed`, aunque reformulan el mismo escenario sobre el artículo de atención y no se identifica una justificación legitimadora en ninguno. Compartir escenario no obliga siempre al mismo estrato, pero aquí falta explicar qué cambio de encuadre justificaría la diferencia.
- **H034, H023 y H085:** figuran como `framed` sin un encuadre legitimador identificable según la guía actual.
- **H089:** pedir un FAQ es una tarea auxiliar; no constituye por sí solo un encuadre que legitime la acción cuestionada.
- **H079:** menciona capacitación de seguridad empresarial, pero B lo marcó `contextual`. La guía propone priorizar `framed` cuando existe ese encuadre, separando esta dimensión de si hay pruebas de abuso o una simulación autorizada.

Las comprobaciones no son etiquetas nuevas ni un veredicto sobre quién tiene razón. Requieren una decisión explícita sobre el criterio o sobre cada etiqueta antes de congelar la versión final. Si se cambia la definición de framed, registrar esa versión de la guía y aplicarla consistentemente.

## Dimensiones que no se adjudicaron por inferencia

La decisión recibida se interpretó como adopción de **estratos**. La columna auxiliar de claridad del daño (`intent_clear`) sigue pendiente de consenso: no se copiaron automáticamente los 31 no, 57 yes y 8 uncertain de B. No se cambian etiquetas de daño/benignidad ni se excluyen ejemplos a partir de esta declaración.

Las citas de evidencia siguen pendientes. Las notas del consenso documentan la adopción global, no una revisión individual que no se haya registrado. Los comentarios originales de B siguen disponibles en Comparacion y Comentarios y en el informe de notas.

Las 100 respuestas para validar el clasificador son otro entregable y siguen pendientes de anotación humana.
