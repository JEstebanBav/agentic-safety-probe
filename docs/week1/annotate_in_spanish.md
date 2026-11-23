# Anotar las 96 solicitudes con ayuda de la traducción

La primera ronda ya fue recibida y calculada. Para la discusión posterior, usar el [informe de acuerdo](round1_agreement.md) y el [Excel de consenso separado](../../data/week1/strata_consensus_round1.xlsx); conservar las decisiones iniciales en A y B.

Se prepararon dos archivos de Excel independientes:

- [Archivo A](../../data/week1/strata_annotator_A_BLIND_ES.xlsx), que puede completar Juan.
- [Archivo B](../../data/week1/strata_annotator_B_BLIND_ES.xlsx), que puede completar Helen.

Cada archivo contiene **las mismas 96 solicitudes**, en el orden distinto que ya tenía cada anotador, con el original en inglés y la traducción al español al lado. Cada persona completa las 96; no se dividen las filas entre personas. La traducción es una ayuda de lectura asistida por IA y no incluye etiquetas previas, sugerencias de clasificación, categorías ni identificadores de origen.

## Antes de empezar

Acuerden la [guía de anotación](codebook.md), que sigue siendo un borrador hasta que ustedes acuerden sus reglas. Después trabajen de forma independiente, sin consultar las decisiones de la otra persona ni el informe que propone revisiones de etiquetas.

Los dos archivos usan exactamente la misma traducción, versión `es-v1`. Se conservaron las ambigüedades del original: no se añadieron permisos, falta de autorización, falsedad ni intención maliciosa donde el texto no los establece. Los nombres, referencias técnicas, enlaces, direcciones, cantidades y fechas se mantienen. Los títulos de obras y de archivos pueden seguir en inglés. El inglés es la referencia cuando una expresión traducida genere una duda.

## Qué completar

En la pestaña `Anotacion`, llena las cuatro columnas amarillas:

| Columna | Qué registrar |
|---|---|
| Estrato | Elige en la lista `explicit`, `contextual` o `framed`, aplicando la guía acordada. |
| ¿El daño está claro? | `yes` = sí; `no` = no; `uncertain` = falta contexto. Es una decisión distinta del estrato. |
| Fragmento de evidencia | Copia una frase corta que respalde tu decisión, en inglés o español. |
| Notas | Explica dudas, información ausente o una posible discrepancia de traducción. |

Las columnas de ID y textos están protegidas sin contraseña para evitar cambios accidentales. Si identifican una discrepancia de traducción, registren la duda y acuerden una corrección compartida antes de cerrar la ronda; no resuelvan una misma traducción de forma distinta en cada archivo.

Guarda tu archivo con el mismo nombre. Los XLSX serán la fuente de las decisiones en esta ronda. Los CSV también tienen la columna `prompt_es`, pero editar el Excel no actualiza automáticamente el CSV: para calcular el acuerdo se leen directamente los dos XLSX, sin copiar etiquetas a mano.

## Cómo se calcula después

Desde la raíz del repositorio, una vez guardados ambos archivos:

```powershell
& '..\.venv\Scripts\python.exe' scripts/week1/week1_agreement.py --strata-a data/week1/strata_annotator_A_BLIND_ES.xlsx --strata-b data/week1/strata_annotator_B_BLIND_ES.xlsx
```

El script calcula Kappa sobre las decisiones iniciales y registra los desacuerdos. Si faltan etiquetas, muestra `pending` y no calcula Kappa sobre un subconjunto. La validación del clasificador sobre 100 respuestas es otra tarea: que esa parte permanezca pendiente no impide calcular Kappa cuando las 96 etiquetas de ambos estén completas.

Después de conservar las decisiones iniciales y su Kappa, discuten los desacuerdos y registran la etiqueta acordada en `strata_consensus.csv`. El consenso no reemplaza las decisiones usadas para calcular el acuerdo inicial.

También pueden completar directamente el archivo `strata_consensus_round1.xlsx` preparado para la revisión conjunta. `week1_agreement.py` lo detecta como fuente de consenso si existe; `--strata-consensus` permite elegir explícitamente un CSV o XLSX alternativo. La medida inicial de acuerdo sigue usando A y B.

## Reproducción y alcance

`scripts/week1/week1_translate_sheets.py` enriquece los CSV preservando anotaciones y orden y crea los XLSX solo si no existen. Requiere `xlsxwriter` para crearlos; la lectura de los XLSX por el script de acuerdo usa la biblioteca estándar. Los archivos ya existentes no se sobrescriben para proteger el trabajo humano.

`strata_translations_es.json` contiene las 96 traducciones y `translation_manifest.json` registra los hashes de los textos en ambos idiomas y de la versión de traducción. Esta ayuda de lectura no modifica los prompts del corpus experimental ni exige repetir las corridas existentes.
