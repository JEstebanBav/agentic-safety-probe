# Semana 1: materiales y estado

Trabajo local realizado el 2 de noviembre de 2026 sobre los 96 pares originales. Las observaciones son de una revision asistida por IA; no sustituyen las dos anotaciones humanas. Versionado en Git y publicado en `main` el 22 de noviembre de 2026, salvo las claves `*_PRIVATE.*`, que siguen solo en local mientras esté pendiente la anotación a ciegas de las 100 respuestas.

La primera ronda humana ya está completa: **κ = 0,180808; 43/96 coincidencias de estrato; 53 desacuerdos**. La meta de 0,6 todavía no se alcanza. Se preservaron las hojas originales en `data/week1/round1/`. Ver el [informe de la primera ronda](round1_agreement.md), la [revisión de sus 20 comentarios](round1_notes_review.md) y el [Excel de consenso separado](../../data/week1/strata_consensus_round1.xlsx). Las 100 respuestas del clasificador siguen pendientes de anotación.

El 3 de noviembre, A declaró que ambos acordaron adoptar los estratos de B: **53 explicit, 27 contextual y 16 framed**. Esa decisión ya está registrada en el consenso, sin cambiar A/B ni su Kappa inicial. Quedan 17 comprobaciones de consistencia con la guía y la claridad del daño por adjudicar; ver [la decisión y su alcance](strata_adjudication.md).

## Orden de uso

Para leer las solicitudes en español, usen los [Excel bilingües A/B y sus instrucciones](annotate_in_spanish.md). Cada persona completa sus 96 filas; el cálculo puede leer los XLSX guardados directamente, sin trasladar decisiones al CSV.

1. Acordar la [guia de anotacion](codebook.md) y anotar independientemente antes de consultar sugerencias de etiquetas en la auditoria. Juan puede usar las hojas A y Helen las B. Cada persona debe completar sus propias 96 solicitudes y 100 respuestas.
2. Usar [strata A](../../data/week1/strata_annotator_A_BLIND.csv), [strata B](../../data/week1/strata_annotator_B_BLIND.csv), [respuestas A](../../data/week1/outputs_annotator_A_BLIND.csv) y [respuestas B](../../data/week1/outputs_annotator_B_BLIND.csv). Los identificadores son opacos y los ordenes son distintos. No consultar archivos PRIVATE ni las etiquetas de la otra persona durante esta fase.
3. Ejecutar el calculo de acuerdo. Discutir los desacuerdos despues de conservar las decisiones iniciales y completar [consenso de estratos](../../data/week1/strata_consensus.csv) y [consenso de respuestas](../../data/week1/outputs_consensus.csv). Con hojas vacias el [informe humano](human_agreement.json) indica pendiente: no hay kappa ni acuerdo validado todavia.
4. Adjudicar los hallazgos del [informe de pares](pair_audit.md) usando [la tabla completa de 96 pares](../../data/week1/pair_review.csv). Registrar `author_decision` y `author_notes`. Revisar las [15 propuestas de texto benigno](../../data/week1/benign_rewrite_proposals.json): son borradores con cambios de contexto declarados, no pares minimos ya validados.
5. Revisar y congelar [grupos tematicos propuestos](../../data/week1/theme_groups_proposed.csv). El [candidato agrupado](../../data/week1/dataset_v2_grouped_candidate.jsonl) solo agrega metadatos: no corrige textos, herramientas ni etiquetas. El loader actual no propaga `theme_group_id`; hace falta integrar el mapa revisado en los folds antes de usarlo en una evaluacion. No basta cambiar de archivo de entrada.

Si cambia el texto danino, las anotaciones actuales no se trasladan automaticamente a la nueva version. Guardar la relacion entre versiones y revisar los casos afectados. Si cambia cualquier texto, sistema o herramienta, generar activaciones nuevas antes de atribuir resultados a esa version corregida.

## Que encontro la revision

No hubo fallos en la estructura: 96 pares, cuatro variantes por par, texto identico de cada brazo entre chat/agente y herramientas/esquemas identicos entre brazos del agente. Esta comprobacion no valida la semantica.

La revision de contenido marco 58 pares con al menos una observacion: 1 contradiccion directa en un benigno, 14 ambiguedades en benignos, 16 ambiguedades en la etiqueta danina, 18 revisiones de estrato, 11 brechas de capacidades, 19 diferencias adicionales entre brazos y 2 revisiones de categoria. Los conjuntos se solapan: no sumar esas cifras como pares distintos. Son candidatos a adjudicacion, no un veredicto de que los 58 sean invalidos.

Se confirmaron 15 pares con herramientas cuyo contexto merece revision. Se proponen 66 grupos de escenario, 28 con mas de un par, para evitar separar variantes cercanas entre entrenamiento y evaluacion. El barrido lexico detecta 25 enlaces con Jaccard >= 0,25; no se presenta como una reproduccion exacta de los 24 cruces del mentor. El [resumen reproducible](audit_summary.json) y las [comprobaciones estructurales](structural_checks.json) registran el hash del corpus.

## Clasificador y resultados

Se corrigieron apostrofos tipograficos en `src/causal_test.py` y se agrego `tool_call_with_refusal` como detalle separado. El detalle mixto conserva `comply` en el recuento binario. Se excluyen los argumentos JSON de la busqueda de rechazo para no confundir el texto de un correo con una negativa del asistente.

La [reclasificacion](classifier_rescore.md) procesa 8.104 registros de ocho archivos conservados, con 27 cambios de etiqueta binaria. En las corridas recientes, Llama agente-danino pasa de 43 a 45 rechazos de 96, y Qwen3-8B chat-benigno de 35 a 36. Mistral no cambia sus recuentos basales. Hay 100 respuestas aleatorias listas para contraste humano; el objetivo 90/100 sigue pendiente.

Los resultados nuevos estan separados en [results/week1_rescore](../../results/week1_rescore/), incluida la lista de [etiquetas cambiadas](../../results/week1_rescore/label_changes.csv) y [diferencias numericas recalculadas](../../results/week1_rescore/metric_changes.csv). Estas incluyen conteos nuevos de casos mixtos y reconstruccion de indicadores de llamadas ausentes en algunas corridas historicas; no todos los renglones corresponden a una cifra publicada. Los resumenes causales describen las cohortes historicas disponibles: si cambia la elegibilidad basal no se reconstruyen intervenciones faltantes.

La [trazabilidad local](results_traceability.md) mapea las 16 tablas a sus fuentes. La extraccion de evidencia de los cuatro modelos coincide con el archivo del articulo y coinciden los 27 hashes de fuentes registrados. Esas fuentes ya estan rastreadas en el Git local, incluida la corrida reciente Qwen3-8B. No se verifico el remoto ni cada celda redondeada del manuscrito de forma independiente.

## Preservacion y reproduccion

La copia de referencia esta en `plan_work/baseline_week1_2026-11-02`, junto al directorio del repositorio. Contiene 49 archivos de datos, codigo y resultados recientes; las cuatro matrices grandes de activaciones siguen en sus rutas originales y tienen hash SHA-256 en el [manifiesto](baseline_manifest.json). Los datos experimentales y resultados originales permanecen iguales a esa referencia; el clasificador si fue actualizado.

Desde la raiz del repositorio, en este equipo:

```powershell
& '..\.venv\Scripts\python.exe' scripts/week1/week1_audit.py
& '..\.venv\Scripts\python.exe' scripts/week1/week1_rescore.py
& '..\.venv\Scripts\python.exe' scripts/week1/week1_agreement.py
& '..\.venv\Scripts\python.exe' scripts/week1/week1_review_annotations.py
& '..\.venv\Scripts\python.exe' scripts/week1/week1_traceability.py
& '..\.venv\Scripts\python.exe' -m unittest discover -s tests -v
```

Los scripts no sobrescriben hojas de anotacion ya creadas. Los informes derivados de auditoria y reclasificacion si se regeneran: conservar un commit/copia de cada ronda. El script de auditoria verifica que el corpus siga siendo el revisado; si cambia, se requiere versionar la revision y sus hojas. La muestra de respuestas tambien se identifica por hashes.

El cálculo de acuerdo prioriza los XLSX A/B si existen y el Excel de consenso de ronda 1 cuando está preparado. Se pueden indicar otros archivos con `--strata-a`, `--strata-b` y `--strata-consensus`. El consenso siempre se mantiene separado y no se utiliza para calcular el Kappa inicial.

## Pendientes de la semana

- Anotacion humana y adjudicacion: kappa >= 0,6, consenso de 96 pares y acuerdo del clasificador >= 90/100.
- Aplicar las correcciones de contenido/herramientas aceptadas, con historial y version nueva; congelar los grupos e integrarlos en el pipeline.
- Revisar Pan, Gupta, TACIT y The Agent Knew y escribir los cuatro parrafos de contribucion. En esta tarea no se han leido ni verificado esos articulos.
- Capturar `tools/list` de 20 servidores MCP reales. Hay un [borrador de analisis de semana 2](week2_analysis_plan.md), con decisiones pendientes explicitadas; todavia no hay capturas ni protocolo congelado.
- Renderizar los prompts con los tokenizadores, adjudicar cambios en tablas conductuales y preparar la publicacion en GitHub. El mapa local de fuentes ya esta comprobado; falta la comprobacion independiente de celdas redondeadas y configuracion de arquitectura. No se ha comprobado ni actualizado el remoto.
- Con GPU: nueva corrida sobre el corpus corregido, control Llama sin instruccion inyectada y comprobacion de rechazo benigno. La reevaluacion por grupos de ejemplos sin cambios podria aprovechar activaciones conservadas despues de verificar la alineacion de IDs.
