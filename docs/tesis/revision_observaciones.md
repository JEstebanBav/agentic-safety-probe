# Validación de las observaciones de la tesis

Fecha de revisión: 15 de octubre de 2026.

La respuesta al tutor no refleja por completo el estado de los archivos actuales. Resumen y abstract ya están redactados; PCA ya tiene resultados locales; la selección de capa ya dispone de una evaluación separada. Sin embargo, varias tablas de la tesis corresponden a corridas anteriores y no hay resultados locales de las corridas de 96 pares que el documento afirma reportar. Por eso no conviene dar las veinte observaciones por cerradas a partir de las marcas de la respuesta.

## Alcance y fuentes

Se revisaron la estructura del proyecto, el pipeline `run_experiment.py`, sus módulos de extracción, direcciones, sondas, controles y validación, los constructores y archivos de datos, la documentación metodológica y los resultados guardados. La aplicación Gradio y varias notas describen etapas anteriores del proyecto; no deben tomarse como fuente de las cifras actuales.

Fuentes principales:

- `docs/tesis/respuesta_observaciones.pdf`: tres páginas, veinte observaciones y su estado declarado.
- `docs/tesis/Tesis_corta_v2.pdf`: versión revisada, 61 páginas físicas.
- `docs/tesis/Tesis_corta_v2_overleaf.zip`: fuentes LaTeX, leídas desde una extracción temporal.
- `docs/Tesis v2.pdf`: versión anterior, 80 páginas físicas.
- `results/corridas/llama33_70b/`: seis corridas completas de análisis sobre 42 pares, además de un ensayo de cuatro pares.
- `results/corridas/mistral7b/v3/run.log`: único archivo encontrado en esa carpeta. **No hay JSON de Mistral V3 allí**.
- Resto de `results/`, `data/dataset_v2.jsonl` y `../docs/tesis/seccion_metodologia_integrada.tex`.

Las anotaciones de ambos PDF de tesis no contienen comentarios de texto en el campo `/Contents`. El listado de observaciones se reconstruye desde la respuesta al tutor; no se puede cotejar literalmente cada comentario original con una nota incrustada. Tampoco se verificó bibliografía externa ni se ejecutaron modelos en GPU. Las páginas citadas abajo son físicas del PDF de 61 páginas.

## Contexto del proyecto

La investigación compara representaciones internas de instrucciones idénticas bajo formato conversacional y formato con herramientas. Cada par dañino–benigno tiene cuatro variantes. Se estima una dirección conversacional por diferencia de medias, se miden proyecciones y su interacción con el brazo benigno, se controla la escala y la circularidad, y se evalúan direcciones y monitores agénticos. El repositorio también contiene pruebas factoriales y conductuales e intervenciones, aunque la tesis revisada mantiene un alcance principalmente diagnóstico.

El corpus actual **sí existe y contiene 96 pares/384 entradas**: 42 explícitos, 24 contextuales y 30 de apariencia legítima, repartidos en siete categorías. Se comprobaron los 96 pares: no hubo diferencias de texto entre formatos dentro de cada brazo, ni diferencias de mensaje de sistema, herramientas o esquemas entre dañino y benigno dentro de cada formato. Hay 64 herramientas distintas utilizadas. La tabla cruzada de categorías y explicitud del capítulo 3 coincide con el archivo.

Esto valida la construcción del corpus; **no demuestra que los resultados reportados se hayan obtenido ejecutándolo**.

## Las veinte observaciones

“Atendida en texto” significa que el cambio solicitado está presente; no certifica por sí solo todas las cifras que lo acompañan.

| N.º | Observación recogida en la respuesta | Estado declarado | Validación actual |
|---|---|---|---|
| 1 | Avanzar y preparar presentación con retroalimentación externa. | No requiere cambio | Es una recomendación de proceso. Los archivos no permiten confirmar que se haya realizado la retroalimentación. |
| 2 | Resumen por redactar. | Por redactar | **Ya redactado** en PDF, p. 2, y `1.3_abstract.tex`. Hay que actualizar el estado y aclarar que Llama usa 42 pares; después sincronizar las cifras. |
| 3 | Abstract por redactar. | Por redactar | **Ya redactado** en PDF, p. 3, y `1.3_abstract_en.tex`. Requiere la misma precisión y sincronización que el resumen. |
| 4 | Acentuación de “estas/éstas”. | Atendido | **No verificable de forma puntual** sin el fragmento original señalado. La revisión aún contiene “éstas” en la pregunta central; no basta para identificar si se corrigió el pasaje observado. |
| 5 | Frase confusa sobre las dos dimensiones. | Atendido | **Atendida en texto**: la pregunta central identifica cambio de formato y grado de explicitud, PDF p. 13, capítulo 1. |
| 6 | Definir desde el inicio un concepto central. | Atendido por rediseño | **Parcialmente verificable**: hay formalización y glosario, pero este aparece al final, p. 57. Sin el comentario literal no se puede asegurar cuál era el concepto. Si se pedía definirlo desde su primera aparición, añadir un glosario final no basta por sí solo. |
| 7 | Remisión a otra observación. | Atendido | Se atiende mediante las aclaraciones de diseño de los puntos 9 y 10; hereda las precisiones pendientes del punto 9. |
| 8 | Contradicción con el antecedente de Pinto. | Atendido | **Atendida en texto**: el capítulo 2 distingue transferencia para steering de diagnóstico de detección, pp. 22–23. La introducción formula la brecha como capacidad de detección. No se auditó externamente la interpretación del artículo. |
| 9 | Si solo varían herramientas, por qué también varían instrucciones. | Atendido | **Atendida la distinción dentro/entre pares**, pero queda una imprecisión: el objetivo dice que solo cambia la disponibilidad de herramientas, mientras métodos y código también cambian el mensaje de sistema. Describir la manipulación como formato compuesto, salvo que se use el análisis factorial para separar componentes. |
| 10 | Mismo catálogo y sistema entre ramas dañina y benigna. | Atendido | **Atendida y verificada** sobre los 96 pares actuales, para herramientas, esquemas y sistema dentro de cada formato. |
| 11 | Autoría de prompts, criterios y revisión asistida. | Atendido | **Atendida documentalmente**: capítulo 3, p. 27, identifica al autor, los criterios y a Claude como apoyo. El archivo no permite acreditar de forma independiente la historia de esa revisión. |
| 12 | La proyección alta no demuestra detección de peligro. | Atendido | **Atendida en texto** en desplazamiento y glosario: diferencia geometría, discriminación dañino–benigno y rechazo efectivo. Mantener esa distinción en resumen y discusión. |
| 13 | Diseño pareado y supuesto de independencia. | Atendido por rediseño | **Parcial**: métodos ya usa cambios de signo pareados y el código actual contempla la interacción intra-par. Pero las cifras antiguas de interacción no acreditan ese procedimiento; deben regenerarse o reemplazarse con resultados pareados identificados. Sigue siendo necesario justificar independencia/intercambiabilidad entre pares; el emparejamiento solo resuelve la dependencia dentro del par. |
| 14 | Falta procedimiento y resultados de PCA. | En curso | **Resultados disponibles, integración pendiente**. Llama V3 tiene PCA en 20 capas y Mistral V3 lo imprime en el log. La tesis solo anuncia PCA y no presenta sus resultados ni detalla suficientemente su construcción. Ya no depende únicamente de esperar una ejecución. |
| 15 | Fortalecer descripción de extracción del flujo residual. | Atendido por rediseño | **Atendida la ampliación matemática**. Falta afinar que se lee el último token del estímulo renderizado antes de generar, que puede ser un delimitador de plantilla; no necesariamente el último token léxico de la instrucción. |
| 16 | La prueba de signos sí depende de la dirección. | Atendido | **Atendida en texto** en métodos y estabilidad: se reconoce la dependencia y se distingue de la invariancia a escala. Expresar esta última para reescalados positivos; invertir el eje cambia los signos. |
| 17 | Cercanía entre sonda lineal y MLP no demuestra equivalencia; curva no monótona. | Atendido por rediseño | **Atendida en la redacción** al retirar equivalencia y describir ascenso/descenso. El código también compara diferencias sobre folds comunes. Al actualizar corridas hay que revisar de nuevo la curva: la figura citada mide la dirección, mientras el párrafo discute la sonda. |
| 18 | Los 39/42 incluyen los 21 usados para seleccionar capa. | En curso | **Corrección computacional disponible; explicación en tesis pendiente**. V3 separa 21 pares de selección y 21 de inferencia, reportando 21/21 decrecientes en inferencia. No hace falta esperar 96 pares para explicar el error original. La réplica de 96 pares sigue sin estar disponible localmente. |
| 19 | Conclusiones y recomendaciones por construir. | Por redactar | **Pendiente**: capítulo 6 solo tiene títulos y está comentado en `main.tex`. Además, el capítulo 5 de discusión está incluido pero también contiene únicamente títulos, PDF p. 56. |
| 20 | Material complementario por construir. | Por redactar | **Pendiente**: hay glosario, pero no anexos que documenten corpus, anotaciones, configuración y resultados. El ZIP no incorpora ese material experimental. |

## Inconsistencias de evidencia que conviene resolver primero

### 1. No están las corridas de 96 pares reportadas

Se inspeccionaron los 17 `results.json` bajo `results/`: catorce corresponden a 42 pares y tres a ensayos de cuatro pares. No hay ninguno de 96. En cambio, la tabla de corridas del capítulo 4, p. 45, reporta Mistral con 96 pares/capa 18, Qwen3-8B con 96/capa 22 y Qwen3-32B con 96/capa 46.

Los archivos locales `results/corridas/qwen3_8b/v1/results.json` y `results/corridas/qwen3_32b/v1/results.json` contienen 42 pares, con capas principales 32 y 46 respectivamente. El primero, además, tiene AUROC held-out conversacional/agéntico de aproximadamente 0,518/0,511 en su capa principal, que no corresponde a la corrida descrita en la tesis. Esto no refuta una ejecución externa de 96 pares; significa que **esa ejecución no es auditable con los archivos entregados aquí**. Las figuras incluidas en Overleaf no sustituyen sus datos fuente.

### 2. Las tablas de Llama describen una corrida anterior

Las cifras de la tesis, capa 56 y Δp ≈ 5,74, coinciden con `results/corridas/llama33_70b/v1/`, no con `results/corridas/llama33_70b/v3/`.

| Medida | Tesis / corrida anterior | Llama V3 guardado |
|---|---:|---:|
| Pares | 42 | 42 |
| Capa principal | 56 | 76 |
| Δp dañino, estimación completa | 5,7394 | 16,2776 |
| Δp benigno | −3,2337 | −2,1755 |
| Interacción | 8,9730 | 18,4530 |
| Δp con cross-fitting | 5,4972 | 15,7244 |
| Pares decrecientes, cross-fitting | 36/42 | 42/42 |
| AUROC dirección conversacional, chat/agente | 0,9688/0,9048, dentro de muestra | 0,9779/0,9195, held-out |

La última fila contrasta protocolos distintos deliberadamente: **las cifras 0,969/0,905 que aparecen en la tabla de escala de la tesis son dentro de muestra**, aunque el capítulo afirma que ninguna conclusión descansa en cifras dentro de muestra. Debe indicarse el protocolo de cada tabla y usar las cifras held-out cuando se argumente generalización.

Hay una razón técnica documentada para revisar la procedencia: `src/activation_extractor.py` explica que la copia asíncrona GPU→CPU podía leerse antes de completarse y alteraba capas profundas. El código actual usa copia bloqueante. Se compararon directamente activaciones guardadas de V2 y V3: en `chat_harmful`, capa 52 coincide exactamente; capa 56 difiere en 1/42 filas; capas 60 y 76 difieren en 42/42, con diferencia máxima de 15,6047 en la 76. Ese patrón coincide con el problema documentado, aunque por sí solo no reconstruye la versión de código de todas las corridas antiguas. También existe una corrección de tokenización para evitar duplicar BOS, con un modo explícito `PROBE_LEGACY_BOS`.

Debe fijarse una corrida de referencia y registrar versión de código, tokenización y corpus. V3 es la candidata más completa disponible para Llama sobre 42 pares; no debe mezclarse con tablas o figuras antiguas sin identificarlo.

### 3. PCA permite avanzar ahora

En Llama V3, capa 76, el JSON reporta coseno PC1–diferencia de medias **0,98944** y varianza explicada **20,1278 %**. En capa 56, **0,99362** y **25,9056 %**. Se recalcularon desde `activations.npz` mediante la matriz de Gram y su autovector principal; coinciden con los resultados guardados.

El procedimiento implementado concatena activaciones conversacionales dañinas y benignas, centra por la media global, aplica PCA y alinea el signo de PC1 con la diferencia de medias. Esto debe describirse e incorporarse como resultados. En capas iniciales PC1 no siempre se alinea; no debe afirmarse validación uniforme en toda la red. Un coseno bajo con PC1 tampoco invalida automáticamente la diferencia de medias.

Mistral V3 imprime, en capa 30, coseno **0,952** y varianza de PC1 **28,7 %**. Son cifras redondeadas de un log, sobre 42 pares.

### 4. La observación 18 tiene una respuesta concreta

`results_llama33_v3/robust_results.json → layer_selection` registra selección por pares, 21 para selección y 21 para inferencia, capa elegida 76, Δp de inferencia 16,5899 e IC [14,0276; 18,9079]. En esa mitad decrecen 21/21, con prueba de signos p ≈ 9,54×10⁻⁷. Los 200 splits guardados seleccionan capa 76 en todos los casos.

Ese resultado es diferente del conteo global 42/42 del cross-fitting. La tesis debe explicar qué conjunto produjo cada denominador y reconocer que el antiguo 39/42 no era un conteo exclusivo de inferencia. Bonferroni controla multiplicidad; no reemplaza separar selección e inferencia.

### 5. Diferencia de medias y sonda logística se están mezclando

Métodos, glosario y resultados definen la dirección agéntica por diferencia de medias, mientras los análisis históricos de `agent_direction` extraen pesos de regresión logística. Son objetos distintos.

En Llama V3, capa 76, `robust_results.json → rotation` distingue ambos métodos: coseno de muestra completa **0,21277** para diferencia de medias y **0,07435** para logística. Los monitores también difieren: F1 OOF **0,84337**, FPR **0,14286** para diferencia de medias; F1 OOF **0,93671**, FPR **0** para logística. Esos valores corresponden al protocolo guardado y no constituyen validación externa de despliegue.

Cada tabla o figura debe identificar el estimador, la capa y la corrección por fiabilidad. No se puede atribuir el monitor de un método al vector del otro. La ortogonalidad debe discutirse junto con la reproducibilidad de las direcciones; un coseno pequeño por sí solo no prueba el mecanismo causal del fallo.

### 6. Mistral V3 está incompleto localmente

El log describe FP16, 42 pares y capa principal 30, no 96 pares/capa 18. Reporta Δp dañino 8,0227, benigno −0,7313 e interacción 8,7540. El AUROC held-out de la dirección conversacional es aproximadamente 0,9478 en chat y 0,7851 en agente. Su selección usa 21/21 pares y obtiene 19/21 decrecientes en inferencia.

El archivo menciona rutas `/workspace/.../results.json` y otros artefactos que **no están en esta carpeta local**. Termina durante la etapa causal, en la dosis 2, sin un cierre de esa etapa. Se puede recuperar evidencia descriptiva del log, pero no afirmar que el paquete causal esté completo ni recalcular sus proyecciones desde él.

### 7. Hay evidencia conductual adicional que la tesis no integra

Llama V3 incluye `causal_results.json` y `behavior_generations.jsonl`. La línea base guardada clasifica como rechazo 38/42 solicitudes dañinas en chat y 23/42 en agente; McNemar p ≈ 0,000275. También contiene intervenciones con controles aleatorios y brazo benigno.

Es evidencia adicional por revisar, no una demostración automática de toda la hipótesis de rotación. El etiquetado de respuestas es heurístico (`src/causal_test.py`), por lo que requiere auditar generaciones y declarar el método; dosis altas también provocan rechazo de solicitudes benignas. Si se incorpora, hay que actualizar el alcance de métodos y la afirmación de que no se analizan respuestas. Si se mantiene la tesis diagnóstica, identificar explícitamente estos ensayos como material complementario fuera de sus conclusiones principales.

## Otros ajustes concretos

- En composición del corpus se dice que el estrato más pequeño tiene 30 pares; es el contextual, con **24**.
- El resumen y el abstract presentan cuatro modelos junto al corpus de 96 pares sin explicitar la excepción de Llama. Añadirla.
- La lectura del último token corresponde al estímulo renderizado: el log de Mistral muestra `[/INST]` como posición registrada. Ajustar terminología sin perder la justificación causal de acceso al prefijo completo.
- El crecimiento del efecto con profundidad no descarta por sí solo un efecto de longitud o de sistema; eliminar esa inferencia fuerte o aportar un control específico. El análisis factorial existente puede ser útil para separar mensaje de sistema y herramientas.
- La fórmula de permutación de métodos aplica corrección +1; varios resultados históricos guardan p=0. Reportar la resolución del remuestreo y un límite compatible, nunca “probabilidad cero”.
- La prueba de signos sobre estimaciones cross-fit comparte datos de entrenamiento entre folds. Debe explicarse el alcance de la inferencia, sin afirmar independencia absoluta de los signos solo por haber excluido cada par de su propia dirección.
- Registrar acuerdo entre anotadores sigue pendiente: la revisión asistida por un modelo no equivale a una segunda anotación independiente.
- `README.md`, `docs/decision_log.md`, las notas y `app.py` conservan afirmaciones históricas sobre rotación, causalidad, número de pares y monitor perfecto. No usarlas sin contraste con los artefactos experimentales elegidos.

## Orden sugerido de cierre

1. Consolidar las corridas fuente: recuperar los artefactos de 96 pares si existen y el paquete de Mistral V3; identificar corpus, revisión de código y tokenización de cada una.
2. Elegir la corrida Llama de referencia y regenerar todas sus tablas y figuras con un protocolo consistente, separando dirección de medias y sonda logística.
3. Integrar PCA y la explicación selección/inferencia; actualizar la respuesta al tutor para resumen y abstract.
4. Resolver las precisiones de formato compuesto, posición de lectura e inferencia pareada.
5. Redactar discusión, conclusiones y anexos a partir de los resultados efectivamente trazables.

Esta revisión agrega únicamente este informe. No modifica la tesis, el ZIP, los resultados ni el código del experimento.
