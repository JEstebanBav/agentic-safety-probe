# PROMPT — PRESENTACIÓN DE SUSTENTACIÓN DE TESIS

## Objetivo

Necesito que diseñes y construyas la presentación completa para la sustentación de mi tesis de pregrado en Ciencia de Datos de la Universidad Externado de Colombia.

**Título:** Transformación de la dirección de rechazo en modelos de lenguaje agénticos.

**Subtítulo:** Un análisis desde la interpretabilidad mecánica.

**Autor:** Juan Esteban Espitia Leiva.

**Tutor:** Julián Fernando Sánchez.

**Institución:** Universidad Externado de Colombia, Departamento de Matemáticas, Pregrado en Ciencia de Datos.

La duración de la exposición es de **20 minutos**, sin contar las preguntas del jurado.

La presentación debe comunicar la tesis como una investigación completa: el problema que la motiva, su pregunta, el fundamento matemático, el diseño experimental, los resultados, su interpretación, los aportes y las limitaciones. Su propósito es sustentar el trabajo y sus conclusiones ante un jurado académico.

**Toma la iniciativa tanto en el diseño como en la estructura.** Después de comprender los documentos, decide el concepto visual, los recursos gráficos, el orden del argumento, el número de diapositivas y la distribución del tiempo. Este prompt establece la intención y el rigor esperados; las decisiones de comunicación y diseño quedan a tu criterio.

## Investigación y fuentes

La tesis estudia cómo cambia la representación interna que distingue instrucciones dañinas de benignas cuando un modelo de lenguaje pasa del formato conversacional a un formato con mensaje de sistema y herramientas simuladas. Examina la dirección conversacional de rechazo, su capacidad discriminativa, la explicitud de la intención dañina, las direcciones estimadas en formato agéntico y su potencial para el monitoreo de seguridad. También incluye ensayos conductuales e intervenciones exploratorias.

La raíz del proyecto es:

`C:\Users\38614180\AgentHarm\agentic-safety-probe`

Las rutas siguientes son relativas a esa raíz. Si trabajas fuera del equipo original, utiliza los archivos equivalentes adjuntos.

Consulta principalmente:

- `docs/tesis/Tesis_avances_actualizada.pdf`: versión corregida del documento de tesis.
- `docs/tesis/Tesis_avances_actualizada_overleaf.zip` y la carpeta `docs/tesis/Tesis_avances_actualizada/`: fuentes LaTeX, bibliografía, figuras y material complementario.
- `docs/tesis/Tesis_avances_actualizada/evidencia/metricas_avance.json`: cifras y protocolos del documento actualizado.
- `docs/tesis/Tesis_avances_actualizada/evidencia/manifiesto_fuentes.json`: identificación de los archivos experimentales utilizados.
- `results/llama33_final/results_llama33_v3/`: resultados, activaciones, validaciones, generaciones e intervenciones de la corrida principal de Llama.
- `results/results_mistralV3/results_mistral7b_v3.log`: registro disponible de Mistral V3.
- `results/qwen8b/` y `results/qwen32b/`: resultados históricos, cuyo alcance debes evaluar antes de incorporarlos.
- `data/dataset_full.jsonl`, `data/dataset_v2.jsonl` y sus informes y etiquetas: construcción de los conjuntos de datos.
- `run_experiment.py`, `src/` y la documentación metodológica: implementación de los procedimientos.
- `docs/tesis/respuesta_observaciones_actualizada.pdf`: aclaraciones y correcciones solicitadas por el tutor.

Algunos archivos conservan la palabra “avances” en su nombre. Eso identifica la versión del archivo; **el entregable solicitado aquí es la presentación de sustentación de toda la tesis**.

Lee los documentos antes de decidir cómo contar la investigación. Identifica su argumento central y contrasta las cifras con los artefactos que las sustentan. Ante una discrepancia, revisa modelo, versión, capa, estimador y protocolo antes de seleccionar el resultado.

Conserva la distinción entre el corpus ampliado construido de **96 pares** y las ejecuciones disponibles de **42 pares por modelo**. Las cuatro variantes de un par no son observaciones independientes. La carpeta de Mistral V3 contiene un registro y no un paquete local completo de resultados; su etapa causal no tiene cierre en ese archivo. Presenta estas condiciones con precisión cuando sean relevantes para interpretar la evidencia.

## Storytelling científico

**La presentación debe contar una historia que haga necesaria la pregunta de investigación.** El jurado debe comprender por qué existe el problema antes de entrar en los detalles del método.

Parte del problema en su contexto actual: qué está cambiando en el uso de los modelos de lenguaje, cómo se están utilizando como agentes con herramientas y por qué ese cambio hace pertinente estudiar sus mecanismos de seguridad. Explica cómo se manifiesta el problema y qué consecuencias tiene, utilizando evidencia verificable.

Construye desde ese contexto la motivación de la tesis: qué sabemos, qué falta comprender y por qué mirar las representaciones internas puede aportar una respuesta. Haz que la pregunta, la matemática, los experimentos y los resultados aparezcan como partes conectadas de esa búsqueda.

Para contextualizar la actualidad, utiliza fuentes primarias pertinentes y comprueba sus fechas. Si necesitas complementar la bibliografía de la tesis, busca evidencia reciente y distingue esa contextualización externa de los resultados propios. No conviertas afirmaciones generales sobre adopción o incidentes en hechos sin fuente. No inventes un caso real para dar dramatismo a la apertura.

Decide libremente cómo desarrollar el resto del relato. El orden debe ayudar al jurado a seguir el razonamiento: entender qué se quiere averiguar, por qué la medición propuesta tiene sentido, qué evidencia permite obtener y qué respuesta sostiene finalmente.

No reproduzcas mecánicamente el índice del documento ni conviertas la exposición en una lista de tareas realizadas. Da continuidad a las ideas mediante transiciones orales y visuales. Cada parte debe preparar la siguiente y el cierre debe responder al problema que abrió la presentación.

El storytelling debe surgir de la investigación y de sus hallazgos. No fuerces una conclusión espectacular ni ocultes un resultado que matice la interpretación.

## Fundamento matemático y enfoque de Ciencia de Datos

**La matemática debe formar parte de la exposición principal.** Esta es una sustentación de Ciencia de Datos y el jurado debe poder comprender el fundamento de las medidas, los estimadores y la validación de la tesis.

Estudia la formalización del documento y decide qué desarrollo matemático es necesario para sostener el argumento dentro de los veinte minutos. No relegues todo el fundamento al material de respaldo ni utilices las ecuaciones como decoración.

La explicación debe conectar intuición, formulación y aplicación. Cuando introduzcas una expresión, define sus símbolos, explica qué objeto describe, por qué se calcula de esa forma y cómo se interpreta en el experimento. Haz visible la relación entre la ecuación y la figura o resultado que depende de ella.

Considera la relevancia de los siguientes objetos presentes en la tesis al seleccionar el fundamento:

- Las activaciones como vectores del flujo residual y la posición en que se realiza la lectura.
- La estimación de una dirección mediante diferencia de medias y su normalización.
- La proyección de una activación sobre esa dirección y su interpretación geométrica.
- El contraste pareado entre formatos y la interacción con el brazo benigno.
- La separación estandarizada y el AUROC como medidas diferentes de las unidades crudas de proyección.
- El coseno entre direcciones, PCA y la fiabilidad de las estimaciones.
- La lógica estadística de la evaluación fuera de muestra y de la separación entre selección e inferencia.

Selecciona y articula estos elementos según lo que necesite el argumento; no los conviertas automáticamente en una diapositiva por concepto. El detalle debe ser suficiente para que el jurado pueda evaluar la validez del método y su conexión con las conclusiones.

Usa la notación de la tesis de manera consistente. Si simplificas una derivación o recurres a una representación de baja dimensión para explicar una idea, aclara el alcance de esa simplificación. Distingue los diagramas pedagógicos de los datos efectivamente medidos.

## Diseño y comunicación visual

Desarrolla una identidad visual propia para esta investigación. Elige la paleta, la tipografía, la composición y los recursos gráficos después de comprender el problema y decidir cómo comunicarlo. Justifica esas decisiones por su aporte a la claridad y a la narrativa.

Busco una presentación profesional, original, visualmente memorable y con calidad editorial. Debe percibirse como un trabajo diseñado cuidadosamente para esta tesis.

Tienes libertad para decidir cómo representar conceptos, métodos y resultados. Mantén jerarquías claras, buena legibilidad en proyección, espacio suficiente y un sistema gráfico coherente. Evita que todas las diapositivas repitan la misma composición. La variedad debe responder al contenido.

Los recursos visuales deben explicar o demostrar algo. Evita imágenes genéricas de inteligencia artificial, efectos decorativos, elementos de plantilla y composiciones que ocupen espacio sin aportar significado. La investigación y sus representaciones deben dar identidad al diseño.

Selecciona las figuras que mejor contribuyan al relato. Las figuras verificadas están disponibles en:

`docs/tesis/Tesis_avances_actualizada/figures/`

Puedes reutilizarlas, adaptar su presentación o reconstruirlas desde los datos cuando eso mejore la explicación. No es obligatorio utilizar todas ni conservar su disposición original. Mantén intactos valores, unidades, escalas pertinentes, denominadores, intervalos y significado científico.

Diseña los elementos matemáticos con el mismo cuidado que las visualizaciones: símbolos legibles, notación consistente y relaciones comprensibles. No uses capturas ilegibles de ecuaciones, tablas o páginas del PDF.

## Resultados, interpretación y rigor

Selecciona los hallazgos que mejor respondan a la pregunta y sustenten la contribución de la tesis. Decide qué merece protagonismo y qué requiere desarrollo adicional en las notas o en el respaldo técnico.

La exposición debe permitir distinguir:

- Una medida geométrica, la discriminación dañino–benigno y el rechazo conductual.
- El desplazamiento crudo, los cambios de escala y la pérdida de margen discriminativo.
- La evaluación dentro de muestra, fuera de fold y en datos externos.
- Las direcciones por diferencia de medias y por regresión logística.
- La descripción de reorientación y la evidencia específica de intervención.

No conviertas un AUROC de entrenamiento perfecto en evidencia de generalización, un coseno pequeño en demostración causal o un cero observado de falsos positivos en garantía de seguridad. Identifica el alcance del control benigno, de las pruebas estadísticas y de los ensayos de intervención, incluido su costo en solicitudes benignas.

Presenta la incertidumbre y las limitaciones en relación con las conclusiones que afectan. Una sustentación completa incluye esos límites y sus implicaciones; no requiere afirmar que todas las extensiones posibles ya fueron ejecutadas.

No inventes resultados, estadísticas, referencias ni procedimientos. Las afirmaciones sobre el contexto actual deben tener fuentes verificadas y los resultados propios deben corresponder a los archivos del proyecto. Si falta evidencia para una afirmación, ajusta su alcance.

## Entregables

Entrega una presentación completa y lista para sustentar en **PowerPoint editable (.pptx)**, con exportación a **PDF** si el entorno lo permite.

Requisitos de entrega:

- Formato panorámico 16:9 y textos en español académico, claros y naturales.
- Número de diapositivas y distribución del tiempo decididos por ti para una exposición de veinte minutos.
- Notas del presentador con explicación oral, transiciones, tiempo orientativo y fuentes.
- Fundamento matemático integrado en el discurso principal.
- Gráficos, diagramas y ecuaciones editables cuando sea posible, con los recursos fuente disponibles.
- Referencias discretas y trazabilidad de las cifras utilizadas.
- Material de respaldo para preguntas del jurado, seleccionado según las necesidades de la investigación.
- Guion oral que mantenga el hilo narrativo de principio a fin.

Revisa visualmente todas las diapositivas y las exportaciones. Comprueba legibilidad, alineación, fórmulas, gráficos, citas, tiempos y ausencia de desbordamientos. Si una exportación o verificación no puede realizarse, indícalo con precisión.

**El objetivo final es que el jurado comprenda un problema relevante en el contexto actual, siga la construcción matemática y experimental de una respuesta y pueda valorar los resultados, la contribución y el alcance de la tesis mediante una historia científica clara y un diseño con identidad propia.**
