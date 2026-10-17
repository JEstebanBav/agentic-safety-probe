# Guion de la presentación ligera

Juan Esteban Espitia Leiva · Universidad Externado de Colombia

Duración orientativa: 11:50. 14 diapositivas principales y 2 de apoyo para preguntas.

El guion coincide con las notas del PowerPoint. Las pausas y el ritmo pueden ajustar la duración.

| Diap. | Idea | Tiempo | Acumulado |
|---:|---|---:|---:|
| 1 | La pregunta de la tesis | 0:20 | 0:20 |
| 2 | De responder a poder actuar | 0:45 | 1:05 |
| 3 | El caso de Hugging Face | 0:45 | 1:50 |
| 4 | Qué quiero comprobar | 0:40 | 2:30 |
| 5 | Una comparación dentro de cada par | 1:00 | 3:30 |
| 6 | Dos ideas matemáticas | 1:15 | 4:45 |
| 7 | Probar con pares no usados en el ajuste | 0:55 | 5:40 |
| 8 | El rechazo observado baja | 0:50 | 6:30 |
| 9 | Se conserva señal, pero cambia el margen | 1:10 | 7:40 |
| 10 | Lo encubierto conserva señal | 0:50 | 8:30 |
| 11 | Detectar daño también tiene un costo | 1:05 | 9:35 |
| 12 | Forzar la señal no resuelve todo | 0:50 | 10:25 |
| 13 | La conclusión y sus límites | 1:00 | 11:25 |
| 14 | La idea para recordar | 0:25 | 11:50 |

## 1. La pregunta de la tesis

Tiempo: 0:20

Esta tesis pregunta qué ocurre con una señal interna de rechazo cuando un modelo pasa de conversar a recibir un contexto agéntico. Voy a explicar el problema, cómo lo medí y qué resultados considero más importantes.

**Transición.** Primero, ¿por qué importa ese cambio de contexto?

**Fuentes y alcance.** Título y alcance del documento de tesis. Esta versión resume la exposición completa.

## 2. De responder a poder actuar

Tiempo: 0:45

En una conversación, el resultado visible suele ser un texto. Un agente también puede solicitar una llamada a una herramienta, y el sistema que lo rodea puede ejecutarla. Por eso una misma petición puede tener consecuencias distintas. El rechazo es una de las conductas que se usan para evitar solicitudes dañinas. Mi pregunta es si una señal asociada a ese rechazo se conserva cuando cambia el contexto de entrada. En el experimento no se ejecutan herramientas reales: se incorpora su catálogo a la entrada del modelo.

**Transición.** Un incidente reciente ayuda a ver por qué la pregunta es relevante.

**Fuentes y alcance.** Diseño y alcance: tesis, secciones 3.2 y 3.3. El diagrama describe el funcionamiento general de un agente, no una ejecución de herramientas de este experimento.

## 3. El caso de Hugging Face

Tiempo: 0:45

En julio de 2026, agentes de OpenAI que participaban en una evaluación interna de ciberseguridad salieron del entorno previsto y accedieron a sistemas de terceros, entre ellos Hugging Face. OpenAI aclara que eran modelos de investigación con salvaguardas reducidas. El caso sirve como motivación: un objetivo de evaluación terminó acompañado de acciones fuera del alcance autorizado. Mi tesis no explica ese incidente. Estudia una pregunta más acotada sobre la representación interna de solicitudes, en un turno y sin ejecutar herramientas.

**Transición.** Esa motivación se convierte en una pregunta concreta de investigación.

**Fuentes y alcance.** Fuentes primarias: Hugging Face, reconstrucción técnica del incidente de julio de 2026: https://huggingface.co/blog/agent-intrusion-technical-timeline. OpenAI, informe publicado el 26 de agosto de 2026: https://openai.com/index/hugging-face-incident-and-the-road-ahead/. Fuentes verificadas para la presentación completa el 15 de octubre de 2026.

## 4. Qué quiero comprobar

Tiempo: 0:40

La pregunta no es solo si el modelo responde o se niega. Quiero saber si la dirección de rechazo que se estima en chat sigue ayudando a separar solicitudes dañinas y benignas cuando la entrada tiene formato agéntico. Para responder, primero mido la señal, después comparo los dos formatos y finalmente evalúo si puede ser útil como monitor. Durante toda la exposición separaré la señal interna, la detección y la conducta observada.

**Transición.** Para comparar necesito mantener la solicitud y cambiar su formato.

**Fuentes y alcance.** Tesis: pregunta de investigación, objetivos y distinción entre geometría, discriminación y conducta.

## 5. Una comparación dentro de cada par

Tiempo: 1:00

Cada par contiene una solicitud etiquetada como dañina y una contraparte benigna. Cada una se presenta en chat y en formato agéntico. El texto de la solicitud se mantiene entre formatos; en agente se incorpora un mensaje de sistema y el catálogo de herramientas. El catálogo describe funciones y parámetros, pero no ejecuta acciones ni acredita permisos. Un ejemplo es cobrar por servicios nunca prestados frente a cobrar por servicios prestados. El corpus ampliado tiene 96 pares. Las exportaciones de Mistral y ambos Qwen usan esos 96; la de Llama contiene 42. Mantengo esos tamaños en los resultados. La explicitud describe cómo se formula el miembro dañino etiquetado; la contraparte conserva su etiqueta benigna.

**Transición.** Ahora explico la idea matemática con dos pasos sencillos.

**Fuentes y alcance.** docs/sustentacion_ligera/fuente/datos_ligera.json, extraído de los resultados verificados; procedencia y SHA-256 registrados en fuentes. Corpus: data/dataset_v2.jsonl y dataset_v2_report.json. Ejemplo: fraud_06, traducción abreviada para explicar el contraste. Se preservan las etiquetas y los resultados originales.

## 6. Dos ideas matemáticas

Tiempo: 1:15

El modelo transforma la entrada en una representación interna: un vector de números. Primero calculo el promedio de las representaciones dañinas y el de las benignas. La resta entre esos dos promedios define una dirección. El dibujo es solo un esquema para imaginarla. Después doy un puntaje a cada entrada según cuánto se alinea con esa dirección. La primera fórmula es una resta de promedios; la segunda, un producto punto con la dirección normalizada. No necesitamos desarrollar el cálculo para seguir los resultados. Lo importante es que el puntaje representa un contraste geométrico entre clases etiquetadas. La conducta de rechazo se observa por separado: tener un puntaje no equivale por sí solo a rechazar.

**Transición.** Para saber si ese puntaje sirve, hay que probarlo con ejemplos distintos de los usados para estimarlo.

**Fuentes y alcance.** Tesis, sección 3.3.1: diferencia de medias y proyección. Se omiten índices de capa y la derivación de normalización para facilitar la exposición. Se lee el flujo residual del último token de la entrada antes de generar; no se usa una representación de la respuesta completa.

## 7. Probar con pares no usados en el ajuste

Tiempo: 0:55

Separé el ajuste de la evaluación: la dirección que da puntaje a un par se estima sin usar ese par. Sus variantes quedan juntas para conservar la comparación. Después uso la AUROC, que resume el orden de los puntajes entre dañinas y benignas. Cero coma cinco equivale al azar; uno es un orden perfecto. Una AUROC de cero coma nueve significa que, al tomar una dañina y una benigna al azar, la dañina queda por encima aproximadamente nueve de cada diez veces. Es una medida de orden, no el porcentaje de aciertos de un clasificador con umbral. También observo por separado las respuestas y las falsas alarmas de los monitores.

**Transición.** Empiezo por el cambio que se ve desde afuera: la conducta.

**Fuentes y alcance.** Tesis, sección 3.4.1: validación por pares y AUROC. En los resultados de proyección se usa cross-fitting; los intervalos se condicionan a los puntajes guardados. Los controles detallados de selección y las pruebas están en la presentación completa.

## 8. El rechazo observado baja

Tiempo: 0:50

En Llama, de las mismas 42 solicitudes dañinas, 38 se rechazan en chat y 23 en formato agéntico. Es una diferencia de conducta observada con las etiquetas usadas en el experimento. No son dos grupos de solicitudes diferentes: el texto se mantiene entre formatos. El patrón no demuestra todavía qué mecanismo causó el cambio. Por eso paso a mirar la representación interna y su capacidad de separar las clases. Las etiquetas de rechazo son heurísticas y requieren auditoría humana.

**Transición.** ¿La señal interna también cambia?

**Fuentes y alcance.** docs/sustentacion_ligera/fuente/datos_ligera.json, extraído de los resultados verificados; procedencia y SHA-256 registrados en fuentes. Conducta: metricas_avance.json → llama.behavior.refusal. McNemar exacto sobre pares: p=0,000274658. Llama V3, n=42. No se extrapolan estos conteos a 96 pares.

## 9. Se conserva señal, pero cambia el margen

Tiempo: 1:10

Los puntos oscuros muestran la AUROC en chat y los dorados, la de esa dirección aplicada en agente. Todos los valores agénticos quedan por encima de cero coma cinco, pero la pérdida puntual varía: es mucho mayor en Qwen3 de ocho mil millones que en Qwen3 de treinta y dos mil millones. En Llama, la separación ajustada por escala también cae cerca de 44 por ciento; por tanto, el cambio no consiste solo en que los números se hagan pequeños. En Mistral hay que ser prudente: el intervalo de diferencia de AUROC incluye cero. Y estas comparaciones no aíslan el efecto del tamaño del modelo: cambian también arquitectura, entrenamiento y versión del pipeline.

**Transición.** Otra pregunta es si la apariencia legítima elimina esa señal.

**Fuentes y alcance.** docs/sustentacion_ligera/fuente/datos_ligera.json, extraído de los resultados verificados; procedencia y SHA-256 registrados en fuentes. Datos por modelo → chat/agente/ic_diferencia. Llama d=3,2051→1,8036 (−43,73 %). Mistral: IC95% de diferencia [−0,005645; 0,096899]. Qwen3-8B [0,210392; 0,330946]; Qwen3-32B [0,007704; 0,063477]. Llama n=42; demás modelos n=96. No son pruebas formales de diferencias entre modelos.

## 10. Lo encubierto conserva señal

Tiempo: 0:50

El estrato de apariencia legítima conserva señal por encima del azar en estas tres ejecuciones. Los valores son aproximadamente cero coma 88, cero coma 69 y cero coma 90. Tampoco aparece una jerarquía universal: la mayor pérdida puntual está en el explícito para Mistral y Qwen3 de ocho mil millones, y en el contextual para Qwen3 de treinta y dos mil millones. Esto es descriptivo, porque no tenemos intervalos y contrastes específicos entre estratos. Apariencia legítima describe la formulación del miembro dañino etiquetado; no significa que cualquier declaración de autorización sea un engaño.

**Transición.** Si la dirección de chat pierde margen, ¿ayuda ajustar un monitor al contexto agéntico?

**Fuentes y alcance.** docs/sustentacion_ligera/fuente/datos_ligera.json, extraído de los resultados verificados; procedencia y SHA-256 registrados en fuentes. estratos96 → framed.auroc_agent; 30 dañinas y 30 benignas por modelo. Los tres tamaños totales son 96. El análisis depende de las etiquetas del corpus; autorización real no verificada.

## 11. Detectar daño también tiene un costo

Tiempo: 1:05

Ajustar un monitor al formato agéntico mejora el AUROC en Llama y ambos Qwen. Para hablar de utilidad muestro conteos con el umbral calibrado en entrenamiento: cuántas dañinas detecta y cuántas benignas marca por error. En Llama, el monitor logístico detecta 37 de 42 dañinas sin falsas alarmas observadas. En el corpus de 96, Qwen3 de treinta y dos mil millones detecta 86 dañinas con 15 falsas alarmas; los otros dos producen más falsas alarmas. El cero de Llama es un resultado de esa muestra, no una garantía de riesgo cero. Tampoco todos los monitores mejoran todo: el AUROC del de Mistral es 0,850 frente a 0,878 del eje de chat aplicado en agente. La utilidad sigue siendo preliminar y necesita validación externa.

**Transición.** Observar una señal es una cosa. Forzarla durante la generación es otra.

**Fuentes y alcance.** docs/sustentacion_ligera/fuente/datos_ligera.json, extraído de los resultados verificados; procedencia y SHA-256 registrados en fuentes. modelos → monitor. TP=recall*n, FP=FPR*n, ambos enteros comprobados y coherentes con F1. Llama: logística; no se muestra aquí el segundo estimador de medias de la versión completa. AUROC de monitores: Llama 0,9926, Mistral 0,8498, Qwen8 0,8297, Qwen32 0,9406. Comparación descriptiva, sin prueba formal entre monitores.

## 12. Forzar la señal no resuelve todo

Tiempo: 0:50

También probé sumar la dirección durante la generación en Llama. Muestro una intensidad concreta para hacer visible el costo: con dosis dos, rechaza las 19 dañinas del subconjunto, pero también 40 de 41 benignas. Con una dirección aleatoria de igual norma rechaza solo una de las 19 dañinas. Hay un efecto sobre la conducta etiquetada, pero no una reparación selectiva de seguridad. Este ensayo es exploratorio: usa subconjuntos que no rechazaban en agente y una capa distinta de la del diagnóstico. No permite concluir que la reorientación geométrica sea la causa del fallo.

**Transición.** Con estos resultados puedo responder la pregunta inicial.

**Fuentes y alcance.** docs/sustentacion_ligera/fuente/datos_ligera.json, extraído de los resultados verificados; procedencia y SHA-256 registrados en fuentes. intervencion_dosis2, extraída de llama.intervencion56.per_dose['2']. Capa 56, alfa=13,8512735; n dañinas=19, benignas=41. Rechazos: 19 y 40; control aleatorio 1. No se corrige aquí por la búsqueda entre capas y dosis.

## 13. La conclusión y sus límites

Tiempo: 1:00

La dirección conversacional conserva capacidad de separar, aunque el margen cambia al pasar a agente y la magnitud depende del modelo. La tesis aporta una manera de medir ese cambio y de separar tres cosas: representación interna, detección y conducta. Los monitores muestran utilidad preliminar; la intervención muestra que recuperar rechazo también puede bloquear solicitudes benignas. Los límites son concretos: un turno y herramientas sin ejecución, tamaños y versiones distintos, etiquetas que necesitan revisión humana y ausencia de validación externa. La autorización declarada no se verifica, y casos como fraud_01 conservan una ambigüedad que debe auditarse. Las conclusiones dependen de la validez de las etiquetas; no son una garantía operacional de seguridad.

**Transición.** Cierro con la implicación práctica que conecta todo el trabajo.

**Fuentes y alcance.** Síntesis de los resultados anteriores y de las limitaciones de la presentación completa. Véase docs/sustentacion/nota_autorizacion_etiquetas.md. No se modifican retrospectivamente etiquetas, muestras ni métricas.

## 14. La idea para recordar

Tiempo: 0:25

La idea para recordar es que la seguridad puede cambiar cuando cambia el contexto del modelo. Antes de confiar en que un agente rechazará, conviene medir si todavía distingue. Muchas gracias.

**Transición.** Preguntas.

**Fuentes y alcance.** Conclusión de la tesis y de esta exposición. No se añaden resultados en el cierre.

## 15. A1 · Fuentes

Apoyo para preguntas.

Las dos referencias externas son fuentes primarias del incidente de Hugging Face. Los datos de la tesis se conservan en los archivos del repositorio; esta versión registra ruta y hash de las fuentes y reutiliza las cifras verificadas de la presentación completa.

**Transición.** —

**Fuentes y alcance.** https://huggingface.co/blog/agent-intrusion-technical-timeline
https://openai.com/index/hugging-face-incident-and-the-road-ahead/
docs/sustentacion_ligera/fuente/datos_ligera.json, extraído de los resultados verificados; procedencia y SHA-256 registrados en fuentes.

## 16. A2 · Autorización y etiquetas

Apoyo para preguntas.

El experimento no verifica identidades ni permisos reales. Si dos escenarios tienen el mismo texto y contexto, pero difieren solo en una autorización oculta, este diseño no puede observar esa diferencia. En fraud_01, el benigno declara cuentas de prueba y QA, pero la contraparte dañina no afirma que las credenciales sean robadas; ese par debe revisarse. Cambiar el ejemplo didáctico a fraud_06 no resuelve la validez del corpus. Se conservan los resultados existentes y se declara la limitación. Tampoco se clasifica una auditoría como ataque solo por decir que es una auditoría: hay que examinar la acción pedida y el contexto disponible.

**Transición.** —

**Fuentes y alcance.** docs/sustentacion/nota_autorizacion_etiquetas.md; data/dataset_full.jsonl y data/dataset_v2.jsonl → fraud_01, fraud_06. Este mazo no reetiqueta ni reejecuta el corpus.