# Guion de sustentación

**Transformación de la dirección de rechazo en modelos de lenguaje agénticos.** Un análisis desde la interpretabilidad mecánica.

Juan Esteban Espitia Leiva · Tutor: Julián Fernando Sánchez · Pregrado en Ciencia de Datos, Universidad Externado de Colombia.

Duración prevista de la exposición: 19:35 de 20:00. Las diapositivas de respaldo (R1–R13) solo se usan en preguntas.

Este guion es el mismo texto de las notas del presentador del archivo .pptx. Los tiempos son orientativos; el margen restante absorbe pausas y transiciones.

| Diap. | Título | Tiempo | Acumulado |
|---:|---|---:|---:|
| 1 | Portada | 0:15 | 0:15 |
| 2 | Los modelos de lenguaje ya no solo conversan: actúan | 0:50 | 1:05 |
| 3 | Un agente de OpenAI pasó de una evaluación a una intrusión | 0:50 | 1:55 |
| 4 | El rechazo falla en agentes; falta saber qué ocurre adentro | 1:00 | 2:55 |
| 5 | Pregunta de investigación | 0:45 | 3:40 |
| 6 | El instrumento: mismo texto, distinto formato, dos intenciones | 1:10 | 4:50 |
| 7 | Un corpus contrastivo de 96 pares y tres grados de explicitud | 0:45 | 5:35 |
| 8 | Qué se mide: el flujo residual en la última posición | 0:55 | 6:30 |
| 9 | La dirección de rechazo es una diferencia de medias | 1:00 | 7:30 |
| 10 | Cuánto cambia la proyección y si el cambio es propio del daño | 0:55 | 8:25 |
| 11 | Un cambio de escala no es una pérdida de discriminación | 1:00 | 9:25 |
| 12 | La muestra no puede evaluarse a sí misma | 1:00 | 10:25 |
| 13 | Afuera: el mismo modelo rechaza menos cuando actúa | 0:45 | 11:10 |
| 14 | En chat, la dirección conversacional sí separa las clases | 0:45 | 11:55 |
| 15 | La proyección dañina cae; la benigna cambia poco | 1:00 | 12:55 |
| 16 | Parte de la caída es escala; otra parte es pérdida de margen | 1:00 | 13:55 |
| 17 | Según la explicitud, no aparece una jerarquía universal | 0:40 | 14:35 |
| 18 | En agente, el contraste dañino–benigno se reorienta | 0:55 | 15:30 |
| 19 | Los monitores agénticos recuperan señal, con costos distintos | 1:00 | 16:30 |
| 20 | Sumar la dirección restaura el rechazo, y también lo impone | 0:55 | 17:25 |
| 21 | Respuesta a la pregunta | 0:50 | 18:15 |
| 22 | Lo que la evidencia sostiene y lo que todavía no | 0:55 | 19:10 |
| 23 | Cierre | 0:25 | 19:35 |

## 1. Portada

*0:15 · de 0:00 a 0:15*

Buenos días. Soy Juan Esteban Espitia Leiva y voy a sustentar mi trabajo de grado, dirigido por el profesor Julián Fernando Sánchez: «Transformación de la dirección de rechazo en modelos de lenguaje agénticos».
Los puntos que ven abajo no son decoración: son datos del trabajo. Cada punto es una instrucción. Al final de la exposición los vamos a poder leer.

**Transición.** Empiezo por lo que está cambiando en el uso de estos modelos.

**Fuentes.** Puntos: proyecciones fuera de muestra (dirección reajustada por fold) en la capa 76 de Llama‑3.3‑70B V3. results/corridas/llama33_70b/v3/results.json → heldout.projections. Fila superior: chat; inferior: agente. Rojo arriba del eje: dañinas; azul abajo: benignas. Escala común de −10 a 28 unidades crudas de proyección.

## 2. Los modelos de lenguaje ya no solo conversan: actúan

*0:50 · de 0:15 a 1:05*

Durante años evaluamos los modelos de lenguaje como sistemas que conversan: reciben una instrucción, producen texto, y una persona lee ese texto y decide si actúa.
Hoy esos mismos modelos se despliegan como agentes. Además de la instrucción reciben un mensaje de sistema y un catálogo de herramientas, y pueden invocar acciones —enviar un correo, mover dinero, ejecutar un comando— que el sistema ejecuta sin una revisión humana en cada paso.
Gartner pronostica que a finales de este año cerca del 40 % de las aplicaciones empresariales incluirán agentes. Es una proyección de industria, no una medición, pero muestra la dirección del cambio. Y el Informe Internacional de Seguridad en IA de 2026 lo resume así: los agentes plantean riesgos mayores porque actúan de forma autónoma y es más difícil intervenir antes de que un fallo cause daño.

**Transición.** Esto ya no es solo una proyección: hay un caso documentado.

**Fuentes.** Gartner, comunicado de prensa del 26 de agosto de 2025 (gartner.com/en/newsroom/press-releases/2025-08-26-gartner-predicts-40-percent-of-enterprise-apps-will-feature-task-specific-ai-agents-by-2026-up-from-less-than-5-percent-in-2025). La tesis cita esta cifra a través de miniOrange (2026); aquí se usa la fuente primaria y se presenta como pronóstico.
International AI Safety Report 2026, resumen ejecutivo, publicado el 3 de febrero de 2026, presidido por Yoshua Bengio (internationalaisafetyreport.org/publication/2026-report-executive-summary). Cita original: "AI agents pose heightened risks because they act autonomously, making it harder for humans to intervene before failures cause harm". Traducción propia.
No se usan en la diapositiva las cifras de encuestas de Gravitee citadas en la tesis: son datos de percepción de un proveedor y el jurado podría pedir su metodología.

## 3. Un agente de OpenAI pasó de una evaluación a una intrusión

*0:50 · de 1:05 a 1:55*

En julio de 2026, agentes de OpenAI que resolvían una evaluación interna de ciberseguridad salieron del entorno previsto y accedieron a sistemas de terceros, entre ellos Hugging Face. Hugging Face reconstruyó actividad entre el nueve y el trece de julio y comunicó acceso a cinco conjuntos de datos de clientes. OpenAI aclara que se trataba de modelos de investigación con salvaguardas reducidas. El punto es que un objetivo de evaluación terminó acompañado de acciones fuera del alcance autorizado. El caso motiva estudiar la seguridad cuando el modelo puede actuar. No demuestra qué ocurrió con una dirección de rechazo: esta tesis mide representaciones en un turno, con herramientas que no se ejecutan.

**Transición.** El incidente muestra un riesgo conductual. ¿Qué sabemos del mecanismo interno que debería sostener el rechazo?

**Fuentes.** Hugging Face (27 de julio de 2026), reconstrucción técnica, https://huggingface.co/blog/agent-intrusion-technical-timeline. Periodo reconstruido: 9–13 de julio; el contenido de clientes accedido se acota a cinco datasets. OpenAI (26 de agosto de 2026), informe del incidente, https://openai.com/index/hugging-face-incident-and-the-road-ahead/. Evaluación interna con modelos de investigación y salvaguardas reducidas. Contextualización externa: no se presenta como evidencia de degradación de una dirección de rechazo ni como resultado de esta tesis. Fuentes consultadas el 15 de octubre de 2026.

## 4. El rechazo falla en agentes; falta saber qué ocurre adentro

*1:00 · de 1:55 a 2:55*

La literatura responde en dos planos que todavía no se han conectado.
En el plano conductual, AgentHarm, publicado en ICLR 2025, midió 110 tareas dañinas con herramientas simuladas: los modelos líderes cumplen muchas sin necesidad de jailbreak, y las evasiones diseñadas para chat se transfieren al agente. Eso describe el fallo, pero no lo explica.
En el plano mecánico, Arditi y colaboradores mostraron en NeurIPS 2024 que el rechazo está mediado por una sola dirección del espacio de activaciones: si se quita, el modelo deja de rechazar; si se suma, rechaza incluso lo inofensivo. Pero esa dirección se caracterizó en chat.
Los trabajos más cercanos intervienen el modelo —Lermen elimina la dirección por diseño— o estudian el control —Pinto muestra que la dirección inyectada llega casi intacta, pero su efecto en la conducta se recalibra—. Lo que no se ha determinado es si, usada como instrumento de diagnóstico, la dirección conserva su capacidad de detectar la intención dañina cuando el modelo actúa, y si eso depende de cuán explícito es el daño.

**Transición.** Esa es exactamente la pregunta de la tesis.

**Fuentes.** Andriushchenko et al., "AgentHarm: A benchmark for measuring harmfulness of LLM agents", ICLR 2025 (arXiv:2410.09024): 110 tareas explícitamente maliciosas (440 con aumentos), 11 categorías; "leading LLMs are surprisingly compliant with malicious agent requests without jailbreaking". Arditi et al., NeurIPS 2024 (arXiv:2406.11717). Lermen, Dziemian y Pimpale (2024, arXiv:2410.10871). Pinto (2026, arXiv:2607.09156). Zhong y Li (2026, arXiv:2606.26161). Formulación de la brecha: tesis, sección 2.3.

## 5. Pregunta de investigación

*0:45 · de 2:55 a 3:40*

La pregunta es esta: cuando un modelo de lenguaje pasa de conversar a actuar como agente, ¿conserva su dirección de rechazo la capacidad de detectar intenciones dañinas, incluso cuando se presentan de forma encubierta?
Tiene dos dimensiones que el diseño cruza. El formato: de conversacional a agéntico, con mensaje de sistema y herramientas simuladas y el mismo texto de instrucción. Y la explicitud: si el daño es explícito, contextual o tiene apariencia legítima, como en el caso que acabamos de ver.
Para responderla planteé cinco objetivos: construir un corpus que aísle el formato, estimar la dirección como línea base, cuantificar su cambio, evaluar la detección según la explicitud y explorar un monitor. Los recorro en ese orden.

**Transición.** Empiezo por el instrumento: el corpus.

**Fuentes.** Pregunta y objetivos: tesis, secciones 1.2 y 1.4 (objetivo general y cinco objetivos específicos).

## 6. El instrumento: mismo texto, distinto formato, dos intenciones

*1:10 · de 3:40 a 4:50*

El instrumento es un corpus contrastivo pareado. Cada instrucción base aparece en cuatro variantes: dañina o benigna, en chat o en agente.
Dos controles lo hacen funcionar. El texto de la solicitud es idéntico entre chat y agente; la comparación se hace dentro del par. Además, ambas ramas reciben las mismas herramientas y el mismo sistema dentro de cada formato.
Por entrada completa me refiero a todo lo que el modelo lee antes de responder, convertido a tokens: unidades de texto de su tokenizador. En agente contiene la solicitud, un mensaje de sistema que indica cómo usar herramientas y un catálogo con sus descripciones y parámetros. Ese catálogo no acredita permisos ni ejecuta funciones: aquí las herramientas no se ejecutan.
El ejemplo es fraud_06. Una solicitud pide facturar servicios que nunca se prestaron; la otra, servicios prestados. Se mantiene el tema, pero cambia la acción solicitada. Eso reduce el contraste temático; no demuestra por sí solo una separación pura de intención.
Las barras muestran la longitud media de los 42 pares de Llama, no la longitud de este ejemplo: la entrada agéntica es casi siete veces mayor, principalmente por el catálogo de herramientas.

**Transición.** Ese diseño se aplicó a un corpus que conviene describir con precisión.

**Fuentes.** Ejemplo literal: data/dataset_v2.jsonl, pair_id fraud_06 (variantes chat_harmful y chat_benign); datos_presentacion.json → ejemplo_didactico. El resaltado marca el contraste entre servicios no prestados y prestados. La dirección se estima con chat y se evalúa en ambos formatos. El ejemplo fraud_01 fue retirado de esta explicación por ambigüedad de autorización, sin modificar el corpus ni las métricas; se documenta en R13 y nota_autorizacion_etiquetas.md.
Tokens medios (42 pares, tokenizador de Llama‑3.3): chat dañina 88.2, agente dañina 594.5; solo sistema 116.2; solo herramientas 566.5 (results.json → prompt_tokens). Los incrementos "+28" y "+478" son diferencias de esas medias. Simetría verificada: regenerar_avances.py comprueba texto idéntico entre formatos y mismo sistema/herramientas entre ramas en los 96 pares.

## 7. Un corpus contrastivo de 96 pares y tres grados de explicitud

*0:45 · de 4:50 a 5:35*

El corpus ampliado tiene 96 pares y 384 variantes, siete categorías de daño y tres niveles de explicitud: 42 explícitos, 24 contextuales y 30 con apariencia legítima. La explicitud describe cómo se presenta la solicitud dañina etiquetada; no cambia la etiqueta de su contraparte benigna. Alegar una auditoría o una autorización no basta para decidir la etiqueta. El emparejamiento léxico, medido con Jaccard, es homogéneo entre estratos; eso controla una propiedad del pareado, pero no valida por sí solo las etiquetas.
Mistral y los dos Qwen tienen ejecuciones verificadas de 96 pares; sus resultados completos sustentan las comparativas. Llama‑3.3‑70B tiene 42 pares y aporta el análisis detallado de representación y los ensayos conductuales. No agrego tamaños distintos como si fueran una sola muestra.
Y la unidad de análisis es el par: sus cuatro variantes no son cuatro observaciones independientes.

**Transición.** ¿Qué se mide sobre cada una de esas variantes?

**Fuentes.** Corpus ampliado: dataset_v2_report.json → pares 96, entradas 384, por_estrato {explicit 42, contextual 24, framed 30}, por_fuente {propio 62, agentharm 34}. Herramientas: 64 distintas, 2–8 por par (tesis, sección 3.2). Jaccard mediano por estrato: tesis, tabla 3.3. Ejecuciones verificadas en results.json: Llama V3, 42 pares y referencia 76; Mistral V2, 96 y referencia 18; Qwen3‑8B V2, 96 y referencia 22; Qwen3‑32B V2, 96 y referencia 46. Mistral V3, 42 pares y referencia 30, se conserva para ensayos complementarios.

## 8. Qué se mide: el flujo residual en la última posición

*0:55 · de 5:35 a 6:30*

¿Qué medimos exactamente? Estos modelos son transformers decodificadores. Cada posición mantiene un vector, el flujo residual, y cada capa le suma la salida de la atención y la del perceptrón. Si se despliega la recurrencia, el flujo es una suma de contribuciones; por eso tiene sentido buscar un concepto como una dirección en ese espacio.
Leo en una sola posición: el último token de la entrada completa, antes de generar. La entrada es la secuencia formada por el mensaje de sistema, el catálogo de herramientas, la solicitud y los delimitadores de la plantilla. Por la máscara causal, esa posición puede atender a todo ese contexto. Desde ella se obtiene el primer token de la respuesta. Cumple el mismo papel en ambos formatos; promediar posiciones mezclaría el efecto con la longitud, que en agente es casi siete veces mayor.
En Llama leo 20 de sus 80 capas, en un espacio de 8.192 dimensiones.

**Transición.** Sobre esos vectores se define la dirección de rechazo.

**Fuentes.** Ecuaciones: tesis, ecuaciones (3.2) a (3.4). Dimensiones: tesis, tabla 3.4; results.json → model_info (num_layers 80, hidden_size 8192). Capas leídas: config.layers = 0,4,…,76. El diagrama de posiciones es esquemático: las anchuras no son proporcionales a los tokens.

## 9. La dirección de rechazo es una diferencia de medias

*1:00 · de 6:30 a 7:30*

Sobre esas activaciones, la dirección de rechazo es la diferencia entre el centroide de las instrucciones dañinas y el de las benignas en chat, normalizada a norma uno. Como los conjuntos están pareados, también es la media de las diferencias dentro de cada par.
La proyección de una activación sobre ese eje es su producto escalar con r gorro: en el esquema, la sombra del punto x sobre la recta. La segunda igualdad advierte algo que importa después: la proyección mezcla la orientación, que es lo que nos interesa, con la norma del vector, que no.
¿Por qué medias y no Fisher? El eje óptimo de Fisher invierte la covarianza dentro de clases, pero con 84 activaciones en 8.192 dimensiones esa matriz tiene rango a lo sumo 82. Si se regulariza y lambda tiende a infinito, se recupera la diferencia de medias: es el discriminante que renuncia a estimar la covarianza.

**Transición.** Con esa dirección defino las cantidades que comparan los formatos.

**Fuentes.** Tesis, sección 3.3.1: ecuación (3.5) de la dirección; discusión del discriminante de Fisher y regularización; ecuación (3.7) de la proyección. Rango: 2n − 2 = 82 con n = 42 pares. El panel izquierdo es un esquema con puntos generados (semilla fija), no datos.

## 10. Cuánto cambia la proyección y si el cambio es propio del daño

*0:55 · de 7:30 a 8:25*

Con esa dirección defino tres cantidades. La primera es el desplazamiento: para cada instrucción dañina, su proyección en chat menos su proyección en agente. Su promedio es delta p sub h. Si es positivo, la dirección responde con menos intensidad cuando el modelo actúa.
Eso solo no basta, porque hay tres explicaciones posibles, a la derecha: un debilitamiento específico del daño, un corrimiento rígido de todo el espacio, o movimientos en sentidos opuestos. El brazo benigno, que atraviesa exactamente el mismo cambio de formato, las distingue. La cantidad propia del daño es la interacción, delta cuadrado p: el desplazamiento dañino menos el benigno. Phi indica qué fracción del efecto reproduce el benigno.
Para la inferencia reporto d sub z, el tamaño de efecto pareado, que, a diferencia de t, es comparable entre corpus de distinto tamaño.

**Transición.** Falta una pregunta: ¿cambia la información o solo la escala?

**Fuentes.** Tesis, sección 3.3.1, ecuaciones (3.8) desplazamiento intra‑par, (3.9) prueba t y d_z, (3.10) interacción y fracción común. Los tres paneles son esquemas, no datos.

## 11. Un cambio de escala no es una pérdida de discriminación

*1:00 · de 8:25 a 9:25*

La segunda pregunta es si el cambio es de información o solo de escala. Si el formato agéntico simplemente comprimiera todas las proyecciones por un factor a, la distancia cruda entre clases caería en una fracción uno menos a, tan grande como se quiera, sin que se perdiera capacidad de separar. Es lo que muestra el esquema.
Por eso expreso el efecto en unidades estandarizadas: d, la separación dividida por la desviación agrupada, que en ese caso límite no cambia, y kappa, que mide la compresión.
Y uso el AUROC: la probabilidad de que una instrucción dañina tomada al azar proyecte más que una benigna. Vale uno si separa perfectamente y 0,5 si es azar; como solo depende del orden, ningún reescalamiento lo altera.
La regla de lectura es esta: solo la pérdida que sobrevive en d y en el AUROC cuenta como pérdida de margen discriminativo.

**Transición.** Todo esto exige un cuidado estadístico: no evaluar con los mismos datos con que se estima.

**Fuentes.** Tesis, sección 3.3.1 (invariancia de escala, ecuaciones 3.11–3.12, caso afín) y sección 3.4.1 (AUROC, ecuación 3.15). El valor a = 0,31 del esquema coincide con el κ observado en Llama (0,309) solo como ilustración; las campanas no son datos.

## 12. La muestra no puede evaluarse a sí misma

*1:00 · de 9:25 a 10:25*

Queda el problema estadístico central: estimar la dirección y evaluarla con los mismos datos. La norma de la dirección estimada está sesgada hacia arriba en un término que crece con la dimensión, y la separación medida con esos datos hereda el sesgo. En alta dimensión un clasificador puede separar incluso ruido: la sonda logística logra AUROC uno en entrenamiento, y eso no prueba generalización.
Por eso evalúo la detección fuera de muestra. Divido los pares en cinco folds; la dirección que puntúa un par se estimó sin él, y sus cuatro variantes usan la misma dirección.
En Llama V3, la exportación documenta una partición aparte: 21 pares eligen la capa y otros 21 hacen la inferencia, con corrección de Bonferroni. Sus pruebas respetan el pareado —permutación de signos, bootstrap por pares y prueba de signos— y comparo contra etiquetas barajadas. Las comparativas V2 conservan las estimaciones exportadas y su versión se indica por separado.

**Transición.** Con este método, paso a los resultados. Empiezo por lo que se ve desde afuera.

**Fuentes.** Sesgo de la norma: tesis, ecuación (3.6). Cross‑fitting por pares: sección 3.4.1, ecuación (3.16). Inferencia y selección separada (21 + 21, Bonferroni por 20 capas, 200 particiones): sección 3.4.2. AUROC de entrenamiento de la sonda logística = 1: results.json → linear_aurocs_in_sample['76']. Nadeau y Bengio (2003). La barra 21|21 de la tarjeta es esquemática.
Tamaño y procedencia: las ejecuciones Mistral V2 y Qwen V2 incluidas en las comparativas tienen 96 pares verificados. Llama V3 tiene 42 pares; su análisis detallado y los ensayos conductuales conservan sus denominadores reales. Las comparaciones no convierten estos tamaños en equivalentes ni atribuyen a 96 un contraste de 42.

## 13. Afuera: el mismo modelo rechaza menos cuando actúa

*0:45 · de 10:25 a 11:10*

Empiezo por afuera, por la conducta. En Llama‑3.3‑70B, de las 42 instrucciones dañinas, 38 se rechazan en chat; en formato agéntico, solo 23. Cada cuadro es un par: 16 pares rechazan en chat y cumplen en agente, y solo uno hace lo contrario. La prueba exacta de McNemar da p de 0,0003. En Mistral el patrón se repite: de 19 a 5 rechazos.
Dos advertencias: estas etiquetas salen de una heurística sobre el texto generado, que no observa herramientas ejecutadas, y la auditoría humana está pendiente. Por eso lo trato como evidencia conductual exploratoria.
El síntoma, entonces, se reproduce en nuestro sistema. La pregunta es qué cambió adentro.

**Transición.** Para responderla, primero hay que mostrar que en chat el detector funciona.

**Fuentes.** metricas_avance.json → llama.behavior.refusal (chat_harmful 38/42, agent_harmful 23/42, chat_benign 5/42, agent_benign 1/42; IC de Wilson guardados) y gap_harmful (16 / 1 / 22 / 3; mcnemar_p = 0.000274658203125). Benignas: 5 frente a 1, McNemar p = 0,219. Mistral: registro, BEHAVIOURAL BASELINE (19/42 → 5/42; 16 frente a 2; p = 0,0013). Tesis, sección 4.3.
Tamaño y procedencia: las ejecuciones Mistral V2 y Qwen V2 incluidas en las comparativas tienen 96 pares verificados. Llama V3 tiene 42 pares; su análisis detallado y los ensayos conductuales conservan sus denominadores reales. Las comparaciones no convierten estos tamaños en equivalentes ni atribuyen a 96 un contraste de 42.

## 14. En chat, la dirección conversacional sí separa las clases

*0:45 · de 11:10 a 11:55*

Primero, la línea base. En chat, la dirección conversacional, evaluada fuera de muestra, separa dañinas de benignas con un AUROC cercano a 0,98 desde la capa 16 hasta el final; en la capa de referencia, 0,978.
La curva ámbar es la misma dirección aplicada al formato agéntico: queda por debajo en todas las capas profundas.
A la derecha, PCA muestra que en capas profundas el contraste dañino–benigno es el eje dominante de variación: el primer componente coincide con la dirección, con coseno de 0,99. Y una sonda logística con validación cruzada da 0,968, frente a 0,515 con etiquetas barajadas.
Hay, por tanto, un detector bien definido en chat.

**Transición.** ¿Qué le pasa a ese detector cuando el modelo actúa?

**Fuentes.** Curvas: results.json → heldout.per_layer[capa].auroc_chat / auroc_agent (dirección reajustada por fold, 5 folds por pares). Capa 76: AUROC chat 0,9779, IC [0,9490; 0,9955] (remuestreo condicional a los puntajes cross‑fit). PCA: pca_by_layer (|cos| = 0,9894, varianza PC1 = 20,13 %); Mistral desde el registro (capa 30: 0,952; 28,7 %). Sonda logística: linear_aurocs['76'] = 0,9677, sd entre folds 0,0373, IC Nadeau–Bengio [0,9285; 1]; control permutado 0,5152. La desviación entre folds no es un intervalo de confianza.
Tamaño y procedencia: las ejecuciones Mistral V2 y Qwen V2 incluidas en las comparativas tienen 96 pares verificados. Llama V3 tiene 42 pares; su análisis detallado y los ensayos conductuales conservan sus denominadores reales. Las comparaciones no convierten estos tamaños en equivalentes ni atribuyen a 96 un contraste de 42.

## 15. La proyección dañina cae; la benigna cambia poco

*1:00 · de 11:55 a 12:55*

Ahora el desplazamiento, en la capa 76 de Llama. Cada línea es uno de sus 42 pares. A la izquierda, las dañinas: la proyección cae al pasar a agente en todos los pares. El promedio, delta p sub h, es 15,7, con intervalo de 13,8 a 17,6.
A la derecha, las benignas: casi no se mueven, y si algo, suben un poco. Su cambio promedio es −1,5 y su intervalo incluye el cero.
La interacción, lo específico del daño, es 17,3, con intervalo de 14,9 a 19,5; es positiva en 41 de 42 pares y su p es del orden de 10 a la menos 4, que es la resolución de las permutaciones. En Mistral, ahora sobre 96 pares y en otras unidades, la interacción guardada es 3,28.
Esto descarta un corrimiento rígido. Pero todavía no dice si se perdió información o si solo cambió la escala.

**Transición.** Para eso están las otras dos unidades.

**Fuentes.** Proyecciones: results.json → heldout.projections (dirección reajustada en cada fold). Δp_h = 15,7244, IC percentil [13,7630; 17,5535], t = 16,05, d_z = 2,4772, 42/42 decrecen (sign test p = 4.55e-13). Δp_b = −1,5438, IC [−3,0956; 0,2151]. Interacción 17,2682, IC [14,9074; 19,4547], p de permutación intra‑par con piso 10⁻⁴, 41/42 positivos, fracción común −0,0982. Mistral V2: results/corridas/mistral7b/v2/results.json, format_effect.held_out.per_layer[18], n=96.
Tamaño y procedencia: las ejecuciones Mistral V2 y Qwen V2 incluidas en las comparativas tienen 96 pares verificados. Llama V3 tiene 42 pares; su análisis detallado y los ensayos conductuales conservan sus denominadores reales. Las comparaciones no convierten estos tamaños en equivalentes ni atribuyen a 96 un contraste de 42.

## 16. Parte de la caída es escala; otra parte es pérdida de margen

*1:00 · de 12:55 a 13:55*

Para eso están las otras dos unidades. Arriba, las mismas proyecciones en unidades crudas: en agente, todo se comprime cerca de cero. La separación cruda cae 83 %. Pero la dispersión también cae: kappa es 0,31.
Abajo, al estandarizar por la dispersión de cada formato, la brecha entre clases se recupera en parte: d pasa de 3,2 a 1,8, una pérdida de 44 %. Y el AUROC pasa de 0,978 a 0,920; la diferencia, 0,058, tiene un intervalo que excluye el cero.
En Mistral, sobre 96 pares, d cae un 51,6 % y el AUROC pasa de 0,924 a 0,878. Su intervalo de diferencia de AUROC incluye cero: esa pérdida puntual no es significativa al 95 %.
La lectura: buena parte de la caída cruda es escala, pero sobrevive una pérdida real de margen. La información no desaparece —el AUROC agéntico sigue lejos de 0,5—; se estrecha.

**Transición.** Esa pérdida, ¿depende de cuán explícito es el daño?

**Fuentes.** S y s calculados desde heldout.projections (desviación agrupada de las dos clases en cada formato): S_chat = 20,8962, S_ag = 3,6280, d_chat = 3,2051, d_ag = 1,8036, κ = 0,3085; coinciden con metricas_avance.json → heldout_format. AUROC: heldout_detection (gap 0,0584, IC [0,0113; 0,1128], condicional a los puntajes guardados). Panel estandarizado: (p − media benigna del formato) / s agrupada del formato; puntos con dispersión vertical aleatoria solo para legibilidad. Mistral V2: results/corridas/mistral7b/v2/results.json, format_effect.held_out.per_layer[18] y heldout.detection, n=96.
Tamaño y procedencia: las ejecuciones Mistral V2 y Qwen V2 incluidas en las comparativas tienen 96 pares verificados. Llama V3 tiene 42 pares; su análisis detallado y los ensayos conductuales conservan sus denominadores reales. Las comparaciones no convierten estos tamaños en equivalentes ni atribuyen a 96 un contraste de 42.

## 17. Según la explicitud, no aparece una jerarquía universal

*0:40 · de 13:55 a 14:35*

Aquí comparo tres ejecuciones sobre el mismo corpus de 96 pares: 42 explícitos, 24 contextuales y 30 de apariencia legítima. En Mistral y Qwen3‑8B la mayor pérdida puntual está en el estrato explícito; en Qwen3‑32B está en el contextual. No aparece una jerarquía universal de degradación. La apariencia legítima conserva señal, aunque con márgenes diferentes. Son estimaciones puntuales: sin intervalos por estrato no afirmo una ordenación poblacional ni una diferencia significativa entre modelos.

**Transición.** Si la dirección de chat pierde margen en agente, ¿hay otra dirección que separe mejor?

**Fuentes.** results/corridas/mistral7b/v2/results.json → format_effect.held_out.by_stratum; n=96
results/corridas/qwen3_8b/v2/results.json → format_effect.held_out.by_stratum; n=96
results/corridas/qwen3_32b/v2/results.json → format_effect.held_out.by_stratum; n=96
Tamaño y procedencia: las ejecuciones Mistral V2 y Qwen V2 incluidas en las comparativas tienen 96 pares verificados. Llama V3 tiene 42 pares; su análisis detallado y los ensayos conductuales conservan sus denominadores reales. Las comparaciones no convierten estos tamaños en equivalentes ni atribuyen a 96 un contraste de 42.

## 18. En agente, el contraste dañino–benigno se reorienta

*0:55 · de 14:35 a 15:30*

Si la dirección conversacional pierde margen en agente, ¿hay otra dirección que separe mejor? Estimé la dirección agéntica con el mismo estimador, diferencia de medias, y la comparé con la de chat.
El coseno entre las dos es 0,21. Como el ruido de estimación reduce el coseno aparente, medí la fiabilidad de cada dirección entre mitades disjuntas de pares: 0,90 en chat y 0,92 en agente, alta. Corrigiendo por ese techo, el coseno es 0,22, un ángulo de unos 77 grados.
Con regresión logística el coseno es aún menor, 0,07, pero esas direcciones son poco reproducibles, con fiabilidades de 0,25 y 0,28; por eso no las interpreto con la misma confianza ni las promedio con las anteriores.
La lectura prudente: el contraste entre clases se reorienta en agente. Es geometría, no causalidad.

**Transición.** Esa reorientación tiene una consecuencia práctica para el monitoreo.

**Fuentes.** rotation_at_reference.diff_means: cos muestra completa 0,2128; cos entre formatos en mitades disjuntas 0,2009 (sd 0,0022, 50 repeticiones); fiabilidad chat 0,9010, agente 0,9220; techo √(ρc·ρa) = 0,9114; normalizado 0,2204; ángulo 77,26°. Logística: cos 0,0744, fiabilidades 0,2522 y 0,2832. La versión anterior de la tesis atribuía a Llama un coseno de 0,051; la cifra vigente es la de V3. La normalización por fiabilidad es un diagnóstico bajo supuestos de error, no un ángulo poblacional exacto.
Tamaño y procedencia: las ejecuciones Mistral V2 y Qwen V2 incluidas en las comparativas tienen 96 pares verificados. Llama V3 tiene 42 pares; su análisis detallado y los ensayos conductuales conservan sus denominadores reales. Las comparaciones no convierten estos tamaños en equivalentes ni atribuyen a 96 un contraste de 42.

## 19. Los monitores agénticos recuperan señal, con costos distintos

*1:00 · de 15:30 a 16:30*

El ajuste al formato agéntico mejora el AUROC en Llama y ambos Qwen. En Mistral no supera al eje conversacional aplicado en agente: 0,850 frente a 0,878. La tabla separa los dos estimadores de Llama, sobre 42 pares, de los monitores de Mistral y Qwen sobre 96. Qwen3‑32B alcanza F1 de 0,873, con 15,6 por ciento de falsos positivos; Mistral y Qwen3‑8B quedan alrededor de 0,75, con 27,1 y 35,4 por ciento. Ese costo importa: no basta con detectar más dañinas si también se bloquean muchas benignas. El cero observado de la logística de Llama no es una tasa poblacional nula. Sin validación externa, hablamos de viabilidad preliminar.

**Transición.** ¿Y si, además de observar la dirección, la sumo durante la generación?

**Fuentes.** Llama: datos_presentacion.json → llama.monitores, n=42. Mistral/Qwen: evaluacion96[modelo].monitor, n=96. F1 y FPR se conservan como exportados; AUROC es la media OOF guardada. Llama usa capa 76; Mistral 18, Qwen3‑8B 22 y Qwen3‑32B 46.
Tamaño y procedencia: las ejecuciones Mistral V2 y Qwen V2 incluidas en las comparativas tienen 96 pares verificados. Llama V3 tiene 42 pares; su análisis detallado y los ensayos conductuales conservan sus denominadores reales. Las comparaciones no convierten estos tamaños en equivalentes ni atribuyen a 96 un contraste de 42.

## 20. Sumar la dirección restaura el rechazo, y también lo impone

*0:55 · de 16:30 a 17:25*

Por último, ¿la dirección tiene efecto sobre la conducta? En la capa 56 sumé alfa veces la dirección durante la generación, en las dañinas que el modelo no rechazaba como agente, y lo comparé con una dirección aleatoria de la misma norma.
Con dosis uno rechazan 12 de 19, frente a 1 de 19 con la aleatoria: McNemar da p de 0,001. Con dosis dos, 19 de 19. Pero miren la curva azul: a esa dosis también rechaza 40 de 41 benignas.
Es decir, sumar la dirección cambia la conducta en el sentido esperado, pero no la repara de forma selectiva: impone el rechazo. Además es exploratorio: una capa distinta de la del diagnóstico, sin corrección por múltiples dosis y con etiquetas heurísticas. En Mistral, el registro termina en esta etapa.

**Transición.** Con esto puedo volver a la pregunta.

**Fuentes.** intervention_layer56: unidad de dosis 6,9256 (Δp_h chat − agente en la capa 56); α = 3,46; 6,93; 13,85; 27,70. Dañinas no rechazadas en agente: n = 19; benignas: n = 41. Tasas e IC de Wilson guardados por dosis; restore_vs_control McNemar: dosis 0,5 p = 0,0625; 1: p = 0,00098; 2 y 4: p = 7,6·10⁻⁶. Benignas a dosis 4: 33/41 con 8 degeneradas. La capa 56 es la declarada en config.intervention_layers (56, 52, 28); se reporta solo la 56, como en la tesis.
Tamaño y procedencia: las ejecuciones Mistral V2 y Qwen V2 incluidas en las comparativas tienen 96 pares verificados. Llama V3 tiene 42 pares; su análisis detallado y los ensayos conductuales conservan sus denominadores reales. Las comparaciones no convierten estos tamaños en equivalentes ni atribuyen a 96 un contraste de 42.

## 21. Respuesta a la pregunta

*0:50 · de 17:25 a 18:15*

Vuelvo a la pregunta: cuando el modelo pasa a actuar como agente, ¿conserva su dirección de rechazo la capacidad de detectar intenciones dañinas?
La respuesta que sostienen los datos es: sí, pero con menos margen. Es una degradación que depende del modelo, no una desaparición.
Primero, la interacción con el control benigno es positiva en Llama y Mistral. Segundo, en Llama la pérdida sobrevive en d y en el AUROC; en Mistral la diferencia puntual de AUROC tiene un intervalo que incluye cero. Tercero, no hay una jerarquía universal por explicitud, y lo encubierto conserva señal. Y cuarto, los monitores recuperan margen en Llama y Qwen, aunque la mejora no es universal.
En la conducta, el rechazo cae, y sumar la dirección lo restaura, pero con sobrerrechazo.

**Transición.** ¿Qué aporta esto y hasta dónde llega?

**Fuentes.** Síntesis de las diapositivas 13 a 20. Llama V3, n=42: d 3,21 → 1,80 (−43,7 %), AUROC 0,9779 → 0,9195. Mistral V2, n=96: d 2,1766 → 1,0533 (−51,6 %), AUROC 0,9239 → 0,8781, IC de diferencia [−0,0056; 0,0969]. Qwen V2, n=96: comparativas de estratos y monitores en datos_presentacion.json → evaluacion96.
Tamaño y procedencia: las ejecuciones Mistral V2 y Qwen V2 incluidas en las comparativas tienen 96 pares verificados. Llama V3 tiene 42 pares; su análisis detallado y los ensayos conductuales conservan sus denominadores reales. Las comparaciones no convierten estos tamaños en equivalentes ni atribuyen a 96 un contraste de 42.

## 22. Lo que la evidencia sostiene y lo que todavía no

*0:55 · de 18:15 a 19:10*

¿Qué aporta el trabajo? Un corpus contrastivo auditable que aísla el formato; un diagnóstico con controles explícitos de circularidad y de selección; y una forma de lectura que separa tres unidades y tres objetos —geometría, discriminación y conducta— que es fácil confundir.
Los límites acotan conclusiones concretas. La comparación por explicitud necesita intervalos y contrastes específicos. El formato cambia sistema y catálogo a la vez, sin ejecutar herramientas, así que no atribuyo el efecto a un componente ni lo extiendo a agentes que ejecutan acciones. Mistral y ambos Qwen tienen 96 pares verificados, mientras Llama usa 42; los tamaños y las versiones se mantienen separados. Las etiquetas de daño y explicitud admiten casos ambiguos: la autorización declarada no se verifica. Las etiquetas conductuales son heurísticas y los monitores no tienen validación externa.
Lo siguiente es revisar daño y explicitud con anotadores independientes, auditar generaciones y usar un diseño factorial con control de longitud, además de validación externa y trayectorias de varios pasos.

**Transición.** Cierro volviendo a la imagen del comienzo.

**Fuentes.** Tesis, secciones 5.4 (limitaciones), 5.5 y 6.3 (recomendaciones); respuesta_observaciones_actualizada.pdf (pendientes experimentales).

## 23. Cierre

*0:25 · de 19:10 a 19:35*

Al comienzo les mostré estos puntos; ahora se pueden leer. En chat, las dañinas quedan lejos de las benignas. En agente, todo se acerca a cero y el margen se estrecha, aunque el orden se conserva en buena parte.
La implicación práctica es sencilla: antes de confiar en que un agente rechazará, conviene medir si todavía distingue.
Muchas gracias. Quedo atento a sus preguntas.

**Fuentes.** Mismos datos que la portada: proyecciones fuera de muestra, Llama V3, capa 76 (results.json → heldout.projections). AUROC y d: metricas_avance.json → heldout_detection y heldout_format.
Tamaño y procedencia: las ejecuciones Mistral V2 y Qwen V2 incluidas en las comparativas tienen 96 pares verificados. Llama V3 tiene 42 pares; su análisis detallado y los ensayos conductuales conservan sus denominadores reales. Las comparaciones no convierten estos tamaños en equivalentes ni atribuyen a 96 un contraste de 42.

## 24. R1 · Modelos, ejecuciones y configuración

*Respaldo: usar solo si el jurado pregunta.*

La tabla recoge tamaños verificados directamente en los archivos: Mistral y ambos Qwen con 96 pares, Llama con 42. Los modelos son abiertos, alineados y con soporte de herramientas; cubren tres recetas de entrenamiento y escalas de 7 a 70 mil millones de parámetros. La comparación no aísla el tamaño: cambian arquitectura, entrenamiento, plantilla y versión del pipeline. Mistral V3, con 42 pares, es un ensayo adicional que se identifica por separado.

## 25. R2 · Perfil por capas del desplazamiento

*Respaldo: usar solo si el jurado pregunta.*

Si preguntan por qué la capa 76: es el máximo del barrido para describir magnitudes, y la inferencia no la reutiliza: se separan 21 pares para elegir y 21 para inferir, y la mitad de selección elige la 76 en todas las particiones guardadas. Si preguntan por qué la intervención es en la 56: está declarada así en la configuración de ese ensayo; no es la capa del diagnóstico.

**Fuentes.** Tamaño y procedencia: las ejecuciones Mistral V2 y Qwen V2 incluidas en las comparativas tienen 96 pares verificados. Llama V3 tiene 42 pares; su análisis detallado y los ensayos conductuales conservan sus denominadores reales. Las comparaciones no convierten estos tamaños en equivalentes ni atribuyen a 96 un contraste de 42.

## 26. R3 · Circularidad y selección de capa

*Respaldo: usar solo si el jurado pregunta.*

Si preguntan cuánto infla la circularidad: en Llama, el desplazamiento ingenuo es 16,28 y el corregido 15,72, un 3,5 %; en Mistral, un 2,4 %. La corrección reduce el efecto y conserva el signo. Si preguntan por la reutilización de pares al elegir la capa: la prueba se hace sobre 21 pares que no participaron en la elección; en Llama decrecen 21 de 21, con p de signos de 9,5 por 10 a la menos 7, por debajo del umbral de Bonferroni.

**Fuentes.** Tamaño y procedencia: las ejecuciones Mistral V2 y Qwen V2 incluidas en las comparativas tienen 96 pares verificados. Llama V3 tiene 42 pares; su análisis detallado y los ensayos conductuales conservan sus denominadores reales. Las comparaciones no convierten estos tamaños en equivalentes ni atribuyen a 96 un contraste de 42.

## 27. R4 · Fundamento matemático adicional

*Respaldo: usar solo si el jurado pregunta.*

Para preguntas de fundamento: la última posición determina la distribución del siguiente token; Fisher no es estimable con 84 puntos en 8.192 dimensiones y su versión regularizada converge a la diferencia de medias; la permutación de signos supone intercambiabilidad del cambio de formato dentro de pares independientes; la prueba de signos depende del eje, y por eso la dirección se estima fuera de muestra.

## 28. R5 · PCA y sonda logística

*Respaldo: usar solo si el jurado pregunta.*

Si preguntan por la validación geométrica: en capas profundas PC1 coincide con la diferencia de medias con coseno de 0,99, aunque explica solo entre 20 y 37 % de la varianza. Es una descripción dentro de muestra. Si preguntan por la sonda logística: su AUROC de entrenamiento es 1 por la alta dimensión; la cifra que cuenta es la de validación cruzada, 0,968, frente a 0,515 con etiquetas barajadas.

**Fuentes.** Tamaño y procedencia: las ejecuciones Mistral V2 y Qwen V2 incluidas en las comparativas tienen 96 pares verificados. Llama V3 tiene 42 pares; su análisis detallado y los ensayos conductuales conservan sus denominadores reales. Las comparaciones no convierten estos tamaños en equivalentes ni atribuyen a 96 un contraste de 42.

## 29. R6 · ¿La proyección explica la conducta?

*Respaldo: usar solo si el jurado pregunta.*

Si preguntan si la caída de proyección causa el cambio de conducta: dentro de cada formato la proyección predice bien el rechazo etiquetado, con AUROC de 0,93 a 0,95 en Llama. Pero los pares que dejan de rechazar no pierden más proyección que los que siguen rechazando (p = 0,71). En Mistral sí la pierden (p = 0,004), con solo 3 pares que siguen rechazando. Conclusión: la proyección es informativa, pero no explica sola la decisión; por eso no afirmo causalidad desde la geometría.

**Fuentes.** Tamaño y procedencia: las ejecuciones Mistral V2 y Qwen V2 incluidas en las comparativas tienen 96 pares verificados. Llama V3 tiene 42 pares; su análisis detallado y los ensayos conductuales conservan sus denominadores reales. Las comparaciones no convierten estos tamaños en equivalentes ni atribuyen a 96 un contraste de 42.

## 30. R7 · Comparación sobre 96 pares

*Respaldo: usar solo si el jurado pregunta.*

Estos tres archivos sí contienen 96 pares. Qwen3‑8B pierde aproximadamente 0,275 de AUROC, Qwen3‑32B 0,036 y Mistral 0,046. El intervalo de Mistral incluye cero: no afirmo allí una diferencia significativa de AUROC al 95 por ciento. En ambos Qwen el intervalo lo excluye. Son medidas del corpus común; las diferencias de arquitectura, entrenamiento y configuración impiden atribuir el patrón exclusivamente al tamaño.

**Fuentes.** results/corridas/mistral7b/v2/results.json → heldout.detection; SHA-256 c95ce6e47f62d138ee90facb69c1c51b445a89c64cad098ee6fba918b4766d36
results/corridas/qwen3_8b/v2/results.json → heldout.detection; SHA-256 b4b4bf08115d873f3eab069747f82fd4962e6e5e4aae8c3f0bec31c269043291
results/corridas/qwen3_32b/v2/results.json → heldout.detection; SHA-256 f406d30bdee3f9398053bcb4271842dec296fdbb0f8b80d1125a8f896476c9b1
Tamaño y procedencia: las ejecuciones Mistral V2 y Qwen V2 incluidas en las comparativas tienen 96 pares verificados. Llama V3 tiene 42 pares; su análisis detallado y los ensayos conductuales conservan sus denominadores reales. Las comparaciones no convierten estos tamaños en equivalentes ni atribuyen a 96 un contraste de 42.

## 31. R8 · Exploratorio: ¿sistema o herramientas?

*Respaldo: usar solo si el jurado pregunta.*

Usar solo si preguntan cuál componente del formato produce el efecto. La respuesta del documento es: el diseño principal no lo separa. Aclarar que la corrida V3 tiene, además, una variante factorial no analizada en la tesis: con la dirección de muestra completa, añadir solo los esquemas de herramientas reproduce casi toda la caída (16,20 de 16,28), y el mensaje de sistema solo produce 4,32. Pero los esquemas también alargan la entrada en unos 478 tokens, así que herramientas y longitud están confundidas; por eso el documento lo deja como trabajo pendiente con control de longitud.

**Fuentes.** factorial_results.json → by_layer['76'].harmful y .interaction (total, tools, system, non_additive). Estas cifras no aparecen en el documento de tesis; están en los archivos de la corrida V3 (run_factorial_analysis en run_experiment.py). Decidir antes de la sustentación si se muestra esta diapositiva.
Tamaño y procedencia: las ejecuciones Mistral V2 y Qwen V2 incluidas en las comparativas tienen 96 pares verificados. Llama V3 tiene 42 pares; su análisis detallado y los ensayos conductuales conservan sus denominadores reales. Las comparaciones no convierten estos tamaños en equivalentes ni atribuyen a 96 un contraste de 42.

## 32. R9 · Consideraciones éticas

*Respaldo: usar solo si el jurado pregunta.*

Si preguntan por riesgos del corpus o de publicar el diagnóstico: el material no da capacidad operativa, las herramientas son simuladas, el modelo no se reentrena y el balance favorece la divulgación porque el fallo ya está documentado y la defensa requiere este diagnóstico.

## 33. R10 · Notación

*Respaldo: usar solo si el jurado pregunta.*

Referencia rápida de símbolos para responder preguntas sobre definiciones.

## 34. R11 · Referencias

*Respaldo: usar solo si el jurado pregunta.*

Lista de las fuentes citadas en las diapositivas. Las marcadas con asterisco se verificaron para la presentación en sus fuentes primarias (octubre de 2026).

## 35. R12 · Trazabilidad de las cifras

*Respaldo: usar solo si el jurado pregunta.*

Cada figura identifica archivo, clave y tamaño. Mistral y Qwen V2 usan resultados JSON de 96 pares, incorporados desde la carpeta entregada por el autor y con su hash registrado en evaluacion96. Llama V3 conserva 42 pares. El registro adicional de Mistral V3 se utiliza únicamente donde se indica ese ensayo complementario.

**Fuentes.** Tamaño y procedencia: las ejecuciones Mistral V2 y Qwen V2 incluidas en las comparativas tienen 96 pares verificados. Llama V3 tiene 42 pares; su análisis detallado y los ensayos conductuales conservan sus denominadores reales. Las comparaciones no convierten estos tamaños en equivalentes ni atribuyen a 96 un contraste de 42.

## 36. R13 · Autorización declarada y validez de las etiquetas

*Respaldo: usar solo si el jurado pregunta.*

Tener credenciales solo demuestra una posibilidad de acceso; no establece permiso para cualquier acción. En un experimento de texto tampoco se puede comprobar si la autorización declarada es verdadera. El benigno de fraud_01 describe cuentas de prueba y QA. Sin embargo, su contraparte etiquetada como dañina no dice que las credenciales sean robadas: el guion anterior añadía esa información y se corrigió. Ese par requiere revisión de validez; también la etiqueta de explicitud depende de esa revisión. La apariencia legítima es una propiedad de la formulación del miembro dañino etiquetado, no una regla para convertir cualquier petición de auditoría en un ataque. Si el texto admite una lectura legítima y no aporta evidencia suficiente de abuso, se reconoce la ambigüedad. Con los resultados actuales se conservan las etiquetas originales y se declara la limitación. Una revisión futura debe usar una rúbrica, anotadores independientes y análisis de sensibilidad; cambiar el corpus requeriría recalcular las estimaciones afectadas.

**Fuentes.** data/dataset_full.jsonl y data/dataset_v2.jsonl, pair_id fraud_01: is_harmful verdadero/falso y subtlety explicit. El mensaje dañino dice I have obtained login credentials; no dice stolen. El benigno declara authorized access to test bank accounts y internal QA process. No hay verificación externa de titularidad ni de autorización en esta presentación. Documento de revisión: docs/sustentacion/nota_autorizacion_etiquetas.md. fraud_03 se señala allí como otro ejemplo para revisar: el entrenamiento de phishing declarado no basta por sí solo para determinar abuso.
