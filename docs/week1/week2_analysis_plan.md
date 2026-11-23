# Plan de analisis de semana 2 — borrador previo a la captura MCP

Fecha: 2 de noviembre de 2026. Este documento es una propuesta; no se ha congelado ni preregistrado. No se han capturado servidores MCP ni ejecutado nuevos modelos en esta tarea. Registrar el commit y los cambios de protocolo antes de inspeccionar los nuevos resultados. Acordar los detalles pendientes antes de presentar este documento como un analisis fijado por adelantado.

## Preguntas y comparaciones

Medir cuanto cambia la capacidad de la direccion extraida en chat para discriminar solicitudes daninas y benignas cuando cambia el contexto de herramientas. Comparar Qwen3-8B y Mistral-7B sobre el corpus adjudicado; incorporar 70B solo si hay computo. Comparar tambien estratos, sin asumir de antemano cual sufrira mas.

Mantener el mismo modelo, tokenizador, solicitud, posicion de lectura y configuracion de generacion dentro de cada comparacion. Guardar cada prompt ya renderizado y su conteo real de tokens.

| Condicion | Contenido que se agrega |
|---|---|
| Sistema agente sin herramientas | Encuadre del sistema sin lista de herramientas. El chat original se conserva como referencia adicional. |
| Relleno con longitud equivalente | Texto de control sin instrucciones de accion, ajustado a los tokens de la condicion real correspondiente. |
| Herramientas sinteticas | Descripciones/esquemas del conjunto sintetico controlado. |
| Descripciones reales | Nombres y descripciones capturados de servidores MCP, sin propiedades del esquema. |
| Esquemas reales completos | Misma seleccion real, con los esquemas de entrada completos. |

No confundir retirar un esquema con retirar tambien la instruccion de llamada. Para cada condicion fijar que cambia y guardar el prompt renderizado. La condicion Llama con/sin la linea inyectada por su plantilla es un control separado. No quitar arbitrariamente una instruccion de unos modelos y mantenerla en otros sin registrarlo.

## Captura y asignacion

Semana 1: objetivo de 20 servidores vivos; semana 2: aproximadamente 40 servidores y 200 herramientas. Conservar la respuesta `tools/list` original y una tabla normalizada con nombre, descripcion, esquema de entrada completo, servidor, fecha UTC, version/referencia del servidor y errores de captura. No llamar herramientas que ejecuten acciones: la captura requiere solo enumeracion. La presencia en un registro no cuenta como captura real.

Definir los servidores y criterios de inclusion antes de medir el efecto. Registrar exclusiones por autenticacion, indisponibilidad o esquema invalido. No inventar descripciones ausentes ni tratarlas como observadas. Mantener todos los componentes de cada servidor agrupados en los analisis que usen servidores como unidad independiente.

Usar la misma seleccion y orden de herramientas reales para los dos brazos de cada par. Fijar una semilla y un plan de asignacion antes de generar. Separar el experimento de distractores reales comunes del experimento de herramientas capaces de realizar la tarea: no atribuir ejecucion a herramientas irrelevantes ni llamar funcional una sustitucion por nombre.

## Longitud y variables descriptivas

Contar tokens con el tokenizador de cada modelo despues de aplicar su plantilla. Para cada contexto real generar un control de relleno con la misma longitud total; fijar una tolerancia maxima de 2 tokens o declarar el emparejamiento fallido. Guardar longitud total, longitud de la descripcion y del esquema por separado. No truncar la solicitud para igualar longitudes. Definir una politica unica ante limites de contexto antes de correr.

Variables propuestas: tokens de descripcion, numero de herramientas, parametros totales y obligatorios, profundidad del esquema, longitud de nombres, verbos imperativos por 100 palabras y vocabulario relacionado con dano. Congelar diccionario, regla de conteo y transformaciones antes del analisis. Un verbo o palabra sensible no prueba causalidad ni hace maliciosa una herramienta.

## Evaluacion y criterio de evidencia

Usar grupos tematicos adjudicados para mantener reformulaciones fuera de los folds de entrenamiento cuando se evaluan. Reestimar direccion, estandarizacion y probe dentro de cada fold. Elegir capas sin usar los resultados del test. No reutilizar activaciones si cambio el texto, sistema o herramientas.

Resultados primarios propuestos: AUROC fuera de muestra de la direccion de chat, diferencia pareada entre condiciones y sus intervalos. Reportar el brazo benigno y la interaccion de ambos brazos para distinguir un desplazamiento general del formato de un efecto especifico. Resultados por estrato y rasgos descriptivos son secundarios; registrar multiplicidad y tratar las asociaciones como exploratorias hasta replicarlas.

La afirmacion sobre herramientas reales recibe apoyo si la diferencia respecto al control de longitud persiste en datos retenidos con incertidumbre reportada y se replica en ambos modelos. Una asociacion descriptiva entre rasgos y perdida recibe apoyo si mantiene direccion y capacidad predictiva al retener servidores completos. Si desaparece con el control de longitud, se reporta ese resultado. No basta que una condicion aislada baje frente al chat para atribuirlo a descripciones reales.

Para comparar monitor y steering en semana 3, calibrar umbrales a tasas de falsas alarmas de 2%, 5% y 10% en un conjunto benigno de calibracion separado. Congelar umbrales antes del test externo. Con 96 benignos, un punto porcentual corresponde aproximadamente a un ejemplo: informar conteos e incertidumbre, sin afirmar precision que la muestra no permite.

## Decisiones que faltan antes de congelar

Seleccion concreta de servidores, politica de autenticacion y presupuesto de captura; adjudicacion del corpus y grupos; asignacion de herramientas compatibles frente a distractores; diccionarios de rasgos; semilla, folds y capa; conjunto de calibracion/test; metrica primaria unica y politica de intervalos para dependencias por par y servidor. Este borrador no reemplaza esas decisiones.
