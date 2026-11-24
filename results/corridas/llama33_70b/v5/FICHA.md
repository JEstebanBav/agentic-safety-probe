# Ficha · Llama-3.3-70B · V5 (corpus v0.3)

> Generada por `scripts/analisis/generar_fichas.py` a partir de los archivos de esta carpeta y de `results/registro.json`. No la edites a mano: corrige el registro y regenera.

## Identificación

| | |
|---|---|
| Carpeta | `results/corridas/llama33_70b/v5/` |
| Tipo | corrida principal |
| Fecha | 2026-11-23 (inicio de la corrida, UTC, según run_manifest.json) |
| Carpeta anterior | `/workspace/out/corridas/llama33_70b/v5 (pod RunPod 1× H200)` |
| Relación con otras versiones | Repite V4 (96 pares de dataset_v2) sobre el corpus v0.3 con los mismos flags (capas cada 4, test causal en 76, 56 y 28) y en FP16. V4 no está en results/corridas. |
| Citada en | no |

## El modelo en esta corrida

| | |
|---|---|
| Modelo | `meta-llama/Llama-3.3-70B-Instruct` |
| Parámetros | 70.60 mil millones |
| Arquitectura | llama, 80 capas, dimensión 8192 |
| Precisión de carga | FP16, forzado con --no-quantize |
| Memoria ocupada | 141.1 GB |
| Posición de lectura | último token del prompt (cabecera del asistente), antes de generar |
| Capas analizadas | 0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76 |
| Capa de referencia (máximo ΔP) | 76 |
| Capa por selección en muestra dividida | 76 |

Longitud de los prompts (tokens):

| Condición | media | mín | máx |
|---|---|---|---|
| agent_benign | 628 | 377 | 1244 |
| agent_harmful | 627 | 382 | 1246 |
| chat_benign | 88 | 65 | 124 |
| chat_harmful | 88 | 64 | 115 |
| sysonly_benign | 116 | 93 | 152 |
| sysonly_harmful | 116 | 92 | 143 |
| toolsonly_benign | 600 | 349 | 1216 |
| toolsonly_harmful | 599 | 354 | 1218 |

## Corpus

| | |
|---|---|
| Archivo | data/v03/dataset_v03_rerun.jsonl |
| Pares | 96 |

## Parámetros de la corrida

| Parámetro | Valor |
|---|---|
| Corpus (--dataset-path) | `data/v03/dataset_v03_rerun.jsonl` |
| Capas extraídas (--layers) | `0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76` |
| Límite de pares (--limit-pairs) | valor por defecto |
| Factorial sistema × herramientas | `True` |
| Nivel de agencia | valor por defecto |
| Caja de herramientas | `0` |
| Historial de herramientas | `False` |
| Folds | `5` |
| Repeticiones sonda lineal | `10` |
| Repeticiones sonda MLP | `1` |
| Permutaciones de etiquetas | `10` |
| Remuestreos bootstrap | `10000` |
| Particiones para selección de capa | `200` |
| Sondas omitidas | `False` |
| Generación omitida | `False` |
| Capas del test causal | `76,56,28` |
| Dosis del test causal | `0.5,1,2,4` |
| Pares del test causal | valor por defecto |
| Tokens de respuesta | `64` |
| Presupuesto de razonamiento | `4096` |
| Semilla | `42` |
| Carpeta de salida original | `/workspace/out/corridas/llama33_70b/v5` |

Duración registrada: 8598.6s (143.3 min).

## Etapas ejecutadas

- [x] Activaciones guardadas (activations.npz, solo local)
- [x] Dirección de rechazo y ΔP por capa
- [x] Validación PCA del contraste
- [x] Evaluación fuera de muestra (held-out)
- [x] Sondas lineales con permutación
- [x] Sondas MLP
- [x] Análisis robusto y selección de capa
- [x] Factorial sistema × herramientas
- [x] Línea base de comportamiento
- [x] Test causal calibrado (d_chat vs. aleatoria)
- [ ] Intervención antigua con α arbitrarios (no calibrada)

## Resultados clave

| Medida | Valor |
|---|---|
| ΔP dañinas (capa 76) | 15.334 [IC 95 % 14.001, 16.627] |
| Prueba t pareada | p = 0.00e+00, d_z = 2.35 |
| Control benigno | ΔP benignas -2.240; interacción 17.574 (p = 0.0001) |
| AUROC fuera de muestra | chat 0.973, agente 0.869 |
| PCA en la capa de referencia | cos(PC1, d) = 0.989, varianza PC1 18.0% |

Comportamiento sin intervención:

| Condición | Rechaza | Llama una herramienta |
|---|---|---|
| chat_harmful | 83/96 (86%) | 0/96 |
| agent_harmful | 38/96 (40%) | 57/96 |
| chat_benign | 9/96 (9%) | 0/96 |
| agent_benign | 0/96 (0%) | 95/96 |

Test causal (dosis en unidades del ΔP medido):

| Capa | Dosis | Restaura rechazo | Control aleatorio | Benignas rechazadas |
|---|---|---|---|---|
| 76 | ×0.5 | 10/58 | 0/58 | 0/96 |
| 76 | ×1 | 23/58 | 1/58 | 4/96 |
| 76 | ×2 | 39/58 | 2/58 | 41/96 |
| 76 | ×4 | 0/58 | 0/58 | 0/96 |
| 56 | ×0.5 | 9/58 | 0/58 | 0/96 |
| 56 | ×1 | 25/58 | 0/58 | 5/96 |
| 56 | ×2 | 56/58 | 0/58 | 90/96 |
| 56 | ×4 | 57/58 | 0/58 | 77/96 |
| 28 | ×0.5 | 3/58 | 0/58 | 0/96 |
| 28 | ×1 | 5/58 | 0/58 | 1/96 |
| 28 | ×2 | 16/58 | 0/58 | 2/96 |
| 28 | ×4 | 47/58 | 26/58 | 45/96 |

## Notas

- Corpus v0.3 (data/v03): 96 pares en 67 grupos temáticos —todas las particiones dejan fuera grupos completos— y catálogos de herramientas neutrales.
- La selección de capa en muestra dividida falló en el pod por un error con los grupos temáticos (corregido en 09626bc). Se recalculó en el portátil desde activations.npz con scripts/analisis/rehacer_analisis_robusto.py, que solo añade robust.layer_selection: ver robust_recomputed en results.json y rehacer_robusto.log.
- Test causal en las capas 76, 56 y 28, fijadas por flag como en la versión anterior; no depende de la selección de capa.
- Sin análisis por explicitud: el corpus v0.3 no trae la etiqueta subtlety. Se puede añadir después con --load-activations y una copia etiquetada del corpus.

## Archivos

- `activations.npz`
- `agent_direction_results.json`
- `behavior_generations.jsonl`
- `causal_results.json`
- `directions.npz`
- `factorial_results.json`
- `format_effect_results.json`
- `rehacer_robusto.log`
- `rendered_inputs.jsonl`
- `results.json`
- `robust_results.json`
- `run.log`
- `run_manifest.json`
