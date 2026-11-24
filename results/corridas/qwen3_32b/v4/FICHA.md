# Ficha · Qwen3-32B · V4 (corpus v0.3)

> Generada por `scripts/analisis/generar_fichas.py` a partir de los archivos de esta carpeta y de `results/registro.json`. No la edites a mano: corrige el registro y regenera.

## Identificación

| | |
|---|---|
| Carpeta | `results/corridas/qwen3_32b/v4/` |
| Tipo | corrida principal |
| Fecha | 2026-11-23 (inicio de la corrida, UTC, según run_manifest.json) |
| Carpeta anterior | `/workspace/out/corridas/qwen3_32b/v4 (pod RunPod 1× H200)` |
| Relación con otras versiones | Primera corrida completa de Qwen3-32B con el pipeline actual (V3 no terminó). Corpus v0.3 y FP16: V1 y V2 se cargaron en 4 bits. |
| Citada en | no |

## El modelo en esta corrida

| | |
|---|---|
| Modelo | `Qwen/Qwen3-32B` |
| Parámetros | 32.80 mil millones |
| Arquitectura | qwen3, 64 capas, dimensión 5120 |
| Precisión de carga | FP16, forzado con --no-quantize |
| Memoria ocupada | 65.5 GB |
| Posición de lectura | último token del prompt (cabecera del asistente), antes de generar |
| Capas analizadas | 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62 |
| Capa de referencia (máximo ΔP) | 62 |
| Capa por selección en muestra dividida | 62 |

Longitud de los prompts (tokens):

| Condición | media | mín | máx |
|---|---|---|---|
| agent_benign | 509 | 316 | 971 |
| agent_harmful | 509 | 321 | 973 |
| chat_benign | 66 | 43 | 102 |
| chat_harmful | 66 | 42 | 95 |
| sysonly_benign | 94 | 71 | 130 |
| sysonly_harmful | 94 | 70 | 123 |
| toolsonly_benign | 481 | 288 | 943 |
| toolsonly_harmful | 481 | 293 | 945 |

## Corpus

| | |
|---|---|
| Archivo | data/v03/dataset_v03_rerun.jsonl |
| Pares | 96 |

## Parámetros de la corrida

| Parámetro | Valor |
|---|---|
| Corpus (--dataset-path) | `data/v03/dataset_v03_rerun.jsonl` |
| Capas extraídas (--layers) | valor por defecto |
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
| Capas del test causal | valor por defecto |
| Dosis del test causal | `0.5,1,2,4` |
| Pares del test causal | valor por defecto |
| Tokens de respuesta | `64` |
| Presupuesto de razonamiento | `4096` |
| Semilla | `42` |
| Carpeta de salida original | `/workspace/out/corridas/qwen3_32b/v4` |

Duración registrada: 47762.1s (796.0 min).

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
| ΔP dañinas (capa 62) | 402.719 [IC 95 % 366.135, 436.616] |
| Prueba t pareada | p = 6.60e-40, d_z = 2.28 |
| Control benigno | ΔP benignas -1.138; interacción 403.857 (p = 0.0001) |
| AUROC fuera de muestra | chat 0.935, agente 0.809 |
| PCA en la capa de referencia | cos(PC1, d) = 0.973, varianza PC1 13.0% |

Comportamiento sin intervención:

| Condición | Rechaza | Llama una herramienta |
|---|---|---|
| chat_harmful | 83/96 (86%) | 0/96 |
| agent_harmful | 33/96 (34%) | 61/96 |
| chat_benign | 44/96 (46%) | 0/96 |
| agent_benign | 0/96 (0%) | 93/96 |

Test causal (dosis en unidades del ΔP medido):

| Capa | Dosis | Restaura rechazo | Control aleatorio | Benignas rechazadas |
|---|---|---|---|---|
| 62 | ×0.5 | 1/62 | 4/62 | 1/94 |
| 62 | ×1 | 7/62 | 5/62 | 3/94 |
| 62 | ×2 | 5/62 | 11/62 | 3/94 |
| 62 | ×4 | 9/62 | 6/62 | 2/94 |

## Notas

- Corpus v0.3 (data/v03): 96 pares en 67 grupos temáticos —todas las particiones dejan fuera grupos completos— y catálogos de herramientas neutrales.
- Razona antes de responder: hasta 4096 tokens de razonamiento, con corte de bucles. Línea base: 3/384 respuestas incompletas (2 cortadas por bucle), etiquetadas aparte de las negativas.
- La selección de capa en muestra dividida falló en el pod por un error con los grupos temáticos (corregido en 09626bc). Se recalculó en el portátil desde activations.npz con scripts/analisis/rehacer_analisis_robusto.py, que solo añade robust.layer_selection: ver robust_recomputed en results.json y rehacer_robusto.log.
- El test causal corrió en la capa de máximo ΔP (62) porque en el pod la selección no estaba disponible; la selección recalculada elige la misma capa, así que el test quedó en la capa pre-registrada.
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
