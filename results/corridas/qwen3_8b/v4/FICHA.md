# Ficha · v4

> Generada por `scripts/analisis/generar_fichas.py` a partir de los archivos de esta carpeta y de `results/registro.json`. No la edites a mano: corrige el registro y regenera.

## Identificación

| | |
|---|---|
| Carpeta | `results/v4/` |
| Tipo | no registrado |
| Fecha | 2026-11-23 (fecha del archivo results.json) |
| Carpeta anterior | no registrado |
| Relación con otras versiones | no registrado |
| Citada en | no |

## El modelo en esta corrida

| | |
|---|---|
| Modelo | `Qwen/Qwen3-8B` |
| Parámetros | 8.20 mil millones |
| Arquitectura | qwen3, 36 capas, dimensión 4096 |
| Precisión de carga | FP16, forzado con --no-quantize |
| Memoria ocupada | 16.4 GB |
| Posición de lectura | último token del prompt (cabecera del asistente), antes de generar |
| Capas analizadas | 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34 |
| Capa de referencia (máximo ΔP) | 32 |
| Capa por selección en muestra dividida | no registrado |

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
| Carpeta de salida original | `/workspace/out/corridas/qwen3_8b/v4` |

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
| ΔP dañinas (capa 32) | 178.206 [IC 95 % 161.558, 194.268] |
| Prueba t pareada | p = 1.34e-38, d_z = 2.20 |
| Control benigno | ΔP benignas 1.694; interacción 176.512 (p = 0.0001) |
| AUROC fuera de muestra | chat 0.913, agente 0.754 |
| PCA en la capa de referencia | cos(PC1, d) = 0.972, varianza PC1 25.6% |

Comportamiento sin intervención:

| Condición | Rechaza | Llama una herramienta |
|---|---|---|
| chat_harmful | 82/96 (85%) | 0/96 |
| agent_harmful | 22/96 (23%) | 66/96 |
| chat_benign | 37/96 (39%) | 0/96 |
| agent_benign | 1/96 (1%) | 89/96 |

Test causal (dosis en unidades del ΔP medido):

| Capa | Dosis | Restaura rechazo | Control aleatorio | Benignas rechazadas |
|---|---|---|---|---|
| 32 | ×0.5 | 8/71 | 6/71 | 0/93 |
| 32 | ×1 | 7/71 | 9/71 | 0/93 |
| 32 | ×2 | 14/71 | 7/71 | 2/93 |
| 32 | ×4 | 22/71 | 13/71 | 5/93 |

## Archivos

- `activations.npz`
- `agent_direction_results.json`
- `behavior_generations.jsonl`
- `causal_results.json`
- `directions.npz`
- `factorial_results.json`
- `format_effect_results.json`
- `rendered_inputs.jsonl`
- `results.json`
- `robust_results.json`
- `run.log`
- `run_manifest.json`
