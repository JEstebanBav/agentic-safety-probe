# Ficha · v4

> Generada por `scripts/generar_fichas.py` a partir de los archivos de esta carpeta y de `results/registro.json`. No la edites a mano: corrige el registro y regenera.

## Identificación

| | |
|---|---|
| Carpeta | `results/v4/` |
| Tipo | no registrado |
| Fecha | 2026-10-17 (fecha del archivo results.json) |
| Carpeta anterior | no registrado |
| Relación con otras versiones | no registrado |
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
| agent_benign | 605 | 377 | 1244 |
| agent_harmful | 607 | 382 | 1246 |
| chat_benign | 85 | 64 | 117 |
| chat_harmful | 87 | 64 | 115 |
| sysonly_benign | 113 | 92 | 145 |
| sysonly_harmful | 115 | 92 | 143 |
| toolsonly_benign | 577 | 349 | 1216 |
| toolsonly_harmful | 579 | 354 | 1218 |

## Corpus

| | |
|---|---|
| Archivo | data/dataset_v2.jsonl |
| Pares | 96 |

## Parámetros de la corrida

| Parámetro | Valor |
|---|---|
| Corpus (--dataset-path) | `data/dataset_v2.jsonl` |
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
| Presupuesto de razonamiento | `1024` |
| Semilla | `42` |
| Carpeta de salida original | `/workspace/out/corridas/llama33_70b/v4` |

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
| ΔP dañinas (capa 76) | 15.486 [IC 95 % 14.078, 16.838] |
| Prueba t pareada | p = 7.89e-40, d_z = 2.28 |
| Control benigno | ΔP benignas -2.104; interacción 17.590 (p = 0.0001) |
| AUROC fuera de muestra | chat 0.968, agente 0.869 |
| PCA en la capa de referencia | cos(PC1, d) = 0.990, varianza PC1 18.7% |

Comportamiento sin intervención:

| Condición | Rechaza | Llama una herramienta |
|---|---|---|
| chat_harmful | 82/96 (85%) | 0/96 |
| agent_harmful | 43/96 (45%) | 51/96 |
| chat_benign | 8/96 (8%) | 0/96 |
| agent_benign | 2/96 (2%) | 93/96 |

Test causal (dosis en unidades del ΔP medido):

| Capa | Dosis | Restaura rechazo | Control aleatorio | Benignas rechazadas |
|---|---|---|---|---|
| 76 | ×0.5 | 7/53 | 0/53 | 0/94 |
| 76 | ×1 | 21/53 | 2/53 | 6/94 |
| 76 | ×2 | 36/53 | 1/53 | 49/94 |
| 76 | ×4 | 0/53 | 1/53 | 1/94 |
| 56 | ×0.5 | 8/53 | 2/53 | 0/94 |
| 56 | ×1 | 25/53 | 2/53 | 7/94 |
| 56 | ×2 | 51/53 | 2/53 | 91/94 |
| 56 | ×4 | 53/53 | 2/53 | 75/94 |
| 28 | ×0.5 | 3/53 | 0/53 | 0/94 |
| 28 | ×1 | 3/53 | 1/53 | 1/94 |
| 28 | ×2 | 13/53 | 2/53 | 2/94 |
| 28 | ×4 | 45/53 | 24/53 | 53/94 |

## Archivos

- `activations.npz`
- `agent_direction_results.json`
- `behavior_generations.jsonl`
- `causal_results.json`
- `directions.npz`
- `factorial_results.json`
- `format_effect_results.json`
- `results.json`
- `robust_results.json`
- `run.log`
- `stratified_results.json`
