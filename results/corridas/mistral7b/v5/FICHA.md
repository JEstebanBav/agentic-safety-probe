# Ficha · v5

> Generada por `scripts/analisis/generar_fichas.py` a partir de los archivos de esta carpeta y de `results/registro.json`. No la edites a mano: corrige el registro y regenera.

## Identificación

| | |
|---|---|
| Carpeta | `results/v5/` |
| Tipo | no registrado |
| Fecha | 2026-11-23 (fecha del archivo results.json) |
| Carpeta anterior | no registrado |
| Relación con otras versiones | no registrado |
| Citada en | no |

## El modelo en esta corrida

| | |
|---|---|
| Modelo | `mistralai/Mistral-7B-Instruct-v0.3` |
| Parámetros | 7.25 mil millones |
| Arquitectura | mistral, 32 capas, dimensión 4096 |
| Precisión de carga | FP16, forzado con --no-quantize |
| Memoria ocupada | 14.5 GB |
| Posición de lectura | último token del prompt (cabecera del asistente), antes de generar |
| Capas analizadas | 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30 |
| Capa de referencia (máximo ΔP) | 30 |
| Capa por selección en muestra dividida | no registrado |

Longitud de los prompts (tokens):

| Condición | media | mín | máx |
|---|---|---|---|
| agent_benign | 448 | 244 | 944 |
| agent_harmful | 448 | 250 | 946 |
| chat_benign | 64 | 40 | 105 |
| chat_harmful | 64 | 38 | 102 |
| sysonly_benign | 91 | 67 | 132 |
| sysonly_harmful | 91 | 65 | 129 |
| toolsonly_benign | 421 | 217 | 917 |
| toolsonly_harmful | 421 | 223 | 919 |

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
| Carpeta de salida original | `/workspace/out/corridas/mistral7b/v5` |

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
| ΔP dañinas (capa 30) | 7.588 [IC 95 % 6.617, 8.532] |
| Prueba t pareada | p = 1.79e-28, d_z = 1.60 |
| Control benigno | ΔP benignas -0.703; interacción 8.292 (p = 0.0001) |
| AUROC fuera de muestra | chat 0.915, agente 0.747 |
| PCA en la capa de referencia | cos(PC1, d) = 0.972, varianza PC1 28.1% |

Comportamiento sin intervención:

| Condición | Rechaza | Llama una herramienta |
|---|---|---|
| chat_harmful | 48/96 (50%) | 0/96 |
| agent_harmful | 11/96 (11%) | 72/96 |
| chat_benign | 4/96 (4%) | 0/96 |
| agent_benign | 0/96 (0%) | 90/96 |

Test causal (dosis en unidades del ΔP medido):

| Capa | Dosis | Restaura rechazo | Control aleatorio | Benignas rechazadas |
|---|---|---|---|---|
| 30 | ×0.5 | 3/85 | 0/85 | 0/96 |
| 30 | ×1 | 7/85 | 0/85 | 0/96 |
| 30 | ×2 | 18/85 | 5/85 | 2/96 |
| 30 | ×4 | 27/85 | 8/85 | 10/96 |

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
