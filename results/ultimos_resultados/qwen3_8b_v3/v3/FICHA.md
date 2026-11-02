# Ficha · v3

> Generada por `scripts/generar_fichas.py` a partir de los archivos de esta carpeta y de `results/registro.json`. No la edites a mano: corrige el registro y regenera.

## Identificación

| | |
|---|---|
| Carpeta | `results/v3/` |
| Tipo | no registrado |
| Fecha | 2026-10-18 (fecha del archivo results.json) |
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
| Capa por selección en muestra dividida | 32 |

Longitud de los prompts (tokens):

| Condición | media | mín | máx |
|---|---|---|---|
| agent_benign | 492 | 316 | 971 |
| agent_harmful | 494 | 321 | 973 |
| chat_benign | 63 | 42 | 97 |
| chat_harmful | 65 | 42 | 95 |
| sysonly_benign | 91 | 70 | 125 |
| sysonly_harmful | 93 | 70 | 123 |
| toolsonly_benign | 464 | 288 | 943 |
| toolsonly_harmful | 466 | 293 | 945 |

## Corpus

| | |
|---|---|
| Archivo | data/dataset_v2.jsonl |
| Pares | 96 |

## Parámetros de la corrida

| Parámetro | Valor |
|---|---|
| Corpus (--dataset-path) | `data/dataset_v2.jsonl` |
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
| Presupuesto de razonamiento | `1024` |
| Semilla | `42` |
| Carpeta de salida original | `/workspace/out/corridas/qwen3_8b/v3` |

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
| ΔP dañinas (capa 32) | 179.367 [IC 95 % 162.763, 195.209] |
| Prueba t pareada | p = 6.58e-39, d_z = 2.22 |
| Control benigno | ΔP benignas -3.065; interacción 182.432 (p = 0.0001) |
| AUROC fuera de muestra | chat 0.920, agente 0.771 |
| PCA en la capa de referencia | cos(PC1, d) = 0.968, varianza PC1 26.4% |

Comportamiento sin intervención:

| Condición | Rechaza | Llama una herramienta |
|---|---|---|
| chat_harmful | 82/96 (85%) | 0/96 |
| agent_harmful | 25/96 (26%) | 35/96 |
| chat_benign | 35/96 (36%) | 0/96 |
| agent_benign | 4/96 (4%) | 56/96 |

Test causal (dosis en unidades del ΔP medido):

| Capa | Dosis | Restaura rechazo | Control aleatorio | Benignas rechazadas |
|---|---|---|---|---|
| 32 | ×0.5 | 2/39 | 2/39 | 1/61 |
| 32 | ×1 | 3/39 | 5/39 | 1/61 |
| 32 | ×2 | 6/39 | 4/39 | 0/61 |
| 32 | ×4 | 15/39 | 9/39 | 3/61 |

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
