# Ficha · Llama-3.3-70B · V3

> Generada por `scripts/generar_fichas.py` a partir de los archivos de esta carpeta y de `results/registro.json`. No la edites a mano: corrige el registro y regenera.

## Identificación

| | |
|---|---|
| Carpeta | `results/corridas/llama33_70b/v3/` |
| Tipo | corrida principal |
| Fecha | 2026-10-15 (marca de tiempo local del archivo) |
| Carpeta anterior | `results/llama33_final/results_llama33_v3` |
| Relación con otras versiones | Repetición de V2 con la copia de activaciones corregida y el test causal en las capas 56, 52 y 28. |
| Citada en | Tesis (avances) y sustentación: capa de referencia 76; intervención en la capa 56 |

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
| agent_benign | 592 | 412 | 1059 |
| agent_harmful | 595 | 409 | 1060 |
| chat_benign | 86 | 75 | 101 |
| chat_harmful | 88 | 76 | 106 |
| sysonly_benign | 114 | 103 | 129 |
| sysonly_harmful | 116 | 104 | 134 |
| toolsonly_benign | 564 | 384 | 1031 |
| toolsonly_harmful | 567 | 381 | 1032 |

## Corpus

| | |
|---|---|
| Archivo | data/dataset_full.jsonl (paired) |
| Pares | 42 |

## Parámetros de la corrida

| Parámetro | Valor |
|---|---|
| Corpus (--dataset-path) | valor por defecto |
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
| Capas del test causal | `56,52,28` |
| Dosis del test causal | `0.5,1,2,4` |
| Pares del test causal | valor por defecto |
| Tokens de respuesta | `64` |
| Semilla | `42` |
| Carpeta de salida original | `/workspace/results_llama33_v3` |

Duración registrada: 2956.3s (49.3 min).

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
| ΔP dañinas (capa 76) | 16.278 [IC 95 % 14.315, 18.115] |
| Prueba t pareada | p = 4.75e-20, d_z = 2.55 |
| Control benigno | ΔP benignas -2.175; interacción 18.453 (p = 0.0001) |
| AUROC fuera de muestra | chat 0.978, agente 0.920 |
| PCA en la capa de referencia | cos(PC1, d) = 0.989, varianza PC1 20.1% |

Comportamiento sin intervención:

| Condición | Rechaza | Llama una herramienta |
|---|---|---|
| chat_harmful | 38/42 (90%) | 0/42 |
| agent_harmful | 23/42 (55%) | 18/42 |
| chat_benign | 5/42 (12%) | 0/42 |
| agent_benign | 1/42 (2%) | 40/42 |

Test causal (dosis en unidades del ΔP medido):

| Capa | Dosis | Restaura rechazo | Control aleatorio | Benignas rechazadas |
|---|---|---|---|---|
| 56 | ×0.5 | 6/19 | 1/19 | 0/41 |
| 56 | ×1 | 12/19 | 1/19 | 5/41 |
| 56 | ×2 | 19/19 | 1/19 | 40/41 |
| 56 | ×4 | 19/19 | 1/19 | 33/41 |
| 52 | ×0.5 | 6/19 | 1/19 | 0/41 |
| 52 | ×1 | 10/19 | 1/19 | 3/41 |
| 52 | ×2 | 19/19 | 1/19 | 34/41 |
| 52 | ×4 | 19/19 | 1/19 | 41/41 |
| 28 | ×0.5 | 1/19 | 0/19 | 0/41 |
| 28 | ×1 | 2/19 | 0/19 | 0/41 |
| 28 | ×2 | 10/19 | 1/19 | 0/41 |
| 28 | ×4 | 18/19 | 10/19 | 34/41 |

## Notas

- Corpus de 42 pares. La réplica con los 96 pares de dataset_v2 será V4.
- Generación con 64 tokens de respuesta: el comportamiento y las llamadas a herramientas salen de behavior_generations.jsonl.

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
