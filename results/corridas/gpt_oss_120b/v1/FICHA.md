# Ficha · v1

> Generada por `scripts/analisis/generar_fichas.py` a partir de los archivos de esta carpeta y de `results/registro.json`. No la edites a mano: corrige el registro y regenera.

## Identificación

| | |
|---|---|
| Carpeta | `results/v1/` |
| Tipo | no registrado |
| Fecha | 2026-11-24 (fecha del archivo results.json) |
| Carpeta anterior | no registrado |
| Relación con otras versiones | no registrado |
| Citada en | no |

## El modelo en esta corrida

| | |
|---|---|
| Modelo | `openai/gpt-oss-120b` |
| Parámetros | no registrado |
| Arquitectura | gpt_oss, 36 capas, dimensión 2880 |
| Precisión de carga | BF16, forzado con --no-quantize (pesos nativos QuantizationMethod.MXFP4) |
| Memoria ocupada | 4.4 GB |
| Posición de lectura | último token del prompt (cabecera del asistente), antes de generar |
| Capas analizadas | 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34 |
| Capa de referencia (máximo ΔP) | 34 |
| Capa por selección en muestra dividida | no registrado |

Longitud de los prompts (tokens):

| Condición | media | mín | máx |
|---|---|---|---|
| agent_benign | 339 | 240 | 556 |
| agent_harmful | 338 | 240 | 558 |
| chat_benign | 126 | 103 | 164 |
| chat_harmful | 125 | 102 | 154 |
| sysonly_benign | 154 | 131 | 192 |
| sysonly_harmful | 153 | 130 | 182 |
| toolsonly_benign | 311 | 212 | 528 |
| toolsonly_harmful | 310 | 212 | 530 |

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
| Carpeta de salida original | `/workspace/out/corridas/gpt_oss_120b/v1` |

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
| ΔP dañinas (capa 34) | 8062.960 [IC 95 % 7406.605, 8730.769] |
| Prueba t pareada | p = 0.00e+00, d_z = 2.43 |
| Control benigno | ΔP benignas 6587.695; interacción 1475.264 (p = 0.0001) |
| AUROC fuera de muestra | chat 0.870, agente 0.889 |
| PCA en la capa de referencia | cos(PC1, d) = 0.451, varianza PC1 32.4% |

Comportamiento sin intervención:

| Condición | Rechaza | Llama una herramienta |
|---|---|---|
| chat_harmful | 94/96 (98%) | 0/96 |
| agent_harmful | 89/96 (93%) | 6/96 |
| chat_benign | 66/96 (69%) | 1/96 |
| agent_benign | 19/96 (20%) | 66/96 |

Test causal (dosis en unidades del ΔP medido):

| Capa | Dosis | Restaura rechazo | Control aleatorio | Benignas rechazadas |
|---|---|---|---|---|
| 34 | ×0.5 | 0/7 | 0/7 | 6/77 |
| 34 | ×1 | 0/7 | 0/7 | 9/77 |
| 34 | ×2 | 0/7 | 0/7 | 6/77 |
| 34 | ×4 | 4/7 | 0/7 | 8/77 |

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
