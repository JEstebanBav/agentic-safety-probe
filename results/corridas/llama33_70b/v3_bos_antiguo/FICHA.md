# Ficha · Llama-3.3-70B · V3 · token inicial antiguo

> Generada por `scripts/generar_fichas.py` a partir de los archivos de esta carpeta y de `results/registro.json`. No la edites a mano: corrige el registro y regenera.

## Identificación

| | |
|---|---|
| Carpeta | `results/corridas/llama33_70b/v3_bos_antiguo/` |
| Tipo | corrida complementaria |
| Fecha | 2026-10-15 (marca de tiempo local del archivo) |
| Carpeta anterior | `results/llama33_final/results_llama33_bos_antiguo_v2` |
| Relación con otras versiones | Sensibilidad al token inicial: misma extracción que V3 con PROBE_LEGACY_BOS=1 (dos BOS, como V1). |
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
| agent_benign | 593 | 413 | 1060 |
| agent_harmful | 596 | 410 | 1061 |
| chat_benign | 87 | 76 | 102 |
| chat_harmful | 89 | 77 | 107 |

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
| Factorial sistema × herramientas | `False` |
| Nivel de agencia | valor por defecto |
| Caja de herramientas | `0` |
| Historial de herramientas | `False` |
| Folds | `5` |
| Repeticiones sonda lineal | `1` |
| Repeticiones sonda MLP | valor por defecto |
| Permutaciones de etiquetas | `10` |
| Remuestreos bootstrap | `10000` |
| Particiones para selección de capa | `0` |
| Sondas omitidas | `True` |
| Generación omitida | `True` |
| Capas del test causal | valor por defecto |
| Dosis del test causal | `0.5,1,2,4` |
| Pares del test causal | valor por defecto |
| Tokens de respuesta | `64` |
| Semilla | `42` |
| Carpeta de salida original | `/workspace/results_llama33_bos_antiguo_v2` |

Duración registrada: 323.4s (5.4 min).

## Etapas ejecutadas

- [x] Activaciones guardadas (activations.npz, solo local)
- [x] Dirección de rechazo y ΔP por capa
- [x] Validación PCA del contraste
- [x] Evaluación fuera de muestra (held-out)
- [ ] Sondas lineales con permutación
- [ ] Sondas MLP
- [x] Análisis robusto y selección de capa
- [ ] Factorial sistema × herramientas
- [ ] Línea base de comportamiento
- [ ] Test causal calibrado (d_chat vs. aleatoria)
- [ ] Intervención antigua con α arbitrarios (no calibrada)

## Resultados clave

| Medida | Valor |
|---|---|
| ΔP dañinas (capa 76) | 15.248 [IC 95 % 13.319, 17.053] |
| Prueba t pareada | p = 2.17e-19, d_z = 2.45 |
| Control benigno | ΔP benignas -3.198; interacción 18.446 (p = 0.0001) |
| AUROC fuera de muestra | chat 0.977, agente 0.922 |
| PCA en la capa de referencia | cos(PC1, d) = 0.991, varianza PC1 20.4% |

## Notas

- La configuración guardada no registra PROBE_LEGACY_BOS porque es una variable de entorno; el modo se deduce del nombre original de la carpeta.

## Archivos

- `activations.npz`
- `agent_direction_results.json`
- `directions.npz`
- `format_effect_results.json`
- `results.json`
- `robust_results.json`
- `run.log`
- `stratified_results.json`
