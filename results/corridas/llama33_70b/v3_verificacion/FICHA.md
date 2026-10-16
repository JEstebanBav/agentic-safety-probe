# Ficha · Llama-3.3-70B · V3 · verificación

> Generada por `scripts/generar_fichas.py` a partir de los archivos de esta carpeta y de `results/registro.json`. No la edites a mano: corrige el registro y regenera.

## Identificación

| | |
|---|---|
| Carpeta | `results/corridas/llama33_70b/v3_verificacion/` |
| Tipo | corrida complementaria |
| Fecha | 2026-10-15 (marca de tiempo local del archivo) |
| Carpeta anterior | `results/llama33_final/results_llama33_check` |
| Relación con otras versiones | Hecha justo después de V3: extracción en las capas 28, 52 y 56 sin sondas ni generación, para comprobar las activaciones. |
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
| Capas analizadas | 28, 52, 56 |
| Capa de referencia (máximo ΔP) | 56 |
| Capa por selección en muestra dividida | 56 |

Longitud de los prompts (tokens):

| Condición | media | mín | máx |
|---|---|---|---|
| agent_benign | 592 | 412 | 1059 |
| agent_harmful | 595 | 409 | 1060 |
| chat_benign | 86 | 75 | 101 |
| chat_harmful | 88 | 76 | 106 |

## Corpus

| | |
|---|---|
| Archivo | data/dataset_full.jsonl (paired) |
| Pares | 42 |

## Parámetros de la corrida

| Parámetro | Valor |
|---|---|
| Corpus (--dataset-path) | valor por defecto |
| Capas extraídas (--layers) | `28,52,56` |
| Límite de pares (--limit-pairs) | valor por defecto |
| Factorial sistema × herramientas | `False` |
| Nivel de agencia | valor por defecto |
| Caja de herramientas | `0` |
| Historial de herramientas | `False` |
| Folds | `5` |
| Repeticiones sonda lineal | `1` |
| Repeticiones sonda MLP | valor por defecto |
| Permutaciones de etiquetas | `10` |
| Remuestreos bootstrap | `1000` |
| Particiones para selección de capa | `0` |
| Sondas omitidas | `True` |
| Generación omitida | `True` |
| Capas del test causal | valor por defecto |
| Dosis del test causal | `0.5,1,2,4` |
| Pares del test causal | valor por defecto |
| Tokens de respuesta | `64` |
| Semilla | `42` |
| Carpeta de salida original | `/workspace/results_llama33_check` |

Duración registrada: 226.2s (3.8 min).

## Etapas ejecutadas

- [x] Activaciones guardadas (activations.npz, solo local)
- [x] Dirección de rechazo y ΔP por capa
- [x] Validación PCA del contraste
- [ ] Evaluación fuera de muestra (held-out)
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
| ΔP dañinas (capa 56) | 6.926 [IC 95 % 6.003, 7.748] |
| Prueba t pareada | p = 9.93e-19, d_z = 2.34 |
| Control benigno | ΔP benignas -2.664; interacción 9.590 (p = 0.0001) |
| PCA en la capa de referencia | cos(PC1, d) = 0.994, varianza PC1 25.9% |

## Archivos

- `activations.npz`
- `agent_direction_results.json`
- `directions.npz`
- `format_effect_results.json`
- `results.json`
- `robust_results.json`
- `run.log`
- `stratified_results.json`
