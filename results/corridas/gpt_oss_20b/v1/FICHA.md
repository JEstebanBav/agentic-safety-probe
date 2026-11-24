# Ficha · gpt-oss-20b · V1 (corpus v0.3)

> Generada por `scripts/analisis/generar_fichas.py` a partir de los archivos de esta carpeta y de `results/registro.json`. No la edites a mano: corrige el registro y regenera.

## Identificación

| | |
|---|---|
| Carpeta | `results/corridas/gpt_oss_20b/v1/` |
| Tipo | corrida principal |
| Fecha | 2026-11-24 (inicio de la corrida, UTC, según run_manifest.json) |
| Carpeta anterior | `/workspace/out/corridas/gpt_oss_20b/v1 (pod RunPod 1× H200)` |
| Relación con otras versiones | Primera corrida del modelo. Corpus v0.3; cálculo en BF16 con los expertos en su formato nativo MXFP4. |
| Citada en | no |

## El modelo en esta corrida

| | |
|---|---|
| Modelo | `openai/gpt-oss-20b` |
| Parámetros | 20.90 mil millones |
| Arquitectura | gpt_oss, 24 capas, dimensión 2880 |
| Precisión de carga | BF16, forzado con --no-quantize (pesos nativos MXFP4) |
| Memoria ocupada | 3.6 GB sin contar los pesos que el kernel cuantizado guarda fuera de los parámetros (en gpt-oss, los expertos MXFP4): la medida de esta corrida no los ve, así que el modelo ocupaba más |
| Posición de lectura | último token del prompt (cabecera del asistente), antes de generar |
| Capas analizadas | 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22 |
| Capa de referencia (máximo ΔP) | 16 |
| Capa por selección en muestra dividida | 16 |

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
| Carpeta de salida original | `/workspace/out/corridas/gpt_oss_20b/v1` |

Duración registrada: 10662.8s (177.7 min).

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
| ΔP dañinas (capa 16) | 109.657 [IC 95 % 91.095, 128.588] |
| Prueba t pareada | p = 5.16e-20, d_z = 1.17 |
| Control benigno | ΔP benignas 103.338; interacción 6.319 (p = 0.5305) |
| AUROC fuera de muestra | chat 0.895, agente 0.900 |
| PCA en la capa de referencia | cos(PC1, d) = 0.877, varianza PC1 19.0% |

Comportamiento sin intervención:

| Condición | Rechaza | Llama una herramienta |
|---|---|---|
| chat_harmful | 91/96 (95%) | 0/96 |
| agent_harmful | 84/96 (88%) | 9/96 |
| chat_benign | 51/96 (53%) | 1/96 |
| agent_benign | 12/96 (12%) | 66/96 |

Test causal (dosis en unidades del ΔP medido):

| Capa | Dosis | Restaura rechazo | Control aleatorio | Benignas rechazadas |
|---|---|---|---|---|
| 16 | ×0.5 | 3/10 | 1/10 | 8/80 |
| 16 | ×1 | 3/10 | 1/10 | 8/80 |
| 16 | ×2 | 3/10 | 3/10 | 9/80 |
| 16 | ×4 | 5/10 | 1/10 | 16/80 |

## Notas

- Corpus v0.3 (data/v03): 96 pares en 67 grupos temáticos —todas las particiones dejan fuera grupos completos— y catálogos de herramientas neutrales.
- Razona antes de responder: hasta 4096 tokens de razonamiento, con corte de bucles (formato harmony). Línea base: 27/384 respuestas incompletas (27 cortadas por bucle), etiquetadas aparte de las negativas.
- La selección de capa en muestra dividida falló en el pod por un error con los grupos temáticos (corregido en 09626bc). Se recalculó en el portátil desde activations.npz con scripts/analisis/rehacer_analisis_robusto.py, que solo añade robust.layer_selection: ver robust_recomputed en results.json y rehacer_robusto.log.
- El test causal corrió en la capa de máximo ΔP (16) porque en el pod la selección no estaba disponible; la selección recalculada elige la misma capa, así que el test quedó en la capa pre-registrada.
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
