# Ficha · Qwen3-32B · V2 (96 pares)

> Generada por `scripts/generar_fichas.py` a partir de los archivos de esta carpeta y de `results/registro.json`. No la edites a mano: corrige el registro y regenera.

## Identificación

| | |
|---|---|
| Carpeta | `results/corridas/qwen3_32b/v2/` |
| Tipo | corrida principal |
| Fecha | no registrada (carpeta no estaba en git) |
| Carpeta anterior | `results/qwen32b_v2` |
| Relación con otras versiones | Primera corrida de Qwen3-32B con el corpus ampliado. |
| Citada en | Sustentación (comparativa de 96 pares, capa 46) |

## El modelo en esta corrida

| | |
|---|---|
| Modelo | `Qwen/Qwen3-32B` |
| Parámetros | 32.80 mil millones |
| Arquitectura | qwen3, 64 capas, dimensión 5120 |
| Precisión de carga | 4-bit NF4 (bitsandbytes) — deducida: ocupaba 18.7 GB para 32.8 mil millones de parámetros |
| Memoria ocupada | 18.7 GB |
| Posición de lectura | último token del prompt (cabecera del asistente), antes de generar |
| Capas analizadas | 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62 |
| Capa de referencia (máximo ΔP) | 46 |
| Capa por selección en muestra dividida | 46 |

## Corpus

| | |
|---|---|
| Archivo | data/dataset_v2.jsonl — deducido: 96 pares, y la sustentación la identifica como V2 de 96 |
| Pares | 96 |

## Parámetros de la corrida

Esta corrida es de una versión del pipeline que no guardaba su configuración en `results.json`.

## Etapas ejecutadas

- [ ] Activaciones guardadas (activations.npz, solo local)
- [x] Dirección de rechazo y ΔP por capa
- [ ] Validación PCA del contraste
- [x] Evaluación fuera de muestra (held-out)
- [x] Sondas lineales con permutación
- [x] Sondas MLP
- [x] Análisis robusto y selección de capa
- [ ] Factorial sistema × herramientas
- [ ] Línea base de comportamiento
- [ ] Test causal calibrado (d_chat vs. aleatoria)
- [x] Intervención antigua con α arbitrarios (no calibrada)

## Resultados clave

| Medida | Valor |
|---|---|
| ΔP dañinas (capa 46) | 71.897 [IC 95 % 66.737, 76.801] |
| Prueba t pareada | p = 0.00e+00, d_z = 2.80 |
| Control benigno | ΔP benignas 11.959; interacción 59.938 (p = 0.0000) |
| AUROC fuera de muestra | chat 0.947, agente 0.911 |

## Notas

- Cargado en 4 bits (memoria ocupada 18,7 GB). Los demás modelos de la comparativa están en FP16: es una diferencia a declarar.
- Versión anterior del pipeline: sin línea base de comportamiento, test causal calibrado, factorial ni PCA, y sin activations.npz.

## Archivos

- `agent_direction_results.json`
- `agent_intervention_results.json`
- `directions.npz`
- `format_effect_results.json`
- `results.json`
- `robust_results.json`
- `stratified_results.json`
