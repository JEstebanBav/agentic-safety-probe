# Ficha · Mistral-7B · V1 · estadística

> Generada por `scripts/generar_fichas.py` a partir de los archivos de esta carpeta y de `results/registro.json`. No la edites a mano: corrige el registro y regenera.

## Identificación

| | |
|---|---|
| Carpeta | `results/corridas/mistral7b/v1_estadistica/` |
| Tipo | corrida complementaria |
| Fecha | ≤ 2026-09-22 (fecha del commit) |
| Carpeta anterior | `results/mistral7b_stats` |
| Relación con otras versiones | Variante de V1; por su nombre original (mistral7b_stats), centrada en el análisis estadístico. |
| Citada en | no |

## El modelo en esta corrida

| | |
|---|---|
| Modelo | `mistralai/Mistral-7B-Instruct-v0.3` |
| Parámetros | 7.25 mil millones |
| Arquitectura | mistral, 32 capas, dimensión 4096 |
| Precisión de carga | FP16, según el log |
| Memoria ocupada | 14.5 GB |
| Posición de lectura | último token del prompt (cabecera del asistente), antes de generar |
| Capas analizadas | 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30 |
| Capa de referencia (máximo ΔP) | 18 |
| Capa por selección en muestra dividida | 18 |

## Corpus

| | |
|---|---|
| Archivo | dataset_full.jsonl (paired, 168 entries) |
| Pares | 42 |

## Parámetros de la corrida

Esta corrida es de una versión del pipeline que no guardaba su configuración en `results.json`.
Del encabezado del log:

- Modelo: `mistralai/Mistral-7B-Instruct-v0.3`
- Precisión: `False`
- Corpus: `dataset_full.jsonl (paired, 168 entries)`

Duración registrada: 1117.9s (18.6 min).

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
| ΔP dañinas (capa 18) | 2.602 [IC 95 % 2.085, 3.100] |
| Prueba t pareada | p = 9.72e-13, d_z = 1.53 |
| Control benigno | ΔP benignas -0.905; interacción 3.507 (p = 0.0000) |
| AUROC fuera de muestra | chat 0.943, agente 0.900 |

## Archivos

- `agent_direction_results.json`
- `agent_intervention_results.json`
- `directions.npz`
- `format_effect_results.json`
- `results.json`
- `robust_results.json`
- `run.log`
- `run.pid`
- `stratified_results.json`
