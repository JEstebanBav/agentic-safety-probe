# Ficha · Qwen3-8B · V1

> Generada por `scripts/generar_fichas.py` a partir de los archivos de esta carpeta y de `results/registro.json`. No la edites a mano: corrige el registro y regenera.

## Identificación

| | |
|---|---|
| Carpeta | `results/corridas/qwen3_8b/v1/` |
| Tipo | corrida principal |
| Fecha | ≤ 2026-09-22 (fecha del commit) |
| Carpeta anterior | `results/qwen8b` |
| Relación con otras versiones | Corrida FP16 con el pipeline original. Su log está compartido con Mistral V1 FP16 en logs/cola.log. |
| Citada en | Tesis (avances): AUROC fuera de muestra |

## El modelo en esta corrida

| | |
|---|---|
| Modelo | `Qwen/Qwen3-8B` |
| Parámetros | 8.20 mil millones |
| Arquitectura | qwen3, 36 capas, dimensión 4096 |
| Precisión de carga | FP16 — deducida: ocupaba 16.4 GB para 8.2 mil millones de parámetros |
| Memoria ocupada | 16.4 GB |
| Posición de lectura | último token del prompt (cabecera del asistente), antes de generar |
| Capas analizadas | 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34 |
| Capa de referencia (máximo ΔP) | 32 |
| Capa por selección en muestra dividida | 22 |

## Corpus

| | |
|---|---|
| Archivo | data/dataset_full.jsonl (por defecto) |
| Pares | 42 |

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
| ΔP dañinas (capa 32) | 50.549 [IC 95 % 17.746, 90.826] |
| Prueba t pareada | p = 6.82e-03, d_z = 0.40 |
| Control benigno | ΔP benignas 15.929; interacción 34.619 (p = 0.1182) |
| AUROC fuera de muestra | chat 0.518, agente 0.511 |

## Archivos

- `agent_direction_results.json`
- `agent_intervention_results.json`
- `directions.npz`
- `format_effect_results.json`
- `results.json`
- `robust_results.json`
- `stratified_results.json`
