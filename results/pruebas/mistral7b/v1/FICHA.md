# Ficha · Mistral-7B · prueba

> Generada por `scripts/generar_fichas.py` a partir de los archivos de esta carpeta y de `results/registro.json`. No la edites a mano: corrige el registro y regenera.

## Identificación

| | |
|---|---|
| Carpeta | `results/pruebas/mistral7b/v1/` |
| Tipo | corrida de prueba (4 pares) |
| Fecha | ≤ 2026-09-22 (fecha del commit) |
| Carpeta anterior | `results/dryrun/mistral7b` |
| Relación con otras versiones | Comprobación del pipeline, época de V1. Sus estadísticas no significan nada. |
| Citada en | no |

## El modelo en esta corrida

| | |
|---|---|
| Modelo | `mistralai/Mistral-7B-Instruct-v0.3` |
| Parámetros | 7.25 mil millones |
| Arquitectura | mistral, 32 capas, dimensión 4096 |
| Precisión de carga | FP16 — deducida: ocupaba 14.5 GB para 7.25 mil millones de parámetros |
| Memoria ocupada | 14.5 GB |
| Posición de lectura | último token del prompt (cabecera del asistente), antes de generar |
| Capas analizadas | 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30 |
| Capa de referencia (máximo ΔP) | 18 |
| Capa por selección en muestra dividida | 18 |

## Corpus

| | |
|---|---|
| Archivo | data/dataset_full.jsonl (por defecto) |
| Pares | 4 |

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
- [ ] Intervención antigua con α arbitrarios (no calibrada)

## Resultados clave

| Medida | Valor |
|---|---|
| ΔP dañinas (capa 18) | 1.834 [IC 95 % 0.687, 2.624] |
| Prueba t pareada | p = 2.73e-02, d_z = 1.53 |
| Control benigno | ΔP benignas -1.563; interacción 3.397 (p = 0.0270) |
| AUROC fuera de muestra | chat 1.000, agente 0.875 |

## Archivos

- `agent_direction_results.json`
- `directions.npz`
- `format_effect_results.json`
- `results.json`
- `robust_results.json`
- `stratified_results.json`
