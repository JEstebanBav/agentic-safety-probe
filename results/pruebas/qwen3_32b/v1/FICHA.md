# Ficha · Qwen3-32B · prueba

> Generada por `scripts/generar_fichas.py` a partir de los archivos de esta carpeta y de `results/registro.json`. No la edites a mano: corrige el registro y regenera.

## Identificación

| | |
|---|---|
| Carpeta | `results/pruebas/qwen3_32b/v1/` |
| Tipo | corrida de prueba (4 pares) |
| Fecha | ≤ 2026-09-22 (fecha del commit) |
| Carpeta anterior | `results/dryrun/qwen32b` |
| Relación con otras versiones | Comprobación del pipeline, época de V1. Sus estadísticas no significan nada. |
| Citada en | no |

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
| ΔP dañinas (capa 46) | 79.649 [IC 95 % 65.114, 92.748] |
| Prueba t pareada | p = 1.34e-03, d_z = 4.62 |
| Control benigno | ΔP benignas -0.331; interacción 79.980 (p = 0.0270) |
| AUROC fuera de muestra | chat 1.000, agente 0.875 |

## Archivos

- `agent_direction_results.json`
- `directions.npz`
- `format_effect_results.json`
- `results.json`
- `robust_results.json`
- `stratified_results.json`
