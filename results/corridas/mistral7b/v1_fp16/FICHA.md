# Ficha · Mistral-7B · V1 · FP16

> Generada por `scripts/generar_fichas.py` a partir de los archivos de esta carpeta y de `results/registro.json`. No la edites a mano: corrige el registro y regenera.

## Identificación

| | |
|---|---|
| Carpeta | `results/corridas/mistral7b/v1_fp16/` |
| Tipo | corrida principal |
| Fecha | ≤ 2026-09-22 (fecha del commit) |
| Carpeta anterior | `results/mistral7b_fp16` |
| Relación con otras versiones | Corrida FP16 con el pipeline original. Su log está compartido con Qwen3-8B V1 en logs/cola.log. |
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
| Capa de referencia (máximo ΔP) | 16 |
| Capa por selección en muestra dividida | 16 |

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
| ΔP dañinas (capa 16) | 1.674 [IC 95 % 1.343, 2.000] |
| Prueba t pareada | p = 1.30e-12, d_z = 1.51 |
| Control benigno | ΔP benignas -0.416; interacción 2.089 (p = 0.0000) |
| AUROC fuera de muestra | chat 0.946, agente 0.885 |

## Notas

- Intervención antigua con α arbitrarios: no es el test causal calibrado.

## Archivos

- `agent_direction_results.json`
- `agent_intervention_results.json`
- `directions.npz`
- `format_effect_results.json`
- `results.json`
- `robust_results.json`
- `stratified_results.json`
