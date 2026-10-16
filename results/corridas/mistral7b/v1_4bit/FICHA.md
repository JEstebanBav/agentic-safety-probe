# Ficha · Mistral-7B · V1 · 4 bits

> Generada por `scripts/generar_fichas.py` a partir de los archivos de esta carpeta y de `results/registro.json`. No la edites a mano: corrige el registro y regenera.

## Identificación

| | |
|---|---|
| Carpeta | `results/corridas/mistral7b/v1_4bit/` |
| Tipo | corrida complementaria |
| Fecha | ≤ 2026-09-22 (fecha del commit) |
| Carpeta anterior | `results/mistral7b_4bit` |
| Relación con otras versiones | Control de cuantización de V1: mismo modelo y prompts, solo cambia la precisión. |
| Citada en | no |

## El modelo en esta corrida

| | |
|---|---|
| Modelo | `mistralai/Mistral-7B-Instruct-v0.3` |
| Parámetros | 7.25 mil millones |
| Arquitectura | mistral, 32 capas, dimensión 4096 |
| Precisión de carga | 4-bit NF4 (bitsandbytes), según el log |
| Memoria ocupada | 4.0 GB |
| Posición de lectura | último token del prompt (cabecera del asistente), antes de generar |
| Capas analizadas | 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30 |
| Capa de referencia (máximo ΔP) | 16 |
| Capa por selección en muestra dividida | 16 |

## Corpus

| | |
|---|---|
| Archivo | dataset_full.jsonl (paired, 168 entries) |
| Pares | 42 |

## Parámetros de la corrida

Esta corrida es de una versión del pipeline que no guardaba su configuración en `results.json`.
Del encabezado del log:

- Modelo: `mistralai/Mistral-7B-Instruct-v0.3`
- Precisión: `4-bit (forzado)`
- Corpus: `dataset_full.jsonl (paired, 168 entries)`

Duración registrada: 1058.7s (17.6 min).

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
| ΔP dañinas (capa 16) | 1.747 [IC 95 % 1.414, 2.072] |
| Prueba t pareada | p = 5.25e-13, d_z = 1.56 |
| Control benigno | ΔP benignas -0.310; interacción 2.057 (p = 0.0000) |
| AUROC fuera de muestra | chat 0.956, agente 0.883 |

## Archivos

- `agent_direction_results.json`
- `agent_intervention_results.json`
- `directions.npz`
- `format_effect_results.json`
- `results.json`
- `robust_results.json`
- `run.log`
- `stratified_results.json`
