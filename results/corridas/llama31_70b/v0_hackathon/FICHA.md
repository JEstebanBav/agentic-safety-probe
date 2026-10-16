# Ficha · Llama-3.1-70B · hackathon

> Generada por `scripts/generar_fichas.py` a partir de los archivos de esta carpeta y de `results/registro.json`. No la edites a mano: corrige el registro y regenera.

## Identificación

| | |
|---|---|
| Carpeta | `results/corridas/llama31_70b/v0_hackathon/` |
| Tipo | corrida exploratoria |
| Fecha | ≤ 2026-09-22 (fecha del commit) |
| Carpeta anterior | `results/llama31_hackathon` |
| Relación con otras versiones | Anterior al corpus pareado de la tesis. La usa app.py (la demo). |
| Citada en | no |

## El modelo en esta corrida

| | |
|---|---|
| Modelo | `meta-llama/Llama-3.1-70B-Instruct` |
| Parámetros | 70.60 mil millones |
| Arquitectura | llama, 80 capas, dimensión 8192 |
| Precisión de carga | FP16 — deducida: ocupaba 141.1 GB para 70.6 mil millones de parámetros |
| Memoria ocupada | 141.1 GB |
| Posición de lectura | último token del prompt (cabecera del asistente), antes de generar |
| Capas analizadas | 1, 5, 7, 10, 15, 20, 25 |
| Capa de referencia (máximo ΔP) | 20 |
| Capa por selección en muestra dividida | no registrado |

## Corpus

| | |
|---|---|
| Archivo | data/dataset_full.jsonl (por defecto) |
| Pares | 511 |

## Parámetros de la corrida

Esta corrida es de una versión del pipeline que no guardaba su configuración en `results.json`.

## Etapas ejecutadas

- [ ] Activaciones guardadas (activations.npz, solo local)
- [x] Dirección de rechazo y ΔP por capa
- [ ] Validación PCA del contraste
- [ ] Evaluación fuera de muestra (held-out)
- [x] Sondas lineales con permutación
- [x] Sondas MLP
- [ ] Análisis robusto y selección de capa
- [ ] Factorial sistema × herramientas
- [ ] Línea base de comportamiento
- [ ] Test causal calibrado (d_chat vs. aleatoria)
- [ ] Intervención antigua con α arbitrarios (no calibrada)

## Resultados clave

| Medida | Valor |
|---|---|
| ΔP dañinas (capa 20) | 0.682 [IC 95 % 0.587, 0.779] |
| Prueba t pareada | p = 5.88e-17, d_z = 2.10 |

## Notas

- Modelo deducido del nombre de la carpeta y de la arquitectura guardada (80 capas, dimensión 8192).
- No usa el corpus de 42 ni de 96 pares.

## Archivos

- `agent_direction_results.json`
- `bootstrap_delta_p.json`
- `results.json`
- `stratified_results.json`
