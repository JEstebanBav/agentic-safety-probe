# Ficha · Llama-3.3-70B · V1

> Generada por `scripts/generar_fichas.py` a partir de los archivos de esta carpeta y de `results/registro.json`. No la edites a mano: corrige el registro y regenera.

## Identificación

| | |
|---|---|
| Carpeta | `results/corridas/llama33_70b/v1/` |
| Tipo | corrida principal |
| Fecha | ≤ 2026-09-22 (fecha del commit; la corrida no la registró) |
| Carpeta anterior | `results/llama33` |
| Relación con otras versiones | Primera corrida de Llama-3.3, con el pipeline original. |
| Citada en | Tesis original (capa 56, ΔP ≈ 5,74), según docs/tesis/revision_observaciones.md |

## El modelo en esta corrida

| | |
|---|---|
| Modelo | `meta-llama/Llama-3.3-70B-Instruct` |
| Parámetros | 70.60 mil millones |
| Arquitectura | llama, 80 capas, dimensión 8192 |
| Precisión de carga | FP16, según el log |
| Memoria ocupada | 141.1 GB |
| Posición de lectura | último token del prompt (cabecera del asistente), antes de generar |
| Capas analizadas | 0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76 |
| Capa de referencia (máximo ΔP) | 56 |
| Capa por selección en muestra dividida | 52 |

## Corpus

| | |
|---|---|
| Archivo | dataset_full.jsonl (paired, 168 entries) |
| Pares | 42 |

## Parámetros de la corrida

Esta corrida es de una versión del pipeline que no guardaba su configuración en `results.json`.
Del encabezado del log:

- Modelo: `meta-llama/Llama-3.3-70B-Instruct`
- Precisión: `False`
- Corpus: `dataset_full.jsonl (paired, 168 entries)`

Duración registrada: 2378.7s (39.6 min).

## Etapas ejecutadas

- [ ] Activaciones guardadas (activations.npz, solo local)
- [x] Dirección de rechazo y ΔP por capa
- [ ] Validación PCA del contraste
- [ ] Evaluación fuera de muestra (held-out)
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
| ΔP dañinas (capa 56) | 5.739 [IC 95 % 4.663, 6.767] |
| Prueba t pareada | p = 1.46e-13, d_z = 1.63 |

## Notas

- Pipeline anterior a las correcciones de los tutores: sin evaluación fuera de muestra por par, sin factorial ni test causal calibrado.
- Tokenizaba cada prompt con dos tokens de inicio de secuencia (BOS).

## Archivos

- `agent_direction_results.json`
- `bootstrap_delta_p.json`
- `directions.npz`
- `format_effect_results.json`
- `reanalisis_observaciones.json`
- `results.json`
- `robust_results.json`
- `run.log`
- `stratified_results.json`
