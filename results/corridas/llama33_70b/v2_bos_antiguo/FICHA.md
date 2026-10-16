# Ficha · Llama-3.3-70B · V2 · token inicial antiguo (solo log)

> Generada por `scripts/generar_fichas.py` a partir de los archivos de esta carpeta y de `results/registro.json`. No la edites a mano: corrige el registro y regenera.

## Identificación

| | |
|---|---|
| Carpeta | `results/corridas/llama33_70b/v2_bos_antiguo/` |
| Tipo | corrida complementaria |
| Fecha | 2026-10-15 (marca de tiempo local del archivo) |
| Carpeta anterior | `results/llama33_final/results_llama33_bos_antiguo.log` |
| Relación con otras versiones | Primer intento de la sensibilidad al token inicial (PROBE_LEGACY_BOS=1). Solo se conserva el log; V3_bos_antiguo es la versión completa. |
| Citada en | no |

## El modelo en esta corrida

| | |
|---|---|
| Modelo | `meta-llama/Llama-3.3-70B-Instruct` |
| Parámetros | 70.60 mil millones |
| Arquitectura | no registrado |
| Precisión de carga | FP16, según el log |
| Memoria ocupada | no registrado |
| Posición de lectura | último token del prompt (cabecera del asistente), antes de generar |
| Capas analizadas | 0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76 |
| Capa de referencia (máximo ΔP) | no registrado |
| Capa por selección en muestra dividida | no registrado |

## Corpus

| | |
|---|---|
| Archivo | data/dataset_full.jsonl (paired) |
| Pares | 42 |

## Parámetros de la corrida

No hay `results.json` en esta carpeta.
Del encabezado del log:

- Modelo: `meta-llama/Llama-3.3-70B-Instruct`
- Precisión: `FP16 (forzado)`
- Corpus: `data/dataset_full.jsonl (paired)`
- Factorial: `False`
- Nivel de agencia: `dataset default`

Duración registrada: 575.1s (9.6 min).

## Etapas ejecutadas

No se pueden comprobar desde archivos: solo hay log.

## Resultados clave

Esta carpeta no tiene `results.json`; ver el log.

## Archivos

- `run.log`
