# Ficha · Mistral-7B · V3 (solo log)

> Generada por `scripts/generar_fichas.py` a partir de los archivos de esta carpeta y de `results/registro.json`. No la edites a mano: corrige el registro y regenera.

## Identificación

| | |
|---|---|
| Carpeta | `results/corridas/mistral7b/v3/` |
| Tipo | corrida principal |
| Fecha | ≤ 2026-10-15 (fecha del commit) |
| Carpeta anterior | `results/results_mistralV3` |
| Relación con otras versiones | Corrida sobre 42 pares (dataset_full, según el log) que ya imprime la validación PCA; solo se conserva el log. |
| Citada en | Tesis (avances): valores redondeados del log, capa 30 |

## El modelo en esta corrida

| | |
|---|---|
| Modelo | `mistralai/Mistral-7B-Instruct-v0.3` |
| Parámetros | 7.25 mil millones |
| Arquitectura | no registrado |
| Precisión de carga | FP16, según el log |
| Memoria ocupada | no registrado |
| Posición de lectura | último token del prompt (cabecera del asistente), antes de generar |
| Capas analizadas | 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30 |
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

- Modelo: `mistralai/Mistral-7B-Instruct-v0.3`
- Precisión: `FP16 (forzado)`
- Corpus: `data/dataset_full.jsonl (paired)`
- Factorial: `True`
- Nivel de agencia: `dataset default`

## Etapas ejecutadas

No se pueden comprobar desde archivos: solo hay log.

## Resultados clave

Esta carpeta no tiene `results.json`; ver el log.

## Notas

- No hay results.json: las cifras salen de run.log y están redondeadas.

## Archivos

- `run.log`
