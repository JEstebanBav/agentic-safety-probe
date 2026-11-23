# Datos: qué es cada archivo

Los archivos se quedan en estas rutas porque las corridas guardadas (`activations.npz`), los manifiestos con hashes y el loader las referencian tal cual.

## Corpus vigente

| Ruta | Uso |
|---|---|
| [v03/](v03/) | **Corpus v0.3 para la nueva corrida**: 96 pares, grupos temáticos y catálogo neutral. Ver su [README](v03/README.md). |

## Corpus anteriores (no borrar: hay resultados que dependen de ellos)

| Archivo | Uso |
|---|---|
| `dataset_v2.jsonl` (+ `_report.json`) | 96 pares. Base de los resultados publicados y de la semana 1. |
| `dataset_full.jsonl` | 42 pares hechos a mano. Valor por defecto del runner si no se pasa `--dataset-path`. |
| `dataset_expanded*.jsonl` (+ `_report.json`) | 93 pares con AgentHarm; variantes `metas` y `procedimientos`. |
| `dataset_agentharm*.jsonl` | Solo los pares importados de AgentHarm. |
| `etiquetado_explicitud_*.csv` | Hojas de etiquetado de explicitud de esos corpus. |

## Anotación y apoyo

| Ruta | Uso |
|---|---|
| [week1/](week1/) | Hojas A/B, consenso de estratos, auditoría de pares y reclasificación de la semana 1. |
| [agentharm_es/](agentharm_es/) | Comportamientos de AgentHarm con la versión benigna traducida al español. |

## Fuentes y herramientas

| Archivo | Uso |
|---|---|
| `harmful_behaviors_*.json`, `benign_behaviors_*.json`, `chat_*.json` | Benchmark AgentHarm original (`--dataset harmagent`). |
| `tool_definitions_agentharm.json` | Herramientas importadas de AgentHarm (catálogo global del repo). |
| `tool_effects.json` | Herramienta → solo lectura / reversible / irreversible. |
| `loader.py` | Loaders de todos los corpus y del mapa de grupos temáticos. |
| `build_dataset.py`, `build_agentharm_pairs.py`, `prepare_dataset_v2.py` | Constructores de los corpus anteriores. |
