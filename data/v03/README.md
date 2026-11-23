# Corpus v0.3 para la nueva corrida de modelos

Copia sin cambios del paquete `Rerun_modelos_v03`. Los cinco archivos que lista [audit.json](audit.json) coinciden byte a byte con sus hashes. Las instrucciones originales del exportador están en [README_runner_EN.txt](README_runner_EN.txt).

| Archivo | Contenido |
|---|---|
| `dataset_v03_rerun.jsonl` | 96 pares, 384 variantes (chat/agente × dañino/benigno) |
| `theme_groups_67.csv` | `pair_id` → `theme_group_id`: 67 grupos (39 de 1 par, 27 de 2, 1 de 3) |
| `catalogos_neutrales/` | catálogo global neutral (85 herramientas); el segundo archivo está vacío a propósito |

## Cómo correrlo

```bash
bash scripts/runpod/run_modelo_v03.sh <modelo> results/corridas/<modelo>/v03 [flags extra]
```

Equivale a `run_modelo.sh` con tres flags más:

- `--dataset-path data/v03/dataset_v03_rerun.jsonl`
- `--theme-groups data/v03/theme_groups_67.csv`: cada partición (folds de cross-fitting y de sondas, mitades de selección de capa y de fiabilidad) deja juntos todos los pares de un grupo temático. Si falta un par en el mapa, la corrida se detiene antes de usar la GPU.
- `--tool-catalog-dir data/v03/catalogos_neutrales`: solo lo usan `--toolbox`, `--agency-level`, `--include-history` y las condiciones `role_*`. Los archivos del repositorio no se modifican. Con `--toolbox 16` y el catálogo del repo, 138 de los 384 prompts reciben herramientas con nombres antiguos (`query_onion_search` y similares); con el catálogo neutral, ninguno.

Para reanalizar sin GPU: `--load-activations <corrida>/activations.npz`. El dataset y los grupos se toman de la corrida guardada.

## Qué guarda cada corrida

Además de los resultados habituales:

- `run_manifest.json`: hashes del dataset, de los grupos y de los catálogos activos; revisión del modelo y del tokenizador; hash de la plantilla de chat; configuración de generación; versiones de las librerías.
- `rendered_inputs.jsonl`: el texto exacto que recibió el modelo, con id, par y variante.

## Límites

- `is_harmful` es el brazo experimental original, no una etiqueta humana adjudicada. `category` es metadato heredado.
- `subtlety` no viene en el archivo: el loader devuelve `unknown` y el análisis estratificado muestra un único estrato. Los estratos de la semana 1 se anotaron sobre los textos anteriores (84 de las 384 variantes cambiaron), así que no se trasladan.
- La columna `fold` del CSV no se usa: el runner dibuja sus propias particiones (con varias semillas y repeticiones) respetando los grupos.
- Los bootstrap siguen remuestreando pares, no grupos. Los intervalos pueden ser algo estrechos cuando hay pares de un mismo grupo.
- Las activaciones de corridas anteriores no sirven para este corpus: los textos y las herramientas cambiaron.
