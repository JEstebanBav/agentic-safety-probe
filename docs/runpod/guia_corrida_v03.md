# Guía: correr los cuatro modelos con el corpus v0.3

Paso a paso para repetir en RunPod las corridas de Mistral-7B, Llama-3.3-70B, Qwen3-8B y Qwen3-32B sobre el corpus v0.3 ([data/v03/](../../data/v03/README.md)). Cada modelo usa los mismos flags que su última corrida publicada; lo único nuevo es el corpus, los grupos temáticos en las particiones y el catálogo neutral de herramientas.

## 1. Qué se va a correr

| Modelo | Flags (los mismos de su corrida anterior) | Tiempo medido antes | Carpeta nueva |
|---|---|---|---|
| Mistral-7B-Instruct-v0.3 | por defecto | 83 min en la H200 (v4) | `mistral7b/v5` |
| Llama-3.3-70B-Instruct | 20 capas, causal en 76, 56 y 28 | 115 min en la H200 (v4) | `llama33_70b/v5` |
| Qwen3-8B | por defecto | 413 min en una GPU de 48 GB (v3) | `qwen3_8b/v4` |
| Qwen3-32B | por defecto | nunca terminó: la v3 se cortó en la generación | `qwen3_32b/v4` |

Todas en FP16, 96 pares, `--factorial`, 5 folds × 10 repeticiones, 10.000 bootstraps. Qwen3 razona antes de responder (hasta 1024 tokens), por eso tarda mucho más en la etapa de generación.

El análisis por explicitud saldrá omitido (`Explicitness strata: SKIPPED`): el corpus v0.3 no trae estratos. Se calcula después, sin GPU, cuando estén las etiquetas (paso 9).

## 2. Qué alquilar

Un solo pod para las cuatro corridas, en cola:

| | |
|---|---|
| GPU | **1× H200 (141 GB)**, la de las corridas v4 |
| Volumen (`/workspace`) | **300 GB**: los cuatro modelos pesan 237 GB |
| Disco del contenedor | 30 GB |
| Plantilla | RunPod PyTorch, CUDA 12.x |
| Puertos | 8888 (JupyterLab) y 22 (SSH) |

Variables de entorno de la plantilla:

| Clave | Valor |
|---|---|
| `HF_HOME` | `/workspace/hf-cache` |
| `HF_TOKEN` | tu token de HuggingFace (lectura) |
| `HF_XET_HIGH_PERFORMANCE` | `1` |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` |

No uses una H100 de 80 GB para Llama: el modelo en FP16 ocupa 131 GiB. Si no hay H200, usa 2× H100 NVL (94 GB) como en la [guía anterior](guia_runpod_llama33.md) y deja `PROBE_GPU_RESERVE` en 0.12.

## 3. Costo estimado

Precios de RunPod en noviembre de 2026 para la H200: unos US$ 3,59/h en Community Cloud y US$ 5,29/h en Secure Cloud. Cambian seguido; revísalos en [runpod.io/pricing](https://www.runpod.io/pricing) el día que alquiles.

| Etapa | Horas |
|---|---|
| Instalación y verificación | 0,3 |
| Descarga de los 4 modelos (237 GB) | 0,5-1 |
| Corridas de prueba (4 × 4 pares) | 1 |
| Mistral-7B | 1,5 |
| Llama-3.3-70B | 2 |
| Qwen3-8B | 3-7 |
| Qwen3-32B | 8-14 |
| Empaquetar y descargar | 0,5 |
| **Total** | **≈ 17-27 h** |

| | Community (US$ 3,59/h) | Secure (US$ 5,29/h) |
|---|---|---|
| Estimado | **US$ 60-100** | **US$ 90-145** |

Los tiempos de Mistral y Llama están medidos en la misma GPU. Los de Qwen son estimaciones: Qwen3-8B se midió en una GPU más lenta, y Qwen3-32B nunca terminó. La corrida de prueba da la mejor medida: el tiempo de su etapa de generación multiplicado por 24 (96/4 pares) acota la real. Deja un margen de 20-30 % para repetir algo. El volumen de 300 GB suma alrededor de US$ 1 por día mientras el pod exista.

**Si el presupuesto aprieta:**
- Mistral y Qwen3-8B caben en una L40S de 48 GB (unos US$ 0,79-1,09/h). Correrlas en un segundo pod ahorra unos US$ 10-20, a cambio de configurar otro pod.
- En Qwen3-32B, `--intervention-pairs 48` reduce a la mitad la generación. El test causal queda con 48 pares en lugar de 96, así que hay que reportarlo.

## 4. Antes de encender el pod

1. Acepta las licencias de `meta-llama/Llama-3.3-70B-Instruct` y `mistralai/Mistral-7B-Instruct-v0.3` en HuggingFace con la cuenta del token. Sin eso, la descarga responde 403.
2. Confirma que `main` en GitHub tiene el corpus v0.3: debe existir `data/v03/dataset_v03_rerun.jsonl`.

Las comprobaciones locales ya se hicieron con esta versión: los tests pasan, `validate_corrections.py` termina en `ALL CHECKS PASSED` y los cuatro tokenizadores caben en el presupuesto (máximo 1246 de 4096 tokens, Llama).

## 5. Preparar el pod

Trabaja siempre dentro de `tmux`: si se cae la conexión, la corrida sigue.

```bash
cd /workspace
git clone https://github.com/JEstebanBav/agentic-safety-probe.git
cd agentic-safety-probe
pip install -r requirements_runpod.txt
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

python -c "
import torch
p = torch.cuda.get_device_properties(0)
print(p.name, round(p.total_memory/1024**3), 'GiB')"
sha256sum data/v03/dataset_v03_rerun.jsonl   # c96d4bd6...3dba
```

La GPU debe decir alrededor de 140 GiB. El hash debe empezar con `c96d4bd6`.

Descarga los cuatro modelos de una vez (30-60 min):

```bash
tmux new -s descarga
for m in mistralai/Mistral-7B-Instruct-v0.3 meta-llama/Llama-3.3-70B-Instruct Qwen/Qwen3-8B Qwen/Qwen3-32B; do
  hf download "$m" || echo "FALLO $m"
done
# salir sin detener: Ctrl-b y luego d      volver: tmux attach -t descarga
```

## 6. Corridas de prueba (obligatorias)

Cuatro pares, todas las etapas, generación incluida. Una por modelo; cada una tarda entre 5 y 20 min:

```bash
P="--limit-pairs 4 --n-repeats 1 --n-label-permutations 0 --n-bootstrap 1000 --n-selection-splits 20"
bash scripts/runpod/run_modelo_v03.sh mistralai/Mistral-7B-Instruct-v0.3 /workspace/prueba/mistral --no-quantize $P 2>&1 | tee /workspace/prueba/mistral.log
PROBE_GPU_RESERVE=0.04 bash scripts/runpod/run_modelo_v03.sh meta-llama/Llama-3.3-70B-Instruct /workspace/prueba/llama --no-quantize $P --layers 28,56,76 --intervention-layers 56 2>&1 | tee /workspace/prueba/llama.log
bash scripts/runpod/run_modelo_v03.sh Qwen/Qwen3-8B  /workspace/prueba/qwen8  --no-quantize $P 2>&1 | tee /workspace/prueba/qwen8.log
bash scripts/runpod/run_modelo_v03.sh Qwen/Qwen3-32B /workspace/prueba/qwen32 --no-quantize $P 2>&1 | tee /workspace/prueba/qwen32.log
```

En cada log revisa:

- [ ] `Source: data/v03/dataset_v03_rerun.jsonl (paired)`, `Total entries: 384` y `DRY RUN: limited to 4 of 96 pairs`.
- [ ] `Theme groups: data/v03/theme_groups_67.csv - 4 pairs in 4 groups` y `Tool catalogue: data/v03/catalogos_neutrales (85 global tools)`.
- [ ] `first tokens` con **un solo** token de inicio; `last tokens` igual en todas las variantes.
- [ ] `Explicitness strata: SKIPPED`. Es lo esperado con este corpus.
- [ ] En `BEHAVIOURAL BASELINE`, la columna `tool call` es alta en `agent_*` y 0 en `chat_*`. Si en `agent_*` sale cerca de 0, las herramientas no llegan al prompt: no sigas.
- [ ] Solo Qwen: el log dice `Reasoning model: up to 1024 <think> tokens` y la columna `incompl.` está en 0.
- [ ] Termina en `EXPERIMENT COMPLETE`, y en la carpeta están `run_manifest.json` y `rendered_inputs.jsonl`.

Abre algunas respuestas para confirmar que son texto del modelo y no la continuación de un JSON:

```bash
head -c 1500 /workspace/prueba/mistral/behavior_generations.jsonl
```

Con el tiempo de cada prueba, recalcula el costo: (tiempo de generación de la prueba) × 24 + extracción y análisis.

## 7. Corridas reales, en cola

```bash
tmux new -s cola
bash scripts/runpod/cola_v03.sh 2>&1 | tee /workspace/cola_v03.log
```

[cola_v03.sh](../../scripts/runpod/cola_v03.sh) corre Mistral → Llama → Qwen3-8B → Qwen3-32B. Cada log queda en `/workspace/out/corridas/<modelo>/<versión>/run.log`. Si una corrida falla, la cola lo anota y sigue con la siguiente. Si la relanzas, omite las que ya tienen `results.json`.

Seguimiento desde otra terminal:

```bash
tail -3 /workspace/cola_v03.log                         # en cuál va
tail -5 /workspace/out/corridas/qwen3_32b/v4/run.log    # detalle de una
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv
```

El orden interno protege el gasto: `activations.npz` se guarda apenas termina la extracción y `results.json` antes de generar. Si falla la generación, el análisis ya está en disco.

## 8. Sacar los resultados y apagar

```bash
cd /workspace/out
tar czf /workspace/resultados_v03.tar.gz corridas ../prueba ../cola_v03.log
ls -lh /workspace/resultados_v03.tar.gz      # varios GB: incluye las activaciones
```

Descárgalo desde el explorador de JupyterLab o con `runpodctl send /workspace/resultados_v03.tar.gz`. En el portátil, confirma que cada modelo trae `activations.npz`, `results.json`, `causal_results.json`, `run_manifest.json` y `rendered_inputs.jsonl`.

Después **termina** el pod, no solo lo detengas: un pod detenido sigue cobrando el volumen.

## 9. De vuelta en el portátil

1. Copia cada carpeta a `results/corridas/<modelo>/<versión>/` (las de la tabla del paso 1), agrega sus entradas en `results/registro.json` y regenera fichas e índice:

   ```bash
   python scripts/analisis/generar_fichas.py
   ```

   `activations.npz` no se sube a GitHub (está en `.gitignore`). Guárdalo aparte: sin él, cualquier reanálisis vuelve a necesitar GPU.

2. **Explicitud, cuando estén los estratos.** Se crea una copia del corpus con la columna `subtlety` llena según el `pair_id` (91 pares del consenso de la semana 1, más los 5 reanotados) y se reanaliza sin GPU:

   ```bash
   python run_experiment.py \
     --load-activations results/corridas/llama33_70b/v5/activations.npz \
     --dataset-path data/v03/dataset_v03_con_estratos.jsonl \
     --output-dir results/corridas/llama33_70b/v5_estratos
   ```

   Usa una carpeta distinta: con `--load-activations` no se regenera texto, así que los resultados conductuales y causales siguen en la carpeta original.

Para leer el test causal y saber qué archivo responde cada pregunta, ver las últimas secciones de la [guía anterior](guia_runpod_llama33.md#qué-archivo-responde-cada-observación); siguen valiendo.
