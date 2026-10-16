# Guía: volver a correr Llama-3.3-70B en RunPod

Paso a paso para repetir la corrida de la tesis con las correcciones de
particiones, la validación PCA, el factorial sistema × herramientas y la etapa
causal nueva. Para elegir otro modelo o entender los problemas de memoria en
detalle, ver `runpod_large_models.md`.

**Corpus: `data/dataset_v2.jsonl` (96 pares, 384 entradas).** El valor por
defecto de `run_experiment.py` sigue siendo `data/dataset_full.jsonl` (42
pares), así que **todos** los comandos de esta guía pasan
`--dataset-path data/dataset_v2.jsonl`. Si lo omites, la corrida termina sin
error pero sobre el corpus equivocado.

## Qué comprar

| | |
|---|---|
| GPU | **2× H100 NVL (94 GB)**, la misma configuración de la corrida anterior |
| Volumen (`/workspace`) | **200 GB** (los pesos pesan 141 GB) |
| Disco del contenedor | 30 GB |
| Plantilla | RunPod PyTorch, CUDA 12.x |
| Puertos | 8888 (JupyterLab) y 22 (SSH) |

**No sirven 2× H100 de 80 GB.** El modelo en FP16 ocupa 132 GiB y el pipeline
reserva 12 % de cada tarjeta: dos de 80 GB dejan 131 GiB de presupuesto y la
carga falla después de haber descargado los pesos. Si no hay H100 NVL, usa
3× A100/H100 de 80 GB. No cambies a 4 bits para ahorrar: la tesis reporta FP16
y la cuantización altera justo lo que se mide.

Variables de entorno de la plantilla (clave y valor en campos separados):

| Clave | Valor |
|---|---|
| `HF_HOME` | `/workspace/hf-cache` |
| `HF_TOKEN` | tu token de HuggingFace (lectura) |
| `HF_XET_HIGH_PERFORMANCE` | `1` |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` |

### Tiempo y presupuesto

La corrida anterior tardó 39,6 minutos en esta misma máquina con 42 pares. La
nueva genera más texto y usa 96 pares: la extracción y la generación (línea
base de comportamiento y test causal, que por defecto usan todos los pares)
crecen aproximadamente en proporción, ≈ 2,3×. Estimación, no medición:

| Etapa | Tiempo |
|---|---|
| Arranque, instalación, verificación | 10 min |
| Descarga de los pesos (141 GB) | 15-30 min |
| Corrida de prueba | 10 min |
| Corrida real | 2,5-3 h |
| Empaquetar y bajar resultados | 10 min |
| **Total** | **≈ 3,5-4,5 h** |

Con 2× H100 NVL eso es del orden de **US$ 25-30**; revisa el precio por hora
el día que lo compres, porque cambia. Cargar **US$ 45** deja margen para
repetir parte de la corrida si algo falla. La corrida de prueba da una medida
mejor: su tiempo por par multiplicado por 96 es una cota razonable de la real.
Si el presupuesto aprieta, `--intervention-layers 52` (una sola capa causal)
o `--intervention-pairs N` recortan la parte de generación.

## Antes de encender el pod

1. **Comprueba que `main` en GitHub está al día.** El pod clona `main`, así
   que cualquier cambio al código o a `data/dataset_v2.jsonl` tiene que estar
   subido:

   ```bash
   git status            # sin cambios pendientes en src/, data/ ni run_experiment.py
   git push origin main
   ```

2. **Acepta la licencia** de `meta-llama/Llama-3.3-70B-Instruct` en
   HuggingFace con la cuenta del token. Sin eso la descarga responde 403.

3. **Corre las verificaciones locales** (no necesitan GPU):

   ```bash
   python scripts/validate_corrections.py          # debe terminar en ALL CHECKS PASSED
   python scripts/reanalisis_observaciones.py      # re-análisis de la corrida anterior
   python scripts/check_token_budget.py --model meta-llama/Llama-3.3-70B-Instruct \
       --dataset-path data/dataset_v2.jsonl      # la fila "(por defecto)" debe decir "cabe"
   ```

   En `dataset_v2` la condición por defecto ocupa como máximo 1246 tokens de
   4096. La única combinación que no cabe es `--agency-level trajectory
   --toolbox 26`, que esta guía no usa.

## En el pod

Abre una terminal (JupyterLab o SSH) y trabaja siempre dentro de `tmux`: si la
conexión se cae, la corrida sigue.

### 1. Código y entorno

```bash
cd /workspace
git clone https://github.com/JEstebanBav/agentic-safety-probe.git   # rama main
cd agentic-safety-probe

pip install -r requirements_runpod.txt
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Usa `requirements_runpod.txt`, no `requirements.txt` (este último puede
degradar el torch de la imagen).

Comprueba antes de descargar nada:

```bash
echo "HF_HOME=$HF_HOME"          # debe decir /workspace/hf-cache
python -c "
import torch, transformers
print('torch', torch.__version__, '| transformers', transformers.__version__)
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(' ', p.name, round(p.total_memory/1e9), 'GB')
from src.model_loader import build_max_memory
print('presupuesto:', build_max_memory())
"
```

Debe listar dos GPUs de ~94 GB y un presupuesto de ~81 GiB cada una. Si
aparece la línea `Disabling PyTorch`, repite el `pip install --upgrade torch`.

### 2. Descargar los pesos

```bash
tmux new -s descarga
hf download meta-llama/Llama-3.3-70B-Instruct
# salir sin detener: Ctrl-b y luego d      volver: tmux attach -t descarga
```

Progreso desde otra terminal: `du -sh $HF_HOME && df -h /workspace`.

### 3. Corrida de prueba (obligatoria)

Cuatro pares, todas las etapas, generación incluida. Es la que evita pagar una
corrida completa por un error de configuración:

```bash
python run_experiment.py --model meta-llama/Llama-3.3-70B-Instruct \
  --dataset-path data/dataset_v2.jsonl \
  --no-quantize --limit-pairs 4 --layers 28,52,56 --factorial \
  --n-repeats 1 --n-label-permutations 0 --n-bootstrap 1000 \
  --intervention-layers 52 --intervention-doses 1,2 \
  --output-dir /workspace/dryrun 2>&1 | tee /workspace/dryrun.log
```

Sus estadísticas no significan nada con 4 pares. Lo que hay que mirar:

- [ ] Al cargar el corpus aparece `Source: data/dataset_v2.jsonl (paired)` y
      `Total entries: 384`, y luego `DRY RUN: limited to 4 of 96 pairs`. Si
      dice `dataset_full.jsonl` o `of 42 pairs`, falta `--dataset-path`.
- [ ] En `EXTRACTING ACTIVATIONS`, `first tokens` empieza con **un solo**
      `<|begin_of_text|>`.
- [ ] `last tokens` termina igual en todas las variantes (la cabecera del
      asistente). Esa es la posición donde se lee la activación.
- [ ] Las variantes `agent_*` y `toolsonly_*` tienen cientos de tokens; las
      `chat_*` y `sysonly_*`, alrededor de cien.
- [ ] En `BEHAVIOURAL BASELINE` las respuestas son texto del modelo. Ábrelas:

      ```bash
      head -c 1500 /workspace/dryrun/behavior_generations.jsonl
      ```

      Si lo que aparece es la continuación de un JSON de herramientas, el
      prompt está mal construido: no sigas.
- [ ] En `BEHAVIOURAL BASELINE`, la columna `tool call` es **alta en
      `agent_*` y 0 en `chat_*`**. Es la comprobación de que el formato
      agéntico funciona como tal: el modelo responde invocando una de las
      herramientas ofrecidas. En la corrida de 42 pares Llama-3.3 llamó una
      herramienta en 40/42 benignos y 18/42 dañinos, y nunca en chat. Si en
      `agent_*` sale cerca de 0, las herramientas no están llegando al prompt:
      no sigas.
- [ ] En `CAUSAL TEST`, la columna `shift` es igual a `alpha`.
- [ ] La corrida llega a `EXPERIMENT COMPLETE`.

### 4. Corrida real

```bash
tmux new -s corrida
mkdir -p /workspace/results_llama33_v2

python run_experiment.py --model meta-llama/Llama-3.3-70B-Instruct \
  --dataset-path data/dataset_v2.jsonl \
  --no-quantize \
  --layers 0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76 \
  --factorial --n-folds 5 --n-repeats 10 --mlp-n-repeats 1 \
  --n-label-permutations 10 --n-bootstrap 10000 \
  --intervention-layers 52,28 --intervention-doses 0.5,1,2,4 \
  --output-dir /workspace/results_llama33_v2 \
  2>&1 | tee /workspace/results_llama33_v2/run.log
```

Qué hace cada grupo de opciones:

- `--dataset-path data/dataset_v2.jsonl`: el corpus de 96 pares. Sin esta
  opción se usa `dataset_full.jsonl` (42 pares).
- `--layers`: las mismas 20 capas de la corrida reportada.
- `--factorial`: extrae dos condiciones intermedias por instrucción (mensaje de
  sistema agéntico sin herramientas, y herramientas con el mensaje de chat).
  Duplica las pasadas de extracción, que son la parte barata.
- `--intervention-layers 52,28`: el test causal en la capa de referencia y en
  la capa donde convergen los criterios libres de escala. Con una sola capa
  (`52`) la corrida se acorta: unos 20 minutos con 42 pares, del orden de
  45 con 96.
- `--intervention-doses`: múltiplos del gap de proyección medido en esa capa;
  `1` devuelve exactamente la proyección que el formato agéntico quitó.
- `--mlp-n-repeats 1`: el MLP es lo más lento del análisis y no necesita GPU.
  Se repite completo en el portátil (ver abajo).

El orden interno protege el gasto: las activaciones se guardan apenas termina
la extracción, y `results.json` se escribe **antes** de empezar a generar. Si
la generación falla, el análisis ya está en disco.

Si solo falla la generación y quieres repetirla sin rehacer el análisis:

```bash
python run_experiment.py --model meta-llama/Llama-3.3-70B-Instruct \
  --dataset-path data/dataset_v2.jsonl \
  --no-quantize --layers 28,52 --skip-probes --skip-heldout \
  --n-bootstrap 1000 --n-selection-splits 0 \
  --intervention-layers 52,28 --output-dir /workspace/results_llama33_gen
```

### 5. Opcional: sensibilidad al token inicial (≈ 20 min)

Las corridas anteriores tokenizaban cada prompt con dos tokens de inicio de
secuencia; ahora se usa uno. Los números nuevos no van a coincidir cifra por
cifra con los de la tesis. Para poder afirmar que la conclusión no depende de
eso, una pasada solo de extracción con el comportamiento anterior:

```bash
PROBE_LEGACY_BOS=1 python run_experiment.py \
  --model meta-llama/Llama-3.3-70B-Instruct --no-quantize \
  --dataset-path data/dataset_v2.jsonl \
  --layers 0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76 \
  --skip-intervention --skip-probes --n-selection-splits 0 \
  --output-dir /workspace/results_llama33_bos_antiguo
```

### 6. Sacar los resultados y apagar

```bash
cd /workspace
tar czf results_llama33_v2.tar.gz results_llama33_v2 results_llama33_bos_antiguo dryrun.log
ls -lh results_llama33_v2.tar.gz      # ~0,5-1,2 GB con 96 pares: incluye las activaciones
```

Descárgalo desde el explorador de JupyterLab o con `runpodctl send`. Verifica
en el portátil que el archivo abre y contiene `activations.npz` y
`results.json`.

Después **termina** el pod (no solo detenerlo): un pod detenido sigue cobrando
el volumen de 200 GB.

## Mistral y Qwen con el mismo código

Las corridas de 96 pares que hay en `results/mistral7b_v2`, `qwen8b_v2` y
`qwen32b_v2` son de una versión anterior del pipeline: no tienen línea base de
comportamiento, test causal, factorial ni validación PCA, y no guardaron
`activations.npz`. Para compararlas con Llama hay que repetirlas en GPU.
`scripts/run_modelo.sh` ya pasa `--dataset-path data/dataset_v2.jsonl` y
`--factorial`:

```bash
./scripts/run_modelo.sh mistralai/Mistral-7B-Instruct-v0.3 /workspace/results_mistral7b_v3 --no-quantize
./scripts/run_modelo.sh Qwen/Qwen3-8B  /workspace/results_qwen8b_v3  --no-quantize
./scripts/run_modelo.sh Qwen/Qwen3-32B /workspace/results_qwen32b_v3 --no-quantize
```

Haz primero la corrida de prueba de cada uno añadiendo `--limit-pairs 4
--n-repeats 1 --n-label-permutations 0 --n-bootstrap 1000`. Tamaños de pod en
`runpod_large_models.md`.

**Qwen3 razona antes de responder.** Por defecto abre un bloque
`<think>...</think>` y solo después contesta. El pipeline lo detecta solo:
deja hasta `--thinking-budget` tokens (1024 por defecto) para el bloque,
genera `--max-new-tokens` de respuesta después de cerrarlo y etiqueta solo la
respuesta. Las activaciones no cambian, porque se leen en el último token del
prompt, antes de que empiece a razonar. En la corrida de prueba de Qwen revisa:

- [ ] El log dice `Reasoning model: up to 1024 <think> tokens`.
- [ ] La columna `incompl.` de `BEHAVIOURAL BASELINE` está en 0. Si no, el
      razonamiento no cabe: sube `--thinking-budget` (por ejemplo, 2048).
- [ ] La generación tarda más que con Llama por par; usa el tiempo de la
      prueba para estimar la corrida real.

## De vuelta en el portátil

Con `activations.npz` descargado no hace falta volver a alquilar GPU para
ninguna pregunta estadística. Todo el análisis se repite así:

```bash
python run_experiment.py \
  --load-activations results/llama33_v2/activations.npz \
  --dataset-path data/dataset_v2.jsonl \
  --n-folds 5 --n-repeats 10 --mlp-n-repeats 10 \
  --n-label-permutations 10 --n-bootstrap 10000 \
  --output-dir results/llama33_v2_local
```

Con `--mlp-n-repeats 10` esta es la corrida que da la comparación sonda lineal
vs. MLP sobre 50 particiones pareadas. En el portátil tarda cerca de hora y
media.

## Qué archivo responde cada observación

| Observación | Dónde mirar |
|---|---|
| Particiones por par (sondas, monitor, dirección agéntica) | `results.json` → `probe_folds_by_pair`, `linear_aurocs`; `robust_results.json` |
| "39 de 42" y selección de capa | `robust_results.json` → `layer_selection.inference` y `.stability` |
| "Grupos independientes" en la interacción | `format_effect_results.json` → `interaction_tested_within_pairs`, `n_pairs_interaction_positive` |
| Validación PCA | `results.json` → `pca_by_layer` (tabla en `run.log`) |
| Sonda lineal vs. MLP | `results.json` → `probe_comparison` |
| Rotación de la dirección entre formatos (`w_ag`) | `robust_results.json` → `rotation` (tabla "ROTATION" en `run.log`): diferencia de medias y sonda logística, cada una contra sí misma |
| Qué varía entre formatos (sistema vs. herramientas) | `factorial_results.json` |
| La proyección ¿se relaciona con el rechazo? | `causal_results.json` → `behavior` |
| Causalidad | `causal_results.json` → `causal`; textos en `behavior_generations.jsonl` |
| Longitud de los prompts por condición | `results.json` → `prompt_tokens` |

## Cómo leer el test causal

Para cada dosis el reporte da tres tasas sobre las mismas instrucciones:

- **d_chat restores**: instrucciones dañinas que el agente cumplía y que pasa
  a rechazar cuando se le suma la dirección de rechazo.
- **random control**: lo mismo, sumando una dirección aleatoria de igual norma.
- **benign refused**: instrucciones benignas que pasa a rechazar con el mismo
  empuje.

El resultado respalda un papel causal si la primera tasa sube con la dosis, la
segunda se queda cerca de cero y la tercera se mantiene baja en la dosis 1. Si
las tres suben juntas, el empuje está rompiendo al modelo, no restaurando el
rechazo. Si ninguna sube, la proyección que se pierde no es lo que decide el
comportamiento, y eso también es un resultado que hay que reportar.

Las etiquetas (rechaza / cumple / degenerado) salen de un clasificador por
palabras clave. Antes de citar las tasas, revisa a mano una muestra de
`behavior_generations.jsonl` y reporta cuántas etiquetas coinciden con tu
lectura.
