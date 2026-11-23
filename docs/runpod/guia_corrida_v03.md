# Guía: corridas del corpus v0.3 en RunPod (10 modelos)

Plan aprobado el 22 de noviembre de 2026. Corpus: [data/v03/](../../data/v03/README.md). Cola de corridas: [scripts/runpod/cola_v03.sh](../../scripts/runpod/cola_v03.sh).

## 1. Plan

| Fase | Máquina | Modelos | Horas | Costo (Community) |
|---|---|---|---|---|
| 1A | 1× H200 141 GB · volumen 600 GB | Llama-3.3-70B, Qwen3-32B, Qwen3.8-27B, Gemma-4-31B-it, Muse-Glimmer-30B, gpt-oss-20b, gpt-oss-120b | 26–47 | US$ 93–171 (3,59/h) |
| 1B | 1× L40S 48 GB · volumen 100 GB, en paralelo con 1A | Mistral-7B, Qwen3-8B | 9,5–11,5 | US$ 7,5–9 (0,79/h) |
| 2 | 4× H200 · volumen 450 GB, solo si queda presupuesto | GLM-5.3-Flash, solo extracción | 2,3–3,8 | US$ 33–55 (14,36/h) |
| | | | **Total** | **US$ 140–240** + ~5 de almacenamiento |

Configuración por modelo (la pone la cola; no hay que escribirla a mano):

| Modelo | Precisión | Pares en el test causal | Otros |
|---|---|---|---|
| Mistral, Qwen3-8B, Qwen3-32B | FP16, como sus corridas anteriores | 96 | — |
| Llama-3.3-70B | FP16, como la v4 | 96 | 20 capas, causal en 76/56/28, `PROBE_GPU_RESERVE=0.04` |
| Qwen3.8, Gemma 4, Muse, gpt-oss ×2 | BF16 (nativa; gpt-oss mantiene MXFP4) | 48 (`NUEVOS_PARES`) | — |
| GLM-5.3-Flash | BF16 + FP8 nativo | sin generación | análisis completo después, en el portátil |

El análisis de activaciones usa siempre los 96 pares. El análisis por explicitud sale omitido (`Explicitness strata: SKIPPED`) hasta que se agreguen los estratos (paso 9).

**Puntos de control del presupuesto:**
1. **Después de las pruebas:** si la proyección de la Fase 1A pasa de 50 h, relanza los modelos nuevos con `NUEVOS_PARES=24` o deja fuera el más caro.
2. **Antes de la Fase 2:** si lo gastado supera US$ 240, pospón GLM.
3. **Si solo hay Secure Cloud:** H200 a ~5,29/h y L40S a ~1,09/h; el total sube a ~US$ 200–350, y GLM es lo primero que sale.

## 2. Ya verificado en el portátil (Fase 0)

Con transformers 5.24.0, la misma versión del pod, y sin descargar pesos:

- Los diez tokenizadores renderizan los 768 prompts (corpus + factorial).
  - El más largo ocupa 1258 de 4096 tokens (Muse).
  - Las herramientas aparecen en todos los prompts agénticos.
  - Ningún prompt tiene el token de inicio duplicado.
- Los diez modelos se construyeron vacíos y el código encuentra sus capas:
  - los cuatro multimodales (Gemma 4, Qwen3.8, Muse, GLM) en `model.language_model.layers`;
  - el resto en `model.layers`.
- Los formatos de razonamiento se detectan bien:
  - `<think>`: Qwen3, Qwen3.8 y GLM;
  - *harmony*: gpt-oss;
  - *recipient*: Muse;
  - Gemma 4 trae el razonamiento apagado por defecto.
- Las llamadas a herramientas de gpt-oss, Muse y Gemma 4 se reconocen, y el corte de generación ignora el razonamiento.
- Tests y una corrida completa de prueba en CPU, en BF16 y con transformers 5.24.0.

## 3. Antes de alquilar

1. El único repositorio con acceso restringido es `meta-llama/Llama-3.3-70B-Instruct`: su licencia debe estar aceptada con la cuenta del token. Los otros nueve se descargan sin aceptar nada (comprobado el 22 de noviembre de 2026).
2. Confirma que `main` en GitHub tiene esta versión: debe existir `src/response_formats.py`.

## 4. Configurar cada pod

| | 1A | 1B | 2 |
|---|---|---|---|
| GPU | 1× H200 | 1× L40S | 4× H200 |
| Volumen `/workspace` | 600 GB | 100 GB | 450 GB |
| Disco del contenedor | 30 GB | 30 GB | 30 GB |

- **Plantilla:** RunPod PyTorch con **CUDA 12.8 o superior**. Filtra por versión de CUDA al elegir la máquina: los kernels de gpt-oss necesitan torch ≥ 2.8.
- **Puertos:** 8888 y 22.
- **Variables de entorno:** `HF_HOME=/workspace/hf-cache`, `HF_TOKEN=<tu token>`, `HF_XET_HIGH_PERFORMANCE=1` y `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

En cada pod, siempre dentro de `tmux`:

```bash
cd /workspace
git clone https://github.com/JEstebanBav/agentic-safety-probe.git
cd agentic-safety-probe
pip install -r requirements_runpod.txt
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

python -c "
import torch, transformers
print('torch', torch.__version__, '| transformers', transformers.__version__)
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i); print(' ', p.name, round(p.total_memory/1024**3), 'GiB')"
sha256sum data/v03/dataset_v03_rerun.jsonl     # debe empezar con c96d4bd6
```

Debe decir `transformers 5.24.0`, torch 2.8 o superior, y no mostrar la línea `Disabling PyTorch`.

Descarga los modelos de esa máquina:

```bash
# 1A (~465 GB, 30-60 min)
for m in meta-llama/Llama-3.3-70B-Instruct Qwen/Qwen3-32B Qwen/Qwen3.8-27B google/gemma-4-31B-it \
         meta-models/Muse-Glimmer-30B openai/gpt-oss-20b openai/gpt-oss-120b; do hf download "$m" || echo "FALLO $m"; done
# 1B
for m in mistralai/Mistral-7B-Instruct-v0.3 Qwen/Qwen3-8B; do hf download "$m" || echo "FALLO $m"; done
# 2
hf download zai-org/GLM-5.3-Flash
```

## 5. Pruebas (obligatorias, en cada pod)

```bash
bash scripts/runpod/cola_v03.sh h200 prueba 2>&1 | tee /workspace/prueba_h200.log    # o l40s / glm
```

Cada modelo corre 4 pares con todas las etapas, en 5–20 min. Revisa cada `/workspace/out/pruebas/<modelo>/<versión>/run.log`:

- [ ] `Source: data/v03/dataset_v03_rerun.jsonl`, `Total entries: 384` y `DRY RUN: limited to 4 of 96 pairs`.
- [ ] `Theme groups: ... 4 pairs in 4 groups` y `Tool catalogue: data/v03/catalogos_neutrales`.
- [ ] `Class:`, `Layers:` y `compute dtype:` coinciden con la tabla de la sección 2 (BF16 en los nuevos).
- [ ] **gpt-oss:**
  - `Native quantization: mxfp4` **sin** la palabra `DEQUANTIZED`;
  - `Memory:` cerca de 13 GB (20b) o de 65 GB (120b).

  Si dice `DEQUANTIZED` o la carga se detiene con `modules were placed on ['cpu']`, faltan los kernels de MXFP4: revisa la versión de torch y `pip show kernels`.
- [ ] `first tokens` con un solo token de inicio, y `last tokens` igual en todas las variantes.
- [ ] El formato de razonamiento es el esperado: `Reasoning model (<think>)` en Qwen y GLM, `(harmony)` en gpt-oss, `(recipient)` en Muse, y ninguna línea en Mistral, Llama ni Gemma.
- [ ] En `BEHAVIOURAL BASELINE`, la columna `tool call` es alta en `agent_*` y 0 en `chat_*`.
- [ ] La columna `incompl.` está en 0. Si no, el razonamiento no cabe en 1024 tokens: relanza ese modelo con `--thinking-budget 2048`. Es probable en Muse, que razona en modo "high" por defecto, y en gpt-oss. Cuesta más tiempo de generación.
- [ ] Termina en `EXPERIMENT COMPLETE`, y en la carpeta están `run_manifest.json` y `rendered_inputs.jsonl`.

**Recalcular el costo:** cada log trae `Generation stages: X min for 4 pairs (Y min per pair)`. Tiempo real ≈ Y × 96 (re-corridas) o Y × 48 (nuevos), más ~1 h por modelo de extracción y análisis. Suma por máquina y aplica los puntos de control de la sección 1.

## 6. Corridas reales

```bash
tmux new -s cola
bash scripts/runpod/cola_v03.sh h200 real 2>&1 | tee /workspace/cola_h200.log     # en la 1A
bash scripts/runpod/cola_v03.sh l40s real 2>&1 | tee /workspace/cola_l40s.log     # en la 1B
```

Para cambiar los pares de los modelos nuevos: `NUEVOS_PARES=24 bash scripts/runpod/cola_v03.sh h200 real`. Si la cola se relanza, omite las corridas que ya tienen `results.json`.

Seguimiento:

```bash
tail -3 /workspace/cola_h200.log
tail -5 /workspace/out/corridas/qwen3_32b/v4/run.log
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv
```

`activations.npz` se guarda apenas termina la extracción y `results.json` antes de generar. Si falla la generación, el análisis ya está en disco.

## 7. Fase 2: GLM-5.3-Flash (solo con presupuesto)

Mismo procedimiento en el pod de 4× H200: instalación, descarga, `cola_v03.sh glm prueba` y después `cola_v03.sh glm real`. Solo extrae activaciones; el pod no genera texto. En la prueba revisa `Class: Glm5NextForConditionalGeneration`, `Native quantization: fp8` y que no aparezca ningún módulo en `cpu`.

## 8. Sacar resultados y apagar

```bash
cd /workspace/out && tar czf /workspace/resultados_<maquina>.tar.gz corridas pruebas ../*.log
```

Descárgalo con `runpodctl send /workspace/resultados_<maquina>.tar.gz` o desde JupyterLab. Comprueba que cada modelo trae `activations.npz`, `results.json`, `run_manifest.json` y `rendered_inputs.jsonl`, más `causal_results.json` en todos menos GLM. Después **termina** el pod, no solo lo detengas: uno detenido sigue cobrando el volumen.

## 9. En el portátil

1. Copia cada carpeta a `results/corridas/<modelo>/<versión>/`, agrega las entradas en `results/registro.json` y regenera las fichas con `python scripts/analisis/generar_fichas.py`. Guarda `activations.npz` aparte: no se sube a GitHub.
2. **GLM:** análisis completo sin GPU:
   ```bash
   python run_experiment.py --load-activations results/corridas/glm53_flash/v1/activations.npz \
     --n-folds 5 --n-repeats 10 --mlp-n-repeats 1 --n-label-permutations 10 --n-bootstrap 10000 \
     --output-dir results/corridas/glm53_flash/v1_analisis
   ```
3. **Explicitud**, cuando estén los estratos (91 del consenso más 5 reanotados): copia del corpus con `subtlety` y reanálisis sin GPU con `--load-activations ... --dataset-path data/v03/dataset_v03_con_estratos.jsonl`, en una carpeta aparte.

Para comparar entre modelos usa las métricas sin escala (AUROC, d de Cohen, separación estandarizada) y la profundidad relativa de la capa, no el ΔP crudo. En gpt-oss y GLM, que son MoE, interpreta la "rotación" teniendo en cuenta el enrutamiento de expertos ([runpod_large_models.md](runpod_large_models.md)).
