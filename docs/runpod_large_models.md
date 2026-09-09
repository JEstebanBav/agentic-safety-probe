# Running the Probe on a Larger Model (RunPod)

The pipeline has been run on Mistral-7B and Llama-3.1-70B. This document covers
choosing a third, more capable model and running it on RunPod without wasting
money on the parts that are easy to get wrong.

---

## Why add a third model at all

The current finding is a **scale trend**: in Llama-70B the refusal direction
rotates partially (cos ≈ 0.4), in Mistral-7B almost completely. Two points do
not establish a trend. A third point, clearly above 70B in capability, either
confirms that rotation shrinks with scale or breaks the story — and both
outcomes are publishable.

There is a second reason specific to your statistical problem. Your bottleneck
is `n = 84` samples in `d = 8192` dimensions. Several strong models have a
**smaller hidden size** than Llama-70B, which improves that ratio directly,
without collecting a single new prompt.

---

## Recommendation

### Primary: `Qwen/Qwen3-235B-A22B-Instruct-2507`

| | |
|---|---|
| Params | 235B total, 22B active (MoE) |
| Why this one | Strongest open-weights agentic model within a thesis budget. Trained natively for tool use, so your agentic condition is a real operating mode for it rather than a simulated one. |
| Hidden size | ~4096 — roughly **half** Llama-70B's 8192, which meaningfully improves the `n << d` geometry |
| Inference speed | Fast for its size: only 22B params fire per token |
| Pod | **2×** 80-94GB (H100 80GB or H100 NVL 94GB), 4-bit |
| Volume | **700 GB** |
| Download | ~470 GB (see the trap below) |

A single 94 GB card is **not** enough: the 4-bit weights alone are ~129 GB.
See the sizing table under Pod configuration.

### Zero-risk alternative: `meta-llama/Llama-3.3-70B-Instruct`

Same architecture, same VRAM, same cost as the 70B you already ran, but better
instruction and tool following. Literally a one-flag change, no new failure
modes. Worth running first as a sanity check that your pipeline reproduces on a
near-identical model before spending on the 235B.

### Middle point: `Qwen/Qwen3-32B`

Dense (no MoE confound), strong tool use, fits FP16 on a single A100 80GB.
If budget is tight this gives you a genuine third scale point for ~$2/hr.

Run `python -c "from src.model_loader import print_model_options; print_model_options()"`
for the table with disk and VRAM figures.

---

## Two traps that cost real money

### 1. 4-bit reduces VRAM, not download

`bitsandbytes` quantizes **at load time**. The full BF16 checkpoint is
downloaded first and compressed in memory afterwards. So "I'll run the 235B in
4-bit on 2× H100" still means pulling ~470 GB onto pod disk before anything
happens — 45-90 minutes of billed time, and a pod configured with 100 GB disk
will simply fail partway through.

**Mitigation:** use a pre-quantized checkpoint (FP8 / AWQ / GPTQ) when one is
published for the model, or provision the disk honestly. Check what is
available under the model's HuggingFace page before booking the pod.

### 2. MoE routing is a confound, not just an implementation detail

In a Mixture-of-Experts model, different prompts activate different experts.
Chat and agent formats are textually very different, so they may well route to
**different experts** — which would be a genuine alternative explanation for
your rotation finding, distinct from "the refusal direction moves."

The residual stream is still a single dense vector, so extraction works
unchanged and the measurement is valid. But if a reviewer asks "is the rotation
a property of the representation or an artefact of expert routing?", you need
an answer. Two options:

- Use a **dense** model for the clean scale comparison (Qwen3-32B, or
  Llama-3.3-70B), and treat the MoE run as a separate question.
- Run the MoE and **log which experts fire** per condition, turning the
  confound into a finding. This is more interesting but is new work.

Decide which before you book the pod.

---

## Pod configuration

### The setting that breaks the run if you get it wrong

RunPod gives a pod **two** disks, and they behave differently:

| | Mount | Survives a pod stop? | Default size |
|---|---|---|---|
| **Container disk** | `/` | No — wiped | 20 GB |
| **Volume disk** | `/workspace` | Yes, while the pod exists | you choose |

HuggingFace caches to `~/.cache/huggingface`, which is on the **container
disk**. Downloading a 470 GB model onto a 20 GB ephemeral disk fails partway
through, and any model that *does* fit is lost the moment you stop the pod.

**Set `HF_HOME=/workspace/hf-cache` before downloading anything.** This is the
single most expensive mistake available here.

Use a **Network Volume** instead of a pod volume if you expect more than one
session — it detaches from the pod, so you can stop the GPU without losing the
weights. It is region-locked, so pick the region before booking.

### Sizing

Two numbers decide whether a configuration works, and they are independent —
either one alone can block the run:

- **VRAM** must hold the weights at the *loaded* precision, plus activations.
- **Volume disk** must hold the weights at the *checkpoint* precision, which
  for a BF16 release is always 2 bytes per parameter regardless of how you
  intend to load them.

| Model | Precision | VRAM needed | Download | GPUs | Volume |
|---|---|---|---|---|---|
| Qwen3-32B | FP16 | 64 GB | 64 GB | 1× 80-94GB | 120 GB |
| Llama-3.3-70B | 8-bit | 70 GB | 141 GB | 1× 80-94GB | 200 GB |
| Llama-3.3-70B | 4-bit | 38 GB | 141 GB | 1× 48GB+ | 200 GB |
| Qwen3-235B-A22B | 4-bit | **129 GB** | **470 GB** | **2× 80-94GB** | **700 GB** |

The 235B row is the one that catches people. A single 94 GB card is 35 GB short
of the 4-bit weights alone, and a 130 GB volume is 340 GB short of the download.
Both must be fixed; fixing one does not help.

Container disk stays at 20-30 GB in every case — only pip packages live there.

Template: **RunPod PyTorch 2.1+ / CUDA 12.x**. Expose ports 8888 (JupyterLab)
and 22 (SSH). Check current per-hour pricing on RunPod — it moves.

### A note on precision and measurement fidelity

Quantization perturbs activations, and activations are exactly what this
project measures. A 0.68 projection gap is being read off vectors that 4-bit
NF4 has rounded. FP16 > 8-bit > 4-bit for measurement fidelity here, which is a
real argument for running a *smaller* model at full precision rather than a
larger one heavily quantized — especially for a first run whose job is to
establish that the pipeline reproduces.

Whatever you choose, keep it consistent with the earlier runs or state the
difference: comparing a 4-bit 235B against an FP16 32B confounds scale with
quantization noise.

### Take PCIe, not SXM

For the 2-GPU configuration, **H100 PCIe is the right choice and it is
cheaper**. `device_map="auto"` uses accelerate's *pipeline* parallelism: it
splits layers across GPUs and passes one activation tensor at each boundary —
a few MB per forward pass. NVLink's bandwidth advantage applies to *tensor*
parallelism (vLLM, DeepSpeed), where every layer synchronises across all GPUs.
This pipeline never does that, so paying the SXM/NVLink premium buys nothing
measurable here.

### Environment variables

Set these in the pod template so they exist before you touch anything. RunPod's
UI takes **Key and Value as two separate fields** — enter the name on the left
and the value on the right, with no `=` sign anywhere.

| Key | Value | Why |
|---|---|---|
| `HF_HOME` | `/workspace/hf-cache` | Cache on the persistent volume, not the ephemeral container disk |
| `HF_XET_HIGH_PERFORMANCE` | `1` | Accelerated downloads |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | Fewer fragmentation OOMs on multi-GPU |
| `HF_TOKEN` | your token, e.g. `hf_xxxxxxxx` | **Only for gated models** |

`HF_TOKEN` is a real credential, not a placeholder — get one at
huggingface.co/settings/tokens (scope: Read). It is required for **Llama and
Mistral**, which are gated and need their licence accepted on the model page.
Qwen models are Apache 2.0 and normally ungated, so you can omit it; add it if
a download returns 401 or 403.

If you would rather not store the token in the template, run
`huggingface-cli login` on the pod instead and paste it when prompted.

---

## Running it

### 1. Get the code onto the pod

The project is ~5 MB without `.venv` and `results`, so any transfer works.

**Prefer `git clone` on the pod.** It sidesteps the problem below entirely and
makes later fixes a one-line pull instead of another upload.

If you do zip: **PowerShell's `Compress-Archive` writes entries with backslash
separators**, which violates the ZIP spec (it mandates `/`). Extracting on Linux
then produces a flat directory of files literally named `src\model_loader.py` —
one file with a backslash in its name, not a `src` directory. The symptom is
`ModuleNotFoundError: No module named 'src'` even though `ls` appears to show
everything. Build the archive with Python instead, which writes correct
separators:

```powershell
cd c:\Users\<you>\AgentHarm\agentic-safety-probe
python -c "import shutil; shutil.make_archive('probe','zip','.')"
```

That includes `.venv` and `results` too. To keep it minimal, stage first:

```powershell
mkdir ..\probe_upload
copy-item src,data -Destination ..\probe_upload -Recurse
copy-item run_experiment.py,bootstrap_delta_p.py,validate_corrections.py,`
  requirements_runpod.txt,tool_definitions.json -Destination ..\probe_upload
python -c "import shutil; shutil.make_archive('probe','zip','../probe_upload')"
```

If you already extracted a backslash-mangled archive on the pod, repair it in
place rather than re-uploading:

```bash
cd /workspace/agentic-safety-probe
python - <<'EOF'
import shutil
from pathlib import Path
root = Path('.')
n = 0
for p in sorted(root.iterdir()):
    if p.is_file() and '\\' in p.name:
        target = root / p.name.replace('\\', '/')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(p), str(target))
        n += 1
print(f"{n} files relocated")
EOF
find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null
```

**On the pod**, pick one:

- **JupyterLab** (simplest): open the pod's 8888 endpoint, navigate to
  `/workspace`, drag `probe.zip` into the file browser.
- **`runpodctl`**: run `runpodctl send probe.zip` locally, then paste the
  one-time code into `runpodctl receive <code>` on the pod.
- **git**: skip the zip entirely and `git clone` — best if you will iterate,
  since pulling a fix later is one command instead of another upload.

Then unpack. The RunPod PyTorch image does **not** ship `unzip`, so use Python,
which is guaranteed present and needs no install:

```bash
cd /workspace
python -c "import zipfile; zipfile.ZipFile('probe.zip').extractall('agentic-safety-probe')"
cd agentic-safety-probe

# Confirm the upload arrived intact before spending on a download
python -c "
from pathlib import Path
need = ['src/model_loader.py','src/probes.py','src/robust_estimation.py',
        'data/dataset_full.jsonl','run_experiment.py','requirements_runpod.txt']
for f in need:
    print(('OK   ' if Path(f).exists() else 'MISSING '), f)
"
```

(`apt-get update && apt-get install -y unzip` also works if you prefer the tool.)

### 2. Install

```bash
pip install -r requirements_runpod.txt
```

**Use `requirements_runpod.txt`, not `requirements.txt`.** The full file pins
`transformer-lens`, which this project never imports, and installing it lets
pip downgrade the CUDA-matched torch that the RunPod image ships with. The
resulting mismatch surfaces as a CUDA error during model loading — after the
weights have already downloaded.

```bash
# Confirm the cache is on the volume and torch still sees the GPUs
echo "HF_HOME=$HF_HOME"
df -h /workspace
python -c "
import torch, transformers
print('torch       :', torch.__version__)
print('transformers:', transformers.__version__)
print('GPUs:', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f'  {p.name}: {p.total_memory/1e9:.0f} GB')
"
```

**Check that output for a `Disabling PyTorch` line.** Recent transformers needs
torch >= 2.5, and the RunPod "PyTorch 2.4.x" templates ship 2.4.1. When they
disagree, transformers does not fail — it prints

```
[transformers] Disabling PyTorch because PyTorch >= 2.5 is required but found 2.4.1+cu124
[transformers] PyTorch was not found. Models won't be available
```

and carries on with its torch backend off. The run then dies at model load with
`NameError: name 'torch' is not defined`, raised from inside transformers'
own `quantization_config.py` — which reads like a bug in this project and is
not. Fix it before downloading weights:

```bash
pip install --upgrade torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124
# or, to keep torch 2.4 instead:  pip install "transformers==4.51.3"
```

Qwen3 support landed in transformers 4.51.0, so that is the floor either way.

**Upgrade torchvision and torchaudio in the same command.** They carry compiled
extensions built against one specific torch; bumping torch alone leaves them
unable to register their operators, and the resulting failure is deeply
misleading:

```
RuntimeError: operator torchvision::nms does not exist
...
ModuleNotFoundError: Could not import module 'Qwen3MoeForCausalLM'
```

The last line is what you see and it is **false** — transformers supports
Qwen3-MoE. It imports torchvision while resolving any architecture, and surfaces
that crash as a missing model class. Downgrading transformers in response makes
it worse. Since this project uses no vision models,
`pip uninstall -y torchvision torchaudio` is an equally valid fix: transformers
detects their absence and skips the import. A broken torchvision is worse than
an absent one.

Verify by exercising the exact import that fails, which costs seconds instead of
a model load:

```bash
python -c "
import torch, torchvision, transformers
print('torch       :', torch.__version__)
print('torchvision :', torchvision.__version__)
print('transformers:', transformers.__version__)
import torchvision.ops; print('torchvision ops: OK')
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM
print('Qwen3MoeForCausalLM: OK')
print('CUDA:', torch.cuda.is_available(), torch.cuda.device_count(), 'GPUs')
"
```

### 3. Download the weights as a separate, resumable step

Do not let the download happen inside the experiment script. Run it on its own
so you can watch progress and resume after a dropped connection, instead of
discovering a partial download three hours in.

At 470 GB this takes 45-90 minutes, so run it inside tmux — a dropped SSH
session otherwise kills the transfer partway through.

```bash
tmux new -s download
hf download Qwen/Qwen3-235B-A22B-Instruct-2507
# detach: Ctrl-b then d      reattach: tmux attach -t download
```

From another shell, watch progress:

```bash
du -sh $HF_HOME && df -h /workspace
```

**On CLI naming:** `huggingface-cli` is deprecated and no longer runs; recent
versions ship the command as `hf`. The old `--local-dir-use-symlinks` flag is
also gone. Without `--local-dir` the weights land in the `$HF_HOME` cache,
which is exactly where `from_pretrained()` looks, so no path wiring is needed.

### 4. Check the pod matches the model

```bash
python -c "
from src.model_loader import estimate_model_memory_gb, total_vram_gb
m = 'Qwen/Qwen3-235B-A22B-Instruct-2507'
print('VRAM present :', round(total_vram_gb()), 'GB')
print('needs FP16   :', round(estimate_model_memory_gb(m, 16)), 'GB')
print('needs 4-bit  :', round(estimate_model_memory_gb(m, 4)), 'GB')
"
```

### 5. Dry run — the flag that actually makes it cheap

```bash
python run_experiment.py \
  --model Qwen/Qwen3-235B-A22B-Instruct-2507 \
  --limit-pairs 4 \
  --skip-intervention --n-repeats 1 --n-label-permutations 0 \
  --output-dir /workspace/dryrun
```

**Use `--limit-pairs`, not `--n-samples`.** `--n-samples` only sizes the
intervention stage; the custom dataset always contributes all 42 pairs to
extraction regardless of it. `--layers` does not help either — every layer is
computed in the same forward pass, so capturing fewer of them saves memory, not
time. `--limit-pairs 4` cuts the forward passes from 168 to 16, which is where
the GPU time actually goes.

The point of the dry run is that the analysis stages execute **after** all
extractions. Without it, a bug in the reporting code costs you the entire
extraction pass. Its statistics are meaningless at N=4 — it is a plumbing check.

Two things to verify in its output:

- the agentic prompt contains the tool schemas. If the tokenizer's chat template
  has no `tools` slot, the code falls back to rendering them into the system
  prompt. That still works, but it is a *different* manipulation and has to be
  stated in the write-up.
- the run reaches "EXPERIMENT COMPLETE" and writes `results.json`.

### 6. The real run — inside tmux

A 3-hour run dies with your SSH session otherwise.

```bash
tmux new -s probe

python run_experiment.py \
  --model Qwen/Qwen3-235B-A22B-Instruct-2507 \
  --layers 10,20,30,40,50,60,70,80 \
  --n-folds 5 --n-repeats 10 --mlp-n-repeats 1 \
  --n-label-permutations 10 --n-bootstrap 10000 \
  --decompose \
  --output-dir /workspace/results_qwen235b \
  2>&1 | tee /workspace/results_qwen235b/run.log

# detach: Ctrl-b then d      reattach: tmux attach -t probe
```

Layer indices must exist in the model — a 94-layer model and an 80-layer model
do not share a layer grid, so do not copy the Llama list blindly. Omit
`--layers` to sweep every second layer automatically.

### 7. Get the results out, then stop paying

```bash
tar czf /workspace/results_qwen235b.tar.gz /workspace/results_qwen235b/
```

Everything you need for the write-up is in that archive; the activations are not
saved, so it is only a few MB. Download it via the RunPod file browser or
`runpodctl send results_qwen235b.tar.gz`.

Then **terminate** the pod. A stopped pod still bills its volume, and at 700 GB
that is not negligible. Everything downstream runs on your laptop.

---

## Cost control

- **Book by the hour, not the day.** The extraction is the only GPU-bound
  stage; everything after it runs on a laptop.
- **Run the statistics locally.** `bootstrap_delta_p.py` and
  `validate_corrections.py` need no GPU. Only the probe cross-validation
  requires the raw activations.
- **Save the activations if you can afford the disk.** They are
  `n_layers × 168 × hidden_size` floats — a few hundred MB. Saving them means
  never re-renting a GPU to re-run an analysis. The pipeline does not currently
  do this, and it is the single highest-value change if you expect to iterate
  on the statistics.

---

## What to report from the new run

| Quantity | Where |
|---|---|
| ΔP with 95% BCa CI | `results.json` → `bootstrap`, or `bootstrap_delta_p.py` |
| Cross-fitted ΔP (bias removed) | `robust_results.json` → `delta_p_crossfit` |
| Probe AUROC, cross-validated | `results.json` → `linear_aurocs` + `linear_auroc_ci_*` |
| Shuffled-label chance level | `results.json` → `linear_aurocs_permuted` |
| Out-of-fold `w_agent` AUROC | `agent_direction_results.json` → `cross_validated` |
| Rotation angle | Only if `split_half_reliability` clears the floor — otherwise report "differs from chance" with no angle |
