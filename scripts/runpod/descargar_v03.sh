#!/bin/bash
# Descarga los modelos de una maquina sin las copias que transformers no usa.
#
#   bash scripts/runpod/descargar_v03.sh <h200|l40s|glm>
#
# Varios repositorios traen los mismos pesos en otros formatos: Llama-3.3 una
# carpeta original/ (~141 GB de .pth), gpt-oss original/ y metal/ (triplican
# su tamano), Mistral un consolidated.safetensors. Bajarlas todas paso de los
# 600 GB del volumen en el primer pod. Se descarga en segundo plano; el
# progreso queda en /workspace/descarga.log.
set -u
case "${1:?uso: descargar_v03.sh <h200|l40s|glm>}" in
  h200) MODELOS="meta-llama/Llama-3.3-70B-Instruct Qwen/Qwen3-32B Qwen/Qwen3.8-27B google/gemma-4-31B-it meta-models/Muse-Glimmer-30B openai/gpt-oss-20b openai/gpt-oss-120b" ;;
  l40s) MODELOS="mistralai/Mistral-7B-Instruct-v0.3 Qwen/Qwen3-8B" ;;
  glm)  MODELOS="zai-org/GLM-5.3-Flash" ;;
  *) echo "maquina desconocida: $1"; exit 1 ;;
esac
nohup python - $MODELOS > /workspace/descarga.log 2>&1 <<'EOF' &
import sys, time
from huggingface_hub import snapshot_download
SKIP = ["original/*", "metal/*", "consolidated*.safetensors", "*.pth", "*.gguf"]
for m in sys.argv[1:]:
    print(f"=== {m} {time.strftime('%H:%M:%S')}", flush=True)
    try:
        snapshot_download(m, ignore_patterns=SKIP)
    except Exception as e:                       # noqa: BLE001
        print(f"FALLO {m}: {type(e).__name__}: {e}", flush=True)
print(f"=== DESCARGA TERMINADA {time.strftime('%H:%M:%S')}", flush=True)
EOF
echo "Descarga en segundo plano. Progreso: tail -3 /workspace/descarga.log"
