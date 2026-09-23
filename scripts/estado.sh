#!/bin/bash
cd ~/agentic-safety-probe
echo "=== PROCESOS ==="
pgrep -u "$USER" -af "run_experiment.py|cola.sh|hf download" || echo "  nada corriendo"
echo
echo "=== GPU ==="
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader
echo
echo "=== DESCARGA Qwen3-8B ==="
tail -2 ~/download_qwen8.log 2>/dev/null || echo "  (aun sin log)"
echo
for L in run_m7b_4bit.log cola.log; do
  [ -f "$L" ] || continue
  echo "=== $L  (escrito hace $(( ($(date +%s) - $(stat -c %Y "$L")) / 60 )) min) ==="
  tail -8 "$L"
  echo
done
echo "=== CORRIDAS TERMINADAS ==="
grep -l "EXPERIMENT COMPLETE" *.log 2>/dev/null || echo "  ninguna aun"
