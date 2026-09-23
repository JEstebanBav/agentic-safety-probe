import json, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RUNS = {
    "Mistral-7B  FP16 ": "results/mistral7b_fp16",
    "Mistral-7B  4-bit": "results/mistral7b_4bit",
    "Qwen3-32B   4-bit": "results/qwen32b",
    "Llama-3.3-70B    ": "results/llama33",
}

def at(pl, frac):
    """Coseno a una profundidad relativa dada."""
    ks = sorted(pl, key=int)
    top = int(ks[-1])
    target = frac * top
    k = min(ks, key=lambda x: abs(int(x) - target))
    return int(k), pl[k]["cosine"], pl[k]["auroc_d_chat"]

print(f"{'modelo':<18}{'capas':>6}{'cos@25%':>9}{'cos@50%':>9}{'cos@75%':>9}"
      f"{'cos_min':>9}{'cos_OOF':>9}{'reliab':>8}{'AUROC_oof':>11}")
print("-" * 88)

for name, d in RUNS.items():
    f = os.path.join(ROOT, d, "agent_direction_results.json")
    if not os.path.exists(f):
        print(f"{name:<18}  (sin datos: {d})")
        continue
    r  = json.load(open(f))
    pl = r.get("per_layer", {})
    cv = r.get("cross_validated", {})
    if not pl:
        print(f"{name:<18}  (formato antiguo, sin per_layer)")
        continue
    nan = float("nan")
    _, c25, _ = at(pl, 0.25)
    _, c50, _ = at(pl, 0.50)
    _, c75, _ = at(pl, 0.75)
    kmin = min(pl, key=lambda k: pl[k]["cosine"])
    print(f"{name:<18}{len(pl):>6}{c25:>9.3f}{c50:>9.3f}{c75:>9.3f}"
          f"{pl[kmin]['cosine']:>9.3f}"
          f"{cv.get('cosine_with_d_chat_oof', nan):>9.3f}"
          f"{cv.get('split_half_reliability', nan):>8.3f}"
          f"{cv.get('auroc_w_agent_oof', nan):>11.3f}")

print()
print("reliab < 0.25 => el angulo NO es interpretable (MIN_RELIABILITY_FOR_CORRECTION)")
print()

for name, d in RUNS.items():
    f = os.path.join(ROOT, d, "agent_direction_results.json")
    if not os.path.exists(f):
        continue
    r  = json.load(open(f))
    pl = r.get("per_layer", {})
    if not pl:
        continue
    top = int(sorted(pl, key=int)[-1])
    print(f"--- perfil {name.strip()} (best_layer={r['best_layer']}) ---")
    for k in sorted(pl, key=int):
        prof = int(k) / top if top else 0
        bar  = "#" * int(max(pl[k]["cosine"], 0) * 40)
        print("  %3s (%3.0f%%)  cos=%+.3f  d_chat_AUROC=%.3f  %s"
              % (k, prof * 100, pl[k]["cosine"], pl[k]["auroc_d_chat"], bar))
    print()
