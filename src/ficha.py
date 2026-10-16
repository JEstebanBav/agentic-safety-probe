"""
Run sheet ("ficha"): what model, in what state, with which parameters.

A results folder holds numbers; it does not say how the model was loaded,
which corpus it saw or which stages ran. Those facts are scattered over
results.json (which older runs saved without their configuration), the run
log and the memory of whoever launched it. The sheet collects them in one
FICHA.md per run so two versions can be told apart without opening JSON.

Sources, in order of trust:
  1. results.json / causal_results.json / the files present in the folder;
  2. the header of run.log, for older runs that did not save their config;
  3. results/registro.json, for what no file records (exact model id, where
     the run came from, what it complements, where it is cited).
Anything none of them records is printed as "no registrado", never guessed.
The one inference made is precision for older runs, from the memory the model
occupied; the sheet says so when it does.

run_experiment.py writes the sheet at the end of every run;
scripts/generar_fichas.py regenerates all of them and the results/ index.

Pure standard library + numpy-free on purpose: it must run on a laptop
against results downloaded from a pod.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

REGISTRO = Path(__file__).resolve().parent.parent / "results" / "registro.json"
NO = "no registrado"

# Parameters in billions, for the precision inference and the sheet.
PARAMS_B = {
    "meta-llama/Llama-3.3-70B-Instruct": 70.6,
    "meta-llama/Llama-3.1-70B-Instruct": 70.6,
    "mistralai/Mistral-7B-Instruct-v0.3": 7.25,
    "Qwen/Qwen3-8B": 8.2,
    "Qwen/Qwen3-32B": 32.8,
}

# Configuration keys shown in the parameter table, with a Spanish label.
CONFIG_KEYS = [
    ("dataset_path", "Corpus (--dataset-path)"),
    ("layers", "Capas extraídas (--layers)"),
    ("limit_pairs", "Límite de pares (--limit-pairs)"),
    ("factorial", "Factorial sistema × herramientas"),
    ("agency_level", "Nivel de agencia"),
    ("toolbox", "Caja de herramientas"),
    ("include_history", "Historial de herramientas"),
    ("n_folds", "Folds"),
    ("n_repeats", "Repeticiones sonda lineal"),
    ("mlp_n_repeats", "Repeticiones sonda MLP"),
    ("n_label_permutations", "Permutaciones de etiquetas"),
    ("n_bootstrap", "Remuestreos bootstrap"),
    ("n_selection_splits", "Particiones para selección de capa"),
    ("skip_probes", "Sondas omitidas"),
    ("skip_intervention", "Generación omitida"),
    ("intervention_layers", "Capas del test causal"),
    ("intervention_doses", "Dosis del test causal"),
    ("intervention_pairs", "Pares del test causal"),
    ("max_new_tokens", "Tokens de respuesta"),
    ("thinking_budget", "Presupuesto de razonamiento"),
    ("seed", "Semilla"),
    ("output_dir", "Carpeta de salida original"),
]


# ------------------------------------------------------------------
# Reading
# ------------------------------------------------------------------

def _load(path: Path) -> Optional[Dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def load_registro(path: Path = REGISTRO) -> Dict[str, Dict]:
    data = _load(path) or {}
    return data.get("corridas", {})


def registro_key(run_dir: Path) -> str:
    """Folder path relative to results/, with forward slashes."""
    parts = run_dir.resolve().parts
    if "results" in parts:
        i = len(parts) - 1 - parts[::-1].index("results")
        return "/".join(parts[i + 1:])
    return run_dir.name


def _log_header(run_dir: Path) -> Dict[str, str]:
    """Model / Quantize / Source lines printed at the start of every run."""
    out: Dict[str, str] = {}
    for log in sorted(run_dir.glob("*.log")):
        try:
            text = log.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for key, pattern in [("model", r"^Model:\s*(.+)$"),
                             ("quantize", r"^Quantize:\s*(.+)$"),
                             ("source", r"Source:\s*(.+)$"),
                             ("factorial", r"^Factorial:\s*(.+)$"),
                             ("agency", r"^Agency level:\s*(.+)$"),
                             ("layers", r"^Analyzing layers:\s*\[(.+)\]$"),
                             ("entries", r"Total entries:\s*(\d+)"),
                             ("total_time", r"^Total time:\s*(.+)$")]:
            # "Best layer:" is deliberately not read: it is printed by more
            # than one stage, with different meanings.
            m = None if key in out else re.search(pattern, text, flags=re.M)
            if m:
                out[key] = m.group(1).strip()
    return out


def _n_pairs(results: Dict) -> Optional[int]:
    for key in ("projections_by_layer", "projections"):
        block = results.get(key)
        if isinstance(block, dict) and block:
            first = next(iter(block.values()))
            if isinstance(first, dict) and "chat_harmful" in first:
                return len(first["chat_harmful"])
            if key == "projections" and "chat_harmful" in block:
                return len(block["chat_harmful"])
    return None


def _precision(config: Dict, log: Dict, model_info: Dict,
               model: Optional[str]) -> str:
    if config.get("quantize"):
        return "4-bit NF4 (bitsandbytes), forzado con --quantize"
    if config.get("no_quantize"):
        return "FP16, forzado con --no-quantize"
    q = (log.get("quantize") or "").lower()
    if "4-bit" in q or q == "true":
        return "4-bit NF4 (bitsandbytes), según el log"
    if "fp16" in q or q == "false":
        return "FP16, según el log"
    mem, params = model_info.get("memory_gb"), PARAMS_B.get(model or "")
    if mem and params:
        ratio = mem / params
        kind = ("FP16" if ratio >= 1.5 else "4-bit NF4 (bitsandbytes)"
                if ratio <= 0.8 else "8-bit")
        return (f"{kind} — deducida: ocupaba {mem:.1f} GB para "
                f"{params:g} mil millones de parámetros")
    return NO


def _tool_calls_from_generations(run_dir: Path) -> Dict[str, List[int]]:
    """Tool-call counts per condition, recomputed for runs older than the field."""
    path = run_dir / "behavior_generations.jsonl"
    if not path.exists():
        return {}
    try:
        from src.causal_test import answer_has_tool_call
    except ImportError:
        return {}
    counts: Dict[str, List[int]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("arm") != "baseline":
            continue
        k, n = counts.setdefault(r["variant"], [0, 0])
        counts[r["variant"]] = [k + int(answer_has_tool_call(r["text"])), n + 1]
    return counts


def collect(run_dir: Path, registro: Optional[Dict[str, Dict]] = None) -> Dict:
    """Everything the sheet shows, as plain data."""
    run_dir = Path(run_dir)
    if registro is None:
        registro = load_registro()
    key = registro_key(run_dir)
    reg = registro.get(key, {})
    results = _load(run_dir / "results.json") or {}
    causal = _load(run_dir / "causal_results.json") or {}
    config = results.get("config") or {}
    log = _log_header(run_dir)
    model_info = results.get("model_info") or {}
    model = config.get("model") or reg.get("modelo") or log.get("model")

    best = results.get("best_layer")
    robust = results.get("robust") or {}
    sel = (robust.get("layer_selection") or {}) if isinstance(robust, dict) else {}
    pca = (results.get("pca_by_layer") or {}).get(str(best)) if best is not None else None

    files = sorted(p.name for p in run_dir.iterdir() if p.is_file() and p.name != "FICHA.md")
    has = set(files)
    stages = [
        ("Activaciones guardadas (activations.npz, solo local)", "activations.npz" in has),
        ("Dirección de rechazo y ΔP por capa", bool(results.get("gaps_by_layer"))),
        ("Validación PCA del contraste", bool(results.get("pca_by_layer"))),
        ("Evaluación fuera de muestra (held-out)", bool(results.get("heldout"))),
        ("Sondas lineales con permutación", bool(results.get("linear_aurocs"))),
        ("Sondas MLP", bool(results.get("mlp_aurocs"))),
        ("Análisis robusto y selección de capa", bool(robust)),
        ("Factorial sistema × herramientas", "factorial_results.json" in has
         or bool(results.get("factorial"))),
        ("Línea base de comportamiento", bool(causal.get("behavior"))),
        ("Test causal calibrado (d_chat vs. aleatoria)", bool(causal.get("causal"))),
        ("Intervención antigua con α arbitrarios (no calibrada)",
         "agent_intervention_results.json" in has and not causal.get("causal")),
    ]

    beh = causal.get("behavior") or {}
    tool = {v: [t["k"], t["n"]] for v, t in (beh.get("tool_call") or {}).items()}
    if not tool:
        tool = _tool_calls_from_generations(run_dir)

    if not reg.get("fecha") and (run_dir / "results.json").exists():
        import datetime
        stamp = datetime.date.fromtimestamp((run_dir / "results.json").stat().st_mtime)
        reg = dict(reg, fecha=f"{stamp.isoformat()} (fecha del archivo results.json)")

    return {
        "carpeta": key,
        "registro": reg,
        "modelo": model or NO,
        "params_b": PARAMS_B.get(model or ""),
        "model_info": model_info,
        "precision": _precision(config, log, model_info, model),
        "corpus": config.get("dataset_path") or log.get("source") or reg.get("corpus")
                  or (None if not results else "data/dataset_full.jsonl (por defecto)"),
        "n_pares": _n_pairs(results) or (
            int(log["entries"]) // 4 if log.get("entries") else None),
        "capas": results.get("layers_analyzed") or sorted(
            int(l) for l in (results.get("gaps_by_layer") or {})) or [
            int(x) for x in log.get("layers", "").split(",") if x.strip()],
        "capa_referencia": best,
        "capa_seleccionada": sel.get("selected_layer"),
        "prompt_tokens": results.get("prompt_tokens") or {},
        "config": config,
        "log": log,
        "etapas": stages,
        "bootstrap": results.get("bootstrap") or {},
        "ttest": results.get("ttest") or {},
        "especificidad": results.get("specificity") or {},
        "heldout": (results.get("heldout") or {}).get("detection") or {},
        "pca": pca,
        "comportamiento": beh.get("refusal") or {},
        "llamadas_herramienta": tool,
        "causal": causal.get("causal") or {},
        "archivos": files,
        "tiene_resultados": bool(results),
    }


# ------------------------------------------------------------------
# Rendering
# ------------------------------------------------------------------

def _fmt(v, nd=3) -> str:
    if v is None:
        return NO
    if isinstance(v, bool):
        return "sí" if v else "no"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def render(d: Dict) -> str:
    reg = d["registro"]
    mi = d["model_info"]
    L: List[str] = []
    titulo = reg.get("etiqueta") or d["carpeta"]
    L += [f"# Ficha · {titulo}", ""]
    if reg.get("resumen"):
        L += [reg["resumen"], ""]
    L += ["> Generada por `scripts/generar_fichas.py` a partir de los archivos de "
          "esta carpeta y de `results/registro.json`. No la edites a mano: "
          "corrige el registro y regenera.", ""]

    L += ["## Identificación", "", "| | |", "|---|---|",
          f"| Carpeta | `results/{d['carpeta']}/` |",
          f"| Tipo | {reg.get('tipo', NO)} |",
          f"| Fecha | {reg.get('fecha', NO)} |",
          f"| Carpeta anterior | {('`' + reg['origen'] + '`') if reg.get('origen') else NO} |",
          f"| Relación con otras versiones | {reg.get('relacion', NO)} |",
          f"| Citada en | {reg.get('citada_en', 'no')} |", ""]

    L += ["## El modelo en esta corrida", "", "| | |", "|---|---|",
          f"| Modelo | `{d['modelo']}` |",
          f"| Parámetros | {(_fmt(d['params_b'], 2) + ' mil millones') if d['params_b'] else NO} |",
          f"| Arquitectura | " + (f"{mi.get('model_type')}, {mi.get('num_layers')} capas, "
                                  f"dimensión {mi.get('hidden_size')}" if mi else NO) + " |",
          f"| Precisión de carga | {d['precision']} |",
          f"| Memoria ocupada | " + (f"{mi['memory_gb']:.1f} GB" if mi.get("memory_gb") else NO) + " |",
          f"| Posición de lectura | último token del prompt (cabecera del asistente), "
          f"antes de generar |",
          f"| Capas analizadas | {', '.join(map(str, d['capas'])) or NO} |",
          f"| Capa de referencia (máximo ΔP) | {_fmt(d['capa_referencia'])} |",
          f"| Capa por selección en muestra dividida | {_fmt(d['capa_seleccionada'])} |", ""]
    pt = d["prompt_tokens"]
    if pt:
        L += ["Longitud de los prompts (tokens):", "",
              "| Condición | media | mín | máx |", "|---|---|---|---|"]
        for v in sorted(pt):
            s = pt[v]
            L.append(f"| {v} | {s['mean']:.0f} | {s['min']} | {s['max']} |")
        L.append("")

    L += ["## Corpus", "", "| | |", "|---|---|",
          f"| Archivo | {d['corpus'] or NO} |",
          f"| Pares | {_fmt(d['n_pares'])} |", ""]

    L += ["## Parámetros de la corrida", ""]
    cfg = d["config"]
    if cfg:
        L += ["| Parámetro | Valor |", "|---|---|"]
        for k, label in CONFIG_KEYS:
            if k in cfg:
                v = cfg[k]
                L.append(f"| {label} | " + ("valor por defecto" if v is None
                                            else f"`{v}`") + " |")
        L.append("")
    else:
        L += ["No hay `results.json` en esta carpeta." if not d["tiene_resultados"]
              else "Esta corrida es de una versión del pipeline que no guardaba su "
                   "configuración en `results.json`."]
        if d["log"]:
            L += ["Del encabezado del log:", ""]
            for k, label in [("model", "Modelo"), ("quantize", "Precisión"),
                             ("source", "Corpus"), ("factorial", "Factorial"),
                             ("agency", "Nivel de agencia")]:
                if k in d["log"]:
                    L.append(f"- {label}: `{d['log'][k]}`")
        L.append("")
    if d["log"].get("total_time"):
        L += [f"Duración registrada: {d['log']['total_time']}.", ""]

    L += ["## Etapas ejecutadas", ""]
    if d["tiene_resultados"]:
        L += [f"- [{'x' if ok else ' '}] {name}" for name, ok in d["etapas"]]
    else:
        L.append("No se pueden comprobar desde archivos: solo hay log.")
    L.append("")

    L += ["## Resultados clave", ""]
    if not d["tiene_resultados"]:
        L += ["Esta carpeta no tiene `results.json`; ver el log.", ""]
    else:
        b, t, sp, h = d["bootstrap"], d["ttest"], d["especificidad"], d["heldout"]
        rows = []
        if b:
            rows.append(f"| ΔP dañinas (capa {_fmt(d['capa_referencia'])}) | "
                        f"{_fmt(b.get('mean_delta_p'))} "
                        f"[IC 95 % {_fmt(b.get('ci_lower'))}, {_fmt(b.get('ci_upper'))}] |")
        if isinstance(t, dict) and t.get("p_value") is not None:
            rows.append(f"| Prueba t pareada | p = {t['p_value']:.2e}, "
                        f"d_z = {_fmt(t.get('effect_size'), 2)} |")
        if isinstance(sp, dict) and sp.get("interaction") is not None:
            rows.append(f"| Control benigno | ΔP benignas {_fmt(sp.get('delta_p_benign'))}; "
                        f"interacción {_fmt(sp.get('interaction'))} "
                        f"(p = {_fmt(sp.get('interaction_p'), 4)}) |")
        if h:
            rows.append(f"| AUROC fuera de muestra | chat {_fmt(h.get('auroc_chat_heldout'))}, "
                        f"agente {_fmt(h.get('auroc_agent_heldout'))} |")
        if d["pca"]:
            p = d["pca"]
            rows.append(f"| PCA en la capa de referencia | cos(PC1, d) = "
                        f"{_fmt(p.get('cosine_similarity_with_diff_means'))}, "
                        f"varianza PC1 {p.get('explained_variance_ratio_pc1', 0):.1%} |")
        if rows:
            L += ["| Medida | Valor |", "|---|---|"] + rows + [""]

        if d["comportamiento"]:
            L += ["Comportamiento sin intervención:", "",
                  "| Condición | Rechaza | Llama una herramienta |", "|---|---|---|"]
            for v in ("chat_harmful", "agent_harmful", "chat_benign", "agent_benign"):
                r = d["comportamiento"].get(v)
                if not r:
                    continue
                tc = d["llamadas_herramienta"].get(v)
                tcs = f"{tc[0]}/{tc[1]}" if tc else NO
                L.append(f"| {v} | {r['k']}/{r['n']} ({r['rate']:.0%}) | {tcs} |")
            L.append("")
        if d["causal"]:
            L += ["Test causal (dosis en unidades del ΔP medido):", "",
                  "| Capa | Dosis | Restaura rechazo | Control aleatorio | Benignas rechazadas |",
                  "|---|---|---|---|---|"]
            for layer, c in d["causal"].items():
                for i, dose in enumerate(c.get("doses", [])):
                    e = c["per_dose"].get(f"{dose:g}", {})
                    cell = lambda a: (f"{e[a]['k']}/{e[a]['n']}" if a in e else "-")
                    L.append(f"| {layer} | ×{dose:g} | {cell('restore')} | "
                             f"{cell('control')} | {cell('benign')} |")
            L.append("")

    if reg.get("notas"):
        L += ["## Notas", ""] + [f"- {n}" for n in reg["notas"]] + [""]

    L += ["## Archivos", ""] + [f"- `{f}`" for f in d["archivos"]] + [""]
    return "\n".join(L)


def write_ficha(run_dir, registro: Optional[Dict[str, Dict]] = None) -> Path:
    run_dir = Path(run_dir)
    path = run_dir / "FICHA.md"
    path.write_text(render(collect(run_dir, registro)), encoding="utf-8")
    return path


# ------------------------------------------------------------------
# Index of results/
# ------------------------------------------------------------------

def _etapas_cortas(d: Dict) -> str:
    marks = dict(d["etapas"])
    short = [("PCA", "Validación PCA del contraste"),
             ("held-out", "Evaluación fuera de muestra (held-out)"),
             ("factorial", "Factorial sistema × herramientas"),
             ("conducta", "Línea base de comportamiento"),
             ("causal", "Test causal calibrado (d_chat vs. aleatoria)")]
    got = [s for s, k in short if marks.get(k)]
    if marks.get("Intervención antigua con α arbitrarios (no calibrada)"):
        got.append("interv. antigua")
    return ", ".join(got) or "-"


def render_index(rows: List[Dict], intro: str) -> str:
    L = [intro.rstrip(), ""]
    for tipo, titulo in (("corridas", "Corridas"), ("pruebas", "Corridas de prueba")):
        sel = [d for d in rows if d["carpeta"].startswith(tipo + "/")]
        if not sel:
            continue
        L += [f"## {titulo}", "",
              "| Versión | Modelo | Precisión | Pares | Capa ref. | Etapas | Uso |",
              "|---|---|---|---|---|---|---|"]
        for d in sel:
            reg = d["registro"]
            prec = d["precision"].split(" ")[0].split(",")[0]
            L.append(
                f"| [`{d['carpeta'][len(tipo) + 1:]}`]({d['carpeta']}/FICHA.md) | "
                f"{d['modelo'].split('/')[-1]} | {prec} | {_fmt(d['n_pares'])} | "
                f"{_fmt(d['capa_referencia'])} | {_etapas_cortas(d)} | "
                f"{reg.get('uso', '')} |")
        L.append("")
    return "\n".join(L)
