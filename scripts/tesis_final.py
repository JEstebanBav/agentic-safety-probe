# -*- coding: utf-8 -*-
"""
Métricas y figuras de la versión final de la tesis (corpus de 96 pares).

Lee las cuatro corridas de results/ultimos_resultados/ y escribe:
  docs/tesis/Tesis_final/evidencia/metricas_finales.json   cifras y procedencia
  docs/tesis/Tesis_final/figures/*.pdf|png                  figuras del documento

Cada cifra del texto de la tesis sale de este archivo de evidencia. El script
no ejecuta modelos: solo lee resultados guardados.

Uso, desde la raíz del repositorio:
  python scripts/tesis_final.py
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.causal_test import answer_has_tool_call  # noqa: E402

RES = ROOT / "results" / "ultimos_resultados"
OUT = ROOT / "docs" / "tesis" / "Tesis_final"
MODELS = {
    "llama": {"dir": "llama3380B_v4", "name": "Llama-3.3-70B", "version": "V4", "L": 80},
    "mistral": {"dir": "mistral7b_v4", "name": "Mistral-7B", "version": "V4", "L": 32},
    "qwen8b": {"dir": "qwen_8B_v3", "name": "Qwen3-8B", "version": "V3", "L": 36},
    "qwen32b": {"dir": "qwen32B_v3", "name": "Qwen3-32B", "version": "V3", "L": 64},
}
STRATA = ["explicit", "contextual", "framed"]
COLORS = {"llama": "#33365d", "mistral": "#a33c32", "qwen8b": "#26778b", "qwen32b": "#c88045"}


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def model_metrics(key):
    spec = MODELS[key]
    base = RES / spec["dir"]
    d = load(base / "results.json")
    rb = load(base / "robust_results.json")
    fac = load(base / "factorial_results.json")
    cfg = d["config"]
    L = str(d["best_layer"])
    ho = d["heldout"]
    fe = d["format_effect"]["held_out"]["per_layer"][L]
    det = ho["detection"]
    lin = d["linear_suite"]["per_layer"][L]
    rpl = rb["per_layer"][L]
    rot = rb["rotation"][L]
    sel = rb["layer_selection"]
    n_pairs = len(ho["projections"]["chat_harmful"])
    assert cfg["dataset_path"] == "data/dataset_v2.jsonl" and n_pairs == 96, key

    m = {
        "modelo": spec["name"], "version": spec["version"],
        "fuente": f"results/ultimos_resultados/{spec['dir']}",
        "hf_id": cfg["model"], "precision": "FP16" if cfg.get("no_quantize") else "auto",
        "n_pares": n_pairs, "capas_total": spec["L"],
        "capas_evaluadas": d["layers_analyzed"],
        "capa_referencia": int(L),
        "deteccion": {
            "auroc_chat": det["auroc_chat_heldout"], "auroc_agente": det["auroc_agent_heldout"],
            "auroc_chat_ic": det["auroc_chat_ci"], "auroc_agente_ic": det["auroc_agent_ci"],
            "brecha": det["auroc_gap_heldout"], "brecha_ic": det["auroc_gap_ci"],
            "brecha_p": det["p_value_gap"], "optimismo_chat": det["optimism_chat"],
            "optimismo_agente": det["optimism_agent"],
            "coseno_entre_folds": det["inter_fold_cosine_mean"],
        },
        "desplazamiento": {
            "dp_h": fe["delta_p_harmful"], "dp_b": fe["delta_p_benign"],
            "interaccion": fe["interaction"], "interaccion_ic": fe["interaction_ci"],
            "interaccion_p": fe["interaction_p"], "fraccion_comun": fe["common_shift_fraction"],
            "pares_interaccion_positiva": fe["n_pairs_interaction_positive"],
            "dp_h_ic": [ho["paired_harmful"]["ci_lower"], ho["paired_harmful"]["ci_upper"]],
            "dp_h_dz": ho["paired_harmful"]["cohens_d_paired"],
            "dp_h_decrecen": ho["paired_harmful"]["n_decreasing"],
            "dp_h_signos_p": ho["paired_harmful"]["sign_test_p"],
            "dp_b_ic": [ho["paired_benign"]["ci_lower"], ho["paired_benign"]["ci_upper"]],
            "benigno_p_bilateral": ho["specificity"]["benign_shift_p"],
            "interaccion_dz": ho["specificity"]["interaction_cohens_d"],
        },
        "escala": {
            "perdida_cruda": fe["separation_loss_raw"],
            "perdida_estandarizada": fe["separation_loss_standardised"],
            "d_chat": fe["cohens_d_chat"], "d_agente": fe["cohens_d_agent"],
            "kappa": fe["scale_contraction"],
        },
        "estratos": {s["stratum"]: {
            "n": s["n_harmful"], "auroc_chat": s["auroc_chat"], "auroc_agente": s["auroc_agent"],
            "perdida_auroc": s["auroc_loss"], "d_chat": s["cohens_d_chat"],
            "d_agente": s["cohens_d_agent"], "perdida_d": s["separation_loss_standardised"],
        } for s in d["format_effect"]["held_out"]["by_stratum"]},
        "sonda_lineal": {
            "auroc": lin["auroc"], "sd_folds": lin["auroc_std"], "ic": [lin["auroc_ci_lower"], lin["auroc_ci_upper"]],
            "permutado": lin["auroc_permuted_mean"], "dentro_muestra": lin["auroc_in_sample"],
            "repeticiones": lin["n_repeats"],
        },
        "lineal_vs_mlp": d.get("probe_comparison", {}).get(L),
        "circularidad": {
            "dp_ingenuo": rpl["delta_p_naive"], "dp_crossfit": rpl["delta_p_crossfit"],
            "sesgo": rpl["bias_estimate"], "decrecen": rpl["n_decreasing"], "pares": rpl["n_pairs"],
            "signos_p": rpl["sign_test_p"],
        },
        "monitores": {
            "medias": {"auroc": rpl["auroc_agent_diff_means_oof"], "f1": rpl["monitor_f1_diff_means_oof"],
                       "fpr": rpl["monitor_fpr_diff_means_oof"]},
            "logistica": {"auroc": rpl["auroc_w_agent_oof"], "f1": rpl["monitor_f1_oof"],
                          "fpr": rpl["monitor_fpr_oof"], "precision": rpl["monitor_precision_oof"],
                          "recall": rpl["monitor_recall_oof"]},
        },
        "rotacion": {meth: {k: rot[meth][k] for k in (
            "cosine_full_sample", "reliability_chat", "reliability_agent", "ceiling",
            "cosine_normalised", "angle_deg_normalised")} for meth in ("diff_means", "logistic")},
        "seleccion": {
            "n_seleccion": sel["n_selection"], "n_inferencia": sel["n_inference"],
            "capa": sel["selected_layer"], "bonferroni": sel["bonferroni_alpha"],
            "dp_inferencia": sel["inference"]["delta_p"],
            "ic": [sel["inference"]["ci_lower"], sel["inference"]["ci_upper"]],
            "decrecen": sel["inference"]["n_decreasing"], "pares": sel["inference"]["n_pairs"],
            "signos_p": sel["inference"]["sign_test_p"],
            "dz": sel["inference"]["cohens_d_paired"],
            "interaccion_inferencia": sel.get("inference_interaction", {}).get("interaction"),
            "interaccion_inferencia_ic": [sel.get("inference_interaction", {}).get("ci_lower"),
                                          sel.get("inference_interaction", {}).get("ci_upper")],
            "estabilidad_frecuencia": sel["stability"]["selected_layer_frequency"],
            "estabilidad_frac_decrecen_media": sel["stability"]["fraction_decreasing_mean"],
        },
        "pca": {l: {"coseno": v["cosine_similarity_with_diff_means"],
                    "varianza_pc1": v["explained_variance_ratio_pc1"],
                    "componente_mejor": v["best_aligned_component"],
                    "entre_clases_sobre_d": v["between_class_share_along_direction"]}
                for l, v in d["pca_by_layer"].items()},
        "factorial_referencia": {
            blk: {k: {"delta": v["delta"], "ic": [v["ci_lower"], v["ci_upper"]], "p": v["p_permutation"]}
                  for k, v in fac["by_layer"][L][blk].items()}
            for blk in ("harmful", "interaction")},
        "por_capa": {l: {
            "dp_h": d["format_effect"]["held_out"]["per_layer"][l]["delta_p_harmful"],
            "dp_b": d["format_effect"]["held_out"]["per_layer"][l]["delta_p_benign"],
            "interaccion": d["format_effect"]["held_out"]["per_layer"][l]["interaction"],
            "auroc_chat": d["format_effect"]["held_out"]["per_layer"][l]["auroc_chat"],
            "auroc_agente": d["format_effect"]["held_out"]["per_layer"][l]["auroc_agent"],
            "coseno_medias": rb["rotation"][l]["diff_means"]["cosine_full_sample"],
            "coseno_logistica": rb["rotation"][l]["logistic"]["cosine_full_sample"],
            "factorial_sistema": fac["by_layer"][l]["interaction"]["system"]["delta"],
            "factorial_herramientas": fac["by_layer"][l]["interaction"]["tools"]["delta"],
            "factorial_total": fac["by_layer"][l]["interaction"]["total"]["delta"],
        } for l in d["format_effect"]["held_out"]["per_layer"]},
        "archivos_sha256": {f.name: sha(f) for f in sorted(base.glob("*.json"))},
    }

    causal_path = base / "causal_results.json"
    if causal_path.exists():
        c = load(causal_path)
        beh = c["behavior"]
        tool = {}
        gens = base / "behavior_generations.jsonl"
        for line in gens.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r["arm"] == "baseline":
                k, n = tool.get(r["variant"], (0, 0))
                tool[r["variant"]] = (k + int(answer_has_tool_call(r["text"])), n + 1)
        m["conducta"] = {
            "rechazo": {v: [x["k"], x["n"], x["ci_lower"], x["ci_upper"]] for v, x in beh["refusal"].items()},
            "degeneradas": beh["degenerate"],
            "llamadas_herramienta": {v: list(x) for v, x in tool.items()},
            "brecha_danino": beh["gap_harmful"], "brecha_benigno": beh["gap_benign"],
            "proyeccion_vs_rechazo": beh["projection_vs_behavior"],
        }
        m["causal"] = {L_: {
            "unidad": s["dose_unit"],
            "dosis": {dd: {
                "alpha": e["alpha"], "desplazamiento": e["projection_shift_restore"],
                "restaura": [e["restore"]["k"], e["restore"]["n"], e["restore"]["ci_lower"], e["restore"]["ci_upper"], e["restore"]["n_degenerate"]],
                "control": [e["control"]["k"], e["control"]["n"], e["control"]["ci_lower"], e["control"]["ci_upper"], e["control"]["n_degenerate"]],
                "benigno": [e["benign"]["k"], e["benign"]["n"], e["benign"]["ci_lower"], e["benign"]["ci_upper"], e["benign"]["n_degenerate"]],
                "mcnemar_p": e["restore_vs_control"]["mcnemar_p"],
            } for dd, e in s["per_dose"].items()},
        } for L_, s in c["causal"].items()}
    return m


# ------------------------------------------------------------------
# Figuras
# ------------------------------------------------------------------

def save(fig, name):
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "figures" / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / "figures" / f"{name}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def figures(M):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.18})
    keys = list(MODELS)

    # 1. Desplazamiento e interacción por capa; AUROC por capa
    fig, ax = plt.subplots(2, 4, figsize=(13, 5.6))
    for j, k in enumerate(keys):
        pc = M[k]["por_capa"]; ls = sorted(int(x) for x in pc)
        for key, lab, col in [("dp_h", "Dañino", "#a33c32"), ("dp_b", "Benigno", "#26778b"),
                              ("interaccion", "Interacción", "#33365d")]:
            ax[0, j].plot(ls, [pc[str(l)][key] for l in ls], label=lab, color=col)
        ax[0, j].axhline(0, color="gray", lw=0.7)
        ax[0, j].axvline(M[k]["capa_referencia"], color="gray", lw=0.7, ls=":")
        ax[0, j].set(title=f"{M[k]['modelo']} {M[k]['version']}", xlabel="Capa")
        ax[1, j].plot(ls, [pc[str(l)]["auroc_chat"] for l in ls], label="Chat", color="#26778b")
        ax[1, j].plot(ls, [pc[str(l)]["auroc_agente"] for l in ls], label="Agente", color="#c88045")
        ax[1, j].axhline(0.5, color="gray", lw=0.7, ls="--")
        ax[1, j].axvline(M[k]["capa_referencia"], color="gray", lw=0.7, ls=":")
        ax[1, j].set(ylim=(0.4, 1.02), xlabel="Capa")
    ax[0, 0].set_ylabel("Cambio de proyección (OOF)")
    ax[1, 0].set_ylabel("AUROC de la dirección\nconversacional (OOF)")
    ax[0, 0].legend(fontsize=7); ax[1, 0].legend(fontsize=7)
    fig.tight_layout(); save(fig, "final_capas")

    # 2. Estratos
    fig, ax = plt.subplots(1, 4, figsize=(13, 3.4), sharey=True)
    x = np.arange(3)
    for j, k in enumerate(keys):
        st = M[k]["estratos"]
        ax[j].bar(x - .18, [st[s]["auroc_chat"] for s in STRATA], .36, label="Chat", color="#26778b")
        ax[j].bar(x + .18, [st[s]["auroc_agente"] for s in STRATA], .36, label="Agente", color="#c88045")
        ax[j].set(xticks=x, xticklabels=[f"Explícito\nn={st['explicit']['n']}", f"Contextual\nn={st['contextual']['n']}",
                                         f"Ap. legítima\nn={st['framed']['n']}"],
                  ylim=(0.5, 1.02), title=f"{M[k]['modelo']} · capa {M[k]['capa_referencia']}")
    ax[0].set_ylabel("AUROC fuera de muestra"); ax[0].legend(fontsize=7)
    fig.tight_layout(); save(fig, "final_estratos")

    # 3. PCA por profundidad relativa
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    for k in keys:
        p = M[k]["pca"]; ls = sorted(int(x) for x in p); rel = [l / M[k]["capas_total"] for l in ls]
        ax[0].plot(rel, [p[str(l)]["coseno"] for l in ls], label=M[k]["modelo"], color=COLORS[k])
        ax[1].plot(rel, [100 * p[str(l)]["varianza_pc1"] for l in ls], label=M[k]["modelo"], color=COLORS[k])
    ax[0].set(xlabel="Profundidad relativa (capa / L)", ylabel="|cos(PC1, dirección)|", ylim=(0, 1.05))
    ax[1].set(xlabel="Profundidad relativa (capa / L)", ylabel="Varianza explicada por PC1 (%)")
    ax[0].legend(fontsize=7)
    fig.tight_layout(); save(fig, "final_pca")

    # 4. Reorientación (coseno entre formatos, mismo estimador)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)
    for k in keys:
        pc = M[k]["por_capa"]; ls = sorted(int(x) for x in pc); rel = [l / M[k]["capas_total"] for l in ls]
        ax[0].plot(rel, [pc[str(l)]["coseno_medias"] for l in ls], label=M[k]["modelo"], color=COLORS[k])
        ax[1].plot(rel, [pc[str(l)]["coseno_logistica"] for l in ls], label=M[k]["modelo"], color=COLORS[k])
    ax[0].set(title="Diferencia de medias", xlabel="Profundidad relativa (capa / L)",
              ylabel="Coseno chat–agente (muestra completa)", ylim=(-0.05, 1.05))
    ax[1].set(title="Regresión logística", xlabel="Profundidad relativa (capa / L)")
    ax[0].legend(fontsize=7)
    fig.tight_layout(); save(fig, "final_rotacion")

    # 5. Factorial en la capa de referencia, como fracción del efecto total
    fig, ax = plt.subplots(figsize=(8, 3.4))
    w = 0.25
    for i, comp in enumerate(["system", "tools", "non_additive"]):
        vals = []
        for k in keys:
            f = M[k]["factorial_referencia"]["interaction"]
            vals.append(f[comp]["delta"] / f["total"]["delta"])
        ax.bar(np.arange(4) + (i - 1) * w, vals, w,
               label={"system": "Solo mensaje de sistema", "tools": "Solo herramientas",
                      "non_additive": "No aditivo"}[comp])
    ax.axhline(0, color="gray", lw=0.7); ax.axhline(1, color="gray", lw=0.7, ls=":")
    ax.set(xticks=np.arange(4), xticklabels=[f"{M[k]['modelo']}\ncapa {M[k]['capa_referencia']}" for k in keys],
           ylabel="Fracción de la interacción total")
    ax.legend(fontsize=7)
    fig.tight_layout(); save(fig, "final_factorial")

    # 6. Conducta sin intervención
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    variants = ["chat_harmful", "agent_harmful", "chat_benign", "agent_benign"]
    labels = ["Chat\ndañina", "Agente\ndañina", "Chat\nbenigna", "Agente\nbenigna"]
    for j, k in enumerate(["llama", "mistral"]):
        cb = M[k]["conducta"]
        ref = [cb["rechazo"][v][0] / cb["rechazo"][v][1] for v in variants]
        lo = [ref[i] - cb["rechazo"][v][2] for i, v in enumerate(variants)]
        hi = [cb["rechazo"][v][3] - ref[i] for i, v in enumerate(variants)]
        tc = [cb["llamadas_herramienta"][v][0] / cb["llamadas_herramienta"][v][1] for v in variants]
        xx = np.arange(4)
        ax[j].bar(xx - .18, ref, .36, yerr=[np.clip(lo, 0, None), np.clip(hi, 0, None)], capsize=3, label="Rechaza", color="#33365d")
        ax[j].bar(xx + .18, tc, .36, label="Llama una herramienta", color="#c88045")
        ax[j].set(xticks=xx, xticklabels=labels, ylim=(0, 1.05), title=f"{M[k]['modelo']} {M[k]['version']} · 96 pares")
    ax[0].set_ylabel("Fracción de respuestas"); ax[0].legend(fontsize=7)
    fig.tight_layout(); save(fig, "final_conducta")

    # 7. Test causal: dosis-respuesta
    panels = [("llama", l) for l in ["76", "56", "28"]] + [("mistral", "30")]
    fig, ax = plt.subplots(1, 4, figsize=(13, 3.4), sharey=True)
    for j, (k, l) in enumerate(panels):
        cz = M[k]["causal"][l]["dosis"]; ds = sorted(float(x) for x in cz)
        for key, lab, col in [("restaura", "Dirección de rechazo · dañinas", "#a33c32"),
                              ("control", "Dirección aleatoria · dañinas", "#7a7a7a"),
                              ("benigno", "Dirección de rechazo · benignas", "#26778b")]:
            v = [cz[f"{x:g}"][key] for x in ds]
            y = np.array([a[0] / a[1] for a in v]); lo = np.array([a[2] for a in v]); hi = np.array([a[3] for a in v])
            ax[j].errorbar(ds, y, yerr=[np.clip(y - lo, 0, None), np.clip(hi - y, 0, None)], marker="o", capsize=3, label=lab, color=col)
        ax[j].set(xscale="log", xticks=ds, xticklabels=[f"{x:g}" for x in ds], ylim=(-0.04, 1.08),
                  title=f"{M[k]['modelo']} · capa {l}", xlabel="Dosis (× ΔP de la capa)")
        ax[j].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax[j].xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax[0].set_ylabel("Fracción etiquetada como rechazo"); ax[0].legend(fontsize=7, loc="upper left")
    fig.tight_layout(); save(fig, "final_intervencion")


def main():
    M = {k: model_metrics(k) for k in MODELS}
    (OUT / "evidencia").mkdir(parents=True, exist_ok=True)
    (OUT / "evidencia" / "metricas_finales.json").write_text(
        json.dumps(M, ensure_ascii=False, indent=2), encoding="utf-8")
    figures(M)
    print("metricas: docs/tesis/Tesis_final/evidencia/metricas_finales.json")
    print("figuras :", ", ".join(sorted(p.stem for p in (OUT / "figures").glob("*.pdf"))))


if __name__ == "__main__":
    main()
