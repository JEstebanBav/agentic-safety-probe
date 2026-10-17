"""Extrae las cifras de la presentación desde los artefactos del repositorio.

Escribe datos_presentacion.json junto a este archivo. Conserva las comprobaciones
de los análisis de la tesis y añade las exportaciones de 96 pares de Mistral y
Qwen, con verificación de tamaño y hash de procedencia. No ejecuta modelos.

Uso, desde la raíz del repositorio:
    python docs/sustentacion/fuente/datos.py
"""
import json
import math
import hashlib
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[2]
LLAMA = RAIZ / "results/corridas/llama33_70b/v3"
TESIS = RAIZ / "docs/tesis/Tesis_avances_actualizada"


def leer(ruta):
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


def auroc(h, b):
    h, b = np.asarray(h), np.asarray(b)
    mayor = (h[:, None] > b[None, :]).sum()
    empate = (h[:, None] == b[None, :]).sum()
    return (mayor + 0.5 * empate) / (len(h) * len(b))


def wilson(k, n, z=1.959963984540054):
    p = k / n
    centro = (p + z * z / (2 * n)) / (1 + z * z / n)
    medio = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centro - medio), min(1.0, centro + medio)


def cerca(a, b, tol=5e-4):
    assert abs(a - b) < tol, (a, b)


def main():
    r = leer(LLAMA / "results.json")
    m = leer(TESIS / "evidencia/metricas_avance.json")
    ll = m["llama"]
    ho = r["heldout"]
    capa = str(r["best_layer"])
    assert capa == "76" and ho["n_pairs"] == 42

    # Proyecciones fuera de muestra en la capa de referencia (dirección reajustada por fold).
    P = {k: np.asarray(v, dtype=float) for k, v in ho["projections"].items()}
    ch, cb, ah, ab = P["chat_harmful"], P["chat_benign"], P["agent_harmful"], P["agent_benign"]
    assert len(ch) == len(cb) == len(ah) == len(ab) == 42

    # Comprobaciones contra la tesis (tablas 4.2, 4.3 y 4.4).
    cerca(auroc(ch, cb), 0.9779)
    cerca(auroc(ah, ab), 0.9195)
    cerca((ch - ah).mean(), 15.7244)
    cerca((cb - ab).mean(), -1.5438)
    cerca((ch - ah).mean() - (cb - ab).mean(), 17.2682)
    assert int(((ch - ah) > 0).sum()) == 42

    def sep(h, b):
        s = math.sqrt((h.var(ddof=1) + b.var(ddof=1)) / 2)
        return h.mean() - b.mean(), s

    S_c, s_c = sep(ch, cb)
    S_a, s_a = sep(ah, ab)
    cerca(S_c / s_c, 3.2051)
    cerca(S_a / s_a, 1.8036)
    cerca(s_a / s_c, 0.30854)
    cerca(1 - S_a / S_c, 0.82638)

    capas = sorted(int(x) for x in ho["per_layer"])
    por_capa = {k: [ho["per_layer"][str(c)][k] for c in capas]
                for k in ["delta_p_harmful", "delta_p_benign", "interaction", "auroc_chat", "auroc_agent"]}
    pca = {"cos": [r["pca_by_layer"][str(c)]["cosine_similarity_with_diff_means"] for c in capas],
           "var": [r["pca_by_layer"][str(c)]["explained_variance_ratio_pc1"] for c in capas]}
    cerca(pca["cos"][capas.index(76)], 0.9894)
    cerca(pca["var"][capas.index(76)], 0.2013)

    # Sonda logística conversacional (capa 76).
    sonda = {"auroc_cv": r["linear_aurocs"][capa], "sd_folds": r["linear_aurocs_std"][capa],
             "ic": [r["linear_auroc_ci_lower"][capa], r["linear_auroc_ci_upper"][capa]],
             "permutado": r["linear_aurocs_permuted"][capa],
             "en_muestra": r["linear_aurocs_in_sample"][capa]}
    cerca(sonda["auroc_cv"], 0.9677)
    cerca(sonda["permutado"], 0.5152)

    tokens = {k: v["mean"] for k, v in r["prompt_tokens"].items()}

    fact = leer(LLAMA / "factorial_results.json")["by_layer"][capa]

    conducta = ll["behavior"]
    assert conducta["refusal"]["chat_harmful"]["k"] == 38 and conducta["refusal"]["agent_harmful"]["k"] == 23
    interv = ll["intervention_layer56"]

    mon = ll["monitors_at_reference"]
    # Conteos implícitos de los monitores: FPR y F1 determinan TP con 42 positivos.
    def tp_desde_f1(f1, fp, n=42):
        return round(f1 * (fp + n) / (2 - f1))

    fp_dm = round(mon["monitor_fpr_diff_means_oof"] * 42)
    tp_dm = tp_desde_f1(mon["monitor_f1_diff_means_oof"], fp_dm)
    tp_lr = round(mon["monitor_recall_oof"] * 42)
    fp_mi = round(0.1190 * 42)
    tp_mi = tp_desde_f1(0.8250, fp_mi)
    assert (fp_dm, tp_dm, tp_lr, fp_mi, tp_mi) == (6, 35, 37, 5, 33)

    q8 = leer(RAIZ / "results/corridas/qwen3_8b/v1/results.json")["auroc_heldout"]
    q32 = leer(RAIZ / "results/corridas/qwen3_32b/v1/results.json")["auroc_heldout"]

    corpus = [json.loads(linea) for linea in (RAIZ / "data/dataset_v2.jsonl").read_text(encoding="utf-8").splitlines() if linea.strip()]
    ejemplo = {v["variant"]: v for v in corpus if v["pair_id"] == "fraud_06"}
    assert len(ejemplo) == 4
    assert ejemplo["chat_harmful"]["prompt"] == ejemplo["agent_harmful"]["prompt"]
    assert ejemplo["chat_benign"]["prompt"] == ejemplo["agent_benign"]["prompt"]

    evaluacion96 = {}
    for key, nombre, etiqueta in [("mistral", "corridas/mistral7b/v2", "Mistral-7B"), ("qwen8b", "corridas/qwen3_8b/v2", "Qwen3-8B"), ("qwen32b", "corridas/qwen3_32b/v2", "Qwen3-32B")]:
        ruta = RAIZ / "results" / nombre / "results.json"
        completo = leer(ruta)
        referencia = str(completo["best_layer"])
        assert all(len(completo["projections"][k]) == 96 for k in
                   ["chat_harmful", "chat_benign", "agent_full_harmful", "agent_full_benign"])
        assert completo["heldout"]["n_pairs"] == 96
        assert all(len(completo["heldout"]["projections"][k]) == 96 for k in
                   ["chat_harmful", "chat_benign", "agent_full_harmful", "agent_full_benign"])
        evaluacion96[key] = {
            "modelo": etiqueta, "n": 96, "capa": int(referencia),
            "deteccion": completo["heldout"]["detection"],
            "formato": completo["format_effect"]["held_out"]["per_layer"][referencia],
            "estratos": completo["format_effect"]["held_out"]["by_stratum"],
            "monitor": completo["robust"]["per_layer"][referencia],
            "fuente": ruta.relative_to(RAIZ).as_posix(),
            "sha256": hashlib.sha256(ruta.read_bytes()).hexdigest(),
        }

    datos = {
        "cobertura": leer(AQUI / "cobertura_estudio.json"),
        "ejemplo_didactico": {"pair_id": "fraud_06", "subtlety": "contextual",
                              "harmful": ejemplo["chat_harmful"]["prompt"],
                              "benign": ejemplo["chat_benign"]["prompt"],
                              "fuente": "data/dataset_v2.jsonl"},
        "evaluacion96": evaluacion96,
        "fuente": {
            "llama": "results/corridas/llama33_70b/v3/results.json (heldout, pca_by_layer, linear_*), "
                     "factorial_results.json",
            "metricas": "docs/tesis/Tesis_avances_actualizada/evidencia/metricas_avance.json",
            "mistral": "results/corridas/mistral7b/v3/run.log (valores impresos, redondeados)",
            "qwen": "results/corridas/qwen3_8b/v1/results.json, results/corridas/qwen3_32b/v1/results.json (auroc_heldout)",
        },
        "llama": {
            "capa": 76,
            "proy76": {"chat_h": ch.tolist(), "chat_b": cb.tolist(), "ag_h": ah.tolist(), "ag_b": ab.tolist()},
            "sep": {"S_chat": S_c, "s_chat": s_c, "S_ag": S_a, "s_ag": s_a,
                    "d_chat": S_c / s_c, "d_ag": S_a / s_a, "kappa": s_a / s_c,
                    "perdida_cruda": 1 - S_a / S_c, "perdida_d": 1 - (S_a / s_a) / (S_c / s_c)},
            "medias": {"chat_h": ch.mean(), "chat_b": cb.mean(), "ag_h": ah.mean(), "ag_b": ab.mean()},
            "capas": capas,
            "por_capa": por_capa,
            "pca": pca,
            "deteccion": ll["heldout_detection"],
            "formato": ll["heldout_format"],
            "pareado_h": ll["heldout_paired"],
            "pareado_b": ho["paired_benign"],
            "seleccion": {k: v for k, v in ll["selection"].items() if k != "selection_scores"},
            "rotacion": ll["rotation_at_reference"],
            "monitores": mon,
            "monitor_conteos": {"dm": {"tp": tp_dm, "fp": fp_dm}, "lr": {"tp": tp_lr, "fp": 0}},
            "fpr_cero_wilson": wilson(0, 42),
            "estratos": ll["strata_oof"],
            "conducta": conducta,
            "intervencion56": interv,
            "sonda": sonda,
            "tokens": tokens,
            "factorial76": fact,
        },
        "mistral": dict(m["mistral"], monitor={"auroc": 0.9347, "f1": 0.8250, "fpr": 0.1190, "tp": tp_mi, "fp": fp_mi},
                        perdida_cruda=0.847, seleccion={"n": 21, "decrecen": 19, "dp": 7.6873,
                                                         "ic": [6.0074, 9.1634], "p_signos": 2.21e-4},
                        ingenuo=8.0227, pca_cap={"18": [0.968, 0.303], "30": [0.952, 0.287]},
                        conducta={"chat_h": 19, "ag_h": 5, "chat_b": 1, "ag_b": 0, "n": 42,
                                  "deja": 16, "inverso": 2, "mcnemar": 0.0013}),
        "qwen": {"8b": {"capa": 32, **q8}, "32b": {"capa": 46, **q32}},
    }
    with open(AQUI / "datos_presentacion.json", "w", encoding="utf-8") as f:
        json.dump(datos, f, ensure_ascii=False, indent=1)
    print("datos_presentacion.json escrito; comprobaciones superadas.")


if __name__ == "__main__":
    main()
