"""Extrae los datos de la versión ligera sin alterar resultados ni el otro mazo."""
import hashlib
import json
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[2]
COMUN = RAIZ / "docs/sustentacion/fuente/datos_presentacion.json"


def main():
    d = json.loads(COMUN.read_text(encoding="utf-8"))
    modelos = []
    fuentes = []
    llama = d["llama"]
    for clave, nombre, n, ruta in [
        ("llama", "Llama-3.3-70B", 42, "results/corridas/llama33_70b/v3/results.json"),
        ("mistral", "Mistral-7B", 96, "results/corridas/mistral7b/v2/results.json"),
        ("qwen8b", "Qwen3-8B", 96, "results/corridas/qwen3_8b/v2/results.json"),
        ("qwen32b", "Qwen3-32B", 96, "results/corridas/qwen3_32b/v2/results.json"),
    ]:
        raw = (RAIZ / ruta).read_bytes()
        original = json.loads(raw)
        assert original["heldout"]["n_pairs"] == n
        det = llama["deteccion"] if clave == "llama" else d["evaluacion96"][clave]["deteccion"]
        mon = llama["monitores"] if clave == "llama" else d["evaluacion96"][clave]["monitor"]
        tp, fp = round(n * mon["monitor_recall_oof"]), round(n * mon["monitor_fpr_oof"])
        assert abs(tp / n - mon["monitor_recall_oof"]) < 1e-10
        assert abs(fp / n - mon["monitor_fpr_oof"]) < 1e-10
        assert abs(2 * tp / (tp + fp + n) - mon["monitor_f1_oof"]) < 1e-10
        modelos.append({"clave": clave, "nombre": nombre, "n": n,
                        "chat": det["auroc_chat_heldout"], "agente": det["auroc_agent_heldout"],
                        "ic_diferencia": det["auroc_gap_ci"],
                        "monitor": {"tp": tp, "fp": fp, "n": n,
                                    "auroc": mon["auroc_w_agent_oof"], "f1": mon["monitor_f1_oof"]}})
        fuentes.append({"archivo": ruta, "sha256": hashlib.sha256(raw).hexdigest()})
    salida = {
        "modelos": modelos, "conducta": llama["conducta"], "separacion_llama": llama["sep"],
        "intervencion_dosis2": llama["intervencion56"]["per_dose"]["2"],
        "estratos96": [{"nombre": d["evaluacion96"][k]["modelo"],
                         "estratos": d["evaluacion96"][k]["estratos"]}
                        for k in ["mistral", "qwen8b", "qwen32b"]],
        "ejemplo": d["ejemplo_didactico"], "fuentes": fuentes,
        "fuente_comun": {"archivo": COMUN.relative_to(RAIZ).as_posix(),
                         "sha256": hashlib.sha256(COMUN.read_bytes()).hexdigest()},
    }
    (AQUI / "datos_ligera.json").write_text(json.dumps(salida, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Datos verificados: Llama 42; Mistral y Qwen 96. Conteos de monitores coherentes con F1 y FPR.")


if __name__ == "__main__":
    main()
