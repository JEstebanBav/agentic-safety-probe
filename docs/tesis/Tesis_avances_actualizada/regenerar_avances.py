"""Regenera la versión de avances desde el ZIP original y evidencia local.

No ejecuta modelos ni altera sus resultados. Las cifras del log se conservan
con la precisión impresa; las de Llama se obtienen de los JSON guardados.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/tesis/Tesis_avances_actualizada"
LLAMA = ROOT / "results/llama33_final/results_llama33_v3"
MLOG = ROOT / "results/results_mistralV3/results_mistral7b_v3.log"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(relative, text):
    dest = OUT / relative
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text.strip() + "\n", encoding="utf-8")


def number(x, digits=3):
    return f"{float(x):.{digits}f}".replace(".", "{,}")


def table(headers, rows, caption, label, spec=None):
    spec = spec or "l" + "r" * (len(headers) - 1)
    body = "\n".join(" & ".join(str(x) for x in row) + r" \\" for row in rows)
    return (r"\begin{table}[htbp]\centering\small" + "\n"
            + r"\begin{adjustbox}{max width=\textwidth}" + "\n"
            + r"\begin{tabular}{@{}" + spec + r"@{}}\toprule" + "\n"
            + " & ".join(headers) + r" \\\midrule" + "\n" + body
            + "\n" + r"\bottomrule\end{tabular}\end{adjustbox}" + "\n"
            + r"\caption{" + caption + r"}\label{" + label + r"}\end{table}" + "\n")


def figure(name, caption, label):
    return (r"\begin{figure}[htbp]\centering" + "\n"
            + r"\includegraphics[width=0.94\textwidth]{" + name + "}\n"
            + r"\caption{" + caption + r"}\label{" + label + r"}\end{figure}" + "\n")


def replace_section(text, start, end, replacement):
    a, b = text.index(start), text.index(end, text.index(start))
    return text[:a] + replacement.strip() + "\n\n" + text[b:]


def auc(harmful, benign):
    nh, nb = len(harmful), len(benign)
    ranks = rankdata(np.r_[harmful, benign])
    return float((ranks[:nh].sum() - nh * (nh + 1) / 2) / (nh * nb))


def log_section(log, start, limit=2500):
    return log[log.index(start):][:limit]


def parse_mistral(log):
    """Lee tablas impresas, sin atribuirles precisión de un JSON ausente."""
    pca_section = log_section(log, "PCA validation of the chat contrast")
    pca = {}
    for line in pca_section.splitlines():
        m = re.match(r"\s*(\d+)\s+([\d.]+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)%", line)
        if m:
            pca[int(m[1])] = {"cos": float(m[2]), "variance": float(m[5]) / 100}
    gap_section = log_section(log, "Format shift by layer, harmful arm vs benign control:")
    gaps = {}
    for line in gap_section.splitlines():
        m = re.match(r"\s*(\d+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s*$", line)
        if m:
            gaps[int(m[1])] = [float(m[i]) for i in (2, 3, 4)]
    assert len(pca) == 16 and len(gaps) == 16
    specificity = log_section(log, "FORMAT EFFECT SPECIFICITY  (layer 30)  [benign control / diff-in-diff, held-out]")
    for phrase in ["+7.8359", "-0.5818", "+8.4178", "+7.0992", "+9.6758"]:
        assert phrase in specificity
    scale = log_section(log, "FORMAT EFFECT IN SCALE-INVARIANT UNITS  (layer 30, held-out)")
    assert "0.9478" in scale and "0.7851" in scale
    return {"n_pairs": 42, "layer": 30, "source_kind": "log_redondeado",
            "delta_h_oof": 7.8359, "delta_b_oof": -0.5818,
            "interaction_oof": 8.4178, "interaction_ci": [7.0992, 9.6758],
            "auroc_chat_oof": 0.9478, "auroc_agent_oof": 0.7851,
            "d_chat_oof": 2.5289, "d_agent_oof": 0.9163,
            "loss_standardised": 0.638, "scale_contraction": 0.422,
            "pca": pca, "gaps_full_sample": gaps,
            "strata_oof": {"contextual": [14, 0.9439, 0.7398],
                           "explicit": [18, 0.9722, 0.8827],
                           "framed": [10, 0.9000, 0.7100]},
            "causal_complete_locally": False}


def save_figure(fig, name):
    fig.savefig(OUT / "figures" / (name + ".pdf"), bbox_inches="tight")
    fig.savefig(OUT / "figures" / (name + ".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    with zipfile.ZipFile(ROOT / "docs/tesis/Tesis_corta_v2_overleaf.zip") as archive:
        for item in archive.infolist():
            if item.is_dir():
                continue
            rel = Path(item.filename).relative_to("Tesis_corta_v2")
            target = (OUT / rel).resolve()
            assert target.is_relative_to(OUT.resolve())
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(item))
    d = read_json(LLAMA / "results.json")
    robust = read_json(LLAMA / "robust_results.json")
    causal = read_json(LLAMA / "causal_results.json")
    log = MLOG.read_text(encoding="utf-8")
    m = parse_mistral(log)
    layer = str(d["best_layer"])
    assert layer == "76" and len(d["projections"]["chat_harmful"]) == 42
    ho = d["heldout"]
    f = d["format_effect"]["held_out"]["per_layer"][layer]
    r = robust["per_layer"][layer]
    rot = robust["rotation"][layer]
    selection = robust["layer_selection"]
    assert selection["n_selection"] == selection["n_inference"] == 21
    # Controles de coherencia sobre las proyecciones guardadas.
    p = {k: np.asarray(v) for k, v in ho["projections"].items()}
    assert abs((p["chat_harmful"] - p["agent_harmful"]).mean() - f["delta_p_harmful"]) < 1e-7
    assert abs(auc(p["chat_harmful"], p["chat_benign"]) - f["auroc_chat"]) < 1e-10
    qwen = {}
    for name in ["qwen8b", "qwen32b"]:
        q = read_json(ROOT / "results" / name / "results.json")
        qwen[name] = {"n_pairs": len(q["projections"]["chat_harmful"]),
                      "layer": q["best_layer"], "heldout": q["auroc_heldout"],
                      "source_kind": "historico_sin_config_completa"}
    # Corpus ampliado: comprobación completa de sus cuatro variantes.
    rows = [json.loads(x) for x in (ROOT / "data/dataset_v2.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    pairs = {}
    for row in rows:
        pairs.setdefault(row["pair_id"], {})[row["variant"]] = row
    assert len(pairs) == 96 and len(rows) == 384
    for pair in pairs.values():
        for arm in ["harmful", "benign"]:
            assert pair["chat_" + arm]["prompt"] == pair["agent_" + arm]["prompt"]
        for form in ["chat", "agent"]:
            for key in ["system_prompt", "tools", "tool_definitions"]:
                assert pair[form + "_harmful"][key] == pair[form + "_benign"][key]

    # Ajustes de planteamiento y métodos, conservando la formalización original.
    intro_path = "chapters/Chapter1/1.5_chapter01-introduction.tex"
    intro = (OUT / intro_path).read_text(encoding="utf-8")
    intro = intro.replace("éstas", "estas")
    intro = intro.replace("de modo que la única diferencia dentro de la comparación es la disponibilidad de herramientas;",
                          "de modo que la manipulación dentro de la comparación es el formato compuesto por mensaje de sistema y herramientas;")
    intro = intro.replace("Dos aspectos de estos mecanismos permanecen, sin embargo, sin examinar.",
                          "Dos aspectos de estos mecanismos requieren, sin embargo, un diagnóstico específico.")
    marker = r"\section{Planteamiento del proyecto}"
    definition = r"""
En este trabajo, \emph{dirección de rechazo} designa operativamente la diferencia normalizada entre los centroides de activaciones de instrucciones dañinas y benignas en formato conversacional. La \emph{proyección} es el producto escalar de una activación con ese eje, y el \emph{AUROC} mide su capacidad para ordenar las dos clases. Son medidas de representación; ninguna equivale por sí sola a observar un rechazo. El \emph{formato agéntico} estudiado combina un mensaje de sistema y esquemas de herramientas simuladas: describe capacidad de acción, sin ejecutar acciones reales.

Esta versión presenta avances sustentados en los artefactos experimentales disponibles. Distingue el corpus ampliado construido, de 96 pares, de las ejecuciones analizadas, de 42 pares por modelo. La evaluación sobre el corpus ampliado y la validación externa permanecen como trabajo pendiente.
"""
    intro = intro.replace(marker, definition + "\n" + marker)
    write(intro_path, intro)

    methods_path = "chapters/Chapter3/chapter03.tex"
    methods = (OUT / methods_path).read_text(encoding="utf-8")
    methods = replace_section(methods, "El estudio es de tipo explicativo.", r"\subsection{Objeto de estudio}", r"""
El estudio tiene un componente descriptivo y comparativo sobre representaciones internas, complementado por ensayos de intervención exploratorios. El contraste de estímulos conserva el contenido de cada instrucción y modifica el formato completo: mensaje de sistema y disponibilidad de herramientas. Este contraste estima el efecto de ese conjunto de cambios; no identifica por sí solo cuál de sus componentes lo produce.

En el análisis diagnóstico los pesos permanecen fijos y se registran activaciones. En los ensayos complementarios sí se modifica transitoriamente el flujo residual durante la generación, sin reentrenar ni modificar los pesos. Esas intervenciones permiten evaluar un efecto sobre el rechazo bajo las condiciones ensayadas, pero no demuestran por sí solas que una rotación observada sea la causa general de fallos en agentes desplegados.
""")
    methods = replace_section(methods, "Los tres primeros modelos se evalúan", r"\section{Datos y fuentes de información}", r"""
El corpus ampliado contiene 96 pares y está validado estructuralmente. Las ejecuciones que pueden sustentarse con los archivos disponibles corresponden al corpus original de 42 pares en los cuatro modelos. Los resultados principales se apoyan en Llama-3.3-70B V3 y en el registro de Mistral-7B V3; las corridas previas de Qwen se presentan como evidencia exploratoria. No se atribuyen a ninguno de esos archivos resultados sobre 96 pares.

La unidad de muestreo es el par base. Las cuatro variantes de un mismo par no son cuatro observaciones independientes. Los 42 pares originales se distribuyen en 18 explícitos, 14 contextuales y 10 de apariencia legítima. El tamaño de esos estratos limita la precisión de las comparaciones.
""")
    methods = methods.replace("El objeto de estudio es la dirección de rechazo, entendida como la representación interna, localizada en el espacio de activaciones de un modelo de lenguaje, que media su capacidad de negarse a cumplir solicitudes dañinas.",
                              "El objeto de estudio es la dirección de rechazo, estimada operativamente como un contraste dañino--benigno en el espacio de activaciones. Su capacidad discriminativa y su relación con la conducta se evalúan por separado; la denominación no presupone que toda dirección así estimada sea un mecanismo causal de rechazo.")
    methods = replace_section(methods, "La redacción de las instrucciones estuvo a cargo", "La construcción contó con una revisión asistida", r"""
La construcción y consolidación de las instrucciones estuvo a cargo del autor. El corpus original reúne 42 pares propios. El ampliado conserva esos pares, añade 20 pares propios y 34 pares derivados de AgentHarm, para un total de 62 propios y 34 derivados del benchmark. La adaptación priorizó identidad de texto entre formatos, contrapartes benignas temáticamente próximas y una clasificación explícita del grado de explicitud. Un caso se conserva cuando admite un pareado adecuado y una asignación justificable; de otro modo se reformula o descarta. El origen queda registrado por par. La adaptación de contrapartes implica que el corpus no constituye una evaluación íntegra del benchmark publicado.
""")
    methods = methods.replace("El tamaño del estrato más pequeño, con 30 pares", "El tamaño del estrato más pequeño, el contextual, con 24 pares")
    methods = methods.replace("El método mide en una única posición, la del último \\textit{token} de la instrucción, por tres razones.",
                              "El método mide en una única posición, la del último \\textit{token} del estímulo renderizado antes de generar, por tres razones. Ese estímulo incluye mensaje de sistema, herramientas, instrucción y delimitadores de plantilla. La posición puede corresponder a un delimitador, como \\texttt{[/INST]}, y no al último token léxico escrito por el usuario.")
    methods = methods.replace("y el de la posición $n$ es el único que contiene la instrucción completa.",
                              "y el de la posición $n$ contiene el estímulo completo. Esa propiedad describe el acceso causal permitido por la arquitectura, sin garantizar que toda la información sea utilizada.")
    methods = methods.replace("en el límite $\\lambda \\to \\infty$ el discriminante se reduce exactamente a la diferencia de medias.",
                              "cuando $\\lambda \\to \\infty$, la dirección normalizada del discriminante converge a la dirección de la diferencia de medias.")
    methods = methods.replace("que es lo que evita ajustar el ruido de la muestra cuando $n \\ll d$.",
                              "aunque la media estimada sigue teniendo ruido de muestreo y requiere evaluación fuera de muestra cuando $n \\ll d$.")
    methods = replace_section(methods, r"\subsubsection*{Dirección agéntica, rotación y monitor}", r"\subsection{Procedimiento de análisis}", r"""
\subsubsection*{Dirección agéntica, reorientación y monitor}

Se distinguen dos estimadores. La dirección agéntica de diferencia de medias, $\hat{\bm{w}}_{\mathrm{DM}}^{(l)}$, se obtiene restando los centroides agénticos dañino y benigno y normalizando. La dirección $\hat{\bm{w}}_{\mathrm{LR}}^{(l)}$ procede de los coeficientes de una regresión logística lineal. Un clasificador no tiene por qué recuperar el mismo vector que el contraste de centroides. Por ello, cada coseno, AUROC y monitor se identifica por estimador.

El coseno de $\hat{\bm{w}}_{\mathrm{DM}}^{(l)}$ con $\hat{\bm{r}}^{(l)}$ compara dos contrastes de medias. El análisis adicional de regresión logística compara direcciones obtenidas con el mismo estimador en ambos formatos. Se reporta la fiabilidad entre mitades disjuntas, pues ruido de estimación puede reducir el coseno aparente. La normalización por fiabilidad es un diagnóstico bajo supuestos de error, no una medición exacta de un ángulo poblacional.

Los monitores clasifican la proyección mediante un umbral calibrado en los datos de entrenamiento de cada fold. La evaluación se hace en pares retenidos y se reportan AUROC, $F_1$ y tasa de falsos positivos. Los resultados de ambos estimadores se conservan separados. No equivalen a una validación en datos externos ni a garantías de seguridad operacional.

\subsubsection*{Validación geométrica mediante PCA}

Para cada capa se concatenan las activaciones conversacionales dañinas y benignas en una matriz $X\in\mathbb{R}^{2n\times d}$ y se resta su media global. Se ajusta PCA sobre esa matriz centrada, sin centrar por separado cada clase: hacerlo eliminaría precisamente la diferencia entre centroides. Se reportan la proporción de varianza explicada por PC1 y $|\cos(\mathrm{PC1},\hat{\bm{r}})|$. El signo de PC1 se orienta para que el coseno sea positivo. También se examinan los cinco primeros componentes y la varianza proyectada sobre la dirección. La alineación con PC1 indica predominio en la varianza total de esa muestra; su ausencia no invalida automáticamente un contraste discriminativo.
""")
    methods = methods.replace("verificando mediante análisis de componentes principales que el contraste dañino--benigno corresponde a un eje dominante de variación",
                              "evaluando mediante componentes principales si el contraste dañino--benigno coincide con un eje dominante de variación")
    methods = methods.replace("Los cuatro modelos se procesaron con la misma rutina de extracción, que",
                              "Las ejecuciones usan variantes del mismo pipeline; la comparación entre versiones requiere registrar los cambios de implementación. La extracción")
    methods = methods.replace("último \\textit{token} de la\ninstrucción", "último \\textit{token} del\nestímulo renderizado")
    methods = methods.replace("Que una corrida termine certifica, por tanto, que toda activación analizada\ncorresponde a una instrucción completa en ambos formatos.",
                              "En la implementación actual se conservan los extremos de longitud y se comprueba que el estímulo no haya sido truncado. Esa comprobación debe acompañar la configuración de cada ejecución histórica.")
    methods = methods.replace("Cabe esperar, por tanto, que una réplica reproduzca las\nmagnitudes reportadas dentro del margen que los intervalos de confianza\ndescriben, y no cifra por cifra.",
                              "La incertidumbre numérica entre ejecuciones no está cuantificada por los intervalos de muestreo del corpus. Se documentan por separado precisión, tokenización, configuración y cambios de implementación; no se atribuye automáticamente una discrepancia de ejecución al margen estadístico.")
    methods = methods.replace("En el régimen $n \\ll d$, cualquier etiquetado de la muestra es linealmente separable, de modo que un AUROC dentro de muestra próximo a la unidad es una propiedad de la geometría y no un hallazgo.",
                              "En el régimen $n \\ll d$, si los puntos tienen rango afín suficiente, incluso etiquetados arbitrarios pueden ser linealmente separables. La dimensión alta por sí sola no garantiza esa condición, pero hace que un AUROC de entrenamiento saturado no sea evidencia de generalización.")
    methods = methods.replace("Por depender solo del orden, es invariante a cualquier cambio de escala.",
                              "Por depender solo del orden, es invariante a transformaciones estrictamente crecientes del puntaje; una inversión del eje cambia su interpretación.")
    methods = methods.replace("lo que estima el nivel de azar empírico, que no coincide necesariamente con $0{,}5$.",
                              "lo que estima una referencia nula empírica con incertidumbre Monte Carlo. Bajo ausencia de asociación, el valor esperado de referencia es $0{,}5$; una realización finita puede apartarse de él.")
    methods = methods.replace("y a cualquier reescalado de la dirección", "y a reescalados positivos de la dirección")
    methods = methods.replace("bajo la hipótesis nula el signo de cada $\\delta_i$ es intercambiable,",
                              "bajo la hipótesis nula y el supuesto de intercambiabilidad del cambio de condición dentro de pares independientes, el signo de cada $\\delta_i$ puede permutarse,")
    methods = replace_section(methods, "Como el análisis se repite en cada capa,", r"\section{Consideraciones éticas}", r"""
Se emplean 10.000 permutaciones y remuestreos en la corrida principal. Cuando un archivo histórico almacena cero excedencias como $p=0$, se informa una resolución del orden de $10^{-4}$, no probabilidad cero. Se distingue el intervalo percentil del BCa y se identifica el que realmente está guardado.

La inferencia exige cautela adicional. El cross-fitting evita que un par determine su propia dirección, pero los folds comparten datos de entrenamiento: sus puntajes no son realizaciones independientes de todo el procedimiento de ajuste. Los intervalos calculados remuestreando puntajes ya obtenidos son condicionales a esos ajustes y no sustituyen un remuestreo que reestime toda la dirección. Las pruebas de signos sobre esos puntajes se presentan como diagnósticos de consistencia.

Para contrastar una capa se utiliza una separación explícita: 21 pares seleccionan la capa y estiman la dirección, y otros 21 producen la inferencia. Las dos ramas de cada par permanecen juntas. Se indica el denominador de los conteos y se compara el valor $p$ con Bonferroni por número de capas. Esta corrección no reemplaza la separación entre selección e inferencia. Se repite el esquema sobre 200 particiones para describir estabilidad, sin tratarlas como nuevas muestras independientes.

\subsection{Ensayos complementarios de comportamiento}

La ejecución Llama V3 conserva generaciones sin intervención y ensayos de adición de activación en capas 56, 52 y 28. Se suma $\alpha\hat{\bm{r}}$ durante la generación y se compara con una dirección aleatoria de igual norma y con solicitudes benignas. La dosis se expresa en múltiplos del desplazamiento de referencia de cada capa; no corresponde a un parámetro universal. Las respuestas se etiquetan como rechazo, cumplimiento o degeneración mediante una heurística textual reproducible. Esa etiqueta no prueba que una herramienta haya ejecutado la tarea. Los contrastes pareados usan McNemar exacto. Las múltiples dosis y capas se consideran exploratorias, y la auditoría humana de las generaciones sigue pendiente. El log de Mistral no contiene un cierre de su etapa causal y no se reporta como experimento causal completo.
""")
    methods = methods.replace("Y el estudio no analiza las respuestas del modelo para obtener contenido dañino, sino sus activaciones internas.",
                              "El análisis principal se centra en activaciones; los ensayos complementarios etiquetan respuestas únicamente para estudiar rechazo, cumplimiento y degeneración, sin ejecutar las herramientas simuladas.")
    write(methods_path, methods)

    # Tablas y figuras construidas desde la misma evidencia que alimenta el texto.
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.18})
    layers = sorted(int(x) for x in ho["per_layer"])
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for key, label_, color in [("delta_p_harmful", "Dañino", "#a33c32"), ("delta_p_benign", "Benigno", "#26778b"), ("interaction", "Interacción", "#33365d")]:
        ax[0].plot(layers, [ho["per_layer"][str(l)][key] for l in layers], label=label_, color=color)
    for key, label_ in [("auroc_chat", "Chat"), ("auroc_agent", "Agente")]:
        ax[1].plot(layers, [ho["per_layer"][str(l)][key] for l in layers], label=label_)
    ax[0].set(ylabel="Cambio de proyección (OOF)", xlabel="Capa", title="Llama V3 · 42 pares")
    ax[1].set(ylabel="AUROC de la dirección conversacional", xlabel="Capa", ylim=(0.45, 1.02), title="Evaluación por pares retenidos")
    ax[0].axhline(0, color="gray", lw=0.7); ax[1].axhline(0.5, color="gray", lw=0.7, ls="--")
    for a in ax:a.legend(fontsize=8)
    fig.tight_layout();save_figure(fig, "avance_llama_capas")
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(layers, [d["pca_by_layer"][str(l)]["cosine_similarity_with_diff_means"] for l in layers], label="Llama V3")
    ax[0].plot(sorted(m["pca"]), [m["pca"][l]["cos"] for l in sorted(m["pca"])], label="Mistral V3 (log)")
    ax[0].set(xlabel="Capa", ylabel="|cos(PC1, dirección)|", ylim=(0, 1.05))
    ax[1].plot(layers, [100*d["pca_by_layer"][str(l)]["explained_variance_ratio_pc1"] for l in layers], label="Llama V3")
    ax[1].plot(sorted(m["pca"]), [100*m["pca"][l]["variance"] for l in sorted(m["pca"])], label="Mistral V3 (log)")
    ax[1].set(xlabel="Capa", ylabel="Varianza explicada de PC1 (%)")
    for a in ax:a.legend(fontsize=8)
    fig.tight_layout();save_figure(fig, "avance_pca")
    strata = d["format_effect"]["held_out"]["by_stratum"]
    order = ["explicit", "contextual", "framed"]
    sl = {x["stratum"]: x for x in strata}
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    x = np.arange(3)
    for a, model in zip(ax, ["Llama V3", "Mistral V3 (log)"]):
        chat = [sl[s]["auroc_chat"] if model.startswith("Llama") else m["strata_oof"][s][1] for s in order]
        agent = [sl[s]["auroc_agent"] if model.startswith("Llama") else m["strata_oof"][s][2] for s in order]
        a.bar(x-.18, chat, .36, label="Chat", color="#26778b");a.bar(x+.18, agent, .36, label="Agente", color="#c88045")
        a.set(xticks=x, xticklabels=["Explícito\nn=18", "Contextual\nn=14", "Ap. legítima\nn=10"], ylim=(0.5, 1.02), ylabel="AUROC fuera de muestra", title=model)
        a.legend(fontsize=8)
    fig.tight_layout();save_figure(fig, "avance_estratos")
    fig, ax = plt.subplots(figsize=(7, 3.7))
    for method, label_ in [("diff_means", "Diferencia de medias"), ("logistic", "Regresión logística")]:
        ax.plot(layers, [robust["rotation"][str(l)][method]["cosine_full_sample"] for l in layers], label=label_)
    ax.set(xlabel="Capa", ylabel="Coseno entre formatos (muestra completa)", title="Llama V3 · mismo estimador en ambos formatos", ylim=(-.05, 1.05));ax.legend()
    fig.tight_layout();save_figure(fig, "avance_rotacion")
    cd = causal["causal"]["56"]["per_dose"]
    doses = sorted(float(x) for x in cd)
    fig, ax = plt.subplots(figsize=(7, 3.7))
    for key, label_ in [("restore", "Dirección de rechazo · dañinos (n=19)"), ("control", "Dirección aleatoria · dañinos (n=19)"), ("benign", "Dirección de rechazo · benignos (n=41)")]:
        vals = [cd[f"{x:g}"][key] for x in doses]
        y=np.array([v["rate"] for v in vals]);lower=np.array([v["ci_lower"] for v in vals]);upper=np.array([v["ci_upper"] for v in vals])
        ax.errorbar(doses,y,yerr=[y-lower,upper-y],marker="o",capsize=3,label=label_)
    ax.set(xlabel="Dosis (múltiplos del desplazamiento de capa 56)", ylabel="Fracción etiquetada como rechazo", ylim=(-.04, 1.08));ax.legend(fontsize=8)
    fig.tight_layout();save_figure(fig, "avance_intervencion")

    ch4 = r"""
\chapter{Resultados y evaluación}
\label{chap:resultados-evaluacion}

Este capítulo presenta los avances que pueden reconstruirse a partir de los resultados guardados. Los resultados principales corresponden a Llama-3.3-70B V3 y al registro de Mistral-7B V3. Se distingue la estimación dentro de muestra de la evaluación fuera de muestra, y los resultados diagnósticos de los ensayos conductuales exploratorios. La discusión y los límites se desarrollan en el Capítulo~\ref{chap:discusion}.

\section{Resultados, análisis e interpretación}
\label{sec:resultados}
\subsection{Alcance de las ejecuciones}

El corpus ampliado está construido y validado con 96 pares, pero las ejecuciones disponibles emplean 42 pares. En cada una, esos pares producen 168 entradas en las cuatro condiciones; para una sonda de un formato hay 84 observaciones dañinas y benignas. Las variantes no incrementan el número de unidades independientes. La Tabla~\ref{tab:corridas} identifica el soporte de cada ejecución.
"""
    ch4 += table(["Modelo", "Pares", "Capas", "Referencia", "Soporte"],
                 [["Llama-3.3-70B V3", 42, "20 de 80", 76, "JSON y activaciones"],
                  ["Mistral-7B V3", 42, "16 de 32", 30, "Registro de ejecución"],
                  ["Qwen3-8B (previo)", 42, "18 de 36", 32, "JSON histórico"],
                  ["Qwen3-32B (previo)", 42, "32 de 64", 46, "JSON histórico"]],
                 "Ejecuciones disponibles. Qwen se conserva como evidencia exploratoria; no se trata como réplica del pipeline corregido. El paquete causal de Mistral no está completo localmente.", "tab:corridas", "lrrrl")
    ch4 += r"""
La capa de referencia maximiza el desplazamiento en el barrido completo y se usa para describir magnitudes. La inferencia separada de selección de capa se reporta más adelante. Los valores de Mistral proceden de tablas impresas y conservan su precisión redondeada; no se reconstruye un JSON ausente ni se añaden decimales inferidos.

\subsection{Línea base y capacidad discriminativa}

La dirección conversacional se estima en cuatro de cinco folds de pares y se evalúa en el restante. La Tabla~\ref{tab:linea_base} reporta los AUROC de sus puntajes retenidos. Mide discriminación entre etiquetas de contenido, no rechazo efectivo del modelo.
"""
    ch4 += table(["Modelo", "Capa", "AUROC chat", "AUROC agente", "Diferencia"],
                 [["Llama V3", 76, number(f["auroc_chat"],4), number(f["auroc_agent"],4), number(f["auroc_chat"]-f["auroc_agent"],4)],
                  ["Mistral V3", 30, number(m["auroc_chat_oof"],4), number(m["auroc_agent_oof"],4), number(m["auroc_chat_oof"]-m["auroc_agent_oof"],4)]],
                 "AUROC de la dirección conversacional fuera de muestra, con cinco folds por pares y 42 pares por modelo. Cada comparación de formato utiliza la misma dirección de entrenamiento.", "tab:linea_base")
    ch4 += r"""
En Llama el IC de remuestreo de la diferencia de AUROC es $[0{,}0113;\ 0{,}1128]$; el de chat es $[0{,}9490;\ 0{,}9955]$ y el de agente $[0{,}8560;\ 0{,}9734]$. Son intervalos condicionales a los puntajes cross-fit guardados. Para Mistral no se dispone localmente del conjunto de puntajes ni de un intervalo equivalente exportado. La pérdida puntual es mayor en Mistral, pero no se realiza una prueba formal entre modelos ni se atribuye la diferencia exclusivamente a su tamaño.

La sonda logística conversacional es otra medida. En Llama, capa 76, su AUROC medio en diez repeticiones de cinco folds es $0{,}9677$, con desviación entre folds $0{,}0373$ e IC corregido Nadeau--Bengio $[0{,}9285;\ 1]$; el control permutado es $0{,}5152$. La desviación entre folds no es un intervalo de confianza. La muestra completa produce AUROC de entrenamiento $1$, que no se usa como evidencia de generalización. La curva de la dirección y la curva de una sonda no deben confundirse, ni la cercanía de dos sondas demuestra equivalencia.
"""
    ch4 += figure("avance_llama_capas", "Llama V3 sobre 42 pares. Izquierda: desplazamientos e interacción con direcciones cross-fit. Derecha: AUROC fuera de muestra de la dirección conversacional. Cada capa es una medición correlacionada del mismo corpus.", "fig:auroc_capa")
    ch4 += r"""
\subsection{Desplazamiento y control benigno}
\label{sec:desplazamiento}

El resultado central se expresa mediante cambios intra-par. Una proyección menor no demuestra por sí sola menor rechazo: es una medida geométrica. El brazo benigno permite descartar un desplazamiento idéntico de ambas clases sobre el eje, pero no identifica por sí solo el mecanismo completo ni elimina toda posible interacción de longitud y contenido.
"""
    ch4 += table(["Modelo", r"$\Delta p_h$", r"$\Delta p_b$", r"$\Delta^2p$", "IC 95\\,\\% interacción"],
                 [["Llama V3",number(f["delta_p_harmful"],4),number(f["delta_p_benign"],4),number(f["interaction"],4),"$["+number(f["interaction_ci"][0],4)+";\\ "+number(f["interaction_ci"][1],4)+"]$"],
                  ["Mistral V3",number(m["delta_h_oof"],4),number(m["delta_b_oof"],4),number(m["interaction_oof"],4),"$[7{,}0992;\\ 9{,}6758]$"]],
                 "Desplazamiento fuera de muestra, con dirección reajustada por fold. La interacción respeta el pareado dañino--benigno. Las unidades crudas no se comparan entre modelos.", "tab:gap")
    ch4 += r"""
Las interacciones retenidas son positivas y sus intervalos excluyen cero. El test intra-par guardado reporta $p\simeq 10^{-4}$ en ambos casos, a la resolución de 10.000 permutaciones. La fracción común es aproximadamente $-9{,}8\,\%$ en Llama y $-7{,}4\,\%$ en Mistral. En ambos, el cambio benigno es pequeño frente al dañino y su contraste individual no permite afirmar un movimiento benigno distinto de cero al nivel $0{,}05$. La evidencia se concentra en el contraste entre brazos, no en declarar ausencia exacta de efecto en el benigno.

\subsection{Escala del efecto}
\label{sec:escala}

La separación cruda mezcla capacidad discriminativa y escala de proyección. En Llama, fuera de muestra, su pérdida es $82{,}6\,\%$, mientras la separación estandarizada cae $43{,}7\,\%$ y el AUROC pierde $0{,}0584$. En Mistral las pérdidas impresas son $84{,}7\,\%$, $63{,}8\,\%$ y $0{,}1627$, respectivamente. Ninguna cifra cruda debe presentarse como porcentaje de pérdida de seguridad.
"""
    ch4 += table(["Modelo", r"$d_{\rm chat}$",r"$d_{\rm ag}$","Pérdida en $d$",r"$\kappa$"],
                 [["Llama V3",number(f["cohens_d_chat"],4),number(f["cohens_d_agent"],4),number(100*f["separation_loss_standardised"],1)+r"\,\%",number(f["scale_contraction"],3)],
                  ["Mistral V3", "2{,}5289", "0{,}9163",r"63{,}8\,\%", "0{,}422"]],
                 "Separación entre clases en unidades estandarizadas, obtenida sobre puntajes fuera de muestra. Estos valores de $d$ difieren del tamaño pareado del cambio de formato.", "tab:escala")
    ch4 += r"""
\subsection{Comportamiento por explicitud}
\label{sec:estratos}

El análisis disponible utiliza los estratos originales de 18, 14 y 10 pares. El corpus ampliado cuenta con 42, 24 y 30, pero esos tamaños no se transfieren a las estimaciones siguientes.
"""
    sr=[]
    labels={"explicit":"Explícito","contextual":"Contextual","framed":"Apariencia legítima"}
    for model in ["Llama V3", "Mistral V3"]:
        for s in order:
            if model.startswith("Llama"):
                vals=sl[s];n=vals["n_harmful"];ac=vals["auroc_chat"];aa=vals["auroc_agent"]
            else:n,ac,aa=m["strata_oof"][s]
            sr.append([model,labels[s],n,number(ac,4),number(aa,4),number(ac-aa,4)])
    ch4 += table(["Modelo","Estrato","Pares","Chat","Agente","Pérdida"],sr,
                 "AUROC por explicitud sobre puntajes fuera de muestra. El pequeño tamaño de los estratos y la ausencia de intervalos por estrato impiden establecer una ordenación poblacional firme.", "tab:estratos","llrrrr")
    ch4 += r"""
La pérdida puntual máxima de Llama aparece en el estrato explícito y la de Mistral en el contextual. Por tanto, no hay una jerarquía universal de degradación según explicitud en estas ejecuciones. El estrato de apariencia legítima conserva señal por encima del azar en ambos modelos, pero no se interpreta como garantía de detección de solicitudes encubiertas nuevas.
"""
    ch4 += figure("avance_estratos", "Discriminación por explicitud en las dos ejecuciones principales. Ambas usan 42 pares; las barras son estimaciones puntuales, no diferencias estadísticamente contrastadas entre modelos.", "fig:cmp_estratos")
    ch4 += r"""
\subsection{Reorientación y monitores por estimador}
\label{sec:rotacion}

En Llama, capa 76, el coseno de muestra completa entre direcciones de medias es $0{,}2128$. Las fiabilidades entre mitades disjuntas son $0{,}9010$ en chat y $0{,}9220$ en agente. La estimación normalizada por el techo de fiabilidad da coseno $0{,}2204$ y ángulo descriptivo de unos $77^\circ$. La alineación es baja, pero no corresponde al coseno $0{,}051$ que la versión anterior atribuía a ese modelo.

La comparación logística produce coseno de muestra completa $0{,}0744$ y fiabilidades mucho menores, $0{,}2522$ y $0{,}2832$. Esa menor reproducibilidad obliga a interpretar con más cautela el ángulo. Las dos medidas no se promedian ni se sustituyen una por otra.
"""
    ch4 += figure("avance_rotacion", "Llama V3: cosenos de muestra completa entre formatos, usando el mismo estimador en ambos. Los valores no son una prueba causal y deben leerse junto con la fiabilidad descrita en el texto.", "fig:cmp_rotacion")
    ch4 += table(["Modelo / estimador", "AUROC OOF",r"$F_1$ OOF","FPR OOF"],
                 [["Llama / medias",number(r["auroc_agent_diff_means_oof"],4),number(r["monitor_f1_diff_means_oof"],4),number(r["monitor_fpr_diff_means_oof"],4)],
                  ["Llama / logística",number(r["auroc_w_agent_oof"],4),number(r["monitor_f1_oof"],4),number(r["monitor_fpr_oof"],4)],
                  ["Mistral / logística (log)","0{,}9347","0{,}8250","0{,}1190"]],
                 "Monitores agénticos evaluados por cinco folds. En Llama se usa capa 76; en Mistral, capa 30. El AUROC corresponde a la media de folds y $F_1$ al protocolo de umbrales retenidos; las métricas no son intercambiables.", "tab:monitores")
    ch4 += r"""
El cero de falsos positivos de la logística de Llama corresponde a 42 benignos de este corpus. No demuestra una tasa poblacional nula ni seguridad de despliegue. La señal agéntica es aprovechable por un monitor ajustado en ese formato; eso no demuestra que el mecanismo interno de rechazo use espontáneamente el mismo eje.

\section{Evaluación y validación}
\label{sec:evaluacion-resultados}
\subsection{PCA del contraste conversacional}

PCA se ajusta sobre activaciones conversacionales concatenadas y centradas globalmente, según el procedimiento de métodos. La Tabla~\ref{tab:pca} incorpora los resultados que faltaban en la versión anterior. Los de Llama se comprobaron además desde las activaciones guardadas mediante el autovector principal de la matriz de Gram.
"""
    pcar=[]
    for l in [28,56,76]:
        v=d["pca_by_layer"][str(l)];pcar.append(["Llama V3",l,number(v["cosine_similarity_with_diff_means"],4),number(100*v["explained_variance_ratio_pc1"],2)+r"\,\%"])
    for l in [18,30]:pcar.append(["Mistral V3",l,number(m["pca"][l]["cos"],3),number(100*m["pca"][l]["variance"],1)+r"\,\%"])
    ch4 += table(["Modelo","Capa",r"$|\cos(\mathrm{PC1},\hat{\bm{r}})|$","Varianza de PC1"],pcar,
                 "Validación geométrica descriptiva, dentro de muestra. La alineación con PC1 no sustituye la evaluación fuera de muestra del AUROC.","tab:pca")
    ch4 += figure("avance_pca", "Alineación de PC1 y proporción de varianza explicada a lo largo de las capas. Las posiciones de capa son propias de cada arquitectura y los puntos de Mistral conservan la precisión impresa del log.","fig:pca")
    ch4 += r"""
En capas iniciales el contraste no siempre coincide con PC1. En las capas profundas mostradas sí presenta alineación elevada, sin que PC1 explique toda la varianza: por ejemplo, en la capa 76 de Llama explica $20{,}13\,\%$. Este resultado respalda una estructura geométrica dominante de la muestra en esas capas, no la identidad entre un componente principal y una causa del rechazo.

\subsection{Circularidad y selección de capa}

En Llama, capa 76, el desplazamiento ingenuo es $16{,}2776$ y el cross-fit es $15{,}7244$: la diferencia equivale al $3{,}5\,\%$ del valor corregido. En Mistral, capa 30, pasa de $8{,}0227$ a $7{,}8359$, un $2{,}4\,\%$. La corrección reduce el efecto y este conserva su signo. Los conteos cross-fit son 42/42 y 39/42 decrecientes, respectivamente; constituyen una comprobación descriptiva sobre el corpus completo.

Para evitar reutilizar los pares que seleccionan una capa, la inferencia separada utiliza solo la otra mitad. La Tabla~\ref{tab:seleccion} resuelve la ambigüedad del conteo antiguo de 39/42: no debe presentarse como un conteo exclusivo de inferencia.
"""
    ch4 += table(["Modelo","Selección","Inferencia","Capa","Decrecen",r"$\Delta p$ inferencia"],
                 [["Llama V3",21,21,76,"21/21",number(selection["delta_p_crossfit"],4)],
                  ["Mistral V3",21,21,30,"19/21","7{,}6873"]],
                 "Selección e inferencia sobre mitades disjuntas de pares. Los 42/42 o 39/42 del cross-fitting global no son el denominador de esta prueba.","tab:seleccion")
    ch4 += r"""
El intervalo de inferencia de Llama es $[14{,}0276;\ 18{,}9079]$ y el de Mistral $[6{,}0074;\ 9{,}1634]$. Sus pruebas de signos arrojan aproximadamente $9{,}54\times10^{-7}$ y $2{,}21\times10^{-4}$; ambas quedan bajo los umbrales Bonferroni $0{,}0025$ y $0{,}003125$. Sobre 200 particiones aleatorias, cada ejecución selecciona su misma capa en el $100\,\%$ de los casos guardados. Estas particiones describen estabilidad sobre el corpus observado; no multiplican el tamaño de muestra.

La prueba de signos depende de la dirección utilizada. Su invariancia se limita a reescalados positivos, no a cambios del eje. La separación entre ajuste y evaluación protege frente a circularidad, sin eliminar todos los supuestos de muestreo.

\section{Ensayos conductuales e intervención exploratoria}
\label{sec:conductual}

Llama V3 conserva las generaciones y sus etiquetas heurísticas. En solicitudes dañinas, el rechazo sin intervención pasa de 38/42 en chat ($90{,}5\,\%$) a 23/42 en agente ($54{,}8\,\%$). Dieciséis pares dejan de rechazar y uno cambia en sentido contrario, con McNemar exacto $p=0{,}000275$. En benignos las tasas son 5/42 y 1/42. Estas etiquetas describen respuestas generadas, no ejecución de tareas mediante herramientas, y su auditoría humana está pendiente.

Como ejemplo de intervención se reporta la capa 56, declarada explícitamente en la configuración de ese ensayo; es distinta de la capa 76 usada para el diagnóstico principal. Se evalúan 19 dañinos y 41 benignos que inicialmente no eran rechazados en formato agéntico.
"""
    cr=[]
    for dose in doses:
        v=cd[f"{dose:g}"]
        cr.append([number(dose,1),f"{v['restore']['k']}/{v['restore']['n']}",f"{v['control']['k']}/{v['control']['n']}",f"{v['benign']['k']}/{v['benign']['n']}"])
    ch4 += table(["Dosis","Rechazo dañino","Control aleatorio","Rechazo benigno"],cr,
                 "Llama V3, intervención exploratoria en capa 56. El control aleatorio tiene igual norma. Los denominadores corresponden a subconjuntos que no rechazaban inicialmente; no son los 42 pares completos.","tab:causal")
    ch4 += figure("avance_intervencion","Respuesta a la dosis en capa 56. Barras: intervalos binomiales guardados. Una dosis que induce rechazo también en benignos no constituye una reparación selectiva.","fig:causal")
    ch4 += r"""
En dosis 1 el rechazo dañino es 12/19 con dirección de rechazo y 1/19 con control aleatorio; McNemar pareado $p=0{,}000977$. El brazo benigno rechaza 5/41. En dosis 2 el rechazo dañino alcanza 19/19, pero también 40/41 benignos. El ensayo apoya un efecto de la adición sobre la conducta etiquetada en estas condiciones y muestra un costo de sobrerrechazo. No demuestra que la reorientación observada explique por sí sola el fallo ni resuelve la calibración de una defensa. Los valores $p$ por dosis son exploratorios y no están corregidos por todos los contrastes de capas y dosis.

Mistral conserva una línea base impresa de 19/42 rechazos dañinos en chat y 5/42 en agente, con McNemar $p\simeq0{,}0013$. Su registro termina durante la intervención; por ello no se presenta una curva causal completa ni se infiere su desenlace.

\section{Evidencia exploratoria de Qwen}

Los JSON históricos de Qwen usan igualmente 42 pares. En Qwen3-8B, capa 32, el AUROC held-out de la dirección conversacional es $0{,}5176$ en chat y $0{,}5108$ en agente: esta ejecución no establece una línea base discriminativa suficiente en esa capa. En Qwen3-32B, capa 46, los valores son $0{,}9694$ y $0{,}9558$. La configuración exportada es menos completa y no acredita todos los controles actuales. Se conservan como antecedentes experimentales, sin utilizarlos para sostener una réplica uniforme entre familias ni una ley de escala. Repetirlos con la versión corregida y el corpus ampliado es parte del trabajo pendiente.
"""
    write("chapters/Chapter4/chapter04.tex",ch4)

    write("chapters/Chapter5/chapter05.tex", r"""
\chapter{Discusión}
\label{chap:discusion}
\section{Síntesis de los avances}

El avance principal es haber convertido una comparación de proyecciones en un diagnóstico con controles explícitos. El contenido se conserva entre formatos, el brazo benigno distingue cambios comunes de contrastes específicos del daño y la evaluación por pares retenidos evita confundir separación de entrenamiento con generalización. La integración de PCA y la separación de selección e inferencia permiten responder a las observaciones sobre validación geométrica y reutilización de la muestra.

Los resultados de Llama V3 y el registro de Mistral V3 sostienen una reducción del margen discriminativo sobre la dirección conversacional. En ambos, la interacción sigue siendo positiva fuera de muestra y el AUROC agéntico se mantiene por encima del azar. Por tanto, los datos disponibles describen una degradación de margen, con intensidad dependiente del modelo, y no una desaparición completa de información discriminativa.

\section{Interpretación del cambio de representación}

La contracción de escala explica parte de la gran caída de separación cruda. La pérdida persiste en unidades estandarizadas y en AUROC, por lo que no se agota en un cambio de unidades. Sin embargo, restar el control benigno tampoco convierte automáticamente el resto en una medida de rechazo: la etiqueta dañino--benigno, la proyección y la decisión conductual son tres objetos distintos.

La baja alineación entre direcciones comparables en Llama es compatible con una reorientación del contraste entre clases. La fiabilidad alta de las direcciones de medias vuelve esa descripción más interpretable que un coseno obtenido de sondas poco reproducibles. Un monitor agéntico logra recuperar parte de la señal, pero su existencia no prueba que el modelo use ese eje para decidir. El comportamiento puede depender de otras direcciones, capas y transformaciones posteriores.

La comparación entre modelos tampoco aísla el número de parámetros. Cambian arquitectura, entrenamiento, plantilla y distribución de proyecciones. Las diferencias puntuales entre Llama y Mistral no permiten concluir que aumentar tamaño preserve la seguridad. Las ejecuciones históricas de Qwen refuerzan la necesidad de revisar cada configuración y de no seleccionar solo resultados favorables.

\section{Lo que aporta la intervención}

El ensayo complementario de Llama permite ir más allá de asociar dos cantidades: añadir activación en una dirección cambia la fracción de respuestas etiquetadas como rechazo respecto de un control aleatorio. Su interpretación se limita a esa manipulación y a los subconjuntos ensayados. No demuestra necesidad del mecanismo ni que la rotación sea el mediador del efecto del formato. El rechazo masivo de benignos a dosis altas descarta interpretar cualquier incremento de rechazo como una mejora selectiva de seguridad.

La heurística de etiquetado hace auditables los cálculos, pero puede confundir una negativa superficial, un rechazo seguido de información o una respuesta que aparenta usar herramientas. La lectura humana independiente de las generaciones es necesaria antes de elevar estos resultados a una conclusión conductual firme. En Mistral la evidencia causal local es incompleta.

\section{Limitaciones y alcance de validez}

La primera limitación es muestral: el corpus construido tiene 96 pares, mientras las ejecuciones analizadas usan 42 y el estrato más pequeño solo diez. Las cuatro variantes, las capas y los 200 splits no son nuevas unidades independientes. La anotación de explicitud no dispone todavía de una segunda clasificación independiente para estimar acuerdo.

La segunda es el alcance del formato. Se describe capacidad de usar herramientas mediante un mensaje y esquemas simulados; no se observa una tarea autónoma ejecutada durante varios pasos. El contraste principal cambia sistema y herramientas conjuntamente. La interacción con el brazo benigno no descarta por completo efectos de longitud, delimitadores o interacciones entre esos componentes y el contenido.

La tercera es la inferencia. El cross-fitting controla reutilización de cada par, pero induce dependencia por entrenamiento compartido. Los intervalos de los puntajes guardados no incorporan toda la incertidumbre de volver a ajustar el pipeline. La inferencia sobre mitades disjuntas reduce el sesgo de selección, pero evalúa solo 21 pares. Los ángulos ajustados por fiabilidad son estimaciones bajo supuestos, no constantes conocidas.

La cuarta es de reproducibilidad. La copia de activaciones y la duplicación de tokens especiales exigieron correcciones. Las cifras de versiones distintas deben permanecer identificadas; no se atribuye su diferencia a fluctuación de muestreo sin comprobar la implementación. Llama V3 ofrece activaciones y resultados estructurados; Mistral V3 solo está disponible como registro redondeado. Los monitores carecen de evaluación externa y sus umbrales pueden no transferirse a otro corpus o despliegue.

\section{Implicaciones para el avance siguiente}

El siguiente paso informativo es repetir el protocolo corregido sobre los 96 pares construidos, con la misma versión de código, configuración exportada y conservación de activaciones y generaciones. Debe fijarse un protocolo de comparación entre modelos y estimadores, reservar evaluación independiente de umbrales e incorporar una segunda anotación. En paralelo, el análisis factorial puede separar mensaje de sistema y herramientas, con controles de longitud y de plantilla. Solo después corresponde valorar una defensa operacional.
""")
    write("chapters/Chapter6/chapter06.tex", r"""
\chapter{Conclusiones y recomendaciones del avance}
\label{chap:conclusiones}
\section{Conclusiones}

Se construyó y verificó un corpus contrastivo ampliado de 96 pares, con cuatro variantes por par y control de identidad de texto y simetría del andamiaje. Ese es un resultado de construcción de datos. Las conclusiones experimentales de esta versión se refieren a las ejecuciones disponibles de 42 pares y no se extrapolan como si provinieran del corpus ampliado.

En Llama V3 y Mistral V3, la dirección conversacional conserva capacidad discriminativa en formato agéntico, aunque con menor margen: el AUROC fuera de muestra pasa de $0{,}9779$ a $0{,}9195$ en Llama y de $0{,}9478$ a $0{,}7851$ en Mistral. La interacción con el control benigno y la pérdida estandarizada muestran que el resultado no se reduce a una traslación idéntica de ambas clases ni a un mero cambio de escala.

La validación geométrica muestra alineación elevada entre PC1 y la diferencia de medias en capas profundas, y Llama presenta baja alineación entre los contrastes de ambos formatos. La interpretación debe mantener separados diferencia de medias y regresión logística e incorporar la fiabilidad de sus direcciones. Estas observaciones caracterizan representaciones; no identifican por sí solas la causa general del fallo del rechazo.

La disponibilidad de monitores fuera de fold acredita una viabilidad preliminar de recuperar señal agéntica. El ensayo de intervención en Llama aporta evidencia exploratoria de que añadir activación puede aumentar el rechazo, con un costo creciente en solicitudes benignas. La calibración, la validación humana y la transferencia siguen abiertas. Las corridas históricas de Qwen no justifican afirmar un patrón uniforme en todas las familias.

\section{Aportes del trabajo hasta este corte}

El trabajo aporta un corpus ampliado auditable, un pipeline de extracción y análisis con controles de circularidad y pareado, resultados geométricos trazables para Llama y un registro de réplica en Mistral, además de ensayos conductuales conservados para auditoría. La separación entre unidades crudas, separación estandarizada y AUROC evita interpretar un cambio de escala como pérdida total de seguridad. La separación de selección e inferencia hace explícito qué pares sustentan cada contraste.

\section{Recomendaciones}

Se recomienda consolidar los artefactos completos de Mistral, ejecutar el corpus de 96 pares en los modelos con configuración común, documentar versiones y hashes y reservar datos para validación externa. Es necesario completar doble anotación de explicitud y revisión independiente de las generaciones, comparar formalmente estimadores sobre las mismas particiones y evaluar umbrales en condiciones distintas de aquellas que los ajustan. La extensión a trayectorias, autonomía y herramientas ejecutadas requiere un diseño adicional y no se considera validada por este avance.
""")

    write("chapters/Chapter1/1.3_abstract.tex", r"""
\chapter*{Resumen}
\addcontentsline{toc}{chapter}{Resumen}

Este avance estudia cómo cambia la representación interna de instrucciones dañinas al pasar de un formato conversacional a otro con mensaje de sistema y herramientas simuladas. Se construyó un corpus contrastivo de 96 pares dañino--benigno, con cuatro variantes por par y tres niveles de explicitud. Las ejecuciones disponibles corresponden al corpus original de 42 pares: Llama-3.3-70B V3 y el registro de Mistral-7B V3 sustentan los resultados principales; Qwen se conserva como evidencia exploratoria previa.

La dirección conversacional se estima por diferencia de medias sobre el flujo residual del último token del estímulo renderizado. Con cinco folds por pares, su AUROC pasa de $0{,}9779$ en chat a $0{,}9195$ en agente para Llama y de $0{,}9478$ a $0{,}7851$ para Mistral. Las interacciones con el brazo benigno son positivas y la pérdida persiste en unidades estandarizadas, aunque es menor que la caída cruda. PCA se alinea con el contraste en capas profundas. En Llama, la dirección agéntica de medias presenta baja alineación con la conversacional y permite un monitor con $F_1$ fuera de fold de $0{,}8434$; la logística constituye un estimador distinto.

Las observaciones sostienen una pérdida de margen discriminativo en las condiciones ensayadas. Las intervenciones complementarias de Llama aumentan el rechazo etiquetado, pero dosis altas también inducen rechazo de benignos y el etiquetado requiere auditoría humana. La generalización al corpus ampliado, a datos externos y a agentes que ejecuten acciones permanece pendiente.

\vspace{0.4cm}\noindent
\textbf{Palabras clave:} interpretabilidad mecánica; dirección de rechazo; formato agéntico; validación cruzada; seguridad en inteligencia artificial.
""")
    write("chapters/Chapter1/1.3_abstract_en.tex", r"""
\chapter*{Abstract}
\addcontentsline{toc}{chapter}{Abstract}

This progress report studies how internal representations of harmful instructions change between a conversational format and a format containing an agent system message and simulated tool schemas. An expanded contrastive corpus of 96 harmful--benign pairs was constructed, with four variants per pair and three explicitness levels. Available experiments use the original 42-pair corpus. Llama-3.3-70B V3 and the Mistral-7B V3 execution log support the main results; earlier Qwen runs are retained as exploratory evidence.

The conversational direction is estimated by difference in means from the residual stream at the final token of the rendered stimulus. With five-fold pair-level evaluation, its AUROC decreases from 0.9779 in chat to 0.9195 in the agent format for Llama, and from 0.9478 to 0.7851 for Mistral. Interactions with the benign arm are positive, and reduced separation persists in standardized units, although it is smaller than the raw decrease. PCA aligns with the contrast in deeper layers. For Llama, the agent mean-difference direction has low alignment with the conversational direction and supports an out-of-fold monitor with $F_1=0.8434$; logistic regression is treated as a separate estimator.

The observations support a reduced discriminative margin under the tested conditions. Complementary Llama interventions increase heuristically labelled refusal, but higher doses also induce benign refusal, and human label auditing remains necessary. Generalization to the expanded corpus, external data and agents executing real actions remains open.

\vspace{0.4cm}\noindent
\textbf{Keywords:} mechanistic interpretability; refusal direction; agentic format; cross-validation; AI safety.
""")

    glossary=(OUT/"glossary.tex").read_text(encoding="utf-8")
    glossary=glossary.replace("de modo que solo el último \\textit{token} tiene como cono la instrucción completa.","de modo que la posición final del estímulo renderizado tiene acceso causal al estímulo completo, incluidos sus delimitadores.")
    glossary=re.sub(r"Depende del modelo: 18 en Mistral-7B, 22 en Qwen3-8B, 46 en Qwen3-32B y 56 en Llama-3\.3-70B\.","En las ejecuciones principales de esta versión es 30 en Mistral V3 y 76 en Llama V3; las ejecuciones históricas se identifican por separado.",glossary)
    glossary=glossary.replace("Cuenta la proporción de comparaciones dañino--benigno que el formato conversacional ordena correctamente y el agéntico deja de ordenar correctamente.","Es la diferencia neta de probabilidades de ordenación correcta; admite cambios en ambos sentidos y no cuenta exclusivamente los aciertos perdidos.")
    glossary=glossary.replace(r"\item[$\bm{w}_{\text{ag}}$] Dirección estimada por diferencia de medias dentro del formato agéntico.",r"\item[$\bm{w}_{\mathrm{DM}}$ y $\bm{w}_{\mathrm{LR}}$] Direcciones agénticas estimadas por diferencia de medias y regresión logística, respectivamente. Se reportan por separado. La primera es el contraste entre centroides dentro del formato agéntico.")
    write("glossary.tex",glossary)
    main=(OUT/"main.tex").read_text(encoding="utf-8")
    main=main.replace(r"% \include{chapters/Chapter6/chapter06}",r"\include{chapters/Chapter6/chapter06}")
    main=main.replace(r"\input{glossary}",r"\input{glossary}"+"\n\\appendix\n\\include{chapters/Appendix/anexos}")
    main=main.replace(r"\setlength{\headheight}{15pt}",r"\setlength{\headheight}{16pt}")
    write("main.tex",main)
    config=(OUT/"config.tex").read_text(encoding="utf-8")
    config=config.replace("Un análisis desde la interpretabilidad mecánica", "Un análisis desde la interpretabilidad mecánica. Avances de investigación")
    write("config.tex",config)

    appendix=r"""
\chapter{Trazabilidad y material complementario}
\label{app:trazabilidad}
\section{Corpus construido y muestra ejecutada}

El archivo \texttt{dataset\_v2.jsonl} contiene 96 pares, 384 entradas, 62 pares propios y 34 derivados de AgentHarm. El informe de construcción se adjunta en \texttt{evidencia/}. La identidad del texto entre formatos y la simetría de sistema, herramientas y esquemas dentro de cada formato se verificaron en 96/96 pares. Hay 64 herramientas distintas utilizadas. Los conteos por explicitud son 42, 24 y 30. El corpus original ejecutado tiene 42 pares, con estratos de 18, 14 y 10. Estos dos conjuntos no se intercambian al informar denominadores.

La revisión asistida de pares no sustituye una segunda anotación humana independiente. El cálculo de acuerdo entre anotadores permanece pendiente. Las etiquetas y el informe del corpus acompañan el material complementario; las instrucciones completas y las generaciones se conservan en el repositorio de investigación y no se reproducen aquí como un catálogo de tareas.

\section{Configuración de la corrida principal}

Llama V3 emplea \texttt{meta-llama/Llama-3.3-70B-Instruct}, precisión FP16, semilla 42, cinco folds, diez repeticiones de la sonda lineal, una de MLP, diez controles permutados, 10.000 remuestreos y 200 divisiones de selección. El barrido usa capas 0, 4, 8, hasta 76. El corpus se carga por la opción predeterminada original, con \texttt{dataset\_path=null}; las activaciones y sus metadatos identifican 42 pares. Las intervenciones declaradas son en capas 56, 52 y 28, con dosis 0,5; 1; 2 y 4 y un máximo de 64 tokens nuevos.

La configuración exportada de Mistral declara FP16, barrido de 16 capas y 42 pares. Su registro contiene PCA, evaluación retenida y selección de capa, pero el archivo termina durante la etapa causal. La ausencia local del JSON y de sus activaciones limita la reproducción de ese análisis. Los resultados redondeados se identifican como tales.

\section{Control de versiones de resultados}

La versión anterior de la tesis empleaba para Llama la capa 56 y $\Delta p\simeq5{,}74$. Esta versión utiliza Llama V3, capa 76, y reporta como principal el valor fuera de muestra $15{,}7244$. El código documenta la sustitución de una copia asíncrona GPU--CPU por una bloqueante y la corrección de duplicación de BOS. Por eso las diferencias entre ejecuciones no se interpretan como fluctuación estadística sin verificar implementación. El repositorio conserva las corridas anteriores para inspección; sus cifras no se mezclan con V3.

Los archivos de evidencia adjuntos incluyen métricas extraídas y un manifiesto SHA-256 de las fuentes originales. El hash identifica el contenido revisado, pero no reconstruye una revisión de código de entrenamiento ni certifica una ejecución externa que no esté archivada. La regeneración de tablas y figuras se realiza con \texttt{scripts/actualizar\_tesis\_avances.py}, incluido también como \texttt{regenerar\_avances.py} en este paquete, para ejecutarse desde el repositorio original con sus resultados.

\section{Estado de las observaciones del tutor}

Se actualizaron resumen y abstract; se definieron los conceptos desde la introducción; se precisó el formato como sistema y herramientas; se verificó simetría entre ramas; se documentó autoría y apoyo de anotación; se distinguió proyección de rechazo; se incorporaron PCA y su procedimiento; se corrigió la interpretación de la prueba de signos; se separaron selección e inferencia y se retiró cualquier equivalencia inferida solo de cercanía entre sonda y MLP. También se añadieron discusión, conclusiones del avance y este material complementario.

La revisión ortográfica normaliza los demostrativos del texto editado. No se dispone de la nota original para certificar el fragmento concreto señalado por el tutor. La recomendación de retroalimentación externa es de proceso y no se marca como realizada a partir de los archivos. La réplica de 96 pares, la segunda anotación, la auditoría humana conductual y la validación externa permanecen abiertas.

\section{Guion breve para presentar el avance}

La presentación puede organizarse en seis pasos: problema y pregunta; diseño pareado y corpus ampliado construido; evidencia ejecutada sobre 42 pares; pérdida de margen y control benigno; validación con PCA, cross-fitting y selección separada; e intervención exploratoria, limitaciones y siguiente ejecución. En cada diapositiva de resultados debe indicarse modelo, capa, estimador y denominador. Las figuras de esta versión están disponibles en PDF y PNG y pueden reutilizarse directamente.
"""
    write("chapters/Appendix/anexos.tex",appendix)
    evidence={"date":"2026-10-15", "expanded_corpus_pairs":96,"executed_pairs":42,
              "llama":{"config":d["config"],"reference_layer":76,"n_pairs":42,
                       "heldout_detection":ho["detection"],"heldout_format":f,
                       "heldout_paired":ho["paired_harmful"],"selection":selection,
                       "pca":d["pca_by_layer"],"rotation_at_reference":rot,
                       "monitors_at_reference":r,"strata_oof":strata,
                       "behavior":causal["behavior"],"intervention_layer56":causal["causal"]["56"]},
              "mistral":m,"historical_qwen":qwen}
    write("evidencia/metricas_avance.json",json.dumps(evidence,ensure_ascii=False,indent=2,allow_nan=False))
    sources=[LLAMA/"results.json",LLAMA/"robust_results.json",LLAMA/"causal_results.json",LLAMA/"activations.npz",MLOG,
             ROOT/"data/dataset_v2.jsonl",ROOT/"data/dataset_full.jsonl",ROOT/"src/activation_extractor.py",ROOT/"src/model_loader.py",ROOT/"run_experiment.py",
             ROOT/"results/qwen8b/results.json",ROOT/"results/qwen32b/results.json"]
    manifest=[]
    for src in sources:
        manifest.append({"path":src.relative_to(ROOT).as_posix(),"bytes":src.stat().st_size,"sha256":hashlib.file_digest(src.open("rb"),"sha256").hexdigest()})
    write("evidencia/manifiesto_fuentes.json",json.dumps(manifest,ensure_ascii=False,indent=2))
    shutil.copy2(MLOG,OUT/"evidencia/mistral_v3_registro.txt")
    for name in ["dataset_v2_report.json","etiquetado_explicitud_v2.csv"]:
        shutil.copy2(ROOT/"data"/name,OUT/"evidencia"/name)
    shutil.copy2(Path(__file__),OUT/"regenerar_avances.py")
    write("LEEME.md", """
# Tesis: versión de avances

Esta versión usa únicamente evidencia disponible: corpus construido de 96 pares y ejecuciones de 42. Resultados principales: Llama V3 y Mistral V3 (registro). Qwen histórico se identifica por separado.

Abrir `main.tex` como documento principal en Overleaf. Compilador: XeLaTeX; bibliografía: Biber, estilo APA. Conservar la estructura de carpetas. Compilar hasta resolver las referencias. El PDF compilado se entrega también junto al ZIP.

`figures/avance_*.pdf` y `figures/avance_*.png` son las figuras actualizadas. Las figuras antiguas extraídas del ZIP original no se utilizan en el documento revisado.

`evidencia/metricas_avance.json` registra las cifras y su procedencia; `manifiesto_fuentes.json` identifica con SHA-256 los archivos originales. No incluye activaciones completas. La reproducción de métricas requiere el repositorio y sus resultados, no solo este ZIP.

Para regenerar desde el repositorio: `python scripts/actualizar_tesis_avances.py`. El script sobrescribe las fuentes de esta versión a partir del ZIP original y los resultados locales. El archivo `regenerar_avances.py` es una copia de consulta; debe ubicarse en `scripts/` del repositorio original para ejecutarlo. No ejecuta modelos.

Se incorporaron PCA, inferencia separada de selección de capa, estimadores agénticos diferenciados, discusión, conclusiones del avance y anexos. Permanecen abiertos la réplica de 96 pares, doble anotación, auditoría humana de generaciones y validación externa.
""")
    write("RESUMEN_CAMBIOS.md", """
# Cambios para la presentación de avances

- Corpus construido y validado: 96 pares, 384 variantes. Ejecuciones disponibles: 42 pares por modelo. Se corrigieron los denominadores de tablas, resúmenes y conclusiones.
- Se actualizaron las cifras de Llama a V3, capa 76. Las tablas principales usan proyecciones y AUROC fuera de muestra. Mistral V3 se identifica como evidencia redondeada de un registro, capa 30.
- Se incorporó PCA con procedimiento y resultados; se distinguieron dirección de medias y sonda logística.
- Se documentaron 21 pares para selección y otros 21 para inferencia. Se corrigieron interpretación de signos, posición de lectura y efectos del formato compuesto.
- Se añadieron discusión, conclusiones del avance, anexos de trazabilidad, evidencia exportada y cinco figuras nuevas en PDF/PNG.
- Se incorporaron los ensayos conductuales de Llama como exploratorios, con control aleatorio, rechazo benigno y límite del etiquetado heurístico. Qwen previo se presenta sin atribuirle una réplica uniforme del pipeline actual.

Guion sugerido: (1) problema y pregunta; (2) corpus y control pareado; (3) AUROC y margen; (4) PCA y selección; (5) monitores e intervención; (6) límites y siguientes pasos.

No se ejecutaron modelos nuevos. No se atribuyen resultados a los 96 pares sin archivos que los acrediten. Los PDF y ZIP originales se conservan.
""")
    response_rows = [
        (1,"Proceso y retroalimentación","De proceso","No requiere edición; la retroalimentación externa no se certifica mediante archivos."),
        (2,"Resumen","Actualizado","Identifica corpus construido de 96 y ejecución de 42 pares; cifras actuales y alcance del avance."),
        (3,"Abstract","Actualizado","Se sincronizó con el resumen y los resultados de V3."),
        (4,"Estas/éstas","Revisado","Se normalizaron demostrativos del texto editado. No se dispone de la nota literal para acreditar el pasaje concreto."),
        (5,"Dos dimensiones","Atendido","Introducción: se distinguen formato y explicitud."),
        (6,"Definir conceptos desde el inicio","Atendido","Definiciones operativas de dirección, proyección, AUROC y formato en la introducción; formalización y glosario complementan."),
        (7,"Remisión a diseño","Atendido","Se atiende junto con 9 y 10."),
        (8,"Brecha y Pinto","Atendido en texto","Se conserva la distinción entre steering y diagnóstico; no se presenta como auditoría bibliográfica externa."),
        (9,"Cambio de instrucciones y herramientas","Corregido","Se separa variación entre pares de comparación pareada. La manipulación combina sistema y herramientas."),
        (10,"Simetría entre ramas","Verificado","96/96 pares del corpus ampliado: mismo sistema, catálogo y esquemas dentro de cada formato."),
        (11,"Autoría y revisión asistida","Documentado","Se precisa composición por origen y responsabilidad del autor; revisión asistida no equivale a doble anotación."),
        (12,"Alcance de la proyección","Corregido","Se distinguen alineación geométrica, discriminación de contenido y rechazo conductual."),
        (13,"Dependencia y pareado","Corregido con límites","Interacción intra-par y evaluación retenida; se explicitan dependencia entre folds y supuestos entre pares."),
        (14,"Procedimiento y resultados PCA","Incorporado","Métodos y resultados: centrado global, PC1, varianza explicada y coseno; figura regenerada."),
        (15,"Extracción residual","Precisado","Último token del estímulo renderizado antes de generar; puede ser delimitador de plantilla."),
        (16,"Dirección y prueba de signos","Corregido","La prueba depende del eje; solo es invariante a reescalados positivos. Se distingue inferencia separada de cross-fitting."),
        (17,"Lineal/MLP y curva","Corregido","No se afirma equivalencia por cercanía. Se separan las métricas de sonda y dirección; las figuras usan la corrida actual."),
        (18,"39/42 y selección de capa","Resuelto para 42","21 pares seleccionan y 21 infieren: Llama 21/21 decrecen, Mistral 19/21. La réplica de 96 permanece pendiente."),
        (19,"Conclusiones y recomendaciones","Redactado como avance","Se completaron discusión y conclusiones acotadas a los resultados disponibles; se habilitó el capítulo 6."),
        (20,"Material complementario","Incorporado","Anexos de corpus, configuración, evidencia, hashes y estado de observaciones. No reemplaza validación externa."),
    ]
    response = r"""\documentclass[10pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[spanish]{babel}
\usepackage[letterpaper,margin=1.8cm]{geometry}
\usepackage{longtable,booktabs,array}
\usepackage[hidelinks]{hyperref}
\setlength{\parindent}{0pt}
\setlength{\parskip}{6pt}
\begin{document}
\begin{center}\Large\textbf{Respuesta actualizada a las observaciones del tutor}\end{center}
\textbf{Corte de avances: 15 de octubre de 2026.}

La versión actualizada se basa en los archivos experimentales disponibles. El corpus ampliado está construido con 96 pares; las ejecuciones analizadas usan 42 pares. La ampliación del corpus no se presenta como una ejecución ya realizada. Los originales se conservan y las referencias siguientes corresponden a las secciones de la versión de avances.

\small
\begin{longtable}{@{}p{0.5cm}p{3.2cm}p{2.5cm}p{10.7cm}@{}}
\toprule N.º & Observación & Estado & Cambio y alcance \\\midrule
\endfirsthead
\toprule N.º & Observación & Estado & Cambio y alcance \\\midrule
\endhead
"""
    for row in response_rows:
        response += " & ".join(str(x) for x in row) + r" \\[4pt]" + "\n"
    response += r"""\bottomrule\end{longtable}
\normalsize
\textbf{Pendientes experimentales.} Réplica sobre 96 pares, recuperación del paquete completo de Mistral V3, segunda anotación de explicitud, auditoría humana de generaciones y evaluación externa de monitores. Las conclusiones incorporadas son del avance, no un cierre de esas tareas.
\end{document}
"""
    write("respuesta_observaciones_actualizada.tex",response)
    # No dejar figuras históricas sin uso que puedan confundirse con las actuales.
    used = set(re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}",ch4))
    assert len(used)==5
    print(f"Versión generada: {OUT}")
    print("Corpus validado: 96 pares; ejecuciones principales: 42 pares; figuras nuevas: 5")


if __name__ == "__main__":
    main()
