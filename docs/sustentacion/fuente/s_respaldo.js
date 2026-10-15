// Respaldo para preguntas del jurado (R1–R12). No forman parte de los 20 minutos.
const B = require("./base");
const { W, MX, C, F, R, txt, m, mx, N, eq, rect, rrect, seg, etiquetaRespaldo, fuente, rotulo } = B;
const { nota } = require("./guion");
const { grafico, f2, pct } = require("./s_resultados");

const TBL = { fontFace: F.body, fontSize: 12, color: C.ink2, border: { type: "solid", pt: 0.75, color: "E3E6EA" }, valign: "middle", margin: [3, 5, 3, 5] };
const cab = (t) => ({ text: t, options: { bold: true, color: C.ink, fill: { color: C.mist }, fontFace: F.demi } });

module.exports = function (pres, D) {
  const L = D.llama, Mi = D.mistral, sec = "Respaldo";
  let n = 0;
  const nueva = (titulo) => {
    n += 1;
    const s = pres.addSlide({ masterName: "RESPALDO", sectionTitle: sec });
    etiquetaRespaldo(s, n);
    s.addText(titulo, { placeholder: "title" });
    return s;
  };

  // R1 · Modelos y configuración
  {
    const s = nueva("Modelos, ejecuciones y configuración");
    const filas = [["Modelo", "L", "d", "H", "H_kv", "d_ff", "Capas leídas", "Referencia", "Soporte"].map(cab),
      ["Mistral‑7B‑Instruct‑v0.3", "32", "4.096", "32", "8", "14.336", "16", "30", "registro de ejecución"],
      ["Qwen3‑8B", "36", "4.096", "32", "8", "12.288", "18", "32", "JSON histórico"],
      ["Qwen3‑32B", "64", "5.120", "64", "8", "25.600", "32", "46", "JSON histórico"],
      ["Llama‑3.3‑70B‑Instruct", "80", "8.192", "64", "8", "28.672", "20", "76", "JSON y activaciones"]];
    s.addTable(filas, Object.assign({}, TBL, { x: MX, y: 1.9, w: W - 2 * MX, colW: [2.9, 0.6, 0.85, 0.6, 0.7, 0.95, 1.25, 1.1, 3.18] }));
    rotulo(s, "Corrida principal · Llama V3", MX, 3.75, 6);
    txt(s, "meta‑llama/Llama‑3.3‑70B‑Instruct en FP16, semilla 42 · 5 folds por pares · sonda lineal con 10 repeticiones y MLP con 1 · 10 controles con etiquetas permutadas · 10.000 permutaciones y remuestreos · 200 particiones de selección · capas 0, 4, …, 76 · intervención en las capas 56, 52 y 28 con dosis 0,5; 1; 2 y 4 y hasta 64 tokens nuevos · 42 pares (dataset_path = null: corpus original).",
      { x: MX, y: 4.05, w: W - 2 * MX, h: 1.1, fontSize: 13.5, color: C.ink2 });
    txt(s, "Los 42 pares son los mismos en los cuatro modelos. Mistral V3 declara FP16, 16 capas y 42 pares; su JSON y sus activaciones no están en el repositorio local. Qwen conserva una configuración exportada menos completa.",
      { x: MX, y: 5.3, w: W - 2 * MX, h: 0.8, fontSize: 13, color: C.mute });
    fuente(s, "Tesis, tablas 3.4 y 4.1 y anexo A.2 · metricas_avance.json → llama.config.");
    nota(s, { titulo: "R1 · Modelos, ejecuciones y configuración", respaldo: true, t: 0,
      guion: "Si preguntan por qué estos modelos: son abiertos, alineados y con soporte de herramientas; cubren tres recetas de entrenamiento y escalas de 7 a 70 mil millones de parámetros. La comparación entre modelos no aísla el tamaño: cambian arquitectura, entrenamiento y plantilla. Si preguntan por qué no los 96 pares: el corpus ampliado está construido y validado, pero las ejecuciones disponibles son las del corpus original; replicarlo es el siguiente paso." });
  }

  // R2 · Perfil por capas del desplazamiento
  {
    const s = nueva("Perfil por capas del desplazamiento · Llama V3");
    const capas = L.capas.map(String), pc = L.por_capa;
    txt(s, [R("━ ", { color: C.harm }), ...N.dph(), R("    "), R("━ ", { color: C.ben }), ...N.dpb(), R("    "), R("━ ", { color: C.ink }), R("interacción "), ...N.d2p(), R("    fuera de muestra, unidades crudas")],
      { x: MX, y: 1.88, w: 9, h: 0.3, fontSize: 12.5, color: C.ink2 });
    grafico(s, pres, [
      { name: "Δp dañinas", labels: capas, values: pc.delta_p_harmful },
      { name: "Δp benignas", labels: capas, values: pc.delta_p_benign },
      { name: "Interacción", labels: capas, values: pc.interaction },
    ], { x: MX, y: 2.25, w: 7.6, h: 4.4 }, { chartColors: [C.harm, C.ben, C.ink], valAxisMinVal: -5, valAxisMaxVal: 20, valAxisMajorUnit: 5, dec: 0,
      catAxisLabelFrequency: 2, showCatAxisTitle: true, catAxisTitle: "capa" });
    const sel = L.seleccion;
    txt(s, [R("Capa de referencia. ", { fontFace: F.demi, color: C.ink }), R("La 76 maximiza el desplazamiento en el barrido completo y se usa para describir magnitudes. La inferencia usa otra mitad de los pares: en la mitad de selección la capa elegida es la 76 y lo es en el "),
      R(`${pct(sel.stability.selected_layer_frequency["76"], 0)} de las 200 particiones`, { fontFace: F.demi, color: C.ink }), R(".")],
    { x: 8.6, y: 2.25, w: W - MX - 8.6, h: 2.2, fontSize: 13.5, color: C.ink2 });
    txt(s, "El efecto crece con la profundidad y el brazo benigno se mantiene cerca de cero en todo el barrido. Cada capa es una medición correlacionada del mismo corpus: las curvas no son 20 réplicas.",
      { x: 8.6, y: 4.55, w: W - MX - 8.6, h: 1.8, fontSize: 13.5, color: C.ink2 });
    fuente(s, "results.json → heldout.per_layer (delta_p_harmful, delta_p_benign, interaction); robust_results.json → layer_selection.stability. Tesis, figura 4.1.");
    nota(s, { titulo: "R2 · Perfil por capas del desplazamiento", respaldo: true, t: 0,
      guion: "Si preguntan por qué la capa 76: es el máximo del barrido para describir magnitudes, y la inferencia no la reutiliza: se separan 21 pares para elegir y 21 para inferir, y la mitad de selección elige la 76 en todas las particiones guardadas. Si preguntan por qué la intervención es en la 56: está declarada así en la configuración de ese ensayo; no es la capa del diagnóstico." });
  }

  // R3 · Circularidad y selección de capa
  {
    const s = nueva("Circularidad y selección de capa");
    const sel = L.seleccion, inf = sel.inference, mon = L.monitores;
    const filas = [["", "Llama V3 · capa 76", "Mistral V3 · capa 30"].map(cab),
      ["Δp ingenuo (dirección de muestra completa)", f2(mon.delta_p_naive, 4), f2(Mi.ingenuo, 4)],
      ["Δp con cross‑fitting por pares", f2(mon.delta_p_crossfit, 4), f2(Mi.delta_h_oof, 4)],
      ["Sesgo de circularidad (fracción del valor corregido)", pct((mon.delta_p_naive - mon.delta_p_crossfit) / mon.delta_p_crossfit), pct((Mi.ingenuo - Mi.delta_h_oof) / Mi.delta_h_oof)],
      ["Pares que decrecen, cross‑fit global (42)", "42/42", "39/42"],
      ["Selección / inferencia (pares disjuntos)", "21 / 21", "21 / 21"],
      ["Decrecen en la mitad de inferencia", `${inf.n_decreasing}/21`, `${Mi.seleccion.decrecen}/21`],
      ["Δp en inferencia · IC 95 %", `${f2(inf.delta_p)} [${f2(inf.ci_lower)}; ${f2(inf.ci_upper)}]`, `${f2(Mi.seleccion.dp)} [${f2(Mi.seleccion.ic[0])}; ${f2(Mi.seleccion.ic[1])}]`],
      ["Prueba de signos (unilateral) · umbral Bonferroni", `${inf.sign_test_p.toExponential(2).replace(".", ",")} < 0,0025 (20 capas)`, "2,21e‑4 < 0,003125 (16 capas)"],
      ["Interacción en inferencia · IC 95 %", `${f2(sel.inference_interaction.interaction)} [${f2(sel.inference_interaction.ci_lower)}; ${f2(sel.inference_interaction.ci_upper)}]`, "8,37 [6,84; 9,79]"],
      ["Capa elegida en 200 particiones", "76 en el 100 %", "30 en el 100 %"]].map((r, i) => i === 0 ? r : r.map((c, j) => (j === 0 ? { text: c, options: { color: C.ink } } : c)));
    s.addTable(filas, Object.assign({}, TBL, { x: MX, y: 1.9, w: W - 2 * MX, colW: [5.3, 3.4, 3.43], fontSize: 12.5 }));
    txt(s, "Los 42/42 y 39/42 del cross‑fitting global no son el denominador de la prueba: la inferencia usa solo la mitad que no eligió la capa. Las 200 particiones describen estabilidad; no multiplican el tamaño de muestra.",
      { x: MX, y: 6.1, w: W - 2 * MX, h: 0.75, fontSize: 13, color: C.mute });
    fuente(s, "robust_results.json → layer_selection; metricas_avance.json → monitors_at_reference; registro de Mistral (LAYER SELECTION). Tesis, tabla 4.9.");
    nota(s, { titulo: "R3 · Circularidad y selección de capa", respaldo: true, t: 0,
      guion: "Si preguntan cuánto infla la circularidad: en Llama, el desplazamiento ingenuo es 16,28 y el corregido 15,72, un 3,5 %; en Mistral, un 2,4 %. La corrección reduce el efecto y conserva el signo. Si preguntan por la reutilización de pares al elegir la capa: la prueba se hace sobre 21 pares que no participaron en la elección; en Llama decrecen 21 de 21, con p de signos de 9,5 por 10 a la menos 7, por debajo del umbral de Bonferroni." });
  }

  // R4 · Fundamento matemático adicional
  {
    const s = nueva("Fundamento matemático adicional");
    const bloques = [["Salida del modelo: el primer token sale de la última posición", "salida", 1.95, 0.6, MX],
      ["Discriminante de Fisher y su límite regularizado", "fisher", 3.15, 0.6, MX], ["", "ridge", 3.8, 0.8, MX],
      ["Permutación de signos dentro del par", "permutacion", 5.2, 1.0, MX],
      ["Prueba de signos exacta", "signos", 5.35, 0.8, 8.4]];
    bloques.forEach(([t, id, y, h, x]) => {
      if (t) rotulo(s, t, x, y - 0.32, x === MX ? 7.4 : 4.3);
      eq(s, id, x, y, x === MX ? 7.4 : 4.3, h, 18);
    });
    rotulo(s, "Varianza de la validación cruzada repetida", 8.4, 1.65, 4.3);
    eq(s, "nadeau", 8.4, 1.95, 4.3, 0.85, 18);
    txt(s, "J estimaciones que comparten datos de entrenamiento: la corrección de Nadeau y Bengio (2003) evita intervalos demasiado estrechos.",
      { x: 8.4, y: 2.85, w: 4.33, h: 1.0, fontSize: 12.5, color: C.ink2 });
    txt(s, "Con n ≪ d, la covarianza muestral tiene rango a lo sumo 2n − 2 y no se invierte; al regularizar y llevar λ → ∞, el discriminante normalizado converge a la diferencia de medias.",
      { x: 8.4, y: 3.95, w: 4.33, h: 1.0, fontSize: 12, color: C.ink2 });
    txt(s, "Usa solo el sentido del cambio: es insensible a reescalados positivos, pero no a cambiar el eje sobre el que se proyecta.",
      { x: 8.4, y: 6.18, w: 4.33, h: 0.7, fontSize: 12, color: C.ink2 });
    fuente(s, "Tesis, ecuaciones (3.4), (3.5), (3.17) y sección 3.4. Con 10.000 permutaciones, un p guardado como 0 se informa como resolución ≈ 10⁻⁴.");
    nota(s, { titulo: "R4 · Fundamento matemático adicional", respaldo: true, t: 0,
      guion: "Para preguntas de fundamento: la última posición determina la distribución del siguiente token; Fisher no es estimable con 84 puntos en 8.192 dimensiones y su versión regularizada converge a la diferencia de medias; la permutación de signos supone intercambiabilidad del cambio de formato dentro de pares independientes; la prueba de signos depende del eje, y por eso la dirección se estima fuera de muestra." });
  }

  // R5 · PCA y sonda logística
  {
    const s = nueva("PCA y sonda logística: validación geométrica");
    const i = (c) => L.capas.indexOf(c);
    const filas = [["Modelo", "Capa", "|cos(PC1, r̂)|", "Varianza de PC1"].map(cab),
      ["Llama V3", "28", f2(L.pca.cos[i(28)], 4), pct(L.pca.var[i(28)], 2)],
      ["Llama V3", "56", f2(L.pca.cos[i(56)], 4), pct(L.pca.var[i(56)], 2)],
      ["Llama V3", "76", f2(L.pca.cos[i(76)], 4), pct(L.pca.var[i(76)], 2)],
      ["Mistral V3", "18", "0,968", "30,3 %"], ["Mistral V3", "30", "0,952", "28,7 %"]];
    s.addTable(filas, Object.assign({}, TBL, { x: MX, y: 1.9, w: 6.2, colW: [1.7, 0.9, 1.8, 1.8], fontSize: 13 }));
    txt(s, [R("Procedimiento. ", { fontFace: F.demi, color: C.ink }), R("Se concatenan las activaciones conversacionales dañinas y benignas, se resta la media global —no la de cada clase, que eliminaría la diferencia de centroides— y se ajusta PCA. El signo de PC1 se orienta para que el coseno sea positivo.")],
      { x: MX, y: 4.35, w: 6.2, h: 1.4, fontSize: 13, color: C.ink2 });
    txt(s, "En capas tempranas el contraste no coincide con PC1 (capa 0: 0,059). Su ausencia no invalida un contraste discriminativo; su presencia no lo convierte en causa.",
      { x: MX, y: 5.8, w: 6.2, h: 0.9, fontSize: 13, color: C.mute });
    const X0 = 7.3, XW = W - MX - X0, so = L.sonda;
    rotulo(s, "Sonda logística conversacional · capa 76", X0, 1.9, XW);
    const datos = [[f2(so.auroc_cv, 4), "AUROC con validación cruzada (10 × 5 folds por pares)"], [f2(so.sd_folds, 4), "desviación entre folds: no es un intervalo de confianza"],
      [`[${f2(so.ic[0], 4)}; 1]`, "IC 95 % corregido por Nadeau y Bengio"], [f2(so.permutado, 4), "control con etiquetas permutadas"], ["1", "AUROC en muestra completa: no es evidencia de generalización"]];
    datos.forEach(([a, b], k) => {
      txt(s, a, { x: X0, y: 2.25 + k * 0.62, w: 1.75, h: 0.5, fontFace: F.med, fontSize: 18, color: C.ink, valign: "middle" });
      txt(s, b, { x: X0 + 1.85, y: 2.25 + k * 0.62, w: XW - 1.85, h: 0.5, fontSize: 12.5, color: C.ink2, valign: "middle" });
    });
    txt(s, "La sonda y la dirección de medias son estimadores distintos: la cercanía de sus curvas no prueba equivalencia.", { x: X0, y: 5.5, w: XW, h: 0.8, fontSize: 13, color: C.ink2, italic: true });
    fuente(s, "results.json → pca_by_layer, linear_aurocs*, linear_auroc_ci_*; registro de Mistral (PCA validation). Tesis, tabla 4.7 y figura 4.4.");
    nota(s, { titulo: "R5 · PCA y sonda logística", respaldo: true, t: 0,
      guion: "Si preguntan por la validación geométrica: en capas profundas PC1 coincide con la diferencia de medias con coseno de 0,99, aunque explica solo entre 20 y 37 % de la varianza. Es una descripción dentro de muestra. Si preguntan por la sonda logística: su AUROC de entrenamiento es 1 por la alta dimensión; la cifra que cuenta es la de validación cruzada, 0,968, frente a 0,515 con etiquetas barajadas." });
  }

  // R6 · Proyección y conducta
  {
    const s = nueva("¿La proyección explica la conducta?");
    const pv = L.conducta.projection_vs_behavior;
    const filas = [["", "Llama V3 · capa 56", "Mistral V3 · capa 30"].map(cab),
      ["AUROC de la proyección para predecir rechazo · chat", `${f2(pv.chat_harmful.auroc_projection_predicts_refusal, 3)} (38 rechazan, 4 cumplen)`, "0,835 (19 y 23)"],
      ["AUROC de la proyección para predecir rechazo · agente", `${f2(pv.agent_harmful.auroc_projection_predicts_refusal, 3)} (23 y 19)`, "0,962 (5 y 37)"],
      ["Caída media de proyección · pares que dejan de rechazar", `${f2(pv.projection_drop_flipped_vs_held.mean_drop_flipped)} (n = 16)`, "10,91 (n = 16)"],
      ["Caída media de proyección · pares que siguen rechazando", `${f2(pv.projection_drop_flipped_vs_held.mean_drop_held)} (n = 22)`, "6,00 (n = 3)"],
      ["Mann–Whitney, caída entre ambos grupos", `p = ${f2(pv.projection_drop_flipped_vs_held.mannwhitney_p, 2)}`, "p = 0,0041"]].map((r, i) => i === 0 ? r : r.map((c, j) => (j === 0 ? { text: c, options: { color: C.ink } } : c)));
    s.addTable(filas, Object.assign({}, TBL, { x: MX, y: 1.9, w: W - 2 * MX, colW: [5.6, 3.4, 3.13], fontSize: 12.5 }));
    txt(s, [R("Lectura. ", { fontFace: F.demi, color: C.ink }), R("Dentro de cada formato, la proyección ordena bien qué respuestas se etiquetan como rechazo. Pero en Llama la magnitud de la caída no distingue a los pares que cambian de conducta de los que no: la geometría por sí sola no explica la decisión. Por eso proyección, discriminación y rechazo se tratan como objetos distintos.")],
      { x: MX, y: 4.6, w: W - 2 * MX, h: 1.3, fontSize: 13.5, color: C.ink2 });
    txt(s, "Proyecciones registradas durante la generación sin intervención, en la capa del ensayo conductual y con la dirección de muestra completa. Etiquetas heurísticas, sin auditoría humana.",
      { x: MX, y: 6.0, w: W - 2 * MX, h: 0.7, fontSize: 12.5, color: C.mute });
    fuente(s, "metricas_avance.json → llama.behavior.projection_vs_behavior · registro de Mistral (Projection vs behaviour). src/causal_test.py → run_behavioral_baseline.");
    nota(s, { titulo: "R6 · ¿La proyección explica la conducta?", respaldo: true, t: 0,
      guion: "Si preguntan si la caída de proyección causa el cambio de conducta: dentro de cada formato la proyección predice bien el rechazo etiquetado, con AUROC de 0,93 a 0,95 en Llama. Pero los pares que dejan de rechazar no pierden más proyección que los que siguen rechazando (p = 0,71). En Mistral sí la pierden (p = 0,004), con solo 3 pares que siguen rechazando. Conclusión: la proyección es informativa, pero no explica sola la decisión; por eso no afirmo causalidad desde la geometría." });
  }

  // R7 · Qwen y control de versiones
  {
    const s = nueva("Evidencia histórica de Qwen y control de versiones");
    const q = D.qwen;
    const filas = [["Ejecución", "Capa", "AUROC chat", "AUROC agente", "Lectura"].map(cab),
      ["Qwen3‑8B (histórico)", `${q["8b"].capa}`, f2(q["8b"].auroc_chat, 4), f2(q["8b"].auroc_agent, 4), "sin línea base discriminativa en esa capa"],
      ["Qwen3‑32B (histórico)", `${q["32b"].capa}`, f2(q["32b"].auroc_chat, 4), f2(q["32b"].auroc_agent, 4), "pérdida pequeña; configuración incompleta"]];
    s.addTable(filas, Object.assign({}, TBL, { x: MX, y: 1.9, w: W - 2 * MX, colW: [2.9, 0.9, 1.6, 1.7, 5.03], fontSize: 13 }));
    txt(s, "Usan 42 pares, pero su configuración exportada no acredita todos los controles actuales. Se conservan como antecedentes: no sostienen una réplica uniforme entre familias ni una ley de escala.",
      { x: MX, y: 3.4, w: W - 2 * MX, h: 0.8, fontSize: 13.5, color: C.ink2 });
    rotulo(s, "Control de versiones de los resultados de Llama", MX, 4.45, 8);
    const v = [["Versión anterior de la tesis", "capa 56, Δp ≈ 5,74; coseno agéntico atribuido 0,051"],
      ["Versión vigente (V3)", "capa 76, Δp fuera de muestra 15,72; coseno de medias 0,213"],
      ["Correcciones de código entre ambas", "copia GPU → CPU de activaciones ahora bloqueante; eliminación del token BOS duplicado"]];
    v.forEach(([a, b], k) => txt(s, [R(a + ": ", { fontFace: F.demi, color: C.ink }), R(b)], { x: MX, y: 4.8 + k * 0.45, w: W - 2 * MX, h: 0.4, fontSize: 13.5, color: C.ink2 }));
    txt(s, "Las diferencias entre versiones no se interpretan como fluctuación estadística sin verificar la implementación.", { x: MX, y: 6.2, w: W - 2 * MX, h: 0.4, fontSize: 13, color: C.mute });
    fuente(s, "results/qwen8b/results.json y results/qwen32b/results.json → auroc_heldout · tesis, sección 4.4 y anexo A.3 · historial del repositorio (PR #3 y #4).");
    nota(s, { titulo: "R7 · Qwen y control de versiones", respaldo: true, t: 0,
      guion: "Si preguntan por Qwen: las corridas son anteriores al pipeline corregido. En Qwen3‑8B, capa 32, la dirección no separa ni en chat (0,52), así que no hay línea base que degradar; en Qwen3‑32B la pérdida es pequeña. No las uso para sostener un patrón entre familias. Si preguntan por qué cambiaron las cifras respecto a la versión anterior: V3 corrige una copia asíncrona de activaciones y un token BOS duplicado; por eso no mezclo cifras de versiones distintas." });
  }

  // R8 · Factorial exploratorio
  {
    const s = nueva("Exploratorio: ¿sistema o herramientas?");
    const f = L.factorial76.harmful, fi = L.factorial76.interaction, tk = L.tokens;
    rrect(s, pres, MX, 1.88, W - 2 * MX, 0.62, C.agL);
    txt(s, [R("No reportado en el documento. ", { fontFace: F.demi, color: C.ink }), R("La corrida V3 incluye dos condiciones derivadas —solo sistema y solo herramientas—. La tesis declara este análisis como pendiente, con control de longitud.")],
      { x: MX + 0.2, y: 1.88, w: W - 2 * MX - 0.4, h: 0.62, fontSize: 13, color: C.ink, valign: "middle" });
    const filas = [["Cambio respecto de chat · dañinas · capa 76", "Δp", "IC 95 %", "Pares > 0", "Tokens añadidos"].map(cab),
      ["Total: sistema y herramientas (agente)", f2(f.total.delta), `[${f2(f.total.ci_lower)}; ${f2(f.total.ci_upper)}]`, `${f.total.n_positive}/42`, `+${Math.round(tk.agent_harmful - tk.chat_harmful)}`],
      ["Solo esquemas de herramientas", f2(f.tools.delta), `[${f2(f.tools.ci_lower)}; ${f2(f.tools.ci_upper)}]`, `${f.tools.n_positive}/42`, `+${Math.round(tk.toolsonly_harmful - tk.chat_harmful)}`],
      ["Solo mensaje de sistema agéntico", f2(f.system.delta), `[${f2(f.system.ci_lower)}; ${f2(f.system.ci_upper)}]`, `${f.system.n_positive}/42`, `+${Math.round(tk.sysonly_harmful - tk.chat_harmful)}`],
      ["Término no aditivo (total − sistema − herramientas)", f2(f.non_additive.delta), `[${f2(f.non_additive.ci_lower)}; ${f2(f.non_additive.ci_upper)}]`, `${f.non_additive.n_positive}/42`, "—"]];
    s.addTable(filas, Object.assign({}, TBL, { x: MX, y: 2.7, w: W - 2 * MX, colW: [5.2, 1.3, 2.3, 1.4, 1.93], fontSize: 12.5 }));
    txt(s, [R("Lo que sugiere: ", { fontFace: F.demi, color: C.ink }), R(`casi toda la caída acompaña a los esquemas de herramientas; el mensaje de sistema solo produce una caída menor y los efectos no se suman. En la interacción con el brazo benigno: total ${f2(fi.total.delta)}, herramientas ${f2(fi.tools.delta)}, sistema ${f2(fi.system.delta)}.`)],
      { x: MX, y: 5.0, w: W - 2 * MX, h: 0.8, fontSize: 13.5, color: C.ink2 });
    txt(s, [R("Por qué no es concluyente: ", { fontFace: F.demi, color: C.ink }), R("dirección de muestra completa (sin cross‑fitting) y sin control de longitud: los esquemas añaden unos 478 tokens frente a 28 del sistema. Efecto de las herramientas y de la longitud quedan confundidos.")],
      { x: MX, y: 5.85, w: W - 2 * MX, h: 0.8, fontSize: 13.5, color: C.ink2 });
    fuente(s, "results/llama33_final/results_llama33_v3/factorial_results.json → by_layer['76']; results.json → prompt_tokens. Intervalos: bootstrap percentil por pares.");
    nota(s, { titulo: "R8 · Exploratorio: ¿sistema o herramientas?", respaldo: true, t: 0,
      guion: "Usar solo si preguntan cuál componente del formato produce el efecto. La respuesta del documento es: el diseño principal no lo separa. Aclarar que la corrida V3 tiene, además, una variante factorial no analizada en la tesis: con la dirección de muestra completa, añadir solo los esquemas de herramientas reproduce casi toda la caída (16,20 de 16,28), y el mensaje de sistema solo produce 4,32. Pero los esquemas también alargan el estímulo en unos 478 tokens, así que herramientas y longitud están confundidas; por eso el documento lo deja como trabajo pendiente con control de longitud.",
      fuentes: "factorial_results.json → by_layer['76'].harmful y .interaction (total, tools, system, non_additive). Estas cifras no aparecen en el documento de tesis; están en los archivos de la corrida V3 (run_factorial_analysis en run_experiment.py). Decidir antes de la sustentación si se muestra esta diapositiva." });
  }

  // R9 · Ética
  {
    const s = nueva("Consideraciones éticas y uso responsable");
    const items = [["Material sin capacidad operativa", "Las instrucciones describen la acción solicitada, pero no contienen procedimientos, datos ni artefactos para llevarla a cabo. Nombres y cuentas son ficticios."],
      ["Herramientas simuladas", "Ninguna está conectada a un sistema: ninguna petición puede producir un efecto real."],
      ["Diagnóstico, no desalineación", "No se reentrena el modelo para suprimir su rechazo; las intervenciones son transitorias y solo se etiquetan respuestas."],
      ["Doble uso", "Saber dónde se degrada un mecanismo de seguridad es información sensible; el fallo ya está documentado en el plano conductual y una defensa basada en representaciones requiere este diagnóstico."],
      ["Licencias y sujetos", "Modelos usados conforme a sus licencias, que contemplan la investigación en seguridad. Sin sujetos humanos ni datos personales."]];
    txt(s, items.flatMap(([a, b], i) => [R(a, { fontFace: F.demi, color: C.ink, breakLine: true }), R(b, { breakLine: i < items.length - 1, paraSpaceAfter: 12 })]),
      { x: MX, y: 1.95, w: 8.6, h: 4.8, fontSize: 14, color: C.ink2 });
    txt(s, "El corpus completo y las generaciones se conservan en el repositorio de investigación y no se reproducen como catálogo de tareas.", { x: 9.6, y: 1.95, w: W - MX - 9.6, h: 2.0, fontFace: F.head, italic: true, fontSize: 15, color: C.ink });
    fuente(s, "Tesis, sección 3.5 y anexo A.1.");
    nota(s, { titulo: "R9 · Consideraciones éticas", respaldo: true, t: 0, guion: "Si preguntan por riesgos del corpus o de publicar el diagnóstico: el material no da capacidad operativa, las herramientas son simuladas, el modelo no se reentrena y el balance favorece la divulgación porque el fallo ya está documentado y la defensa requiere este diagnóstico." });
  }

  // R10 · Notación
  {
    const s = nueva("Notación");
    const sm = (...p) => ({ text: mx({ fontSize: 14, color: C.ink }, ...p) });
    const filas = [["Símbolo", "Significado"].map(cab),
      [sm(m.b("x"), m.supi("(l)"), m.u(" ∈ ℝ"), m.supi("d")), "flujo residual de la última posición del estímulo en la capa l"],
      [sm(m.hat("r"), m.supi("(l)")), "dirección de rechazo: diferencia normalizada de centroides dañino y benigno en chat"],
      [sm(m.i("p"), m.u(" = "), m.b("x"), m.u(" · "), m.hat("r")), "proyección: alineación geométrica con la dirección (no es un rechazo observado)"],
      [sm(m.u("Δ"), m.i("p"), m.sub("h"), m.u(",  Δ"), m.i("p"), m.sub("b")), "desplazamiento chat − agente de la proyección, por brazo, promediado dentro del par"],
      [sm(m.u("Δ"), m.sup("2"), m.i("p"), m.u(",  "), m.i("φ")), "interacción contenido × formato y fracción del efecto que reproduce el brazo benigno"],
      [sm(m.i("d"), m.u(",  "), m.i("κ")), "separación estandarizada S/s y contracción de escala s_ag / s_chat"],
      [sm(m.u("AUROC")), "probabilidad de que una dañina proyecte más que una benigna (0,5 = azar)"],
      [sm(m.hat("w"), m.sub("DM", false), m.u(",  "), m.hat("w"), m.sub("LR", false)), "dirección agéntica por diferencia de medias y por regresión logística"],
      [sm(m.i("ρ"), m.u(",  cos"), m.sub("norm", false)), "fiabilidad entre mitades y coseno normalizado por el techo de fiabilidad"],
      [sm(m.u("OOF")), "fuera de fold: evaluado en pares que no estimaron la dirección"]];
    s.addTable(filas, Object.assign({}, TBL, { x: MX, y: 1.9, w: W - 2 * MX, colW: [2.4, 9.73], fontSize: 13 }));
    fuente(s, "Tesis, glosario de notación y sección 3.3.1.");
    nota(s, { titulo: "R10 · Notación", respaldo: true, t: 0, guion: "Referencia rápida de símbolos para responder preguntas sobre definiciones." });
  }

  // R11 · Referencias
  {
    const s = nueva("Referencias");
    const refs = [
      "Andriushchenko, M., Souly, A., Dziemian, M., et al. (2024). AgentHarm: A benchmark for measuring harmfulness of LLM agents. arXiv:2410.09024; ICLR 2025.",
      "Anthropic. (2025). Disrupting the first reported AI‑orchestrated cyber espionage campaign [informe completo, noviembre]. *",
      "Arditi, A., Obeso, O., Syed, A., Paleka, D., Panickssery, N., Gurnee, W. y Nanda, N. (2024). Refusal in language models is mediated by a single direction. NeurIPS 2024.",
      "Bengio, Y. (pres.) et al. (2026). International AI Safety Report 2026 [resumen ejecutivo, 3 de febrero]. *",
      "Gartner. (2025, 26 de agosto). Gartner predicts 40% of enterprise apps will feature task‑specific AI agents by 2026, up from less than 5% in 2025 [comunicado de prensa]. *",
      "Lermen, S., Dziemian, M. y Pimpale, G. (2024). Applying refusal‑vector ablation to Llama 3.1 70B agents. arXiv:2410.10871.",
      "Nadeau, C. y Bengio, Y. (2003). Inference for the generalization error. Machine Learning, 52(3), 239–281.",
      "Pinto, L. (2026). Present but rescaled: Chat‑to‑agent transfer of additive activation steering. arXiv:2607.09156.",
      "Zhong, V. y Li, Q. (2026). Refusal lives downstream of persona in chat models. ICML 2026 Mechanistic Interpretability Workshop. arXiv:2606.26161.",
    ];
    txt(s, refs.map((t, i) => R(t, { breakLine: i < refs.length - 1, paraSpaceAfter: 8, indentLevel: 0 })), { x: MX, y: 1.9, w: W - 2 * MX, h: 4.6, fontSize: 12.5, color: C.ink2 });
    txt(s, "* Contextualización externa añadida para la presentación; no forma parte de la bibliografía del documento. El resto de referencias de la tesis está en references.bib.",
      { x: MX, y: 6.45, w: W - 2 * MX, h: 0.45, fontSize: 11.5, color: C.mute });
    nota(s, { titulo: "R11 · Referencias", respaldo: true, t: 0, guion: "Lista de las fuentes citadas en las diapositivas. Las marcadas con asterisco se verificaron para la presentación en sus fuentes primarias (octubre de 2026)." });
  }

  // R12 · Trazabilidad
  {
    const s = nueva("Trazabilidad de las cifras");
    const filas = [["Diap.", "Cifras", "Archivo → clave"].map(cab),
      ["1, 15, 16, 23", "proyecciones fuera de muestra, capa 76; S, d, κ", "results_llama33_v3/results.json → heldout.projections"],
      ["13", "rechazo 38/42 → 23/42; McNemar; Wilson", "evidencia/metricas_avance.json → llama.behavior"],
      ["14", "AUROC por capa; PCA; sonda logística", "results.json → heldout.per_layer, pca_by_layer, linear_aurocs*"],
      ["15", "desplazamientos, interacción e intervalos", "results.json → heldout.paired_harmful, paired_benign; metricas → heldout_format"],
      ["16", "AUROC y su diferencia, con IC", "metricas_avance.json → llama.heldout_detection"],
      ["17", "AUROC por estrato", "metricas → llama.strata_oof; registro de Mistral"],
      ["18 · 19", "cosenos y fiabilidad; monitores", "metricas → rotation_at_reference, monitors_at_reference"],
      ["20", "respuesta a la dosis, capa 56", "metricas → intervention_layer56"],
      ["Mistral", "todas sus cifras (redondeadas)", "results/results_mistralV3/results_mistral7b_v3.log"],
      ["6 · 7", "corpus 96/384; 42 ejecutados; tokens", "data/dataset_v2_report.json; data/dataset_full.jsonl; results.json → prompt_tokens"]];
    s.addTable(filas, Object.assign({}, TBL, { x: MX, y: 1.9, w: W - 2 * MX, colW: [1.35, 4.4, 6.38], fontSize: 12 }));
    txt(s, "SHA‑256 de las fuentes: evidencia/manifiesto_fuentes.json (p. ej., results.json 046cd1db…, registro de Mistral 5db8d72e…). Las cifras de la presentación se extraen con docs/sustentacion/fuente/datos.py, que verifica con aserciones su coincidencia con las tablas de la tesis.",
      { x: MX, y: 6.05, w: W - 2 * MX, h: 0.8, fontSize: 12, color: C.mute });
    nota(s, { titulo: "R12 · Trazabilidad de las cifras", respaldo: true, t: 0, guion: "Si preguntan de dónde sale una cifra: cada diapositiva tiene la fuente al pie y en las notas; esta tabla resume el archivo y la clave. Los resultados de Mistral provienen de un registro con valores impresos y redondeados; no se reconstruyó un JSON ausente." });
  }
};
