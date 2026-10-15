// III. Resultados: conducta, línea base, desplazamiento, margen, explicitud, reorientación,
// monitor e intervención. Todas las cifras salen de datos_presentacion.json.
const B = require("./base");
const { W, MX, C, F, R, txt, m, mx, N, eq, rect, rrect, dot, seg, flecha, nav, fuente, rotulo } = B;
const { nota } = require("./guion");

const f2 = (v, d = 2) => v.toLocaleString("es-CO", { minimumFractionDigits: d, maximumFractionDigits: d }).replace("-", "−");
const pct = (v, d = 1) => `${f2(100 * v, d)} %`;

function ejesGrafico(o) {
  return Object.assign({
    catAxisLabelFontFace: F.body, catAxisLabelFontSize: 11, catAxisLabelColor: C.mute,
    valAxisLabelFontFace: F.body, valAxisLabelFontSize: 11, valAxisLabelColor: C.mute,
    valGridLine: { color: "E3E6EA", size: 0.75 }, catGridLine: { style: "none" },
    catAxisLineShow: true, catAxisLineColor: "AEB4BC", valAxisLineShow: false,
    showLegend: false, lineDataSymbol: "none", lineSize: 2.25,
    catAxisTitleFontFace: F.body, catAxisTitleFontSize: 11, catAxisTitleColor: C.mute,
    valAxisTitleFontFace: F.body, valAxisTitleFontSize: 11, valAxisTitleColor: C.mute,
  }, o);
}

// Gráfico de líneas nativo con área de trazado fija; las etiquetas del eje Y se escriben
// aparte con coma decimal para que no dependan de la configuración regional del equipo.
function grafico(s, pres, series, box, o) {
  const lay = { x: 0.085, y: 0.05, w: 0.895, h: 0.76 };
  s.addChart(pres.charts.LINE, series, ejesGrafico(Object.assign({ x: box.x, y: box.y, w: box.w, h: box.h, layout: lay, valAxisLabelColor: "FFFFFF" }, o)));
  const lo = o.valAxisMinVal, hi = o.valAxisMaxVal, u = o.valAxisMajorUnit;
  for (let k = 0; lo + k * u <= hi + 1e-9; k++) {
    const v = lo + k * u;
    const y = box.y + box.h * (lay.y + lay.h * (1 - (v - lo) / (hi - lo)));
    txt(s, o.fmt ? o.fmt(v) : f2(v, o.dec), { x: box.x - 0.1, y: y - 0.11, w: box.w * lay.x, h: 0.22, align: "right", fontSize: 11, color: C.mute });
  }
}

module.exports = function (pres, D, ctx) {
  const L = D.llama, Mi = D.mistral;
  const sec = "III. Resultados";
  const capas = L.capas.map(String);

  // ------------------------------------------------------------ 13. Conducta
  {
    const s = pres.addSlide({ masterName: "CONTENIDO", sectionTitle: sec });
    nav(s, 2);
    s.addText("Afuera: el mismo modelo rechaza menos cuando actúa", { placeholder: "title" });
    const g = L.conducta.gap_harmful;
    const tipos = [[g.refuse_both, C.ink, "rechaza en ambos formatos"], [g.refuse_chat_comply_agent, C.harm, "rechaza en chat, cumple en agente"],
      [g.comply_chat_refuse_agent, C.ben, "cumple en chat, rechaza en agente"], [g.comply_both, "C9CED6", "cumple en ambos"]];
    const celdas = tipos.flatMap(([n, col]) => Array(n).fill(col));
    const t = 0.52, gp = 0.08;
    celdas.forEach((col, k) => rrect(s, pres, MX + (k % 7) * (t + gp), 1.95 + Math.floor(k / 7) * (t + gp), t, t, col, { rectRadius: 0.05 }));
    tipos.forEach(([n, col, lab], i) => {
      rect(s, pres, MX, 5.68 + i * 0.3, 0.2, 0.2, col);
      txt(s, [R(`${n}  `, { fontFace: F.demi, color: C.ink }), R(lab)], { x: MX + 0.32, y: 5.63 + i * 0.3, w: 4.2, h: 0.3, fontSize: 12.5, color: C.ink2, valign: "middle" });
    });
    const X0 = 5.35;
    rotulo(s, "Llama‑3.3‑70B · instrucciones dañinas rechazadas", X0, 1.95, 7);
    const rf = L.conducta.refusal;
    txt(s, `${rf.chat_harmful.k}/42`, { x: X0, y: 2.25, w: 2.6, h: 0.95, fontFace: F.med, fontSize: 56, color: C.ink });
    flecha(s, pres, X0 + 2.75, 2.75, X0 + 3.55, 2.75, C.mute, 2);
    txt(s, `${rf.agent_harmful.k}/42`, { x: X0 + 3.75, y: 2.25, w: 2.6, h: 0.95, fontFace: F.med, fontSize: 56, color: C.harm });
    txt(s, `chat · ${pct(rf.chat_harmful.rate)}`, { x: X0, y: 3.2, w: 2.6, h: 0.3, fontSize: 13, color: C.ink2 });
    txt(s, `agente · ${pct(rf.agent_harmful.rate)}`, { x: X0 + 3.75, y: 3.2, w: 2.6, h: 0.3, fontSize: 13, color: C.harm });
    txt(s, [R("McNemar exacto: ", { fontFace: F.demi, color: C.ink }), R(`p = ${f2(g.mcnemar_p, 5)} · ${g.refuse_chat_comply_agent} frente a ${g.comply_chat_refuse_agent} pares discordantes`)],
      { x: X0, y: 3.62, w: 7.3, h: 0.32, fontSize: 14, color: C.ink2 });
    txt(s, [R("Benignas rechazadas: ", { fontFace: F.demi, color: C.ink }), R(`${rf.chat_benign.k}/42 en chat y ${rf.agent_benign.k}/42 en agente.`)],
      { x: X0, y: 4.0, w: 7.3, h: 0.32, fontSize: 14, color: C.ink2 });
    txt(s, [R("Mistral‑7B V3 (registro): ", { fontFace: F.demi, color: C.ink }), R(`${Mi.conducta.chat_h}/42 → ${Mi.conducta.ag_h}/42 · ${Mi.conducta.deja} frente a ${Mi.conducta.inverso} · p ≈ 0,0013`)],
      { x: X0, y: 4.38, w: 7.3, h: 0.32, fontSize: 14, color: C.ink2 });
    rrect(s, pres, X0, 5.0, W - MX - X0, 1.62, C.mist);
    txt(s, [R("Cómo leerlo. ", { fontFace: F.demi, color: C.ink }),
      R("Cada respuesta (64 tokens) se etiqueta con una heurística textual reproducible: rechazo, cumplimiento o degeneración. La etiqueta no observa herramientas ejecutadas y la auditoría humana está pendiente: es evidencia conductual exploratoria.")],
    { x: X0 + 0.25, y: 5.12, w: W - MX - X0 - 0.5, h: 1.4, fontSize: 13.5, color: C.ink2 });
    fuente(s, `Cada cuadro es un par (42). IC 95 % de Wilson: chat ${pct(rf.chat_harmful.ci_lower)}–${pct(rf.chat_harmful.ci_upper)}; agente ${pct(rf.agent_harmful.ci_lower)}–${pct(rf.agent_harmful.ci_upper)}. Fuente: metricas_avance.json → llama.behavior; tesis, sección 4.3.`);
    nota(s, {
      titulo: "Afuera: el mismo modelo rechaza menos cuando actúa", t: 45,
      guion: `Empiezo por afuera, por la conducta. En Llama‑3.3‑70B, de las 42 instrucciones dañinas, 38 se rechazan en chat; en formato agéntico, solo 23. Cada cuadro es un par: 16 pares rechazan en chat y cumplen en agente, y solo uno hace lo contrario. La prueba exacta de McNemar da p de 0,0003. En Mistral el patrón se repite: de 19 a 5 rechazos.
Dos advertencias: estas etiquetas salen de una heurística sobre el texto generado, que no observa herramientas ejecutadas, y la auditoría humana está pendiente. Por eso lo trato como evidencia conductual exploratoria.
El síntoma, entonces, se reproduce en nuestro sistema. La pregunta es qué cambió adentro.`,
      transicion: "Para responderla, primero hay que mostrar que en chat el detector funciona.",
      fuentes: `metricas_avance.json → llama.behavior.refusal (chat_harmful 38/42, agent_harmful 23/42, chat_benign 5/42, agent_benign 1/42; IC de Wilson guardados) y gap_harmful (16 / 1 / 22 / 3; mcnemar_p = ${g.mcnemar_p}). Benignas: 5 frente a 1, McNemar p = 0,219. Mistral: registro, BEHAVIOURAL BASELINE (19/42 → 5/42; 16 frente a 2; p = 0,0013). Tesis, sección 4.3.`,
    });
  }

  // ------------------------------------------------------------ 14. Línea base
  {
    const s = pres.addSlide({ masterName: "CONTENIDO", sectionTitle: sec });
    nav(s, 2);
    s.addText("En chat, la dirección conversacional sí separa las clases", { placeholder: "title" });
    rotulo(s, "AUROC fuera de muestra por capa · Llama V3, 42 pares", MX, 1.88, 6);
    txt(s, [R("━ ", { color: C.ink }), R("chat    "), R("━ ", { color: C.ag }), R("agente (misma dirección)")], { x: MX, y: 2.16, w: 6, h: 0.28, fontSize: 12, color: C.ink2 });
    grafico(s, pres, [
      { name: "Chat", labels: capas, values: L.por_capa.auroc_chat },
      { name: "Agente", labels: capas, values: L.por_capa.auroc_agent },
    ], { x: MX, y: 2.45, w: 6.0, h: 3.15 }, { chartColors: [C.ink, C.ag], valAxisMinVal: 0.5, valAxisMaxVal: 1.0,
      valAxisMajorUnit: 0.1, dec: 1, catAxisLabelFrequency: 2, showCatAxisTitle: true, catAxisTitle: "capa" });
    const X0 = 6.95, XW = W - MX - X0;
    rotulo(s, "PCA · alineación de PC1 con la dirección", X0, 1.88, XW);
    txt(s, [R("━ ", { color: C.ink2 }), R("|cos(PC1, "), ...N.rhat(), R(")|, chat, centrado global")], { x: X0, y: 2.16, w: XW, h: 0.28, fontSize: 12, color: C.ink2 });
    grafico(s, pres, [{ name: "|cos(PC1, r)|", labels: capas, values: L.pca.cos }],
      { x: X0, y: 2.45, w: XW, h: 3.15 }, { chartColors: [C.ink2], valAxisMinVal: 0, valAxisMaxVal: 1.0,
        valAxisMajorUnit: 0.25, dec: 2, catAxisLabelFrequency: 2, showCatAxisTitle: true, catAxisTitle: "capa" });
    const det = L.deteccion, i76 = L.capas.indexOf(76);
    const bloques = [
      [f2(det.auroc_chat_heldout, 3), "AUROC chat, capa 76", `IC [${f2(det.auroc_chat_ci[0], 3)}; ${f2(det.auroc_chat_ci[1], 3)}] · en agente ${f2(det.auroc_agent_heldout, 3)}`],
      [f2(L.pca.cos[i76], 3), "|cos(PC1, r̂)|, capa 76", `PC1 explica ${pct(L.pca.var[i76])} · Mistral, capa 30: 0,952 (28,7 %)`],
      [f2(L.sonda.auroc_cv, 3), "sonda logística, validación cruzada", `IC Nadeau–Bengio [${f2(L.sonda.ic[0], 3)}; 1] · barajadas ${f2(L.sonda.permutado, 3)} · en muestra 1`]];
    bloques.forEach(([n, a, b], i) => {
      const x = MX + i * 4.12;
      txt(s, n, { x, y: 5.75, w: 1.45, h: 0.55, fontFace: F.med, fontSize: 28, color: C.ink });
      txt(s, [R(a, { fontFace: F.demi, color: C.ink, breakLine: true }), R(b)], { x: x + 1.5, y: 5.75, w: 2.5, h: 0.95, fontSize: 11.5, color: C.ink2 });
    });
    fuente(s, "Cada capa es una medición correlacionada del mismo corpus. PCA es descriptivo y dentro de muestra. results.json → heldout.per_layer, pca_by_layer, linear_aurocs*; tesis, tablas 4.2 y 4.7.");
    nota(s, {
      titulo: "En chat, la dirección conversacional sí separa las clases", t: 45,
      guion: `Primero, la línea base. En chat, la dirección conversacional, evaluada fuera de muestra, separa dañinas de benignas con un AUROC cercano a 0,98 desde la capa 16 hasta el final; en la capa de referencia, 0,978.
La curva ámbar es la misma dirección aplicada al formato agéntico: queda por debajo en todas las capas profundas.
A la derecha, PCA muestra que en capas profundas el contraste dañino–benigno es el eje dominante de variación: el primer componente coincide con la dirección, con coseno de 0,99. Y una sonda logística con validación cruzada da 0,968, frente a 0,515 con etiquetas barajadas.
Hay, por tanto, un detector bien definido en chat.`,
      transicion: "¿Qué le pasa a ese detector cuando el modelo actúa?",
      fuentes: `Curvas: results.json → heldout.per_layer[capa].auroc_chat / auroc_agent (dirección reajustada por fold, 5 folds por pares). Capa 76: AUROC chat ${f2(det.auroc_chat_heldout, 4)}, IC [${f2(det.auroc_chat_ci[0], 4)}; ${f2(det.auroc_chat_ci[1], 4)}] (remuestreo condicional a los puntajes cross‑fit). PCA: pca_by_layer (|cos| = ${f2(L.pca.cos[i76], 4)}, varianza PC1 = ${pct(L.pca.var[i76], 2)}); Mistral desde el registro (capa 30: 0,952; 28,7 %). Sonda logística: linear_aurocs['76'] = ${f2(L.sonda.auroc_cv, 4)}, sd entre folds ${f2(L.sonda.sd_folds, 4)}, IC Nadeau–Bengio [${f2(L.sonda.ic[0], 4)}; 1]; control permutado ${f2(L.sonda.permutado, 4)}. La desviación entre folds no es un intervalo de confianza.`,
    });
  }

  // ------------------------------------------------------------ 15. Desplazamiento
  {
    const s = pres.addSlide({ masterName: "CONTENIDO", sectionTitle: sec });
    nav(s, 2);
    s.addText("La proyección dañina cae en los 42 pares; la benigna, apenas", { placeholder: "title" });
    const p = L.proy76, lo = -10, hi = 28, yT = 2.3, yB = 6.3;
    const ey = (v) => yB - ((v - lo) / (hi - lo)) * (yB - yT);
    const panel = (x0, w, a, b, col, tit) => {
      rotulo(s, tit, x0, 1.88, w, col);
      const xa = x0 + 0.45, xb = x0 + w - 0.35;
      seg(s, pres, xa, ey(0), xb, ey(0), "AEB4BC", 0.75, { line: { dashType: "dash" } });
      a.forEach((va, i) => seg(s, pres, xa, ey(va), xb, ey(b[i]), col, 0.75, { line: { transparency: 45 } }));
      a.forEach((va, i) => { dot(s, pres, xa, ey(va), 0.07, col); dot(s, pres, xb, ey(b[i]), 0.07, col); });
      const ma = a.reduce((u, v) => u + v, 0) / a.length, mb = b.reduce((u, v) => u + v, 0) / b.length;
      seg(s, pres, xa, ey(ma), xb, ey(mb), C.ink, 3);
      dot(s, pres, xa, ey(ma), 0.16, C.ink); dot(s, pres, xb, ey(mb), 0.16, C.ink);
      txt(s, "chat", { x: xa - 0.4, y: yB + 0.1, w: 0.8, h: 0.26, align: "center", fontSize: 12, color: C.ink2 });
      txt(s, "agente", { x: xb - 0.45, y: yB + 0.1, w: 0.9, h: 0.26, align: "center", fontSize: 12, color: C.ag });
    };
    [-10, 0, 10, 20].forEach((v) => txt(s, v < 0 ? `−${-v}` : `${v}`, { x: MX - 0.05, y: ey(v) - 0.12, w: 0.4, h: 0.24, align: "right", fontSize: 11, color: C.mute }));
    panel(MX + 0.45, 2.9, p.chat_h, p.ag_h, C.harm, "Dañinas · 42 pares");
    panel(MX + 3.6, 2.9, p.chat_b, p.ag_b, C.ben, "Benignas · 42 pares");
    txt(s, "proyección fuera de muestra, capa 76 (unidades crudas)", { x: MX + 0.45, y: 6.68, w: 6.0, h: 0.26, fontSize: 11, color: C.mute });
    const X0 = 7.55, XW = W - MX - X0;
    const ph = L.pareado_h, pb = L.pareado_b, fo = L.formato;
    rotulo(s, "Llama‑3.3‑70B · capa 76 · fuera de muestra", X0, 1.88, XW);
    const filas = [
      [N.dph({ fontSize: 20 }), f2(ph.delta_p), [R(`IC [${f2(ph.ci_lower)}; ${f2(ph.ci_upper)}]`, { breakLine: true }), R(`${ph.n_decreasing}/42 pares decrecen · `), ...N.dz(), R(` = ${f2(ph.cohens_d_paired)}`)], C.harm],
      [N.dpb({ fontSize: 20 }), f2(pb.delta_p), [R(`IC [${f2(pb.ci_lower)}; ${f2(pb.ci_upper)}]`, { breakLine: true }), R("el intervalo incluye 0")], C.ben],
      [N.d2p({ fontSize: 20 }), f2(fo.interaction), [R(`IC [${f2(fo.interaction_ci[0])}; ${f2(fo.interaction_ci[1])}] · p ≈ 10⁻⁴`, { breakLine: true }), R(`${fo.n_pairs_interaction_positive}/42 positivos · `), ...N.phi(), R(` = ${pct(fo.common_shift_fraction)}`)], C.ink]];
    filas.forEach(([sim, val, det, col], i) => {
      const y = 2.22 + i * 0.95;
      txt(s, sim, { x: X0, y, w: 0.85, h: 0.5, valign: "middle" });
      txt(s, val, { x: X0 + 0.85, y: y - 0.04, w: 1.6, h: 0.58, fontFace: F.med, fontSize: 30, color: col, valign: "middle" });
      txt(s, det, { x: X0 + 2.5, y, w: XW - 2.5, h: 0.62, fontSize: 12, color: C.ink2, valign: "middle" });
    });
    rotulo(s, "Mistral‑7B · capa 30 · registro redondeado", X0, 5.15, XW);
    txt(s, [...N.dph(), R(` = ${f2(Mi.delta_h_oof)}   ·   `), ...N.dpb(), R(` = ${f2(Mi.delta_b_oof)}   ·   `), ...N.d2p(), R(` = ${f2(Mi.interaction_oof)} [${f2(Mi.interaction_ci[0])}; ${f2(Mi.interaction_ci[1])}]`)],
      { x: X0, y: 5.45, w: XW, h: 0.32, fontSize: 13.5, color: C.ink2 });
    txt(s, "Medida geométrica en unidades crudas: no se compara entre modelos ni equivale a rechazo.",
      { x: X0, y: 6.0, w: XW, h: 0.55, fontSize: 12.5, color: C.ink, italic: true });
    fuente(s, "IC 95 %: bootstrap percentil por pares (10.000). p ≈ 10⁻⁴ es la resolución de 10.000 permutaciones de signos. results.json → heldout.projections, paired_*; tesis, tabla 4.3.");
    nota(s, {
      titulo: "La proyección dañina cae en los 42 pares; la benigna, apenas", t: 60,
      guion: `Ahora el desplazamiento, en la capa 76. Cada línea es un par. A la izquierda, las dañinas: la proyección cae al pasar a agente en los 42 pares, sin excepción. El promedio, delta p sub h, es 15,7, con intervalo de 13,8 a 17,6.
A la derecha, las benignas: casi no se mueven, y si algo, suben un poco. Su cambio promedio es −1,5 y su intervalo incluye el cero.
La interacción, lo específico del daño, es 17,3, con intervalo de 14,9 a 19,5; es positiva en 41 de 42 pares y su p es del orden de 10 a la menos 4, que es la resolución de las permutaciones. En Mistral, en otras unidades, el patrón se repite: interacción de 8,4.
Esto descarta un corrimiento rígido. Pero todavía no dice si se perdió información o si solo cambió la escala.`,
      transicion: "Para eso están las otras dos unidades.",
      fuentes: `Proyecciones: results.json → heldout.projections (dirección reajustada en cada fold). Δp_h = ${f2(ph.delta_p, 4)}, IC percentil [${f2(ph.ci_lower, 4)}; ${f2(ph.ci_upper, 4)}], t = ${f2(ph.t_statistic, 2)}, d_z = ${f2(ph.cohens_d_paired, 4)}, 42/42 decrecen (sign test p = ${ph.sign_test_p.toExponential(2)}). Δp_b = ${f2(pb.delta_p, 4)}, IC [${f2(pb.ci_lower, 4)}; ${f2(pb.ci_upper, 4)}]. Interacción ${f2(fo.interaction, 4)}, IC [${f2(fo.interaction_ci[0], 4)}; ${f2(fo.interaction_ci[1], 4)}], p de permutación intra‑par con piso 10⁻⁴, 41/42 positivos, fracción común ${f2(fo.common_shift_fraction, 4)}. Mistral: registro (FORMAT EFFECT SPECIFICITY, held‑out). Tesis, tabla 4.3 y sección 4.1.3.`,
    });
  }

  // ------------------------------------------------------------ 16. Escala y margen
  {
    const s = pres.addSlide({ masterName: "CONTENIDO", sectionTitle: sec });
    nav(s, 2);
    s.addText("Parte de la caída es escala; otra parte es pérdida de margen", { placeholder: "title" });
    const p = L.proy76, sp = L.sep;
    const rnd = (() => { let z = 11; return () => ((z = (z * 1664525 + 1013904223) >>> 0) / 4294967296); })();
    const tira = (vals, esc, y, col) => {
      vals.forEach((v) => dot(s, pres, esc(v), y + (rnd() - 0.5) * 0.24, 0.085, col, { transparency: 30 }));
      const mu = vals.reduce((a, b) => a + b, 0) / vals.length;
      seg(s, pres, esc(mu), y - 0.18, esc(mu), y + 0.18, C.ink, 2.5);
    };
    const panel = (y0, tit, lo, hi, ticks, tr, nota_) => {
      rotulo(s, tit, MX, y0, 6.3);
      const x0 = MX + 0.95, x1 = MX + 6.3;
      const esc = (v) => x0 + ((v - lo) / (hi - lo)) * (x1 - x0);
      [["chat", y0 + 0.5, p.chat_h, p.chat_b, "chat"], ["agente", y0 + 1.25, p.ag_h, p.ag_b, "ag"]].forEach(([nom, y, h, b, k]) => {
        txt(s, nom, { x: MX, y: y - 0.02, w: 0.9, h: 0.3, fontFace: F.demi, fontSize: 12, color: nom === "agente" ? C.ag : C.ink, valign: "middle" });
        tira(h.map((v) => tr(v, k)), esc, y + 0.0, C.harm);
        tira(b.map((v) => tr(v, k)), esc, y + 0.32, C.ben);
      });
      const ya = y0 + 1.82;
      seg(s, pres, x0, ya, x1, ya, "AEB4BC", 0.75);
      ticks.forEach((v) => txt(s, f2(v, 0), { x: esc(v) - 0.25, y: ya + 0.03, w: 0.5, h: 0.22, align: "center", fontSize: 10.5, color: C.mute }));
      txt(s, nota_, { x: x0, y: y0, w: x1 - x0, h: 0.26, align: "right", fontSize: 11, color: C.mute });
    };
    panel(1.88, "Unidades crudas", -10, 28, [-10, 0, 10, 20], (v) => v, "proyección fuera de muestra, capa 76");
    const z = (v, k) => (k === "chat" ? (v - L.medias.chat_b) / sp.s_chat : (v - L.medias.ag_b) / sp.s_ag);
    panel(4.15, "Estandarizadas por formato", -2, 6, [-2, 0, 2, 4, 6], z, "(p − media benigna) / s agrupada del formato");
    // Tres unidades
    const X0 = 7.45, XW = W - MX - X0;
    rotulo(s, "Tres unidades · Llama V3, capa 76", X0, 1.88, XW);
    const barra = (y, etq, conserva, tc, ta, extra) => {
      txt(s, etq, { x: X0, y, w: XW, h: 0.28, fontFace: F.demi, fontSize: 13, color: C.ink });
      const bw = XW - 1.25;
      rect(s, pres, X0, y + 0.34, bw, 0.2, C.ink);
      rect(s, pres, X0, y + 0.6, bw * conserva, 0.2, C.ag);
      txt(s, tc, { x: X0 + bw + 0.08, y: y + 0.3, w: 1.15, h: 0.26, fontSize: 12, color: C.ink2 });
      txt(s, ta, { x: X0 + bw * conserva + 0.08, y: y + 0.56, w: 1.6, h: 0.26, fontSize: 12, color: C.ag });
      txt(s, extra, { x: X0, y: y + 0.84, w: XW, h: 0.28, fontSize: 12, color: C.ink2 });
    };
    barra(2.2, "Separación cruda S", sp.S_ag / sp.S_chat, f2(sp.S_chat), f2(sp.S_ag), `pérdida ${pct(sp.perdida_cruda)} · κ = ${f2(sp.kappa)}`);
    barra(3.38, "Separación estandarizada d", sp.d_ag / sp.d_chat, f2(sp.d_chat), f2(sp.d_ag), `pérdida ${pct(sp.perdida_d)}`);
    const det = L.deteccion;
    barra(4.56, "AUROC (escala 0,5–1)", (det.auroc_agent_heldout - 0.5) / (det.auroc_chat_heldout - 0.5), f2(det.auroc_chat_heldout, 3), f2(det.auroc_agent_heldout, 3),
      `diferencia ${f2(det.auroc_gap_heldout, 3)} · IC [${f2(det.auroc_gap_ci[0], 3)}; ${f2(det.auroc_gap_ci[1], 3)}]`);
    rrect(s, pres, X0, 5.8, XW, 0.92, C.mist);
    txt(s, [R("Mistral‑7B (registro): ", { fontFace: F.demi, color: C.ink }), R(`S −${pct(Mi.perdida_cruda)} · d ${f2(Mi.d_chat_oof)} → ${f2(Mi.d_agent_oof)} (−${pct(Mi.loss_standardised)}) · AUROC ${f2(Mi.auroc_chat_oof, 3)} → ${f2(Mi.auroc_agent_oof, 3)} · κ = ${f2(Mi.scale_contraction, 3)}`)],
      { x: X0 + 0.2, y: 5.86, w: XW - 0.4, h: 0.8, fontSize: 12.5, color: C.ink2, valign: "middle" });
    fuente(s, "Barras: chat (tinta) y agente (ámbar), en proporción al valor de chat. Sin prueba formal entre modelos. results.json → heldout.projections; metricas_avance.json; tesis, tablas 4.2 y 4.4.");
    nota(s, {
      titulo: "Parte de la caída es escala; otra parte es pérdida de margen", t: 60,
      guion: `Para eso están las otras dos unidades. Arriba, las mismas proyecciones en unidades crudas: en agente, todo se comprime cerca de cero. La separación cruda cae 83 %. Pero la dispersión también cae: kappa es 0,31.
Abajo, al estandarizar por la dispersión de cada formato, la brecha entre clases se recupera en parte: d pasa de 3,2 a 1,8, una pérdida de 44 %. Y el AUROC pasa de 0,978 a 0,920; la diferencia, 0,058, tiene un intervalo que excluye el cero.
En Mistral la pérdida es mayor: d cae 64 % y el AUROC llega a 0,785.
La lectura: buena parte de la caída cruda es escala, pero sobrevive una pérdida real de margen. La información no desaparece —el AUROC agéntico sigue lejos de 0,5—; se estrecha.`,
      transicion: "Esa pérdida, ¿depende de cuán explícito es el daño?",
      fuentes: `S y s calculados desde heldout.projections (desviación agrupada de las dos clases en cada formato): S_chat = ${f2(sp.S_chat, 4)}, S_ag = ${f2(sp.S_ag, 4)}, d_chat = ${f2(sp.d_chat, 4)}, d_ag = ${f2(sp.d_ag, 4)}, κ = ${f2(sp.kappa, 4)}; coinciden con metricas_avance.json → heldout_format. AUROC: heldout_detection (gap ${f2(det.auroc_gap_heldout, 4)}, IC [${f2(det.auroc_gap_ci[0], 4)}; ${f2(det.auroc_gap_ci[1], 4)}], condicional a los puntajes guardados). Panel estandarizado: (p − media benigna del formato) / s agrupada del formato; puntos con dispersión vertical aleatoria solo para legibilidad. Mistral: registro, FORMAT EFFECT IN SCALE‑INVARIANT UNITS (held‑out).`,
    });
  }

  // ------------------------------------------------------------ 17. Explicitud
  {
    const s = pres.addSlide({ masterName: "CONTENIDO", sectionTitle: sec });
    nav(s, 2);
    s.addText("Según la explicitud, no aparece una jerarquía universal", { placeholder: "title" });
    const est = Object.fromEntries(L.estratos.map((e) => [e.stratum, e]));
    const nombres = [["explicit", "Explícito", 18], ["contextual", "Contextual", 14], ["framed", "Apariencia legítima", 10]];
    const x0 = 3.65, x1 = 8.2, esc = (v) => x0 + ((v - 0.5) / 0.5) * (x1 - x0);
    txt(s, [R("● ", { color: C.ink }), R("chat    "), R("● ", { color: C.ag }), R("agente")], { x: x1 - 2.2, y: 1.88, w: 2.2, h: 0.26, fontSize: 12, color: C.ink2, align: "right" });
    const grupos = [["Llama‑3.3‑70B · capa 76", nombres.map(([k, n, c]) => [n, c, est[k].auroc_chat, est[k].auroc_agent])],
      ["Mistral‑7B · capa 30 · registro", nombres.map(([k, n, c]) => [n, c, Mi.strata_oof[k][1], Mi.strata_oof[k][2]])]];
    [0.5, 0.6, 0.7, 0.8, 0.9, 1.0].forEach((v) => {
      seg(s, pres, esc(v), 2.25, esc(v), 5.75, v === 0.5 ? "AEB4BC" : "E6E8EB", 0.75, v === 0.5 ? { line: { dashType: "dash" } } : {});
      txt(s, f2(v, 1), { x: esc(v) - 0.3, y: 5.8, w: 0.6, h: 0.24, align: "center", fontSize: 11, color: C.mute });
    });
    txt(s, "AUROC fuera de muestra de la dirección conversacional (0,5 = azar)", { x: x0, y: 6.08, w: x1 - x0, h: 0.26, align: "center", fontSize: 11, color: C.mute });
    grupos.forEach(([g, filas], gi) => {
      const yg = 2.3 + gi * 1.75;
      rotulo(s, g, MX, yg, 3.0, C.ink);
      filas.forEach(([n, c, a, b], i) => {
        const y = yg + 0.42 + i * 0.42;
        txt(s, [R(n, { color: C.ink }), R(`  ·  ${c}`, { color: C.mute })], { x: MX, y: y - 0.15, w: 2.6, h: 0.3, fontSize: 12.5, valign: "middle" });
        seg(s, pres, esc(b), y, esc(a), y, "B7BDC5", 4);
        dot(s, pres, esc(a), y, 0.2, C.ink);
        dot(s, pres, esc(b), y, 0.2, C.ag);
        txt(s, `−${f2(a - b, 3)}`, { x: esc(a) + 0.14, y: y - 0.14, w: 0.75, h: 0.28, fontSize: 11.5, color: C.ink2, valign: "middle" });
      });
    });
    const X0 = 9.0, XW = W - MX - X0;
    const bloques = [["Pérdida máxima", "Llama en el estrato explícito (0,991 → 0,929); Mistral en el contextual (0,944 → 0,740)."],
      ["Apariencia legítima", "conserva señal por encima del azar en ambos modelos: 0,89 y 0,71 en agente."],
      ["Precisión", "18, 14 y 10 pares, sin intervalos por estrato: no permiten ordenar los estratos en la población. El corpus ampliado (42 · 24 · 30) se construyó para dar esa precisión."]];
    txt(s, bloques.flatMap(([a, b], i) => [R(a, { fontFace: F.demi, color: C.ink, breakLine: true }), R(b, { breakLine: i < bloques.length - 1, paraSpaceAfter: 14 })]),
      { x: X0, y: 1.95, w: XW, h: 4.7, fontSize: 13.5, color: C.ink2 });
    fuente(s, "metricas_avance.json → llama.strata_oof · registro de Mistral (DETECTION CAPABILITY BY EXPLICITNESS, held‑out) · tesis, tabla 4.5.");
    nota(s, {
      titulo: "Según la explicitud, no aparece una jerarquía universal", t: 40,
      guion: `Segunda dimensión: la explicitud. En Llama, la mayor pérdida aparece en el estrato explícito; en Mistral, en el contextual. No hay, entonces, una jerarquía universal del tipo «cuanto más encubierto, peor».
Lo que sí se observa es que el estrato de apariencia legítima conserva señal por encima del azar en ambos modelos: 0,89 y 0,71 en agente.
Con 18, 14 y 10 pares y sin intervalos por estrato, esto es exploratorio; precisamente para eso se construyó el corpus ampliado de 42, 24 y 30 pares.`,
      transicion: "Si la dirección de chat pierde margen en agente, ¿hay otra dirección que separe mejor?",
      fuentes: `Llama (capa 76): explícito ${f2(est.explicit.auroc_chat, 4)} → ${f2(est.explicit.auroc_agent, 4)}; contextual ${f2(est.contextual.auroc_chat, 4)} → ${f2(est.contextual.auroc_agent, 4)}; apariencia legítima ${f2(est.framed.auroc_chat, 4)} → ${f2(est.framed.auroc_agent, 4)}. Mistral (capa 30, registro): 0,9722 → 0,8827; 0,9439 → 0,7398; 0,9000 → 0,7100. Barras: estimaciones puntuales, sin contraste estadístico entre estratos ni entre modelos.`,
    });
  }

  // ------------------------------------------------------------ 18. Reorientación
  {
    const s = pres.addSlide({ masterName: "CONTENIDO", sectionTitle: sec });
    nav(s, 2);
    s.addText("En agente, el contraste dañino–benigno se reorienta", { placeholder: "title" });
    const rot = L.rotacion, dm = rot.diff_means, lr = rot.logistic;
    rrect(s, pres, MX, 1.88, 5.0, 4.62, C.mist);
    const o = [1.35, 5.85], Lv = 3.75, th = (dm.angle_deg_normalised * Math.PI) / 180;
    const e2 = [o[0] + Lv * Math.cos(th), o[1] - Lv * Math.sin(th)];
    s.addShape(pres.shapes.ARC, { x: o[0] - 1.0, y: o[1] - 1.0, w: 2.0, h: 2.0, angleRange: [360 - dm.angle_deg_normalised, 359.9], line: { color: C.ag, width: 1.5 }, fill: { type: "none" } });
    flecha(s, pres, o[0], o[1], o[0] + Lv, o[1], C.ink, 3);
    flecha(s, pres, o[0], o[1], e2[0], e2[1], C.ag, 3);
    dot(s, pres, o[0], o[1], 0.12, C.ink);
    txt(s, [...mx({ fontSize: 17 }, m.hat("r")), R("  chat", { fontSize: 13, color: C.ink2 })], { x: o[0] + Lv - 1.1, y: o[1] + 0.12, w: 1.4, h: 0.34 });
    txt(s, [...mx({ fontSize: 17, color: C.ag }, m.hat("w"), m.sub("DM", false)), R("  agente", { fontSize: 13, color: C.ag })], { x: e2[0] + 0.12, y: e2[1] - 0.1, w: 1.9, h: 0.34 });
    txt(s, [...mx({ fontSize: 16, color: C.ag }, m.i("θ")), R(` ≈ ${f2(dm.angle_deg_normalised, 0)}°`, { fontSize: 16, color: C.ag })], { x: o[0] + 0.85, y: o[1] - 1.05, w: 1.2, h: 0.34 });
    txt(s, "Ángulo a escala con el valor estimado; plano esquemático.", { x: MX, y: 6.55, w: 5.0, h: 0.3, fontSize: 11, color: C.mute });
    const X0 = 6.0, XW = W - MX - X0;
    eq(s, "coseno", X0, 1.88, XW, 0.9, 20);
    txt(s, [...mx({}, m.u("cos"), m.sub("×", false)), R(": coseno entre direcciones estimadas en mitades disjuntas de pares, una por formato. "),
      ...mx({}, m.i("ρ")), R(": fiabilidad entre mitades del mismo formato; corrige la atenuación por ruido de estimación.")],
    { x: X0, y: 2.85, w: XW, h: 0.8, fontSize: 13, color: C.ink2 });
    // tabla con barras de fiabilidad
    const cols = [X0, X0 + 2.05, X0 + 3.1, X0 + 4.55];
    ["Estimador", "cos, muestra completa", "fiabilidad ρ (chat · agente)", "lectura"].forEach((h, i) =>
      txt(s, h, { x: cols[i], y: 3.78, w: i === 3 ? XW - (cols[3] - X0) : cols[i + 1] - cols[i] - 0.1, h: 0.42, fontSize: 11, color: C.mute }));
    const filas = [["Diferencia de medias", dm.cosine_full_sample, dm.reliability_chat, dm.reliability_agent, `normalizado ${f2(dm.cosine_normalised, 3)} → θ ≈ ${f2(dm.angle_deg_normalised, 0)}°`],
      ["Regresión logística", lr.cosine_full_sample, lr.reliability_chat, lr.reliability_agent, "poco reproducible: no se interpreta igual"]];
    filas.forEach(([n, c, rc, ra, lect], i) => {
      const y = 4.28 + i * 0.68;
      txt(s, n, { x: cols[0], y, w: 2.1, h: 0.5, fontFace: F.demi, fontSize: 13, valign: "middle" });
      txt(s, f2(c, 3), { x: cols[1], y, w: 1.1, h: 0.5, fontFace: F.med, fontSize: 20, valign: "middle" });
      [[rc, C.ink], [ra, C.ag]].forEach(([v, col], k) => {
        rect(s, pres, cols[2], y + 0.08 + k * 0.2, 1.05, 0.14, "E3E6EA");
        rect(s, pres, cols[2], y + 0.08 + k * 0.2, 1.05 * v, 0.14, col);
        txt(s, f2(v, 2), { x: cols[2] + 1.1, y: y + 0.02 + k * 0.2, w: 0.4, h: 0.24, fontSize: 10.5, color: C.ink2 });
      });
      txt(s, lect, { x: cols[3], y, w: XW - (cols[3] - X0), h: 0.5, fontSize: 11.5, color: C.ink2, valign: "middle" });
    });
    rrect(s, pres, X0, 5.75, XW, 0.95, C.mist);
    txt(s, [R("Lectura prudente: ", { fontFace: F.demi, color: C.ink }), R("compatible con una reorientación del contraste entre clases. Es una descripción geométrica bajo supuestos de error: no prueba que el modelo use ese eje para decidir.")],
      { x: X0 + 0.2, y: 5.8, w: XW - 0.4, h: 0.85, fontSize: 13, color: C.ink2, valign: "middle" });
    fuente(s, "Llama V3, capa 76, 42 pares; mismo estimador en ambos formatos. metricas_avance.json → rotation_at_reference; tesis, sección 4.1.6 y figura 4.3.");
    nota(s, {
      titulo: "En agente, el contraste dañino–benigno se reorienta", t: 55,
      guion: `Si la dirección conversacional pierde margen en agente, ¿hay otra dirección que separe mejor? Estimé la dirección agéntica con el mismo estimador, diferencia de medias, y la comparé con la de chat.
El coseno entre las dos es 0,21. Como el ruido de estimación reduce el coseno aparente, medí la fiabilidad de cada dirección entre mitades disjuntas de pares: 0,90 en chat y 0,92 en agente, alta. Corrigiendo por ese techo, el coseno es 0,22, un ángulo de unos 77 grados.
Con regresión logística el coseno es aún menor, 0,07, pero esas direcciones son poco reproducibles, con fiabilidades de 0,25 y 0,28; por eso no las interpreto con la misma confianza ni las promedio con las anteriores.
La lectura prudente: el contraste entre clases se reorienta en agente. Es geometría, no causalidad.`,
      transicion: "Esa reorientación tiene una consecuencia práctica para el monitoreo.",
      fuentes: `rotation_at_reference.diff_means: cos muestra completa ${f2(dm.cosine_full_sample, 4)}; cos entre formatos en mitades disjuntas ${f2(dm.cosine_across_formats, 4)} (sd ${f2(dm.cosine_across_formats_sd, 4)}, 50 repeticiones); fiabilidad chat ${f2(dm.reliability_chat, 4)}, agente ${f2(dm.reliability_agent, 4)}; techo √(ρc·ρa) = ${f2(dm.ceiling, 4)}; normalizado ${f2(dm.cosine_normalised, 4)}; ángulo ${f2(dm.angle_deg_normalised, 2)}°. Logística: cos ${f2(lr.cosine_full_sample, 4)}, fiabilidades ${f2(lr.reliability_chat, 4)} y ${f2(lr.reliability_agent, 4)}. La versión anterior de la tesis atribuía a Llama un coseno de 0,051; la cifra vigente es la de V3. La normalización por fiabilidad es un diagnóstico bajo supuestos de error, no un ángulo poblacional exacto.`,
    });
  }

  // ------------------------------------------------------------ 19. Monitor
  {
    const s = pres.addSlide({ masterName: "CONTENIDO", sectionTitle: sec });
    nav(s, 2);
    s.addText("Un monitor ajustado en formato agéntico recupera la señal", { placeholder: "title" });
    const mo = L.monitores, cc = L.monitor_conteos;
    const filas = [["Llama · diferencia de medias", "capa 76", cc.dm.tp, cc.dm.fp, mo.auroc_agent_diff_means_oof, mo.monitor_f1_diff_means_oof, mo.monitor_fpr_diff_means_oof],
      ["Llama · regresión logística", "capa 76", cc.lr.tp, cc.lr.fp, mo.auroc_w_agent_oof, mo.monitor_f1_oof, mo.monitor_fpr_oof],
      ["Mistral · regresión logística", "capa 30 · registro", Mi.monitor.tp, Mi.monitor.fp, Mi.monitor.auroc, Mi.monitor.f1, Mi.monitor.fpr]];
    const xh = 3.35, xb = 7.0, dd = 0.13, pitch = 0.163;
    rotulo(s, "42 dañinas", xh, 1.88, 3.4, C.harm);
    rotulo(s, "42 benignas", xb, 1.88, 3.4, C.ben);
    ["AUROC", "F1", "FPR"].forEach((h, i) => txt(s, h, { x: 10.62 + i * 0.72, y: 1.88, w: 0.7, h: 0.26, fontSize: 11, color: C.mute, align: "right" }));
    filas.forEach(([n, sub, tp, fp, au, f1, fpr], i) => {
      const y = 2.3 + i * 1.18;
      txt(s, [R(n, { fontFace: F.demi, color: C.ink, breakLine: true }), R(sub, { color: C.mute })], { x: MX, y, w: 2.65, h: 0.62, fontSize: 13 });
      for (let k = 0; k < 42; k++) {
        const cx = xh + (k % 21) * pitch + dd / 2, cy = y + 0.12 + Math.floor(k / 21) * pitch + dd / 2;
        if (k < tp) dot(s, pres, cx, cy, dd, C.harm);
        else s.addShape(pres.shapes.OVAL, { x: cx - dd / 2, y: cy - dd / 2, w: dd, h: dd, fill: { color: C.paper }, line: { color: C.harm, width: 1 } });
        const bx = xb + (k % 21) * pitch + dd / 2;
        if (k < fp) dot(s, pres, bx, cy, dd, C.ben);
        else s.addShape(pres.shapes.OVAL, { x: bx - dd / 2, y: cy - dd / 2, w: dd, h: dd, fill: { color: C.paper }, line: { color: "9DBAD6", width: 1 } });
      }
      txt(s, `${tp}/42 detectadas`, { x: xh, y: y + 0.5, w: 3.4, h: 0.26, fontSize: 11.5, color: C.ink2 });
      txt(s, `${fp}/42 falsos positivos`, { x: xb, y: y + 0.5, w: 3.4, h: 0.26, fontSize: 11.5, color: C.ink2 });
      [au, f1, fpr].forEach((v, j) => txt(s, f2(v, 3), { x: 10.62 + j * 0.72, y: y + 0.05, w: 0.7, h: 0.4, fontSize: j === 1 ? 17 : 14, fontFace: j === 1 ? F.med : F.body, color: C.ink, align: "right", valign: "middle" }));
    });
    txt(s, [R("●", { color: C.harm }), R(" detectada  "), R("○", { color: C.harm }), R(" no detectada      "), R("●", { color: C.ben }), R(" falso positivo  "), R("○", { color: "9DBAD6" }), R(" benigna aceptada")],
      { x: xh, y: 5.75, w: 7.2, h: 0.28, fontSize: 11.5, color: C.ink2 });
    const w0 = L.fpr_cero_wilson;
    txt(s, [R("Cero no es una tasa nula. ", { fontFace: F.demi, color: C.ink }), R(`0/42 falsos positivos tiene un IC 95 % de Wilson de [0 %; ${f2(100 * w0[1], 1)} %]. AUROC: media de 5 folds; F1 y FPR: umbral calibrado en el entrenamiento de cada fold y aplicado a pares retenidos. Sin validación externa: viabilidad preliminar, no garantía de seguridad.`)],
      { x: MX, y: 6.12, w: W - 2 * MX, h: 0.75, fontSize: 12.5, color: C.ink2 });
    fuente(s, "metricas_avance.json → monitors_at_reference; registro de Mistral (AUROC out‑of‑fold, F1 out‑of‑fold, FPR). Conteos de puntos deducidos de F1 y FPR con 42 por clase. Tesis, tabla 4.6.");
    nota(s, {
      titulo: "Un monitor ajustado en formato agéntico recupera la señal", t: 40,
      guion: `Eso tiene una consecuencia práctica, el objetivo cinco. Un monitor que proyecta sobre la dirección agéntica, con umbral calibrado en entrenamiento y evaluado en pares retenidos, recupera la señal.
Con diferencia de medias detecta 35 de 42 dañinas con seis falsos positivos; con regresión logística, 37 de 42 sin falsos positivos; en Mistral, 33 con cinco.
Pero cero de 42 no es una tasa nula: su intervalo de Wilson llega a 8,4 %. Y no hay validación en datos externos. Es viabilidad preliminar, no una garantía de seguridad.`,
      transicion: "Queda una pregunta: ¿la dirección tiene efecto sobre la conducta?",
      fuentes: `Llama, capa 76: diferencia de medias AUROC OOF ${f2(mo.auroc_agent_diff_means_oof, 4)}, F1 ${f2(mo.monitor_f1_diff_means_oof, 4)}, FPR ${f2(mo.monitor_fpr_diff_means_oof, 4)}; logística AUROC ${f2(mo.auroc_w_agent_oof, 4)} (sd ${f2(mo.auroc_w_agent_oof_sd, 4)}), F1 ${f2(mo.monitor_f1_oof, 4)}, precisión 1, sensibilidad ${f2(mo.monitor_recall_oof, 4)}, FPR 0. En muestra la logística da AUROC 1 y F1 1, que no se usan. Mistral, logística (registro): AUROC 0,9347 (sd 0,0311), F1 0,8250, FPR 0,1190. Los conteos (35/6, 37/0, 33/5) se deducen de F1 y FPR con 42 por clase: FP = FPR·42 y TP = F1·(FP + 42)/(2 − F1). IC de Wilson para 0/42 calculado aquí: [0; 0,0838].`,
    });
  }

  // ------------------------------------------------------------ 20. Intervención
  {
    const s = pres.addSlide({ masterName: "CONTENIDO", sectionTitle: sec });
    nav(s, 2);
    s.addText("Sumar la dirección restaura el rechazo, y también lo impone", { placeholder: "title" });
    const iv = L.intervencion56, dosis = ["0.5", "1", "2", "4"];
    eq(s, "adicion", MX, 1.85, 7.4, 0.55, 18);
    const serie = (clave) => dosis.map((k) => iv.per_dose[k][clave]);
    const rates = (clave) => serie(clave).map((o) => o.rate);
    const labels = ["0,5×", "1×", "2×", "4×"];
    txt(s, [R("● ", { color: C.harm }), R("dirección · dañinas (n = 19)    "), R("● ", { color: C.mute }), R("aleatoria de igual norma · dañinas (n = 19)    "), R("● ", { color: C.ben }), R("dirección · benignas (n = 41)")],
      { x: MX, y: 2.55, w: 7.6, h: 0.3, fontSize: 11.5, color: C.ink2 });
    s.addChart(pres.charts.LINE, [
      { name: "Rechazo · dañinas", labels, values: rates("restore") },
      { name: "Aleatoria · dañinas", labels, values: rates("control") },
      { name: "Rechazo · benignas", labels, values: rates("benign") },
    ], ejesGrafico({ x: MX - 0.1, y: 2.95, w: 7.6, h: 3.75, chartColors: [C.harm, C.mute, C.ben], lineDataSymbol: "circle", lineDataSymbolSize: 9,
      valAxisMinVal: 0, valAxisMaxVal: 1, valAxisMajorUnit: 0.25, valAxisLabelFormatCode: "0%", showTitle: true, title: "ERR__dosis",
      showCatAxisTitle: true, catAxisTitle: "dosis (múltiplos de Δp en la capa 56)", showValAxisTitle: true, valAxisTitle: "fracción etiquetada como rechazo" }));
    ctx.errores.dosis = [["restore", C.harm], ["control", C.mute], ["benign", C.ben]].map(([k, col]) => ({
      mas: serie(k).map((o) => o.ci_upper - o.rate), menos: serie(k).map((o) => o.rate - o.ci_lower), color: col }));
    const X0 = 8.35, XW = W - MX - X0;
    const d1 = iv.per_dose["1"], d2 = iv.per_dose["2"], d4 = iv.per_dose["4"];
    const lect = [["Dosis 1", `${d1.restore.k}/19 dañinas rechazan con la dirección y ${d1.control.k}/19 con la aleatoria (McNemar exacto p = ${f2(d1.restore_vs_control.mcnemar_p, 5)}). Benignas: ${d1.benign.k}/41.`],
      ["Dosis 2", `${d2.restore.k}/19 dañinas… y ${d2.benign.k}/41 benignas.`],
      ["Dosis 4", `${d4.restore.k}/19 dañinas; ${d4.benign.k}/41 benignas y ${d4.benign.n_degenerate} respuestas degeneradas.`]];
    txt(s, lect.flatMap(([a, b], i) => [R(a + "  ", { fontFace: F.demi, color: C.ink }), R(b, { breakLine: i < lect.length - 1, paraSpaceAfter: 10 })]),
      { x: X0, y: 1.9, w: XW, h: 2.4, fontSize: 13.5, color: C.ink2 });
    rrect(s, pres, X0, 4.0, XW, 1.2, C.mist);
    txt(s, [R("Un efecto sobre la conducta etiquetada, ", { fontFace: F.demi, color: C.ink }), R("no una reparación selectiva. Tampoco prueba que la reorientación cause el fallo.")],
      { x: X0 + 0.2, y: 4.05, w: XW - 0.4, h: 1.1, fontSize: 13.5, color: C.ink2, valign: "middle" });
    txt(s, "Capa 56, declarada en la configuración (≠ capa 76 del diagnóstico). Subconjuntos que no rechazaban en agente. Valores p exploratorios, sin corrección por capas y dosis. Mistral: el registro termina durante esta etapa.",
      { x: X0, y: 5.4, w: XW, h: 1.2, fontSize: 12.5, color: C.mute });
    fuente(s, "Barras: IC 95 % de Wilson guardados. Unidad de dosis: desplazamiento dañino de la capa 56 = 6,93. metricas_avance.json → intervention_layer56; tesis, tabla 4.8 y sección 4.3.");
    nota(s, {
      titulo: "Sumar la dirección restaura el rechazo, y también lo impone", t: 55,
      guion: `Por último, ¿la dirección tiene efecto sobre la conducta? En la capa 56 sumé alfa veces la dirección durante la generación, en las dañinas que el modelo no rechazaba como agente, y lo comparé con una dirección aleatoria de la misma norma.
Con dosis uno rechazan 12 de 19, frente a 1 de 19 con la aleatoria: McNemar da p de 0,001. Con dosis dos, 19 de 19. Pero miren la curva azul: a esa dosis también rechaza 40 de 41 benignas.
Es decir, sumar la dirección cambia la conducta en el sentido esperado, pero no la repara de forma selectiva: impone el rechazo. Además es exploratorio: una capa distinta de la del diagnóstico, sin corrección por múltiples dosis y con etiquetas heurísticas. En Mistral, el registro termina en esta etapa.`,
      transicion: "Con esto puedo volver a la pregunta.",
      fuentes: `intervention_layer56: unidad de dosis ${f2(iv.dose_unit, 4)} (Δp_h chat − agente en la capa 56); α = 3,46; 6,93; 13,85; 27,70. Dañinas no rechazadas en agente: n = 19; benignas: n = 41. Tasas e IC de Wilson guardados por dosis; restore_vs_control McNemar: dosis 0,5 p = 0,0625; 1: p = 0,00098; 2 y 4: p = 7,6·10⁻⁶. Benignas a dosis 4: 33/41 con 8 degeneradas. La capa 56 es la declarada en config.intervention_layers (56, 52, 28); se reporta solo la 56, como en la tesis.`,
    });
  }
};

module.exports.grafico = grafico;
module.exports.f2 = f2;
module.exports.pct = pct;
