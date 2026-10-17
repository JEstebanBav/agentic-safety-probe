// I. Problema: portada, contexto, caso documentado, brecha y pregunta.
const B = require("./base");
const { W, MX, C, F, R, txt, rect, rrect, dot, seg, flecha, nav, fuente, rotulo, enjambre } = B;
const { nota } = require("./guion");

function caja(slide, pres, x, y, w, h, texto, o = {}) {
  rrect(slide, pres, x, y, w, h, o.fill || C.mist, o.line ? { line: { color: o.line, width: 1 } } : {});
  txt(slide, texto, { x: x + 0.05, y, w: w - 0.1, h, align: "center", valign: "middle", fontSize: o.size || 13,
    color: o.color || C.ink, fontFace: o.font || F.body });
}

// Dos ejes con las proyecciones reales: motivo de portada y de cierre.
function ejes(slide, pres, D, x0, x1, yChat, yAg, oscuro) {
  const p = D.llama.proy76;
  const lo = -10, hi = 28;
  const esc = (v) => x0 + ((v - lo) / (hi - lo)) * (x1 - x0);
  const dia = 0.08;
  const eje = oscuro ? C.darkRule : C.rule;
  const lab = oscuro ? C.darkMute : C.ink2;
  [[yChat, "CHAT", p.chat_h, p.chat_b], [yAg, "AGENTE", p.ag_h, p.ag_b]].forEach(([y, nombre, h, b]) => {
    seg(slide, pres, x0, y, x1, y, eje, 0.75);
    slide.addText(nombre, { x: MX, y: y - 0.13, w: 1.0, h: 0.26, isTextBox: true, margin: 0, fontFace: F.demi, fontSize: 11,
      charSpacing: 1.5, color: lab, valign: "middle" });
    enjambre(h, esc, y, dia, -1, 0.8).forEach((q) => dot(slide, pres, q.x, q.y, dia, C.harm, { transparency: 12 }));
    enjambre(b, esc, y, dia, +1, 0.8).forEach((q) => dot(slide, pres, q.x, q.y, dia, oscuro ? "5C93C6" : C.ben, { transparency: 12 }));
  });
  // Referencia del cero común a ambos ejes.
  seg(slide, pres, esc(0), yChat - 0.55, esc(0), yAg + 0.62, eje, 0.75, { line: { dashType: "dash" } });
  slide.addText("0", { x: esc(0) - 0.2, y: yAg + 0.62, w: 0.4, h: 0.22, isTextBox: true, margin: 0, align: "center",
    fontFace: F.body, fontSize: 10, color: lab });
  return esc;
}

module.exports = function (pres, D) {
  // ------------------------------------------------------------ 1. Portada
  {
    const s = pres.addSlide({ masterName: "PORTADA", sectionTitle: "I. Problema" });
    s.addImage({ path: require("path").join(__dirname, "../../tesis/Tesis_avances_actualizada/images/logo-externado-matematicas.png"),
      x: MX, y: 0.42, w: 1.96, h: 0.62, altText: "Universidad Externado de Colombia, Departamento de Matemáticas" });
    txt(s, "Sustentación de trabajo de grado · Modalidad de investigación", { x: 7.4, y: 0.6, w: W - MX - 7.4, h: 0.3,
      align: "right", fontSize: 11, color: C.mute });
    txt(s, "Transformación de la dirección de rechazo en modelos de lenguaje agénticos",
      { x: MX, y: 1.42, w: 11.6, h: 1.3, fontFace: F.head, fontSize: 36, color: C.ink, valign: "top" });
    txt(s, "Un análisis desde la interpretabilidad mecánica", { x: MX, y: 2.78, w: 9, h: 0.45, fontFace: F.head,
      italic: true, fontSize: 20, color: C.ink2 });
    ejes(s, pres, D, 1.75, W - MX, 4.12, 5.3, false);
    txt(s, "Juan Esteban Espitia Leiva", { x: MX, y: 6.22, w: 6, h: 0.34, fontFace: F.demi, fontSize: 16 });
    txt(s, "Tutor: Julián Fernando Sánchez", { x: MX, y: 6.56, w: 6, h: 0.28, fontSize: 12.5, color: C.ink2 });
    txt(s, "Pregrado en Ciencia de Datos · Departamento de Matemáticas · Bogotá, 2026", { x: MX, y: 6.86, w: 7, h: 0.28,
      fontSize: 11, color: C.mute });
    txt(s, [R("●", { color: C.harm }), R(" dañina    "), R("●", { color: C.ben }), R(" benigna\n"),
      R("Cada punto es una instrucción del corpus: su proyección fuera de muestra sobre la dirección de rechazo. Llama‑3.3‑70B, capa 76, 42 pares.")],
    { x: 7.25, y: 6.22, w: W - MX - 7.25, h: 0.8, fontSize: 10, color: C.mute, align: "right" });
    nota(s, {
      titulo: "Portada", t: 15,
      guion: `Buenos días. Soy Juan Esteban Espitia Leiva y voy a sustentar mi trabajo de grado, dirigido por el profesor Julián Fernando Sánchez: «Transformación de la dirección de rechazo en modelos de lenguaje agénticos».
Los puntos que ven abajo no son decoración: son datos del trabajo. Cada punto es una instrucción. Al final de la exposición los vamos a poder leer.`,
      transicion: "Empiezo por lo que está cambiando en el uso de estos modelos.",
      fuentes: `Puntos: proyecciones fuera de muestra (dirección reajustada por fold) en la capa 76 de Llama‑3.3‑70B V3. results/corridas/llama33_70b/v3/results.json → heldout.projections. Fila superior: chat; inferior: agente. Rojo arriba del eje: dañinas; azul abajo: benignas. Escala común de −10 a 28 unidades crudas de proyección.`,
    });
  }

  // ------------------------------------------------------------ 2. Contexto
  {
    const s = pres.addSlide({ masterName: "CONTENIDO", sectionTitle: "I. Problema" });
    nav(s, 0);
    s.addText("Los modelos de lenguaje ya no solo conversan: actúan", { placeholder: "title" });
    const xs = [MX, 2.43, 4.01, 5.59], ws = [1.45, 1.2, 1.2, 2.0];
    rotulo(s, "Formato conversacional", MX, 1.82, 5);
    const conv = ["Instrucción", "Modelo", "Texto", "Una persona lee y decide si actuar"];
    conv.forEach((t, i) => caja(s, pres, xs[i], 2.15, ws[i], 0.78, t, i === 3 ? { fill: C.paper, line: C.ink2 } : {}));
    rotulo(s, "Formato agéntico", MX, 3.3, 5, C.ag);
    const ag = ["Sistema + herramientas + instrucción", "Modelo", "Llamada a función", "El sistema ejecuta: correo, pago, terminal…"];
    ag.forEach((t, i) => caja(s, pres, xs[i], 3.63, ws[i], 0.78, t, i === 0 || i === 2 ? { fill: C.agL } : {}));
    [2.54, 4.02].forEach((y) => [0, 1, 2].forEach((i) => flecha(s, pres, xs[i] + ws[i] + 0.04, y, xs[i + 1] - 0.05, y, C.ink2, 1.25)));
    txt(s, [R("Entre la respuesta y su efecto ya no hay una persona en cada paso.", { italic: true })],
      { x: xs[3] - 1.2, y: 4.5, w: 3.2, h: 0.5, fontSize: 12.5, color: C.ag, align: "right" });
    txt(s, [R("Cambia lo que acompaña a la instrucción, no el modelo. ", { fontFace: F.demi, color: C.ink }),
      R("El mismo sistema recibe un rol, un catálogo de herramientas y una salida adicional: invocar una acción. La decisión de cumplir o rechazar, tomada al comienzo, gobierna la secuencia que sigue.")],
    { x: MX, y: 5.25, w: 7.0, h: 1.3, fontSize: 15, color: C.ink2 });
    // Evidencia de contexto
    rrect(s, pres, 8.2, 1.82, W - MX - 8.2, 2.3, C.mist);
    rotulo(s, "Adopción · pronóstico de industria", 8.45, 1.98, 4);
    txt(s, "40 %", { x: 8.45, y: 2.25, w: 2.2, h: 0.75, fontFace: F.med, fontSize: 44, color: C.ink });
    txt(s, "de las aplicaciones empresariales incluiría agentes de tarea específica a fines de 2026, frente a menos del 5 % en 2025.",
      { x: 8.45, y: 3.0, w: 4.05, h: 0.75, fontSize: 13, color: C.ink2 });
    txt(s, "Gartner, comunicado del 26‑08‑2025. Proyección, no medición.", { x: 8.45, y: 3.72, w: 4.05, h: 0.28, fontSize: 10, color: C.mute });
    rrect(s, pres, 8.2, 4.3, W - MX - 8.2, 2.3, C.mist);
    rotulo(s, "Riesgo · evaluación internacional", 8.45, 4.46, 4);
    txt(s, "«Los agentes de IA plantean riesgos mayores porque actúan de forma autónoma, lo que dificulta que las personas intervengan antes de que los fallos causen daño».",
      { x: 8.45, y: 4.78, w: 4.05, h: 1.35, fontFace: F.head, italic: true, fontSize: 14, color: C.ink });
    txt(s, "International AI Safety Report 2026, resumen ejecutivo (3‑02‑2026). Traducción propia.",
      { x: 8.45, y: 6.13, w: 4.05, h: 0.4, fontSize: 10, color: C.mute });
    fuente(s, "Gartner (2025). Gartner predicts 40% of enterprise apps will feature task‑specific AI agents by 2026 · Bengio et al. (2026). International AI Safety Report 2026.");
    nota(s, {
      titulo: "Los modelos de lenguaje ya no solo conversan: actúan", t: 50,
      guion: `Durante años evaluamos los modelos de lenguaje como sistemas que conversan: reciben una instrucción, producen texto, y una persona lee ese texto y decide si actúa.
Hoy esos mismos modelos se despliegan como agentes. Además de la instrucción reciben un mensaje de sistema y un catálogo de herramientas, y pueden invocar acciones —enviar un correo, mover dinero, ejecutar un comando— que el sistema ejecuta sin una revisión humana en cada paso.
Gartner pronostica que a finales de este año cerca del 40 % de las aplicaciones empresariales incluirán agentes. Es una proyección de industria, no una medición, pero muestra la dirección del cambio. Y el Informe Internacional de Seguridad en IA de 2026 lo resume así: los agentes plantean riesgos mayores porque actúan de forma autónoma y es más difícil intervenir antes de que un fallo cause daño.`,
      transicion: "Esto ya no es solo una proyección: hay un caso documentado.",
      fuentes: `Gartner, comunicado de prensa del 26 de agosto de 2025 (gartner.com/en/newsroom/press-releases/2025-08-26-gartner-predicts-40-percent-of-enterprise-apps-will-feature-task-specific-ai-agents-by-2026-up-from-less-than-5-percent-in-2025). La tesis cita esta cifra a través de miniOrange (2026); aquí se usa la fuente primaria y se presenta como pronóstico.
International AI Safety Report 2026, resumen ejecutivo, publicado el 3 de febrero de 2026, presidido por Yoshua Bengio (internationalaisafetyreport.org/publication/2026-report-executive-summary). Cita original: "AI agents pose heightened risks because they act autonomously, making it harder for humans to intervene before failures cause harm". Traducción propia.
No se usan en la diapositiva las cifras de encuestas de Gravitee citadas en la tesis: son datos de percepción de un proveedor y el jurado podría pedir su metodología.`,
    });
  }

  // 3. Caso documentado de Hugging Face y OpenAI
  require("./s_caso_hf")(pres);

  // ------------------------------------------------------------ 4. Brecha
  {
    const s = pres.addSlide({ masterName: "CONTENIDO", sectionTitle: "I. Problema" });
    nav(s, 0);
    s.addText("El rechazo falla en agentes; falta saber qué ocurre adentro", { placeholder: "title" });
    const tarjeta = (x, rot, cab, cita, items) => {
      rrect(s, pres, x, 1.82, 4.6, 3.45, C.mist);
      rotulo(s, rot, x + 0.25, 1.98, 4.2);
      txt(s, [R(cab, { fontFace: F.head, fontSize: 18, breakLine: true }), R(cita, { fontSize: 11, color: C.mute })],
        { x: x + 0.25, y: 2.3, w: 4.1, h: 0.66 });
      txt(s, items.map((t, i) => R(t, { bullet: { indent: 12 }, breakLine: i < items.length - 1, paraSpaceAfter: 6 })),
        { x: x + 0.25, y: 3.05, w: 4.12, h: 2.1, fontSize: 13.5, color: C.ink2 });
    };
    tarjeta(MX, "Plano conductual · qué hace", "AgentHarm", "Andriushchenko et al., ICLR 2025", [
      "110 tareas explícitamente dañinas en 11 categorías, con herramientas simuladas.",
      "Modelos líderes cumplen solicitudes maliciosas de agente sin necesidad de jailbreak.",
      "Las evasiones de chat se transfieren y el agente conserva sus capacidades."]);
    tarjeta(W - MX - 4.6, "Plano mecánico · cómo lo representa", "La dirección de rechazo", "Arditi et al., NeurIPS 2024", [
      "El rechazo está mediado por una dirección del flujo residual, en 13 modelos abiertos de hasta 72 B parámetros.",
      "Quitarla suprime el rechazo; sumarla lo induce incluso ante peticiones inofensivas.",
      "Se caracterizó en formato conversacional."]);
    txt(s, "?", { x: 5.4, y: 1.85, w: 2.53, h: 1.25, fontFace: F.head, fontSize: 80, color: C.ag, align: "center", valign: "middle" });
    txt(s, "No se ha determinado si la dirección conserva su capacidad de detectar la intención dañina cuando el modelo actúa, ni si eso depende de la explicitud.",
      { x: 5.38, y: 3.15, w: 2.57, h: 2.0, fontFace: F.head, italic: true, fontSize: 13.5, color: C.ink, align: "center" });
    rotulo(s, "Trabajos cercanos", MX, 5.5, 4);
    const cerca = [["Lermen et al. (2024)", "ablación del vector de rechazo en un agente Llama 3.1 70B: elimina el mecanismo por diseño."],
      ["Pinto (2026)", "steering chat → agente: la dirección llega casi intacta; su acoplamiento con la conducta se recalibra."],
      ["Zhong y Li (2026)", "el rechazo depende de la representación de la persona del modelo."]];
    cerca.forEach(([a, t], i) => txt(s, [R(a, { fontFace: F.demi, color: C.ink }), R(" · " + t)],
      { x: MX + i * 4.12, y: 5.82, w: 3.85, h: 0.95, fontSize: 12.5, color: C.ink2 }));
    fuente(s, "Andriushchenko et al. (2024; ICLR 2025) · Arditi et al. (2024), NeurIPS · Lermen et al. (2024) · Pinto (2026) · Zhong y Li (2026). Referencias completas en R11.");
    nota(s, {
      titulo: "El rechazo falla en agentes; falta saber qué ocurre adentro", t: 60,
      guion: `La literatura responde en dos planos que todavía no se han conectado.
En el plano conductual, AgentHarm, publicado en ICLR 2025, midió 110 tareas dañinas con herramientas simuladas: los modelos líderes cumplen muchas sin necesidad de jailbreak, y las evasiones diseñadas para chat se transfieren al agente. Eso describe el fallo, pero no lo explica.
En el plano mecánico, Arditi y colaboradores mostraron en NeurIPS 2024 que el rechazo está mediado por una sola dirección del espacio de activaciones: si se quita, el modelo deja de rechazar; si se suma, rechaza incluso lo inofensivo. Pero esa dirección se caracterizó en chat.
Los trabajos más cercanos intervienen el modelo —Lermen elimina la dirección por diseño— o estudian el control —Pinto muestra que la dirección inyectada llega casi intacta, pero su efecto en la conducta se recalibra—. Lo que no se ha determinado es si, usada como instrumento de diagnóstico, la dirección conserva su capacidad de detectar la intención dañina cuando el modelo actúa, y si eso depende de cuán explícito es el daño.`,
      transicion: "Esa es exactamente la pregunta de la tesis.",
      fuentes: `Andriushchenko et al., "AgentHarm: A benchmark for measuring harmfulness of LLM agents", ICLR 2025 (arXiv:2410.09024): 110 tareas explícitamente maliciosas (440 con aumentos), 11 categorías; "leading LLMs are surprisingly compliant with malicious agent requests without jailbreaking". Arditi et al., NeurIPS 2024 (arXiv:2406.11717). Lermen, Dziemian y Pimpale (2024, arXiv:2410.10871). Pinto (2026, arXiv:2607.09156). Zhong y Li (2026, arXiv:2606.26161). Formulación de la brecha: tesis, sección 2.3.`,
    });
  }

  // ------------------------------------------------------------ 5. Pregunta
  {
    const s = pres.addSlide({ masterName: "OSCURO", sectionTitle: "I. Problema" });
    nav(s, 0, true);
    rotulo(s, "Pregunta de investigación", MX, 1.0, 6, C.darkMute);
    txt(s, "Cuando un modelo de lenguaje pasa de conversar a actuar como agente, ¿conserva su dirección de rechazo la capacidad de detectar intenciones dañinas, incluso cuando estas se presentan de forma encubierta?",
      { x: MX, y: 1.38, w: 11.4, h: 2.3, fontFace: F.head, fontSize: 30, color: C.darkTxt });
    const dims = [["Dimensión 1 · Formato", "conversacional → agéntico: mensaje de sistema y herramientas simuladas, con el mismo texto de instrucción", C.ag],
      ["Dimensión 2 · Explicitud", "explícita · contextual · apariencia legítima: cuánto delata la superficie del texto la intención", "E9735B"]];
    dims.forEach(([r, t, col], i) => {
      const x = MX + i * 6.15;
      dot(s, pres, x + 0.08, 4.12, 0.14, col);
      rotulo(s, r, x + 0.3, 3.99, 5.2, C.darkTxt);
      txt(s, t, { x: x + 0.3, y: 4.3, w: 5.5, h: 0.62, fontSize: 14, color: C.darkMute });
    });
    const oe = ["Construir un corpus contrastivo que aísle el formato", "Estimar la dirección y caracterizarla por capa",
      "Cuantificar el cambio de proyección y su significancia", "Evaluar la detección según la explicitud",
      "Explorar, de forma preliminar, un monitor"];
    oe.forEach((t, i) => {
      const x = MX + i * 2.45;
      rrect(s, pres, x, 5.3, 2.3, 1.38, C.dark2);
      rotulo(s, `OE${i + 1}`, x + 0.18, 5.42, 1.5, C.ag);
      txt(s, t, { x: x + 0.18, y: 5.72, w: 1.98, h: 0.9, fontSize: 12.5, color: C.darkTxt });
    });
    nota(s, {
      titulo: "Pregunta de investigación", t: 45,
      guion: `La pregunta es esta: cuando un modelo de lenguaje pasa de conversar a actuar como agente, ¿conserva su dirección de rechazo la capacidad de detectar intenciones dañinas, incluso cuando se presentan de forma encubierta?
Tiene dos dimensiones que el diseño cruza. El formato: de conversacional a agéntico, con mensaje de sistema y herramientas simuladas y el mismo texto de instrucción. Y la explicitud: si el daño es explícito, contextual o tiene apariencia legítima, como en el caso que acabamos de ver.
Para responderla planteé cinco objetivos: construir un corpus que aísle el formato, estimar la dirección como línea base, cuantificar su cambio, evaluar la detección según la explicitud y explorar un monitor. Los recorro en ese orden.`,
      transicion: "Empiezo por el instrumento: el corpus.",
      fuentes: "Pregunta y objetivos: tesis, secciones 1.2 y 1.4 (objetivo general y cinco objetivos específicos).",
    });
  }
};
module.exports.ejes = ejes;
module.exports.caja = caja;
