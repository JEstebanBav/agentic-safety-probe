// II. Método: diseño pareado, corpus, lectura, dirección, medidas y validación.
const B = require("./base");
const { W, MX, C, F, R, txt, m, mx, N, eq, rect, rrect, dot, seg, flecha, curva, nav, fuente, rotulo } = B;
const { nota } = require("./guion");
const { caja } = require("./s_problema");

// Generador pseudoaleatorio reproducible para los esquemas pedagógicos.
function azar(semilla) {
  let s = semilla >>> 0;
  const u = () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
  return { u, n: () => Math.sqrt(-2 * Math.log(u() + 1e-12)) * Math.cos(2 * Math.PI * u()) };
}

module.exports = function (pres, D) {
  const L = D.llama;
  const tk = L.tokens;
  const sec = "II. Método";

  // ------------------------------------------------------------ 6. Diseño pareado
  {
    const s = pres.addSlide({ masterName: "CONTENIDO", sectionTitle: sec });
    nav(s, 1);
    s.addText("El instrumento: mismo texto, distinto formato, dos intenciones", { placeholder: "title" });
    const cx = [1.55, 3.7], cw = 1.6, ry = [2.25, 3.75], rh = 1.2;
    rotulo(s, "Chat", cx[0], 1.88, cw, C.ink, { align: "center" });
    rotulo(s, "Agente", cx[1], 1.88, cw, C.ag, { align: "center" });
    rotulo(s, "Dañina", MX, ry[0] + rh / 2 - 0.13, 0.9, C.harm);
    rotulo(s, "Benigna", MX, ry[1] + rh / 2 - 0.13, 0.9, C.ben);
    const celdas = [["Instrucción dañina", "sin herramientas"], ["Misma instrucción", "+ sistema + herramientas"],
      ["Contraparte benigna", "sin herramientas"], ["Misma contraparte", "+ sistema + herramientas"]];
    celdas.forEach(([a, b], k) => {
      const i = k % 2, j = Math.floor(k / 2);
      const fill = j === 0 ? C.harmL : C.benL;
      rrect(s, pres, cx[i], ry[j], cw, rh, fill, i === 1 ? { line: { color: C.ag, width: 2 } } : {});
      txt(s, [R(a, { fontFace: F.demi, breakLine: true }), R(b, { color: C.ink2 })],
        { x: cx[i] + 0.1, y: ry[j], w: cw - 0.2, h: rh, valign: "middle", align: "center", fontSize: 12.5 });
    });
    [0, 1].forEach((j) => {
      const y = ry[j] + rh / 2;
      flecha(s, pres, cx[0] + cw + 0.04, y, cx[1] - 0.06, y, j === 0 ? C.harm : C.ben, 1.75);
      txt(s, j === 0 ? N.dph({ fontSize: 15, color: C.harm }) : N.dpb({ fontSize: 15, color: C.ben }),
        { x: cx[0] + cw, y: y - 0.42, w: cx[1] - cx[0] - cw, h: 0.3, align: "center" });
    });
    txt(s, [R("Entrada agéntica = ", { fontFace: F.demi }), R("solicitud + mensaje de sistema + catálogo de herramientas.")],
      { x: MX, y: 5.2, w: 4.75, h: 0.75, fontSize: 15, color: C.ink });
    txt(s, "El catálogo describe funciones y parámetros. Aquí no se ejecutan: cambia la entrada que lee el modelo.",
      { x: MX, y: 6.05, w: 4.75, h: 0.65, fontSize: 12.5, color: C.ink2 });
    // Ejemplo real
    const X0 = 5.9, XW = W - MX - X0;
    rotulo(s, "Ejemplo literal · fraud_06 · nivel contextual", X0, 1.88, XW);
    const mono = { fontFace: F.mono, fontSize: 12, color: C.ink2 };
    const dif = (t, col) => R(t, Object.assign({}, mono, { color: col, bold: true }));
    const resaltar = (texto, fragmento, col) => {
      const i = texto.indexOf(fragmento);
      if (i < 0) throw new Error("Fragmento ausente en el ejemplo del corpus");
      return [R(texto.slice(0, i), mono), dif(fragmento, col), R(texto.slice(i + fragmento.length), mono)];
    };
    rotulo(s, "Dañina", X0, 2.18, 2, C.harm);
    txt(s, resaltar(D.ejemplo_didactico.harmful, "services that were never provided", C.harm),
      { x: X0, y: 2.44, w: XW, h: 0.95 });
    rotulo(s, "Benigna", X0, 3.48, 2, C.ben);
    txt(s, resaltar(D.ejemplo_didactico.benign, "services we provided", C.ben),
      { x: X0, y: 3.74, w: XW, h: 1.0 });
    // Longitud de la entrada completa (tokens medios, Llama V3).
    rotulo(s, "Entrada completa · longitud media en tokens · Llama V3", X0, 4.9, XW);
    const esc = (XW - 1.05) / tk.agent_harmful;
    const yb = [5.24, 5.74];
    txt(s, "Chat", { x: X0, y: yb[0], w: 1.0, h: 0.38, fontFace: F.demi, fontSize: 12, valign: "middle" });
    txt(s, "Agente", { x: X0, y: yb[1], w: 1.0, h: 0.38, fontFace: F.demi, fontSize: 12, valign: "middle", color: C.ag });
    const sis = tk.sysonly_harmful - tk.chat_harmful, her = tk.toolsonly_harmful - tk.chat_harmful;
    rect(s, pres, X0 + 1.05, yb[0], tk.chat_harmful * esc, 0.38, C.ink2);
    txt(s, `${Math.round(tk.chat_harmful)}`, { x: X0 + 1.1 + tk.chat_harmful * esc, y: yb[0], w: 1.2, h: 0.38, fontSize: 12, valign: "middle", color: C.ink2 });
    let xx = X0 + 1.05;
    [[sis, C.ag, "sistema"], [her, "E2B66A", "esquemas de herramientas"], [tk.agent_harmful - sis - her, C.ink2, ""]].forEach(([v, col, lab]) => {
      rect(s, pres, xx, yb[1], v * esc, 0.38, col);
      if (lab === "esquemas de herramientas") txt(s, `${lab} · +${Math.round(v)}`, { x: xx + 0.1, y: yb[1], w: v * esc - 0.2, h: 0.38, fontSize: 12, valign: "middle", color: C.ink });
      xx += v * esc;
    });
    txt(s, `${Math.round(tk.agent_harmful)}`, { x: xx - 0.6, y: yb[1] + 0.4, w: 0.6, h: 0.26, fontSize: 12, align: "right", color: C.ag });
    txt(s, `sistema +${Math.round(sis)}`, { x: X0 + 1.05, y: yb[1] + 0.4, w: 2.0, h: 0.26, fontSize: 11, color: C.ag });
    fuente(s, "Mismo sistema y catálogo en ambas ramas de cada formato (96/96 pares). Ejemplo: dataset_v2.jsonl → fraud_06. Barras: medias de 42 pares de Llama, results.json → prompt_tokens.");
    nota(s, {
      titulo: "El instrumento: mismo texto, distinto formato, dos intenciones", t: 70,
      guion: `El instrumento es un corpus contrastivo pareado. Cada instrucción base aparece en cuatro variantes: dañina o benigna, en chat o en agente.
Dos controles lo hacen funcionar. El texto de la solicitud es idéntico entre chat y agente; la comparación se hace dentro del par. Además, ambas ramas reciben las mismas herramientas y el mismo sistema dentro de cada formato.
Por entrada completa me refiero a todo lo que el modelo lee antes de responder, convertido a tokens: unidades de texto de su tokenizador. En agente contiene la solicitud, un mensaje de sistema que indica cómo usar herramientas y un catálogo con sus descripciones y parámetros. Ese catálogo no acredita permisos ni ejecuta funciones: aquí las herramientas no se ejecutan.
El ejemplo es fraud_06. Una solicitud pide facturar servicios que nunca se prestaron; la otra, servicios prestados. Se mantiene el tema, pero cambia la acción solicitada. Eso reduce el contraste temático; no demuestra por sí solo una separación pura de intención.
Las barras muestran la longitud media de los 42 pares de Llama, no la longitud de este ejemplo: la entrada agéntica es casi siete veces mayor, principalmente por el catálogo de herramientas.`,
      transicion: "Ese diseño se aplicó a un corpus que conviene describir con precisión.",
      fuentes: `Ejemplo literal: data/dataset_v2.jsonl, pair_id fraud_06 (variantes chat_harmful y chat_benign); datos_presentacion.json → ejemplo_didactico. El resaltado marca el contraste entre servicios no prestados y prestados. La dirección se estima con chat y se evalúa en ambos formatos. El ejemplo fraud_01 fue retirado de esta explicación por ambigüedad de autorización, sin modificar el corpus ni las métricas; se documenta en R13 y nota_autorizacion_etiquetas.md.
Tokens medios (42 pares, tokenizador de Llama‑3.3): chat dañina ${tk.chat_harmful.toFixed(1)}, agente dañina ${tk.agent_harmful.toFixed(1)}; solo sistema ${tk.sysonly_harmful.toFixed(1)}; solo herramientas ${tk.toolsonly_harmful.toFixed(1)} (results.json → prompt_tokens). Los incrementos "+${Math.round(sis)}" y "+${Math.round(her)}" son diferencias de esas medias. Simetría verificada: regenerar_avances.py comprueba texto idéntico entre formatos y mismo sistema/herramientas entre ramas en los 96 pares.`,
    });
  }

  // ------------------------------------------------------------ 7. Corpus
  {
    const s = pres.addSlide({ masterName: "CONTENIDO", sectionTitle: sec });
    nav(s, 1);
    s.addText("Un corpus contrastivo de 96 pares y tres grados de explicitud", { placeholder: "title" });
    const bloques = [["Explícito", 42], ["Contextual", 24], ["Apariencia legítima", 30]];
    const q = 0.25, pitch = 0.31, filas = 6;
    let x = MX;
    const y0 = 2.05;
    bloques.forEach(([nom, tot]) => {
      const cols = Math.ceil(tot / filas);
      for (let k = 0; k < tot; k++) {
        const c = Math.floor(k / filas), f = k % filas;
        const xx = x + c * pitch, yy = y0 + f * pitch;
        rect(s, pres, xx, yy, q, q, C.ink);
      }
      txt(s, [R(nom, { fontFace: F.demi, color: C.ink, breakLine: true }), R(`${tot} pares`, { color: C.ink2 })],
        { x, y: y0 + filas * pitch + 0.08, w: cols * pitch + 0.45, h: 0.62, fontSize: 12 });
      x += cols * pitch + 0.5;
    });
    txt(s, "Mistral y Qwen: 96 pares · Llama: 42 pares", { x: MX, y: 4.65, w: 6.6, h: 0.4, fontSize: 16, fontFace: F.head, color: C.ink });
    rotulo(s, "Modelos y familias", MX, 5.25, 6.4);
    const corr = [["Llama‑3.3‑70B · V3", "42 pares · ref. 76 · análisis geométrico y conductual"],
      ["Mistral‑7B · V2", "96 pares · ref. 18 · resultados estructurados"],
      ["Qwen3‑8B y 32B · V2", "96 pares · referencias 22 y 46"]];
    corr.forEach(([a, b], i) => txt(s, [R(a, { fontFace: F.demi, color: C.ink }), R("   " + b)],
      { x: MX, y: 5.55 + i * 0.4, w: 6.6, h: 0.36, fontSize: 12.5, color: C.ink2, valign: "middle" }));
    const datos = [["96 pares · 384 variantes", "7 categorías de daño; cada celda categoría × explicitud tiene entre 2 y 8 pares."],
      ["62 propios + 34 derivados de AgentHarm", "adaptados para el pareado: no es una evaluación del benchmark publicado."],
      ["Catálogo de 64 herramientas", "de 2 a 8 por tarea (media 3,4); el modelo recibe descripciones, sin ejecución."],
      ["Simetría léxica homogénea", "Jaccard mediano 0,408 · 0,413 · 0,422 por estrato (rango 0,014)."],
      ["La unidad es el par", "incluye una dañina y su benigna; el estrato describe al miembro dañino."],
      ["Un anotador", "con revisión asistida por un modelo (Claude); la segunda anotación independiente está pendiente."]];
    txt(s, datos.flatMap(([a, b], i) => [R(a, { fontFace: F.demi, color: C.ink, breakLine: true }),
      R(b, { color: C.ink2, breakLine: i < datos.length - 1, paraSpaceAfter: 11 })]),
    { x: 7.45, y: 1.9, w: W - MX - 7.45, h: 4.9, fontSize: 13 });
    fuente(s, "Composición: dataset_v2_report.json. Tamaños comprobados en las exportaciones results.json: Mistral/Qwen V2 y Llama V3. Detalle en anexo R1.");
    nota(s, {
      titulo: "Un corpus contrastivo de 96 pares y tres grados de explicitud", t: 45,
      guion: `El corpus ampliado tiene 96 pares y 384 variantes, siete categorías de daño y tres niveles de explicitud: 42 explícitos, 24 contextuales y 30 con apariencia legítima. La explicitud describe cómo se presenta la solicitud dañina etiquetada; no cambia la etiqueta de su contraparte benigna. Alegar una auditoría o una autorización no basta para decidir la etiqueta. El emparejamiento léxico, medido con Jaccard, es homogéneo entre estratos; eso controla una propiedad del pareado, pero no valida por sí solo las etiquetas.
Mistral y los dos Qwen tienen ejecuciones verificadas de 96 pares; sus resultados completos sustentan las comparativas. Llama‑3.3‑70B tiene 42 pares y aporta el análisis detallado de representación y los ensayos conductuales. No agrego tamaños distintos como si fueran una sola muestra.
Y la unidad de análisis es el par: sus cuatro variantes no son cuatro observaciones independientes.`,
      transicion: "¿Qué se mide sobre cada una de esas variantes?",
      fuentes: `Corpus ampliado: dataset_v2_report.json → pares 96, entradas 384, por_estrato {explicit 42, contextual 24, framed 30}, por_fuente {propio 62, agentharm 34}. Herramientas: 64 distintas, 2–8 por par (tesis, sección 3.2). Jaccard mediano por estrato: tesis, tabla 3.3. Ejecuciones verificadas en results.json: Llama V3, 42 pares y referencia 76; Mistral V2, 96 y referencia 18; Qwen3‑8B V2, 96 y referencia 22; Qwen3‑32B V2, 96 y referencia 46. Mistral V3, 42 pares y referencia 30, se conserva para ensayos complementarios.`,
    });
  }

  // ------------------------------------------------------------ 8. Flujo residual
  {
    const s = pres.addSlide({ masterName: "CONTENIDO", sectionTitle: sec });
    nav(s, 1);
    s.addText("Qué se mide: el flujo residual en la última posición", { placeholder: "title" });
    rotulo(s, "Cada capa suma su contribución", MX, 1.88, 6);
    eq(s, "residual", MX, 2.2, 6.2, 0.6, 22);
    txt(s, [R("atención: ", { fontFace: F.demi }), R("mezcla posiciones   ·   "), R("perceptrón: ", { fontFace: F.demi }), R("actúa en cada posición")],
      { x: MX, y: 2.85, w: 6.2, h: 0.3, fontSize: 12.5, color: C.mute });
    eq(s, "desplegado", MX, 3.2, 6.2, 0.85, 22);
    txt(s, [R("Una estructura aditiva: tiene sentido buscar un concepto como una "), R("dirección", { fontFace: F.demi }), R(" de "), R("ℝ", { fontFace: F.math }), R("d", { fontFace: F.math, superscript: true }), R(".")],
      { x: MX, y: 4.2, w: 6.2, h: 0.35, fontSize: 14, color: C.ink2 });
    rotulo(s, "La atención solo mira hacia atrás", MX, 4.7, 6);
    eq(s, "mascara", MX, 5.0, 6.4, 1.0, 19);
    txt(s, [R("Por inducción sobre las capas, la posición "), ...mx({}, m.i("i")), R(" depende solo de "), ...mx({}, m.i("t"), m.sub("1"), m.u(", …, "), m.i("t"), m.sub("i")), R(".")],
      { x: MX, y: 6.15, w: 6.2, h: 0.35, fontSize: 14, color: C.ink2 });
    // Diagrama de posiciones
    const X0 = 7.1, XW = W - MX - X0;
    rotulo(s, "Dónde se lee: la última posición, antes de generar", X0, 1.88, XW);
    const fila = (y, partes) => {
      let x = X0;
      partes.forEach(([w, col, t, tc]) => {
        rect(s, pres, x, y, w, 0.42, col);
        if (t) txt(s, t, { x, y, w, h: 0.42, align: "center", valign: "middle", fontSize: 11, color: tc || C.ink });
        x += w + 0.04;
      });
      return x - 0.04;
    };
    txt(s, "chat", { x: X0, y: 2.22, w: 1, h: 0.25, fontSize: 11, color: C.mute });
    const finC = fila(2.48, [[0.42, "C9CED6", "sist."], [1.55, C.mist, "instrucción"], [0.24, C.ink, "", C.paper]]);
    txt(s, "agente", { x: X0, y: 3.1, w: 1, h: 0.25, fontSize: 11, color: C.ag });
    const finA = fila(3.36, [[0.42, "C9CED6", "sist."], [2.9, C.agL, "esquemas de herramientas"], [1.0, C.mist, "instrucción"], [0.24, C.ink, "", C.paper]]);
    // Cono de influencia hacia el último token de la entrada agéntica.
    const xn = finA - 0.12, yn = 3.36;
    [X0 + 0.2, X0 + 1.2, X0 + 2.2, X0 + 3.2, X0 + 4.0, X0 + 4.6].forEach((x) => seg(s, pres, x, 3.36 + 0.42 + 0.08, xn, 3.36 + 0.42 + 0.5, C.ag, 0.75));
    dot(s, pres, xn, 3.36 + 0.42 + 0.5, 0.09, C.ink);
    txt(s, [R("posición "), ...mx({}, m.i("n")), R(": último token de la entrada completa")],
      { x: X0, y: 4.35, w: XW, h: 0.3, fontSize: 12, color: C.ag, align: "right" });
    const razones = [
      [R("Puede atender a toda la entrada: "), R("sistema, catálogo, solicitud y delimitadores.", { color: C.ink2, fontFace: F.body })],
      [R("Desde ahí se decide la respuesta: "), R("el primer token —rechazo o cumplimiento— sale de ", { color: C.ink2, fontFace: F.body }), ...mx({ color: C.ink2 }, m.b("x"), m.sub("n")), R(" en la última capa.", { color: C.ink2, fontFace: F.body })],
      [R("Mismo papel en ambos formatos: "), R("promediar posiciones mezclaría el efecto con la longitud.", { color: C.ink2, fontFace: F.body })]];
    razones.forEach((rs, i) => {
      txt(s, `${i + 1}`, { x: X0, y: 4.8 + i * 0.55, w: 0.3, h: 0.5, fontFace: F.med, fontSize: 18, color: C.ag });
      txt(s, rs.map((r) => R(r.text, Object.assign({ fontFace: F.demi, color: C.ink }, r.options))), { x: X0 + 0.35, y: 4.8 + i * 0.55, w: XW - 0.35, h: 0.52, fontSize: 13 });
    });
    fuente(s, "Llama‑3.3‑70B: L = 80 capas, d = 8.192; se leen 20 capas (0, 4, …, 76). La posición n puede ser un delimitador de plantilla y no la última palabra del usuario. Tesis, sección 3.3.1.");
    nota(s, {
      titulo: "Qué se mide: el flujo residual en la última posición", t: 55,
      guion: `¿Qué medimos exactamente? Estos modelos son transformers decodificadores. Cada posición mantiene un vector, el flujo residual, y cada capa le suma la salida de la atención y la del perceptrón. Si se despliega la recurrencia, el flujo es una suma de contribuciones; por eso tiene sentido buscar un concepto como una dirección en ese espacio.
Leo en una sola posición: el último token de la entrada completa, antes de generar. La entrada es la secuencia formada por el mensaje de sistema, el catálogo de herramientas, la solicitud y los delimitadores de la plantilla. Por la máscara causal, esa posición puede atender a todo ese contexto. Desde ella se obtiene el primer token de la respuesta. Cumple el mismo papel en ambos formatos; promediar posiciones mezclaría el efecto con la longitud, que en agente es casi siete veces mayor.
En Llama leo 20 de sus 80 capas, en un espacio de 8.192 dimensiones.`,
      transicion: "Sobre esos vectores se define la dirección de rechazo.",
      fuentes: "Ecuaciones: tesis, ecuaciones (3.2) a (3.4). Dimensiones: tesis, tabla 3.4; results.json → model_info (num_layers 80, hidden_size 8192). Capas leídas: config.layers = 0,4,…,76. El diagrama de posiciones es esquemático: las anchuras no son proporcionales a los tokens.",
    });
  }

  // ------------------------------------------------------------ 9. Dirección y proyección
  {
    const s = pres.addSlide({ masterName: "CONTENIDO", sectionTitle: sec });
    nav(s, 1);
    s.addText("La dirección de rechazo es una diferencia de medias", { placeholder: "title" });
    const px = MX, py = 1.85, pw = 5.3, ph = 4.55;
    rrect(s, pres, px, py, pw, ph, C.mist);
    const rnd = azar(7);
    const mb = [1.6, 5.3], mh = [4.6, 3.2];
    const nube = (c, col) => { for (let k = 0; k < 22; k++) dot(s, pres, c[0] + 0.36 * rnd.n(), c[1] + 0.3 * rnd.n(), 0.1, col, { transparency: 25 }); };
    nube(mb, C.ben); nube(mh, C.harm);
    // eje de la dirección, prolongado
    const ux = mh[0] - mb[0], uy = mh[1] - mb[1], nrm = Math.hypot(ux, uy), ex = ux / nrm, ey = uy / nrm;
    seg(s, pres, mb[0] - ex * 1.0, mb[1] - ey * 1.0, mh[0] + ex * 0.75, mh[1] + ey * 0.75, "8D949E", 1, { line: { dashType: "dash" } });
    flecha(s, pres, mb[0], mb[1], mh[0] - ex * 0.08, mh[1] - ey * 0.08, C.ink, 2.25);
    [mb, mh].forEach((c) => s.addShape(pres.shapes.OVAL, { x: c[0] - 0.11, y: c[1] - 0.11, w: 0.22, h: 0.22, fill: { color: C.paper }, line: { color: C.ink, width: 1.75 } }));
    txt(s, mx({ fontSize: 16, color: C.ben }, m.b("μ"), m.sub("b")), { x: mb[0] + 0.62, y: mb[1] + 0.32, w: 0.5, h: 0.34, align: "center", valign: "middle", fill: { color: C.mist } });
    txt(s, mx({ fontSize: 16, color: C.harm }, m.b("μ"), m.sub("h")), { x: mh[0] - 0.25, y: mh[1] - 0.92, w: 0.5, h: 0.34, align: "center", valign: "middle", fill: { color: C.mist } });
    const nx = -ey, ny = ex; // normal hacia abajo a la derecha
    const lr = [mb[0] + 0.72 * ux - 0.3 * nx, mb[1] + 0.72 * uy - 0.3 * ny];
    txt(s, mx({ fontSize: 16 }, m.b("r")), { x: lr[0] - 0.15, y: lr[1] - 0.17, w: 0.3, h: 0.34, align: "center" });
    // punto de ejemplo y su proyección ortogonal
    const xp = [2.2, 2.6];
    const t = (xp[0] - mb[0]) * ex + (xp[1] - mb[1]) * ey, fx = mb[0] + t * ex, fy = mb[1] + t * ey;
    seg(s, pres, xp[0], xp[1], fx, fy, C.ink, 1, { line: { dashType: "sysDot" } });
    dot(s, pres, xp[0], xp[1], 0.16, C.ink);
    dot(s, pres, fx, fy, 0.12, C.ag);
    txt(s, mx({ fontSize: 16 }, m.b("x")), { x: xp[0] - 0.35, y: xp[1] - 0.38, w: 0.3, h: 0.3 });
    txt(s, mx({ fontSize: 15, color: C.ag }, m.i("p"), m.u(" = "), m.b("x"), m.u("·"), m.hat("r")), { x: fx + 0.25 * nx, y: fy + 0.25 * ny - 0.05, w: 1.5, h: 0.32 });
    txt(s, [R("Esquema pedagógico en 2 dimensiones; las activaciones reales viven en "), R("ℝ", { fontFace: F.math }), R("8.192", { superscript: true }), R(".")],
      { x: px, y: py + ph + 0.08, w: pw, h: 0.3, fontSize: 11, color: C.mute });
    const X0 = 6.3, XW = W - MX - X0;
    eq(s, "direccion", X0, 1.85, XW, 0.95, 22);
    txt(s, "Como los conjuntos están pareados, es la media de las diferencias dentro de cada par: cada par contrasta dos instrucciones del mismo tema.",
      { x: X0, y: 2.85, w: XW, h: 0.55, fontSize: 13.5, color: C.ink2 });
    eq(s, "unitaria", X0, 3.45, 3.0, 0.9, 22);
    eq(s, "proyeccion", X0, 4.38, XW, 0.55, 22);
    txt(s, "Mezcla la orientación de la activación, que interesa, con su norma, que no.",
      { x: X0, y: 4.98, w: XW, h: 0.3, fontSize: 13.5, color: C.ink2 });
    rrect(s, pres, X0, 5.45, XW, 1.3, C.mist);
    txt(s, [R("¿Por qué no el discriminante de Fisher?  ", { fontFace: F.demi, color: C.ink }),
      R("Con 2n = 84 activaciones en d = 8.192 dimensiones, la covarianza dentro de clases tiene rango ≤ 82 y no se invierte. Con la covarianza regularizada, "),
      ...mx({}, m.u("𝚺"), m.sub("W", false), m.u(" + "), m.i("λ"), m.u("𝐈")), R(", la dirección converge a "), ...N.rhat(), R(" cuando "), ...mx({}, m.i("λ"), m.u(" → ∞")),
      R(": es el discriminante que renuncia a estimar la covarianza.")],
    { x: X0 + 0.2, y: 5.55, w: XW - 0.4, h: 1.1, fontSize: 13, color: C.ink2 });
    fuente(s, "Tesis, ecuaciones (3.5) y (3.7); Arditi et al. (2024). Esquema generado para la explicación: no son datos medidos.");
    nota(s, {
      titulo: "La dirección de rechazo es una diferencia de medias", t: 60,
      guion: `Sobre esas activaciones, la dirección de rechazo es la diferencia entre el centroide de las instrucciones dañinas y el de las benignas en chat, normalizada a norma uno. Como los conjuntos están pareados, también es la media de las diferencias dentro de cada par.
La proyección de una activación sobre ese eje es su producto escalar con r gorro: en el esquema, la sombra del punto x sobre la recta. La segunda igualdad advierte algo que importa después: la proyección mezcla la orientación, que es lo que nos interesa, con la norma del vector, que no.
¿Por qué medias y no Fisher? El eje óptimo de Fisher invierte la covarianza dentro de clases, pero con 84 activaciones en 8.192 dimensiones esa matriz tiene rango a lo sumo 82. Si se regulariza y lambda tiende a infinito, se recupera la diferencia de medias: es el discriminante que renuncia a estimar la covarianza.`,
      transicion: "Con esa dirección defino las cantidades que comparan los formatos.",
      fuentes: "Tesis, sección 3.3.1: ecuación (3.5) de la dirección; discusión del discriminante de Fisher y regularización; ecuación (3.7) de la proyección. Rango: 2n − 2 = 82 con n = 42 pares. El panel izquierdo es un esquema con puntos generados (semilla fija), no datos.",
    });
  }

  // ------------------------------------------------------------ 10. Desplazamiento y control benigno
  {
    const s = pres.addSlide({ masterName: "CONTENIDO", sectionTitle: sec });
    nav(s, 1);
    s.addText("Cuánto cambia la proyección y si el cambio es propio del daño", { placeholder: "title" });
    rotulo(s, "Cambio dentro del par", MX, 1.88, 5);
    eq(s, "delta", MX, 2.15, 6.0, 0.6, 20);
    eq(s, "gap", MX, 2.78, 6.0, 0.9, 20);
    txt(s, [...N.dph(), R(" > 0: la dirección responde con menor intensidad en agente.")],
      { x: MX, y: 3.66, w: 6.0, h: 0.32, fontSize: 14, color: C.ink2 });
    rotulo(s, "Control benigno: la interacción", MX, 4.15, 5);
    eq(s, "interaccion", MX, 4.42, 6.0, 0.8, 20);
    txt(s, [R("El brazo benigno atraviesa el mismo cambio de formato: solo la interacción se atribuye al daño. Hipótesis de trabajo: "), ...N.d2p(), R(" > 0.")],
      { x: MX, y: 5.24, w: 6.0, h: 0.6, fontSize: 14, color: C.ink2 });
    eq(s, "prueba_t", MX, 5.92, 4.1, 0.68, 16);
    txt(s, [...N.dz(), R(", y no "), ...mx({}, m.i("t")), R(", es comparable entre corpus de distinto tamaño.")],
      { x: 4.8, y: 5.98, w: 2.0, h: 0.62, fontSize: 12, color: C.mute });
    // Tres esquemas
    const X0 = 7.05, XW = W - MX - X0;
    const casos = [["Debilitamiento específico", [" > 0", " ≈ 0"], [3.0, 1.1], [0.0, 0.0]],
      ["Desplazamiento rígido", [" ≈ 0", " ≈ 1"], [3.0, 1.1], [0.0, -1.9]],
      ["Sentidos opuestos", [" > 0", " < 0"], [3.0, 1.1], [0.0, 0.6]]];
    casos.forEach(([tit, res, h, b], i) => {
      const y = 1.88 + i * 1.58;
      rrect(s, pres, X0, y, XW, 1.45, C.mist);
      txt(s, [R(tit, { fontFace: F.demi, breakLine: true }), ...mx({ color: C.ink2 }, m.u("Δ"), m.sup("2"), m.i("p"), m.u(res[0] + "   ·   "), m.i("φ"), m.u(res[1]))],
        { x: X0 + 0.2, y: y + 0.15, w: 2.0, h: 1.1, fontSize: 13, valign: "middle" });
      const a0 = X0 + 2.35, a1 = X0 + XW - 0.25, ya = y + 0.73;
      const esc = (v) => a0 + ((v + 2.4) / 6.0) * (a1 - a0);
      seg(s, pres, a0, ya, a1, ya, "AEB4BC", 0.75);
      const fl = (v0, v1, yy, col) => {
        dot(s, pres, esc(v0), yy, 0.17, col);
        const quieto = Math.abs(v1 - v0) < 0.2, dr = quieto ? 0.3 : 0.17;
        s.addShape(pres.shapes.OVAL, { x: esc(v1) - dr / 2, y: yy - dr / 2, w: dr, h: dr, fill: quieto ? { type: "none" } : { color: C.paper }, line: { color: col, width: 1.75 } });
        if (Math.abs(v1 - v0) > 0.2) flecha(s, pres, esc(v0) + (v1 < v0 ? -0.12 : 0.12), yy, esc(v1) + (v1 < v0 ? 0.13 : -0.13), yy, col, 1.5);
      };
      fl(h[0], h[1], ya - 0.27, C.harm);
      fl(b[0], b[1], ya + 0.27, C.ben);
    });
    txt(s, "Esquemas: punto lleno = chat; punto hueco = agente. Rojo, dañinas; azul, benignas.", { x: X0, y: 6.62, w: XW, h: 0.3, fontSize: 11, color: C.mute });
    fuente(s, "Tesis, ecuaciones (3.8) a (3.10).");
    nota(s, {
      titulo: "Cuánto cambia la proyección y si el cambio es propio del daño", t: 55,
      guion: `Con esa dirección defino tres cantidades. La primera es el desplazamiento: para cada instrucción dañina, su proyección en chat menos su proyección en agente. Su promedio es delta p sub h. Si es positivo, la dirección responde con menos intensidad cuando el modelo actúa.
Eso solo no basta, porque hay tres explicaciones posibles, a la derecha: un debilitamiento específico del daño, un corrimiento rígido de todo el espacio, o movimientos en sentidos opuestos. El brazo benigno, que atraviesa exactamente el mismo cambio de formato, las distingue. La cantidad propia del daño es la interacción, delta cuadrado p: el desplazamiento dañino menos el benigno. Phi indica qué fracción del efecto reproduce el benigno.
Para la inferencia reporto d sub z, el tamaño de efecto pareado, que, a diferencia de t, es comparable entre corpus de distinto tamaño.`,
      transicion: "Falta una pregunta: ¿cambia la información o solo la escala?",
      fuentes: "Tesis, sección 3.3.1, ecuaciones (3.8) desplazamiento intra‑par, (3.9) prueba t y d_z, (3.10) interacción y fracción común. Los tres paneles son esquemas, no datos.",
    });
  }

  // ------------------------------------------------------------ 11. Escala y orden
  {
    const s = pres.addSlide({ masterName: "CONTENIDO", sectionTitle: sec });
    nav(s, 1);
    s.addText("Un cambio de escala no es una pérdida de discriminación", { placeholder: "title" });
    rotulo(s, "Unidades estandarizadas", MX, 1.88, 5);
    eq(s, "estandar", MX, 2.15, 5.6, 0.85, 22);
    txt(s, [...mx({}, m.i("S")), R(": distancia entre las proyecciones medias de las dos clases dentro de un formato; "), ...mx({}, m.i("s")), R(": su desviación estándar agrupada.")],
      { x: MX, y: 3.05, w: 5.6, h: 0.55, fontSize: 13.5, color: C.ink2 });
    rotulo(s, "Caso límite: el formato solo reescala", MX, 3.75, 5.6);
    eq(s, "afin", MX, 4.02, 5.6, 0.45, 19);
    eq(s, "afin2", MX, 4.48, 5.6, 0.45, 19);
    // esquema de campanas: chat amplio, agente comprimido por a
    const X0 = 6.7, XW = W - MX - X0;
    rrect(s, pres, X0, 1.88, XW, 3.05, C.mist);
    const campana = (mu, sd, y0, alto, x0, x1, lo, hi, col) => {
      const pts = [];
      for (let k = 0; k <= 60; k++) {
        const v = mu - 3.2 * sd + (6.4 * sd * k) / 60;
        const xx = x0 + ((v - lo) / (hi - lo)) * (x1 - x0);
        pts.push([xx, y0 - alto * Math.exp(-0.5 * ((v - mu) / sd) ** 2)]);
      }
      curva(s, pres, pts, { fill: col, transparency: 35, line: col, width: 1.25, cerrar: true });
    };
    const lo = -6, hi = 16, x0 = X0 + 1.05, x1 = X0 + XW - 0.25, a = 0.31;
    const esc = (v) => x0 + ((v - lo) / (hi - lo)) * (x1 - x0);
    [[3.05, 1, "chat"], [4.45, a, "agente"]].forEach(([y, k, nom]) => {
      seg(s, pres, x0, y, x1, y, "AEB4BC", 0.75);
      txt(s, nom, { x: X0 + 0.2, y: y - 0.3, w: 0.8, h: 0.28, fontFace: F.demi, fontSize: 12, color: nom === "agente" ? C.ag : C.ink });
      campana(0, 2.2 * k, y, 0.62 / Math.sqrt(k) * 0.55, x0, x1, lo, hi, C.ben);
      campana(10 * k, 2.2 * k, y, 0.62 / Math.sqrt(k) * 0.55, x0, x1, lo, hi, C.harm);
      // separación S
      seg(s, pres, esc(0), y + 0.14, esc(10 * k), y + 0.14, C.ink, 1, { line: { beginArrowType: "arrow", endArrowType: "arrow" } });
      txt(s, mx({ fontSize: 13 }, m.i("S"), m.sub(nom === "chat" ? "chat" : "ag", false)), { x: esc(5 * k) - 0.5, y: y + 0.16, w: 1.0, h: 0.28, align: "center" });
    });
    txt(s, [R("Si "), ...mx({}, m.i("a")), R(" = 0,31: la separación cruda cae 69 %, pero "), ...N.d(), R(" y el AUROC no cambian.")],
      { x: X0 + 0.2, y: 1.98, w: XW - 0.4, h: 0.32, fontSize: 12.5, color: C.ink2 });
    // AUROC
    rotulo(s, "Unidades de rango", MX, 5.15, 5);
    eq(s, "auroc", MX, 5.42, 7.6, 0.95, 18);
    txt(s, "Probabilidad de que una dañina tomada al azar proyecte más que una benigna: 1 separa perfectamente; 0,5 es azar. Solo depende del orden, así que ningún reescalamiento creciente lo altera.",
      { x: 8.45, y: 5.18, w: W - MX - 8.45, h: 1.25, fontSize: 13, color: C.ink2 });
    fuente(s, "Tesis, ecuaciones (3.11), (3.12) y (3.15). Regla de lectura: solo la pérdida que sobrevive en d y en el AUROC cuenta como pérdida de margen. Campanas: esquema.");
    nota(s, {
      titulo: "Un cambio de escala no es una pérdida de discriminación", t: 60,
      guion: `La segunda pregunta es si el cambio es de información o solo de escala. Si el formato agéntico simplemente comprimiera todas las proyecciones por un factor a, la distancia cruda entre clases caería en una fracción uno menos a, tan grande como se quiera, sin que se perdiera capacidad de separar. Es lo que muestra el esquema.
Por eso expreso el efecto en unidades estandarizadas: d, la separación dividida por la desviación agrupada, que en ese caso límite no cambia, y kappa, que mide la compresión.
Y uso el AUROC: la probabilidad de que una instrucción dañina tomada al azar proyecte más que una benigna. Vale uno si separa perfectamente y 0,5 si es azar; como solo depende del orden, ningún reescalamiento lo altera.
La regla de lectura es esta: solo la pérdida que sobrevive en d y en el AUROC cuenta como pérdida de margen discriminativo.`,
      transicion: "Todo esto exige un cuidado estadístico: no evaluar con los mismos datos con que se estima.",
      fuentes: "Tesis, sección 3.3.1 (invariancia de escala, ecuaciones 3.11–3.12, caso afín) y sección 3.4.1 (AUROC, ecuación 3.15). El valor a = 0,31 del esquema coincide con el κ observado en Llama (0,309) solo como ilustración; las campanas no son datos.",
    });
  }

  // ------------------------------------------------------------ 12. Validación
  {
    const s = pres.addSlide({ masterName: "CONTENIDO", sectionTitle: sec });
    nav(s, 1);
    s.addText("La muestra no puede evaluarse a sí misma", { placeholder: "title" });
    rotulo(s, "El riesgo: circularidad en alta dimensión", MX, 1.88, 5.4);
    eq(s, "sesgo", MX, 2.15, 5.3, 0.95, 22);
    txt(s, [R("La norma estimada está sesgada hacia arriba; la separación medida con los mismos datos hereda el sesgo y la condición agéntica no. "),
      R("En alta dimensión, un AUROC de entrenamiento perfecto no prueba generalización: ", { fontFace: F.demi, color: C.ink }),
      R("la sonda logística alcanza 1 en muestra completa.")],
    { x: MX, y: 3.15, w: 5.4, h: 1.15, fontSize: 13.5, color: C.ink2 });
    const X0 = 6.45, XW = W - MX - X0;
    rotulo(s, "Cross‑fitting por pares · K = 5 · ejemplo de Llama V3", X0, 1.88, XW);
    const rnd = azar(42);
    const orden = Array.from({ length: 42 }, (_, i) => i).sort(() => rnd.u() - 0.5);
    const fold = new Array(42);
    orden.forEach((p, k) => { fold[p] = k % 5; });
    const pitch = (XW - 0.55) / 42, tw = pitch * 0.78;
    for (let f = 0; f < 5; f++) {
      const y = 2.22 + f * 0.24;
      txt(s, `${f + 1}`, { x: X0, y, w: 0.3, h: 0.18, fontSize: 10, color: C.mute, valign: "middle" });
      for (let p = 0; p < 42; p++) rect(s, pres, X0 + 0.4 + p * pitch, y, tw, 0.18, fold[p] === f ? C.ink : "D9DDE2");
    }
    txt(s, [R("■", { color: C.ink }), R(" pares evaluados en el fold   "), R("■", { color: "C9CED6" }), R(" pares que estiman la dirección")],
      { x: X0 + 0.4, y: 3.46, w: XW - 0.4, h: 0.26, fontSize: 11, color: C.mute });
    eq(s, "crossfit", X0 + 0.4, 3.78, 4.0, 0.5, 18);
    txt(s, "La dirección que puntúa un par nunca lo vio; sus cuatro variantes usan la misma.", { x: X0 + 0.4, y: 4.28, w: XW - 0.4, h: 0.3, fontSize: 12.5, color: C.ink2 });
    const cards = [
      ["Selección ≠ inferencia · Llama V3", "Partición documentada: 21 pares eligen la capa y otros 21 producen la prueba. Bonferroni: 0,05 / 20 = 0,0025. Las 200 particiones describen estabilidad; no son muestras nuevas."],
      ["Pruebas pareadas · Llama V3", "Permutación de signos (10.000; resolución ≈ 10⁻⁴), t pareada con tamaño de efecto pareado, bootstrap por pares y prueba de signos exacta, que depende del eje usado."],
      ["Referencias nulas", "Etiquetas barajadas como nulo empírico; varianza de la validación repetida corregida (Nadeau y Bengio, 2003); optimismo dentro frente a fuera de muestra, reportado."]];
    cards.forEach(([t, b], i) => {
      const x = MX + i * 4.12;
      rrect(s, pres, x, 4.8, 3.9, 1.95, C.mist);
      txt(s, [R(t, { fontFace: F.demi, color: C.ink, breakLine: true }), R(b, { color: C.ink2 })], { x: x + 0.2, y: 4.92, w: 3.5, h: 1.75, fontSize: 12.5 });
    });
    rect(s, pres, MX + 0.2, 6.3, 1.72, 0.28, C.ink);
    rect(s, pres, MX + 1.96, 6.3, 1.72, 0.28, C.ag);
    txt(s, "21 · selección", { x: MX + 0.2, y: 6.3, w: 1.72, h: 0.28, fontSize: 11, color: C.paper, align: "center", valign: "middle" });
    txt(s, "21 · inferencia", { x: MX + 1.96, y: 6.3, w: 1.72, h: 0.28, fontSize: 11, color: C.ink, align: "center", valign: "middle" });
    fuente(s, "Tesis, ecuaciones (3.6) y (3.16) y secciones 3.4.1–3.4.2. Controles detallados: Llama V3, 42 pares. Folds del diagrama: asignación ilustrativa con semilla fija.");
    nota(s, {
      titulo: "La muestra no puede evaluarse a sí misma", t: 60,
      guion: `Queda el problema estadístico central: estimar la dirección y evaluarla con los mismos datos. La norma de la dirección estimada está sesgada hacia arriba en un término que crece con la dimensión, y la separación medida con esos datos hereda el sesgo. En alta dimensión un clasificador puede separar incluso ruido: la sonda logística logra AUROC uno en entrenamiento, y eso no prueba generalización.
Por eso evalúo la detección fuera de muestra. Divido los pares en cinco folds; la dirección que puntúa un par se estimó sin él, y sus cuatro variantes usan la misma dirección.
En Llama V3, la exportación documenta una partición aparte: 21 pares eligen la capa y otros 21 hacen la inferencia, con corrección de Bonferroni. Sus pruebas respetan el pareado —permutación de signos, bootstrap por pares y prueba de signos— y comparo contra etiquetas barajadas. Las comparativas V2 conservan las estimaciones exportadas y su versión se indica por separado.`,
      transicion: "Con este método, paso a los resultados. Empiezo por lo que se ve desde afuera.",
      fuentes: "Sesgo de la norma: tesis, ecuación (3.6). Cross‑fitting por pares: sección 3.4.1, ecuación (3.16). Inferencia y selección separada (21 + 21, Bonferroni por 20 capas, 200 particiones): sección 3.4.2. AUROC de entrenamiento de la sonda logística = 1: results.json → linear_aurocs_in_sample['76']. Nadeau y Bengio (2003). La barra 21|21 de la tarjeta es esquemática.",
    });
  }
};
