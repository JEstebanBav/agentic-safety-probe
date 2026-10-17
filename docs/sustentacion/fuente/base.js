// Sistema editorial sobrio: papel cálido, tipografía de libro y figuras abiertas.
// Se conserva el código semántico de los gráficos; las antiguas tarjetas se
// sustituyen por separaciones tipográficas y reglas, sin paneles decorativos.

const W = 13.333, H = 7.5, MX = 0.6;

const C = {
  ink: "26282A", ink2: "4C4D4A", mute: "77766F", rule: "D1CEC5", mist: "F0EDE6", paper: "FAF9F6",
  harm: "A14F3D", harmL: "F0DED7", ben: "416879", benL: "DFE8E9", ag: "9B743E", agL: "EEE5D4",
  dark: "FAF9F6", dark2: "FAF9F6", darkTxt: "26282A", darkMute: "4C4D4A", darkRule: "D1CEC5",
};

const F = {
  head: "Palatino Linotype", body: "Calibri", demi: "Calibri",
  med: "Calibri", math: "Cambria Math", mono: "Consolas",
};

const THEME = {
  name: "Direccion de rechazo",
  headFontFace: F.head, bodyFontFace: F.body,
  colors: {
    dk1: C.ink, lt1: C.paper, dk2: C.dark, lt2: C.mist,
    accent1: C.harm, accent2: C.ben, accent3: C.ag, accent4: C.ink2, accent5: C.mute, accent6: C.rule,
    hlink: C.ben, folHlink: C.ink2,
  },
};

const SECCIONES = ["Problema", "Método", "Resultados", "Alcance"];

function definirLayouts(pres) {
  pres.defineSlideMaster({
    title: "CONTENIDO", background: { color: C.paper },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: MX, y: 0.62, w: W - 2 * MX, h: 0.95,
        fontFace: F.head, fontSize: 28, color: C.ink, valign: "top", margin: 0, align: "left" }, text: "" } },
    ],
    slideNumber: { x: W - MX - 0.6, y: 7.02, w: 0.6, h: 0.3, fontFace: F.body, fontSize: 10, color: C.mute, align: "right" },
  });
  pres.defineSlideMaster({
    title: "RESPALDO", background: { color: C.paper },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: MX, y: 0.62, w: W - 2 * MX, h: 0.95,
        fontFace: F.head, fontSize: 26, color: C.ink, valign: "top", margin: 0, align: "left" }, text: "" } },
    ],
    slideNumber: { x: W - MX - 0.6, y: 7.02, w: 0.6, h: 0.3, fontFace: F.body, fontSize: 10, color: C.mute, align: "right" },
  });
  pres.defineSlideMaster({
    title: "OSCURO", background: { color: C.dark },
    slideNumber: { x: W - MX - 0.6, y: 7.02, w: 0.6, h: 0.3, fontFace: F.body, fontSize: 10, color: C.darkMute, align: "right" },
  });
  pres.defineSlideMaster({ title: "PORTADA", background: { color: C.paper } });
}

// ------------------------------------------------------------------ texto
function R(text, o = {}) {
  return { text, options: { ...(o.fontFace === F.demi ? { bold: true } : {}), ...o } };
}

// Párrafo: aplica las opciones de párrafo a la primera corrida y cierra con salto de línea.
function P(runs, po = {}, ultimo = false) {
  const rs = (Array.isArray(runs) ? runs : [R(runs)]).map((r) => (typeof r === "string" ? R(r) : { text: r.text, options: { ...r.options } }));
  Object.assign(rs[0].options, po);
  if (!ultimo) rs[rs.length - 1].options.breakLine = true;
  return rs;
}

function parrafos(lista) {
  // lista de [runs, opcionesDePárrafo]
  return lista.flatMap(([runs, po], i) => P(runs, po || {}, i === lista.length - 1));
}

function txt(slide, content, o = {}) {
  const base = { isTextBox: true, fontFace: F.body, fontSize: 15, color: C.ink, margin: 0, valign: "top", align: "left", paraSpaceAfter: 0 };
  if (o.fontFace === F.demi) base.bold = true;
  slide.addText(content, Object.assign(base, o));
}

// Notación en línea con Cambria Math: letras en itálica matemática, vectores en negrita itálica.
const MI = (c) => {
  const o = c.codePointAt(0);
  if (c >= "a" && c <= "z") return c === "h" ? "ℎ" : String.fromCodePoint(0x1D44E + o - 97);
  if (c >= "A" && c <= "Z") return String.fromCodePoint(0x1D434 + o - 65);
  if (o >= 0x3B1 && o <= 0x3C9) return String.fromCodePoint(0x1D6FC + o - 0x3B1);
  return c;
};
const MB = (c) => {
  const o = c.codePointAt(0);
  if (c >= "a" && c <= "z") return String.fromCodePoint(0x1D482 + o - 97);
  if (c >= "A" && c <= "Z") return String.fromCodePoint(0x1D468 + o - 65);
  if (o >= 0x3B1 && o <= 0x3C9) return String.fromCodePoint(0x1D736 + o - 0x3B1);
  return c;
};
const m = {
  i: (s) => ({ t: [...s].map(MI).join("") }),
  b: (s) => ({ t: [...s].map(MB).join("") }),
  u: (s) => ({ t: s }),
  sub: (s, it = true) => ({ t: it ? [...s].map(MI).join("") : s, sub: true }),
  sup: (s) => ({ t: s, sup: true }),
  supi: (s) => ({ t: [...s].map(MI).join(""), sup: true }),
  hat: (s) => ({ t: MB(s) + "̂" }),
};
// mx(opciones, ...piezas) -> corridas de texto con la fuente matemática
function mx(o, ...piezas) {
  return piezas.map((p) => R(p.t, Object.assign({ fontFace: F.math }, o, p.sub ? { subscript: true } : {}, p.sup ? { superscript: true } : {})));
}
// Atajos de notación frecuente
const N = {
  dph: (o = {}) => mx(o, m.u("Δ"), m.i("p"), m.sub("h")),
  dpb: (o = {}) => mx(o, m.u("Δ"), m.i("p"), m.sub("b")),
  d2p: (o = {}) => mx(o, m.u("Δ"), m.sup("2"), m.i("p")),
  rhat: (o = {}) => mx(o, m.hat("r")),
  kappa: (o = {}) => mx(o, m.i("κ")),
  phi: (o = {}) => mx(o, m.i("φ")),
  dz: (o = {}) => mx(o, m.i("d"), m.sub("z")),
  d: (o = {}) => mx(o, m.i("d")),
  dsub: (s, o = {}) => mx(o, m.i("d"), m.sub(s, false)),
};

// Marcador de ecuación: postproceso.py sustituye el contenido por OMML nativo.
function eq(slide, id, x, y, w, h, pt = 20, color = C.ink, alinear = "left") {
  slide.addText(`[${id}]`, { x, y, w, h, objectName: `EQ__${id}__${pt}__${color}__${alinear}`,
    isTextBox: true, fontFace: F.math, fontSize: pt, color, margin: 0, valign: "middle" });
}

// ------------------------------------------------------------------ formas
function rect(slide, pres, x, y, w, h, fill, o = {}) {
  slide.addShape(pres.shapes.RECTANGLE, Object.assign({ x, y, w, h, fill: { color: fill }, line: { color: fill, width: 0 } }, o));
}
function rrect(slide, pres, x, y, w, h, fill, o = {}) {
  // Los glifos de conteo y los nodos de diagramas sí conservan el relleno:
  // su color codifica datos o condiciones experimentales.
  if ((w < 1 && h < 1) || (w <= 2 && h <= 0.8)) {
    rect(slide, pres, x, y, w, h, fill);
    return;
  }
  // Agrupar sin encerrar el contenido en tarjetas repetidas.
  seg(slide, pres, x, y, x + w, y, C.rule, 0.65);
}
function dot(slide, pres, cx, cy, dia, fill, o = {}) {
  const base = { x: cx - dia / 2, y: cy - dia / 2, w: dia, h: dia, fill: { color: fill }, line: { color: fill, width: 0 } };
  slide.addShape(pres.shapes.OVAL, Object.assign(base, o));
}
function seg(slide, pres, x1, y1, x2, y2, color, width = 1, o = {}) {
  const x = Math.min(x1, x2), y = Math.min(y1, y2);
  const w = Math.max(Math.abs(x2 - x1), 0.0001), h = Math.max(Math.abs(y2 - y1), 0.0001);
  const flipH = x2 < x1, flipV = y2 < y1;
  const line = Object.assign({ color, width }, o.line || {});
  const extra = Object.assign({}, o); delete extra.line;
  slide.addShape(pres.shapes.LINE, Object.assign({ x, y, w, h, flipH, flipV, line }, extra));
}
function flecha(slide, pres, x1, y1, x2, y2, color, width = 1.25, o = {}) {
  seg(slide, pres, x1, y1, x2, y2, color, width, { line: Object.assign({ endArrowType: "triangle" }, o) });
}

// Polilínea o polígono con geometría personalizada; pts en pulgadas absolutas.
function curva(slide, pres, pts, o = {}) {
  const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
  const x = Math.min(...xs), y = Math.min(...ys);
  const w = Math.max(Math.max(...xs) - x, 0.001), h = Math.max(Math.max(...ys) - y, 0.001);
  const points = pts.map((p, i) => ({ x: p[0] - x, y: p[1] - y, ...(i === 0 ? { moveTo: true } : {}) }));
  if (o.cerrar) points.push({ close: true });
  slide.addShape(pres.shapes.CUSTOM_GEOMETRY, { x, y, w, h, points,
    fill: o.fill ? { color: o.fill, transparency: o.transparency || 0 } : { type: "none" },
    line: o.line ? { color: o.line, width: o.width || 1, dashType: o.dash || "solid" } : { type: "none" } });
}

// Barra de navegación por secciones (arriba a la izquierda). Es funcional: indica en qué
// parte del argumento está el jurado.
function nav(slide, actual, oscuro = false) {
  slide.addText(`Ciencia de Datos   /   ${SECCIONES[actual]}`, {
    x: MX, y: 0.28, w: 8, h: 0.24, isTextBox: true, margin: 0,
    fontFace: F.body, fontSize: 10, color: C.mute, valign: "middle",
  });
}

function etiquetaRespaldo(slide, n) {
  slide.addText([R(`Anexo R${n}`, { fontFace: F.demi, fontSize: 10, color: C.ink }),
    R("   /   Preguntas del jurado", { fontFace: F.body, fontSize: 10, color: C.mute })],
  { x: MX, y: 0.28, w: 9, h: 0.24, isTextBox: true, margin: 0, valign: "middle" });
}

function fuente(slide, texto, oscuro = false) {
  slide.addText(texto, { x: MX, y: 6.98, w: W - 2 * MX - 0.9, h: 0.36, isTextBox: true, margin: 0,
    fontFace: F.body, fontSize: 10, color: oscuro ? C.darkMute : C.mute, valign: "top" });
}

// Etiqueta pequeña en versalitas espaciadas
function rotulo(slide, texto, x, y, w, color = C.ink2, o = {}) {
  slide.addText(texto, Object.assign({ x, y, w, h: 0.26, isTextBox: true, margin: 0, fontFace: F.demi,
    fontSize: 11, bold: true, charSpacing: 0, color, valign: "middle" }, o));
}

// Enjambre unidimensional compacto: cada punto se coloca a la menor distancia del eje que no
// solape los ya colocados. dir = -1 hacia arriba, +1 hacia abajo.
function enjambre(valores, escala, base, dia, dir, solape) {
  const orden = valores.map((v, i) => [escala(v), i]).sort((a, b) => a[0] - b[0]);
  const puestos = [];
  const out = new Array(valores.length);
  const d2 = dia * dia * (solape || 0.995);
  for (const [x, i] of orden) {
    const cand = [0];
    for (const q of puestos) {
      const dx = Math.abs(x - q.x);
      if (dx * dx < d2) cand.push(q.y + Math.sqrt(d2 - dx * dx) + 1e-6);
    }
    cand.sort((a, b) => a - b);
    const y = cand.find((c) => puestos.every((q) => (x - q.x) ** 2 + (c - q.y) ** 2 >= d2 - 1e-9)) ?? Math.max(...cand);
    puestos.push({ x, y });
    out[i] = { x, y: base + dir * (y + dia * 0.62) };
  }
  return out;
}

module.exports = { W, H, MX, C, F, THEME, definirLayouts, R, P, parrafos, txt, m, mx, N, eq, rect, rrect, dot, seg, flecha, curva, nav, etiquetaRespaldo, fuente, rotulo, enjambre };
