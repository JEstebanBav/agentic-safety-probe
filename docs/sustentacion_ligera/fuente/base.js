// Una idea por diapositiva: títulos directos, texto grande y figuras abiertas.
const W = 13.333, H = 7.5, MX = 0.68;
const C = { paper: "FFFFFF", ink: "26282A", body: "50565C", mute: "70777D", rule: "D9DDE0",
  pale: "F1F3F4", harm: "A14F3D", benign: "416879", agent: "9B743E", harmL: "F1E3DF", benignL: "E4EEF0" };
const F = { head: "Calibri", body: "Calibri", math: "Cambria Math" };

function text(s, t, x, y, w, h, o = {}) {
  s.addText(t, { x, y, w, h, fontFace: F.body, fontSize: 21, color: C.ink, margin: 0,
    breakLine: false, valign: "top", paraSpaceAfter: 0, ...o });
}
function run(t, options = {}) { return { text: t, options }; }
function line(s, p, x1, y1, x2, y2, color = C.rule, width = 1, opts = {}) {
  s.addShape(p.ShapeType.line, { x: Math.min(x1, x2), y: Math.min(y1, y2),
    w: Math.max(Math.abs(x2 - x1), 0.0001), h: Math.max(Math.abs(y2 - y1), 0.0001),
    flipH: x2 < x1, flipV: y2 < y1, line: { color, width, ...opts } });
}
function rect(s, p, x, y, w, h, color) {
  s.addShape(p.ShapeType.rect, { x, y, w, h, fill: { color }, line: { color, transparency: 100 } });
}
function dot(s, p, x, y, r, color, hollow = false) {
  s.addShape(p.ShapeType.ellipse, { x: x - r, y: y - r, w: 2 * r, h: 2 * r,
    fill: { color: hollow ? C.paper : color }, line: { color, width: hollow ? 1.3 : 0.5 } });
}
function arrow(s, p, x1, y1, x2, y2, color = C.ink, width = 1.8) {
  line(s, p, x1, y1, x2, y2, color, width, { beginArrowType: "none", endArrowType: "triangle" });
}
function label(s, t, x, y, w, color = C.mute) {
  text(s, t, x, y, w, 0.28, { fontSize: 12, bold: true, color });
}
function footer(s, t) { text(s, t, MX, 6.91, W - 2 * MX - 0.45, 0.3, { fontSize: 10.5, color: C.mute }); }
function slide(p, title, section, n, respaldo = false) {
  const s = p.addSlide(); s.background = { color: C.paper };
  label(s, respaldo ? "Apoyo para preguntas" : "Ciencia de Datos · Sustentación", MX, 0.28, 7);
  text(s, section, W - MX - 3.6, 0.28, 3.6, 0.28, { fontSize: 11, color: C.mute, align: "right" });
  text(s, title, MX, 0.9, W - 2 * MX, 1.0, { fontSize: 32, bold: true });
  text(s, String(n).padStart(2, "0"), W - MX - 0.5, 6.96, 0.5, 0.25,
    { fontSize: 11, color: C.mute, align: "right" });
  return s;
}
function eq(s, id, x, y, w, h, pt = 25) {
  text(s, "Ecuación", x, y, w, h, { fontFace: F.math, fontSize: pt,
    objectName: `EQ__${id}__${pt}__${C.ink}__left` });
}
const num = (v, n = 3) => v.toLocaleString("es-CO", { minimumFractionDigits: n, maximumFractionDigits: n });
module.exports = { W, H, MX, C, F, text, run, line, rect, dot, arrow, label, footer, slide, eq, num };
