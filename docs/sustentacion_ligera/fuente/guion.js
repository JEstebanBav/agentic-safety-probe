// Notas propias de la versión ligera; no modifica el guion de la versión completa.
const registro = [];
let acumulado = 0;
const mmss = s => `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, "0")}`;
function nota(s, n) {
  const inicio = acumulado;
  if (!n.respaldo) acumulado += n.t;
  const r = { numero: registro.length + 1, inicio, fin: acumulado, ...n };
  registro.push(r);
  s.addNotes([n.respaldo ? "APOYO · usar solo en preguntas" : `TIEMPO ORIENTATIVO: ${mmss(n.t)} · ${mmss(inicio)}–${mmss(acumulado)}`,
    "", "GUION", n.guion, "", "TRANSICIÓN", n.transicion || "", "", "FUENTES Y ALCANCE", n.fuentes || ""].join("\n"));
}
function markdown() {
  const lineas = ["# Guion de la presentación ligera", "", "Juan Esteban Espitia Leiva · Universidad Externado de Colombia", "",
    `Duración orientativa: ${mmss(acumulado)}. 14 diapositivas principales y 2 de apoyo para preguntas.`, "",
    "El guion coincide con las notas del PowerPoint. Las pausas y el ritmo pueden ajustar la duración.", "",
    "| Diap. | Idea | Tiempo | Acumulado |", "|---:|---|---:|---:|"];
  registro.filter(r => !r.respaldo).forEach(r => lineas.push(`| ${r.numero} | ${r.titulo} | ${mmss(r.t)} | ${mmss(r.fin)} |`));
  registro.forEach(r => { lineas.push("", `## ${r.numero}. ${r.titulo}`, "", r.respaldo ? "Apoyo para preguntas." : `Tiempo: ${mmss(r.t)}`, "", r.guion,
    "", `**Transición.** ${r.transicion || "—"}`, "", `**Fuentes y alcance.** ${r.fuentes || "—"}`); });
  return lineas.join("\n");
}
module.exports = { nota, markdown, registro: () => registro, total: () => acumulado, mmss };
