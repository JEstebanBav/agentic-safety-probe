// Notas del presentador: una sola fuente para las notas del .pptx y para guion_sustentacion.md.
const registro = [];
let acumulado = 0;

function mmss(seg) {
  const m = Math.floor(seg / 60), s = Math.round(seg % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

// n: { titulo, t (segundos; 0 en respaldo), guion, transicion, fuentes, respaldo }
function nota(slide, n) {
  const ini = acumulado;
  if (!n.respaldo) acumulado += n.t;
  const cab = n.respaldo
    ? "RESPALDO · usar solo si el jurado pregunta."
    : `TIEMPO ORIENTATIVO: ${mmss(n.t)} (de ${mmss(ini)} a ${mmss(acumulado)} de 20:00)`;
  const partes = [cab, "", "GUION", n.guion.trim()];
  if (n.transicion) partes.push("", "TRANSICIÓN", n.transicion.trim());
  if (n.fuentes) partes.push("", "FUENTES Y TRAZABILIDAD", n.fuentes.trim());
  slide.addNotes(partes.join("\n"));
  registro.push(Object.assign({ numero: registro.length + 1, inicio: ini, fin: acumulado }, n));
}

function markdown() {
  const lineas = [
    "# Guion de sustentación",
    "",
    "**Transformación de la dirección de rechazo en modelos de lenguaje agénticos.** Un análisis desde la interpretabilidad mecánica.",
    "",
    "Juan Esteban Espitia Leiva · Tutor: Julián Fernando Sánchez · Pregrado en Ciencia de Datos, Universidad Externado de Colombia.",
    "",
    `Duración prevista de la exposición: ${mmss(acumulado)} de 20:00. Las diapositivas de respaldo (R1–R${registro.filter((r) => r.respaldo).length}) solo se usan en preguntas.`,
    "",
    "Este guion es el mismo texto de las notas del presentador del archivo .pptx. Los tiempos son orientativos; el margen restante absorbe pausas y transiciones.",
    "",
    "| Diap. | Título | Tiempo | Acumulado |",
    "|---:|---|---:|---:|",
  ];
  registro.filter((r) => !r.respaldo).forEach((r) => lineas.push(`| ${r.numero} | ${r.titulo} | ${mmss(r.t)} | ${mmss(r.fin)} |`));
  lineas.push("");
  for (const r of registro) {
    lineas.push(`## ${r.numero}. ${r.titulo}`, "");
    lineas.push(r.respaldo ? "*Respaldo: usar solo si el jurado pregunta.*" : `*${mmss(r.t)} · de ${mmss(r.inicio)} a ${mmss(r.fin)}*`, "");
    lineas.push(r.guion.trim(), "");
    if (r.transicion) lineas.push(`**Transición.** ${r.transicion.trim()}`, "");
    if (r.fuentes) lineas.push(`**Fuentes.** ${r.fuentes.trim()}`, "");
  }
  return lineas.join("\n");
}

module.exports = { nota, markdown, mmss, total: () => acumulado, registro: () => registro };
