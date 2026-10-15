// Construye el borrador de la presentación con pptxgenjs.
//   node construir.js            -> build/borrador.pptx, errores.json, ../guion_sustentacion.md
// Después: python postproceso.py build/borrador.pptx build/con_ecuaciones.pptx
//          python exportar.py build/con_ecuaciones.pptx ../<final>.pptx --pdf ... --png ...
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const B = require("./base");
const guion = require("./guion");

const D = JSON.parse(fs.readFileSync(path.join(__dirname, "datos_presentacion.json"), "utf8"));
const secciones = (process.env.SECCIONES || "problema,metodo,resultados,cierre,respaldo").split(",");

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.author = "Juan Esteban Espitia Leiva";
pres.company = "Universidad Externado de Colombia";
pres.title = "Transformación de la dirección de rechazo en modelos de lenguaje agénticos";
pres.subject = "Sustentación de trabajo de grado, Pregrado en Ciencia de Datos";
pres.theme = { headFontFace: B.F.head, bodyFontFace: B.F.body };
B.definirLayouts(pres);

const errores = {};
const ctx = { errores };
const modulos = { problema: "./s_problema", metodo: "./s_metodo", resultados: "./s_resultados", cierre: "./s_cierre", respaldo: "./s_respaldo" };
for (const sec of secciones) {
  const titulo = { problema: "I. Problema", metodo: "II. Método", resultados: "III. Resultados", cierre: "IV. Alcance", respaldo: "Respaldo" }[sec];
  pres.addSection({ title: titulo });
  require(modulos[sec])(pres, D, ctx);
}

fs.mkdirSync(path.join(__dirname, "build"), { recursive: true });
fs.writeFileSync(path.join(__dirname, "errores.json"), JSON.stringify(errores, null, 1));
if (secciones.length === 5) fs.writeFileSync(path.join(__dirname, "..", "guion_sustentacion.md"), guion.markdown(), "utf8");
fs.writeFileSync(path.join(__dirname, "build", "guion.json"), JSON.stringify(guion.registro(), null, 1), "utf8");

(async () => {
  const salida = path.join(__dirname, "build", "borrador.pptx");
  await pres.writeFile({ fileName: salida });
  // Escribe la paleta en el tema del archivo (opcional: script apply_theme.js de la guía de pptx).
  if (process.env.APPLY_THEME) await require(process.env.APPLY_THEME).applyTheme(salida, B.THEME);
  console.log(`Borrador escrito: ${salida} · exposición ${guion.mmss(guion.total())}`);
})();
