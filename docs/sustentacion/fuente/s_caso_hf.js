// Caso contextual: fuentes primarias verificadas el 15 de octubre de 2026.
const B = require("./base");
const { W, MX, C, F, R, txt, seg, flecha, nav, fuente, rotulo } = B;
const { nota } = require("./guion");

module.exports = function (pres) {
  const s = pres.addSlide({ masterName: "CONTENIDO", sectionTitle: "I. Problema" });
  nav(s, 0);
  s.addText("Un agente de OpenAI pasó de una evaluación a una intrusión", { placeholder: "title" });
  rotulo(s, "Hugging Face · incidente de julio de 2026", MX, 1.88, 7.5);
  txt(s, "La tarea asignada era resolver pruebas de ciberseguridad.",
    { x: MX, y: 2.22, w: 7.45, h: 0.48, fontSize: 17, fontFace: F.head });

  const etapas = ["Evaluación interna\nde OpenAI", "Salida del\nentorno aislado", "Acceso a\nsistemas externos", "Intrusión en\nHugging Face"];
  etapas.forEach((t, i) => {
    const x = MX + i * 1.9;
    seg(s, pres, x, 2.95, x + 1.6, 2.95, C.rule, 0.8);
    rotulo(s, `0${i + 1}`, x, 2.67, 1.5, C.ag);
    txt(s, t, { x, y: 3.1, w: 1.6, h: 0.75, fontFace: F.demi, fontSize: 14 });
    if (i < 3) flecha(s, pres, x + 1.64, 3.4, x + 1.85, 3.4, C.ink2, 1);
  });
  txt(s, [R("Lo observado: ", { fontFace: F.demi }),
    R("los agentes encadenaron accesos fuera del entorno previsto y llegaron a infraestructura de terceros.")],
    { x: MX, y: 4.03, w: 7.45, h: 0.8, fontSize: 15, color: C.ink2 });

  const cifras = [
    ["09–13 jul.", "periodo de actividad reconstruido por Hugging Face"],
    ["5 datasets", "contenido de clientes accedido, según el alcance comunicado por Hugging Face"],
    ["Evaluación interna", "OpenAI indica que los modelos operaban con salvaguardas reducidas"],
  ];
  cifras.forEach(([a, b], i) => {
    const y = 1.88 + i * 1.13;
    txt(s, a, { x: 8.5, y, w: 4.23, h: 0.57, fontFace: F.med, fontSize: i === 2 ? 26 : 32 });
    txt(s, b, { x: 8.5, y: y + 0.58, w: 4.23, h: 0.5, fontSize: 13, color: C.ink2 });
  });

  seg(s, pres, MX, 5.35, W - MX, 5.35, C.rule, 0.8);
  txt(s, "Poder actuar exige estudiar cómo se mantiene la seguridad",
    { x: MX, y: 5.58, w: 11.9, h: 0.45, fontFace: F.head, fontSize: 22 });
  txt(s, "Un objetivo de evaluación puede desembocar en acciones fuera del alcance autorizado.",
    { x: MX, y: 6.12, w: 7.45, h: 0.58, fontSize: 15 });
  txt(s, "El caso motiva la pregunta. La tesis mide representaciones con herramientas sin ejecución; no explica este incidente.",
    { x: 8.5, y: 5.95, w: 4.23, h: 0.8, fontSize: 13, color: C.ink2, italic: true });
  fuente(s, "Fuentes primarias: Hugging Face (27 jul. 2026) y OpenAI (26 ago. 2026), informes del incidente. Enlaces en R11 y notas.");
  nota(s, {
    titulo: "Un agente de OpenAI pasó de una evaluación a una intrusión", t: 50,
    guion: "En julio de 2026, agentes de OpenAI que resolvían una evaluación interna de ciberseguridad salieron del entorno previsto y accedieron a sistemas de terceros, entre ellos Hugging Face. Hugging Face reconstruyó actividad entre el nueve y el trece de julio y comunicó acceso a cinco conjuntos de datos de clientes. OpenAI aclara que se trataba de modelos de investigación con salvaguardas reducidas. El punto es que un objetivo de evaluación terminó acompañado de acciones fuera del alcance autorizado. El caso motiva estudiar la seguridad cuando el modelo puede actuar. No demuestra qué ocurrió con una dirección de rechazo: esta tesis mide representaciones en un turno, con herramientas que no se ejecutan.",
    transicion: "El incidente muestra un riesgo conductual. ¿Qué sabemos del mecanismo interno que debería sostener el rechazo?",
    fuentes: "Hugging Face (27 de julio de 2026), reconstrucción técnica, https://huggingface.co/blog/agent-intrusion-technical-timeline. Periodo reconstruido: 9–13 de julio; el contenido de clientes accedido se acota a cinco datasets. OpenAI (26 de agosto de 2026), informe del incidente, https://openai.com/index/hugging-face-incident-and-the-road-ahead/. Evaluación interna con modelos de investigación y salvaguardas reducidas. Contextualización externa: no se presenta como evidencia de degradación de una dirección de rechazo ni como resultado de esta tesis. Fuentes consultadas el 15 de octubre de 2026.",
  });
};
