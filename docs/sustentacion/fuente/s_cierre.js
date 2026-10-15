// IV. Alcance: respuesta, aportes y límites, cierre.
const B = require("./base");
const { W, MX, C, F, R, txt, rect, rrect, seg, nav, fuente, rotulo } = B;
const { nota } = require("./guion");
const { ejes } = require("./s_problema");

module.exports = function (pres, D) {
  const sec = "IV. Alcance";

  // ------------------------------------------------------------ 21. Respuesta
  {
    const s = pres.addSlide({ masterName: "OSCURO", sectionTitle: sec });
    nav(s, 3, true);
    rotulo(s, "Respuesta a la pregunta", MX, 0.95, 6, C.darkMute);
    txt(s, "Sí conserva capacidad de detección, pero con menos margen: una degradación que depende del modelo, no una desaparición.",
      { x: MX, y: 1.3, w: 12.1, h: 1.5, fontFace: F.head, fontSize: 28, color: C.darkTxt });
    const hall = [["Específica del daño", "La interacción con el control benigno es positiva fuera de muestra en Llama y en Mistral; en Llama, 42 de 42 pares decrecen."],
      ["Más que escala", "La pérdida persiste en unidades estandarizadas (d: −44 % y −64 %) y en AUROC (0,978 → 0,920 y 0,948 → 0,785)."],
      ["Sin jerarquía por explicitud", "La pérdida máxima cambia de estrato según el modelo, y la apariencia legítima conserva señal sobre el azar."],
      ["Señal recuperable", "El contraste agéntico se reorienta (≈ 77° en Llama) y un monitor ajustado en agente lo aprovecha fuera de fold."]];
    hall.forEach(([t, b], i) => {
      const x = MX + i * 3.08;
      rrect(s, pres, x, 2.9, 2.9, 2.55, C.dark2);
      txt(s, `${i + 1}`, { x: x + 0.22, y: 3.02, w: 0.6, h: 0.55, fontFace: F.med, fontSize: 26, color: C.ag });
      txt(s, [R(t, { fontFace: F.demi, color: C.darkTxt, breakLine: true }), R(b, { color: C.darkMute })],
        { x: x + 0.22, y: 3.6, w: 2.48, h: 1.8, fontSize: 13 });
    });
    txt(s, [R("En la conducta, ", { fontFace: F.demi, color: C.darkTxt }), R("el rechazo etiquetado cae (38/42 → 23/42) y sumar la dirección lo restaura con un costo de sobrerrechazo benigno. Geometría, discriminación y conducta se leyeron como tres objetos distintos.")],
      { x: MX, y: 5.75, w: 11.9, h: 0.75, fontSize: 14, color: C.darkMute });
    nota(s, {
      titulo: "Respuesta a la pregunta", t: 50,
      guion: `Vuelvo a la pregunta: cuando el modelo pasa a actuar como agente, ¿conserva su dirección de rechazo la capacidad de detectar intenciones dañinas?
La respuesta que sostienen los datos es: sí, pero con menos margen. Es una degradación que depende del modelo, no una desaparición.
Primero, el cambio es específico del daño: la interacción con el control benigno es positiva en ambos modelos. Segundo, no es solo escala: la pérdida sobrevive en d y en el AUROC. Tercero, no hay una jerarquía universal por explicitud, y lo encubierto conserva señal. Y cuarto, la señal es recuperable: el contraste se reorienta y un monitor ajustado en formato agéntico la aprovecha.
En la conducta, el rechazo cae, y sumar la dirección lo restaura, pero con sobrerrechazo.`,
      transicion: "¿Qué aporta esto y hasta dónde llega?",
      fuentes: "Síntesis de las diapositivas 13 a 20; tesis, capítulos 5 y 6. Cifras: d de Llama 3,21 → 1,80 (−43,7 %), de Mistral 2,53 → 0,92 (−63,8 %); AUROC fuera de muestra 0,9779 → 0,9195 y 0,9478 → 0,7851; F1 fuera de fold de los monitores agénticos 0,8434 (medias) y 0,9367 (logística) en Llama.",
    });
  }

  // ------------------------------------------------------------ 22. Aportes y límites
  {
    const s = pres.addSlide({ masterName: "CONTENIDO", sectionTitle: sec });
    nav(s, 3);
    s.addText("Lo que la evidencia sostiene y lo que todavía no", { placeholder: "title" });
    rotulo(s, "Aportes", MX, 1.88, 4);
    const ap = [["Un corpus contrastivo auditable", "96 pares, texto idéntico entre formatos, explicitud estratificada y simetría verificada."],
      ["Un diagnóstico con controles", "cross‑fitting por pares, selección de capa separada de la inferencia y nulos empíricos."],
      ["Tres unidades, tres objetos", "cruda, d y AUROC; geometría, discriminación y conducta, sin confundirlos."],
      ["Resultados trazables", "cada cifra ligada a un archivo de resultados y a su hash."]];
    txt(s, ap.flatMap(([a, b], i) => [R(a, { fontFace: F.demi, color: C.ink, breakLine: true }), R(b, { breakLine: i < ap.length - 1, paraSpaceAfter: 12 })]),
      { x: MX, y: 2.2, w: 4.1, h: 4.0, fontSize: 13.5, color: C.ink2 });
    const X0 = 5.15, XW = W - MX - X0;
    rotulo(s, "Límite", X0, 1.88, 3.6);
    rotulo(s, "Qué conclusión acota", X0 + 3.85, 1.88, 3.6);
    const lim = [["42 pares ejecutados; estratos de 18, 14 y 10", "la comparación por explicitud es exploratoria"],
      ["Formato = sistema + herramientas simuladas, en un turno", "no se atribuye a un componente ni se extiende a agentes que ejecutan acciones"],
      ["Mistral solo como registro; Qwen histórico", "réplica entre familias parcial; ninguna ley de escala"],
      ["Etiquetas conductuales heurísticas; un solo anotador de explicitud", "resultados conductuales y por estrato, exploratorios"],
      ["Monitores sin datos externos", "viabilidad preliminar, no garantía operacional"],
      ["Folds con entrenamiento compartido", "intervalos condicionales a los ajustes guardados"]];
    lim.forEach(([a, b], i) => {
      const y = 2.2 + i * 0.64;
      seg(s, pres, X0, y - 0.04, W - MX, y - 0.04, "E3E6EA", 0.75);
      txt(s, a, { x: X0, y, w: 3.65, h: 0.58, fontFace: F.demi, fontSize: 12.5, color: C.ink, valign: "middle" });
      txt(s, b, { x: X0 + 3.85, y, w: XW - 3.85, h: 0.58, fontSize: 12.5, color: C.ink2, valign: "middle" });
    });
    rrect(s, pres, MX, 6.2, W - 2 * MX, 0.62, C.mist);
    txt(s, [R("Siguiente:  ", { fontFace: F.demi, color: C.ink }), R("réplica en los 96 pares con el código V3 · doble anotación · auditoría humana de generaciones · factorial sistema/herramientas con control de longitud · validación externa · trayectorias de varios pasos")],
      { x: MX + 0.2, y: 6.2, w: W - 2 * MX - 0.4, h: 0.62, fontSize: 12.5, color: C.ink2, valign: "middle" });
    nota(s, {
      titulo: "Lo que la evidencia sostiene y lo que todavía no", t: 55,
      guion: `¿Qué aporta el trabajo? Un corpus contrastivo auditable que aísla el formato; un diagnóstico con controles explícitos de circularidad y de selección; y una forma de lectura que separa tres unidades y tres objetos —geometría, discriminación y conducta— que es fácil confundir.
Los límites acotan conclusiones concretas. Con 42 pares y estratos pequeños, la comparación por explicitud es exploratoria. El formato cambia sistema y herramientas a la vez, con herramientas simuladas, así que no atribuyo el efecto a un componente ni lo extiendo a agentes que ejecutan acciones. Mistral solo existe como registro. Las etiquetas conductuales esperan auditoría humana. Y los monitores no tienen validación externa.
Lo siguiente es replicar en los 96 pares con el mismo código, con doble anotación, auditoría humana y un diseño factorial con control de longitud.`,
      transicion: "Cierro volviendo a la imagen del comienzo.",
      fuentes: "Tesis, secciones 5.4 (limitaciones), 5.5 y 6.3 (recomendaciones); respuesta_observaciones_actualizada.pdf (pendientes experimentales).",
    });
  }

  // ------------------------------------------------------------ 23. Cierre
  {
    const s = pres.addSlide({ masterName: "OSCURO", sectionTitle: sec });
    nav(s, 3, true);
    ejes(s, pres, D, 1.75, W - MX, 1.75, 2.95, true);
    txt(s, "AUROC 0,978 · d 3,21", { x: 9.6, y: 2.05, w: W - MX - 9.6, h: 0.28, align: "right", fontSize: 11.5, color: C.darkMute });
    txt(s, "AUROC 0,920 · d 1,80: el margen se estrecha", { x: 7.6, y: 3.2, w: W - MX - 7.6, h: 0.28, align: "right", fontSize: 11.5, color: C.darkMute });
    txt(s, "Antes de confiar en que un agente rechazará, conviene medir si todavía distingue.",
      { x: MX, y: 4.35, w: 11.4, h: 1.0, fontFace: F.head, italic: true, fontSize: 28, color: C.darkTxt });
    txt(s, "Gracias", { x: MX, y: 5.6, w: 6, h: 0.6, fontFace: F.head, fontSize: 30, color: C.ag });
    txt(s, [R("Juan Esteban Espitia Leiva", { fontFace: F.demi, color: C.darkTxt, breakLine: true }), R("Tutor: Julián Fernando Sánchez · Pregrado en Ciencia de Datos · Universidad Externado de Colombia", { color: C.darkMute })],
      { x: MX, y: 6.25, w: 11, h: 0.6, fontSize: 12.5 });
    nota(s, {
      titulo: "Cierre", t: 25,
      guion: `Al comienzo les mostré estos puntos; ahora se pueden leer. En chat, las dañinas quedan lejos de las benignas. En agente, todo se acerca a cero y el margen se estrecha, aunque el orden se conserva en buena parte.
La implicación práctica es sencilla: antes de confiar en que un agente rechazará, conviene medir si todavía distingue.
Muchas gracias. Quedo atento a sus preguntas.`,
      fuentes: "Mismos datos que la portada: proyecciones fuera de muestra, Llama V3, capa 76 (results.json → heldout.projections). AUROC y d: metricas_avance.json → heldout_detection y heldout_format.",
    });
  }
};
