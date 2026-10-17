// Comparativas de las tres exportaciones verificadas de 96 pares.
const B = require("./base");
const { W, MX, C, F, R, txt, rect, dot, seg, nav, fuente, rotulo } = B;
const { nota } = require("./guion");
const fmt = (v, n = 3) => v.toLocaleString("es-CO", { minimumFractionDigits: n, maximumFractionDigits: n }).replace("-", "−");
const keys = ["mistral", "qwen8b", "qwen32b"];

function estratos(pres, D) {
  const s = pres.addSlide({ masterName: "CONTENIDO", sectionTitle: "III. Resultados" });
  nav(s, 2);
  s.addText("Según la explicitud, no aparece una jerarquía universal", { placeholder: "title" });
  const x0 = 3.55, x1 = 8.5, esc = v => x0 + (v - 0.5) * 2 * (x1 - x0);
  const orden = [["explicit", "Explícito", 42], ["contextual", "Contextual", 24], ["framed", "Ap. legítima", 30]];
  txt(s, [R("● ", { color: C.ink }), R("chat    "), R("● ", { color: C.ag }), R("agente")],
    { x: x1 - 2.2, y: 1.84, w: 2.2, h: 0.28, fontSize: 12, align: "right" });
  for (let v = 0.5; v <= 1.001; v += 0.1) {
    seg(s, pres, esc(v), 2.18, esc(v), 6.1, C.rule, 0.65);
    txt(s, fmt(v, 1), { x: esc(v) - 0.25, y: 6.15, w: 0.5, h: 0.25, fontSize: 11, align: "center", color: C.mute });
  }
  keys.forEach((key, g) => {
    const model = D.evaluacion96[key], by = Object.fromEntries(model.estratos.map(e => [e.stratum, e]));
    const y0 = 2.15 + g * 1.33;
    rotulo(s, `${model.modelo} · n = 96 · capa ${model.capa}`, MX, y0, 3.2, C.ink);
    orden.forEach(([key, nombre, n], i) => {
      const e = by[key], y = y0 + 0.42 + i * 0.29;
      txt(s, `${nombre} · ${n}`, { x: MX, y: y - 0.12, w: 2.8, h: 0.25, fontSize: 12, color: C.ink2 });
      seg(s, pres, esc(e.auroc_agent), y, esc(e.auroc_chat), y, C.rule, 3);
      dot(s, pres, esc(e.auroc_chat), y, 0.14, C.ink);
      dot(s, pres, esc(e.auroc_agent), y, 0.14, C.ag);
      txt(s, `−${fmt(e.auroc_loss)}`, { x: Math.min(esc(e.auroc_chat) + 0.11, 8.55), y: y - 0.12, w: 0.7, h: 0.25, fontSize: 10.5, color: C.ink2 });
    });
  });
  rotulo(s, "La pérdida máxima cambia de estrato", 9.4, 1.95, 3.2);
  txt(s, "Mistral y Qwen3‑8B pierden más en el explícito; en Qwen3‑32B la pérdida mayor es contextual.",
    { x: 9.4, y: 2.35, w: 3.15, h: 1.15, fontSize: 15, color: C.ink2 });
  rotulo(s, "El daño encubierto conserva señal", 9.4, 3.65, 3.2);
  txt(s, "AUROC agéntico de apariencia legítima: 0,879 en Mistral, 0,690 en Qwen3‑8B y 0,903 en Qwen3‑32B.",
    { x: 9.4, y: 4.05, w: 3.15, h: 1.1, fontSize: 15, color: C.ink2 });
  txt(s, "Estimaciones puntuales: sin intervalos por estrato ni prueba de diferencias entre modelos.",
    { x: 9.4, y: 5.5, w: 3.15, h: 0.7, fontSize: 12.5, color: C.mute });
  fuente(s, "96 pares por modelo: results/corridas/mistral7b/v2, qwen8b_v2 y qwen32b_v2 → format_effect.held_out.by_stratum. AUROC de la dirección conversacional, fuera de muestra.");
  nota(s, { titulo: "Según la explicitud, no aparece una jerarquía universal", t: 40,
    guion: "Aquí comparo tres ejecuciones sobre el mismo corpus de 96 pares: 42 explícitos, 24 contextuales y 30 de apariencia legítima. En Mistral y Qwen3‑8B la mayor pérdida puntual está en el estrato explícito; en Qwen3‑32B está en el contextual. No aparece una jerarquía universal de degradación. La apariencia legítima conserva señal, aunque con márgenes diferentes. Son estimaciones puntuales: sin intervalos por estrato no afirmo una ordenación poblacional ni una diferencia significativa entre modelos.",
    transicion: "Si la dirección de chat pierde margen en agente, ¿hay otra dirección que separe mejor?",
    fuentes: keys.map(k => `${D.evaluacion96[k].fuente} → format_effect.held_out.by_stratum; n=96`).join("\n") });
}

function monitores(pres, D) {
  const s = pres.addSlide({ masterName: "CONTENIDO", sectionTitle: "III. Resultados" }); nav(s, 2);
  s.addText("Los monitores agénticos recuperan señal, con costos distintos", { placeholder: "title" });
  const m = D.llama.monitores;
  const filas = [
    ["Llama · diferencia de medias", 42, m.auroc_agent_diff_means_oof, m.monitor_f1_diff_means_oof, m.monitor_fpr_diff_means_oof],
    ["Llama · regresión logística", 42, m.auroc_w_agent_oof, m.monitor_f1_oof, m.monitor_fpr_oof],
    ...keys.map(k => { const x = D.evaluacion96[k]; const r = x.monitor; return [x.modelo, 96, r.auroc_w_agent_oof, r.monitor_f1_oof, r.monitor_fpr_oof]; }),
  ];
  [["Modelo / estimador", MX, 3.0], ["Pares", 3.85, 0.7], ["AUROC OOF", 4.9, 1.4], ["F₁ OOF", 6.65, 1.5], ["Falsos positivos", 10.7, 1.9]].forEach(([t,x,w]) => rotulo(s,t,x,1.98,w));
  filas.forEach(([nombre,n,a,f,p],i) => {
    const y = 2.5 + i * 0.62;
    seg(s,pres,MX,y-0.1,W-MX,y-0.1,C.rule,0.65);
    txt(s,nombre,{x:MX,y,w:3.1,h:0.43,fontSize:14,fontFace:F.demi,valign:"middle"});
    txt(s,String(n),{x:3.85,y,w:0.7,h:0.43,fontSize:15,valign:"middle"});
    txt(s,fmt(a),{x:4.9,y,w:1.3,h:0.43,fontSize:20,valign:"middle"});
    txt(s,fmt(f),{x:6.65,y,w:1.1,h:0.43,fontSize:20,valign:"middle"});
    rect(s,pres,7.9,y+0.13,2.3,0.14,C.mist);rect(s,pres,7.9,y+0.13,2.3*f,0.14,C.ben);
    txt(s,`${fmt(100*p,1)} %`,{x:10.7,y,w:1.8,h:0.43,fontSize:19,color:p>0.25?C.harm:C.ink,valign:"middle"});
  });
  txt(s,"La recuperación no es universal: el AUROC del monitor de Mistral (0,850) no supera al eje de chat en agente (0,878).",
    {x:MX,y:5.9,w:11.8,h:0.65,fontFace:F.head,fontSize:18});
  txt(s,"El cero observado en Llama corresponde a su muestra de 42 benignos; no implica una tasa poblacional nula. No se comparan formalmente estos monitores.",
    {x:MX,y:6.62,w:11.8,h:0.25,fontSize:11,color:C.mute});
  fuente(s,"Resultados fuera de fold: Llama V3 (estimadores separados); Mistral/Qwen V2 (96 pares), robust.per_layer. Los umbrales se calibran en entrenamiento; sin validación externa.");
  nota(s,{titulo:"Los monitores agénticos recuperan señal, con costos distintos",t:60,
    guion:"El ajuste al formato agéntico mejora el AUROC en Llama y ambos Qwen. En Mistral no supera al eje conversacional aplicado en agente: 0,850 frente a 0,878. La tabla separa los dos estimadores de Llama, sobre 42 pares, de los monitores de Mistral y Qwen sobre 96. Qwen3‑32B alcanza F1 de 0,873, con 15,6 por ciento de falsos positivos; Mistral y Qwen3‑8B quedan alrededor de 0,75, con 27,1 y 35,4 por ciento. Ese costo importa: no basta con detectar más dañinas si también se bloquean muchas benignas. El cero observado de la logística de Llama no es una tasa poblacional nula. Sin validación externa, hablamos de viabilidad preliminar.",
    transicion:"¿Y si, además de observar la dirección, la sumo durante la generación?",
    fuentes:"Llama: datos_presentacion.json → llama.monitores, n=42. Mistral/Qwen: evaluacion96[modelo].monitor, n=96. F1 y FPR se conservan como exportados; AUROC es la media OOF guardada. Llama usa capa 76; Mistral 18, Qwen3‑8B 22 y Qwen3‑32B 46."});
}

function respaldo(s,pres,D) {
  const xs=[MX,3.25,4.15,5.65,7.15,8.65];
  [["Modelo",0],["Capa",1],["Chat",2],["Agente",3],["Pérdida",4],["IC 95 % de la diferencia",5]].forEach(([t,k])=>rotulo(s,t,xs[k],1.95,k===5?3.7:1.5));
  keys.forEach((k,i)=>{
    const e=D.evaluacion96[k],d=e.deteccion,y=2.5+i*0.78;
    seg(s,pres,MX,y-0.12,W-MX,y-0.12,C.rule,0.7);
    const values=[e.modelo,String(e.capa),fmt(d.auroc_chat_heldout,4),fmt(d.auroc_agent_heldout,4),fmt(d.auroc_gap_heldout,4),`[${fmt(d.auroc_gap_ci[0],4)}; ${fmt(d.auroc_gap_ci[1],4)}]`];
    values.forEach((v,j)=>txt(s,v,{x:xs[j],y,w:j===0?2.5:j===5?3.8:1.4,h:0.45,fontSize:j===0?15:14,fontFace:j===0?F.demi:F.body}));
  });
  txt(s,"Tres ejecuciones de 96 pares: mismo denominador, diferencias puntuales e incertidumbre distintas.",
    {x:MX,y:5.0,w:11.8,h:0.6,fontFace:F.head,fontSize:21});
  txt(s,"En Mistral el intervalo de la pérdida de AUROC incluye cero; en ambos Qwen lo excluye. La comparación no aísla el tamaño del modelo ni constituye una ley de escala. Llama se mantiene separado por usar 42 pares y una versión distinta del pipeline.",
    {x:MX,y:5.8,w:11.8,h:0.9,fontSize:14,color:C.ink2});
  fuente(s,"results/corridas/{mistral7b,qwen3_8b,qwen3_32b}/v2/results.json → heldout.detection. IC por remuestreo de puntajes retenidos; sin prueba formal entre modelos.");
  nota(s,{titulo:"R7 · Comparación sobre 96 pares",respaldo:true,t:0,
    guion:"Estos tres archivos sí contienen 96 pares. Qwen3‑8B pierde aproximadamente 0,275 de AUROC, Qwen3‑32B 0,036 y Mistral 0,046. El intervalo de Mistral incluye cero: no afirmo allí una diferencia significativa de AUROC al 95 por ciento. En ambos Qwen el intervalo lo excluye. Son medidas del corpus común; las diferencias de arquitectura, entrenamiento y configuración impiden atribuir el patrón exclusivamente al tamaño.",
    fuentes:keys.map(k=>`${D.evaluacion96[k].fuente} → heldout.detection; SHA-256 ${D.evaluacion96[k].sha256}`).join("\n")});
}
module.exports={estratos,monitores,respaldo};
