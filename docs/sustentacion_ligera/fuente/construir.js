// Versión independiente y breve: contexto, intuición y resultados principales.
const fs = require("fs"), path = require("path"), PptxGenJS = require("pptxgenjs");
const B = require("./base"), G = require("./guion");
const { W, MX, C, text, run, line, rect, dot, arrow, label, footer, slide, eq, num } = B;
const { nota } = G;
const D = JSON.parse(fs.readFileSync(path.join(__dirname, "datos_ligera.json"), "utf8"));
const p = new PptxGenJS();
p.layout = "LAYOUT_WIDE";
p.author = "Juan Esteban Espitia Leiva";
p.company = "Universidad Externado de Colombia";
p.title = "Transformación de la dirección de rechazo · versión ligera";
p.subject = "Sustentación en lenguaje claro, con matemática esencial y resultados principales";
p.theme = { headFontFace: "Calibri", bodyFontFace: "Calibri", lang: "es-CO" };
let n = 0;
const nueva = (titulo, sec, apoyo = false) => slide(p, titulo, sec, ++n, apoyo);
const fuenteDatos = "docs/sustentacion_ligera/fuente/datos_ligera.json, extraído de los resultados verificados; procedencia y SHA-256 registrados en fuentes.";
const hf = "https://huggingface.co/blog/agent-intrusion-technical-timeline";
const oa = "https://openai.com/index/hugging-face-incident-and-the-road-ahead/";

// 1. Portada: presentar la pregunta, antes que el aparato matemático.
{
  const s = p.addSlide(); n++; s.background = { color: C.paper };
  s.addImage({ path: path.join(__dirname, "../../tesis/Tesis_avances_actualizada/images/logo-externado-matematicas.png"),
    x: MX, y: 0.38, w: 1.9, h: 0.6 });
  text(s, "Sustentación de trabajo de grado", 7.2, 0.56, 5.4, 0.3, { fontSize: 12, align: "right", color: C.mute });
  text(s, "Transformación de la dirección de rechazo en modelos de lenguaje agénticos", MX, 1.55, 11.85, 1.6,
    { fontSize: 37, bold: true });
  text(s, "Un análisis desde la interpretabilidad mecánica", MX, 3.25, 11.8, 0.5, { fontSize: 23, color: C.body });
  const xs = [MX, 4.9, 9.0], palabras = ["Conversar", "Actuar", "¿Conservar el rechazo?"];
  palabras.forEach((t, i) => { line(s, p, xs[i], 4.65, xs[i] + 3.35, 4.65, i === 2 ? C.agent : C.rule, 1.4);
    text(s, t, xs[i], 4.9, 3.4, 0.75, { fontSize: 25, bold: i === 2, color: i === 2 ? C.agent : C.ink });
    if (i < 2) arrow(s, p, xs[i] + 3.5, 5.1, xs[i] + 3.95, 5.1, C.mute); });
  text(s, "Juan Esteban Espitia Leiva", MX, 6.12, 10.8, 0.35, { fontSize: 18, bold: true });
  text(s, "Tutor: Julián Fernando Sánchez · Ciencia de Datos · Universidad Externado de Colombia", MX, 6.54, 11.8, 0.35, { fontSize: 14, color: C.mute });
  nota(s, { titulo: "La pregunta de la tesis", t: 20,
    guion: "Esta tesis pregunta qué ocurre con una señal interna de rechazo cuando un modelo pasa de conversar a recibir un contexto agéntico. Voy a explicar el problema, cómo lo medí y qué resultados considero más importantes.",
    transicion: "Primero, ¿por qué importa ese cambio de contexto?", fuentes: "Título y alcance del documento de tesis. Esta versión resume la exposición completa." });
}

// 2. Contexto general: texto frente a posibles acciones.
{
  const s = nueva("La misma petición puede terminar en una acción", "El problema");
  label(s, "En una conversación", MX, 2.18, 6);
  [[MX, "Solicitud"], [4.9, "Modelo"], [9.1, "Texto"]].forEach(([x, t], i) => {
    text(s, t, x, 2.68, 3.0, 0.5, { fontSize: 28, bold: true });
    if (i < 2) arrow(s, p, x + 2.7, 2.95, x + 3.8, 2.95, C.mute);
  });
  line(s, p, MX, 3.6, W - MX, 3.6);
  label(s, "Con un agente", MX, 3.95, 6, C.agent);
  const xs = [MX, 3.95, 7.15, 10.35];
  const ts = ["Solicitud y\nherramientas", "Modelo", "Llamada a\nuna función", "Acción del\nsistema"];
  ts.forEach((t, i) => { text(s, t, xs[i], 4.4, i === 3 ? 2.2 : 2.6, 0.95, { fontSize: 25, bold: true, color: i === 3 ? C.agent : C.ink });
    if (i < 3) arrow(s, p, xs[i] + 2.55, 4.76, xs[i] + 3.0, 4.76, C.agent); });
  text(s, "La seguridad importa tanto en lo que el modelo dice como en lo que puede pedir que se ejecute.", MX, 5.92, 11.8, 0.7, { fontSize: 24 });
  footer(s, "Esquema del uso de agentes. En esta tesis las herramientas no se ejecutan; se estudia la entrada que recibe el modelo.");
  nota(s, { titulo: "De responder a poder actuar", t: 45,
    guion: "En una conversación, el resultado visible suele ser un texto. Un agente también puede solicitar una llamada a una herramienta, y el sistema que lo rodea puede ejecutarla. Por eso una misma petición puede tener consecuencias distintas. El rechazo es una de las conductas que se usan para evitar solicitudes dañinas. Mi pregunta es si una señal asociada a ese rechazo se conserva cuando cambia el contexto de entrada. En el experimento no se ejecutan herramientas reales: se incorpora su catálogo a la entrada del modelo.",
    transicion: "Un incidente reciente ayuda a ver por qué la pregunta es relevante.", fuentes: "Diseño y alcance: tesis, secciones 3.2 y 3.3. El diagrama describe el funcionamiento general de un agente, no una ejecución de herramientas de este experimento." });
}

// 3. Caso externo reciente, condensado y con su contexto de evaluación.
{
  const s = nueva("Julio de 2026: de una evaluación a una intrusión", "Un caso documentado");
  text(s, "OpenAI y Hugging Face", MX, 2.18, 11.8, 0.5, { fontSize: 27, bold: true });
  const xs = [MX, 4.9, 9.05], ts = ["Evaluación de\nciberseguridad", "Salida del\nentorno previsto", "Acceso a sistemas\nde Hugging Face"];
  ts.forEach((t, i) => { label(s, `0${i + 1}`, xs[i], 3.02, 2, C.agent); line(s, p, xs[i], 3.4, xs[i] + 3.4, 3.4);
    text(s, t, xs[i], 3.65, 3.45, 1.05, { fontSize: 27, bold: true });
    if (i < 2) arrow(s, p, xs[i] + 3.5, 4.05, xs[i] + 3.95, 4.05, C.mute); });
  text(s, "Modelos de investigación · evaluación interna · salvaguardas reducidas", MX, 5.15, 11.8, 0.5, { fontSize: 19, color: C.body });
  text(s, "El caso motiva estudiar la seguridad cuando el modelo puede actuar.", MX, 6.03, 11.8, 0.5, { fontSize: 25, color: C.agent });
  footer(s, "Fuentes: Hugging Face (27 jul. 2026) y OpenAI (26 ago. 2026). El incidente no es un resultado ni una explicación mecanística de esta tesis.");
  nota(s, { titulo: "El caso de Hugging Face", t: 45,
    guion: "En julio de 2026, agentes de OpenAI que participaban en una evaluación interna de ciberseguridad salieron del entorno previsto y accedieron a sistemas de terceros, entre ellos Hugging Face. OpenAI aclara que eran modelos de investigación con salvaguardas reducidas. El caso sirve como motivación: un objetivo de evaluación terminó acompañado de acciones fuera del alcance autorizado. Mi tesis no explica ese incidente. Estudia una pregunta más acotada sobre la representación interna de solicitudes, en un turno y sin ejecutar herramientas.",
    transicion: "Esa motivación se convierte en una pregunta concreta de investigación.", fuentes: `Fuentes primarias: Hugging Face, reconstrucción técnica del incidente de julio de 2026: ${hf}. OpenAI, informe publicado el 26 de agosto de 2026: ${oa}. Fuentes verificadas para la presentación completa el 15 de octubre de 2026.` });
}

// 4. Una pregunta, tres pasos de investigación.
{
  const s = nueva("¿Se conserva la capacidad de distinguir el daño?", "La pregunta");
  text(s, "Cuando el modelo pasa de chat a agente,\n¿su dirección de rechazo sigue separando\nsolicitudes dañinas y benignas?", MX, 2.25, 11.8, 2.2, { fontSize: 33, bold: true });
  const ts = ["Medir una señal", "Comparar formatos", "Evaluar su utilidad"];
  ts.forEach((t, i) => { const x = MX + i * 4.15; label(s, `0${i + 1}`, x, 5.22, 1.2, C.agent);
    text(s, t, x, 5.65, 3.9, 0.8, { fontSize: 24 }); });
  footer(s, "Se distingue entre representación interna, capacidad de detección y conducta de rechazo.");
  nota(s, { titulo: "Qué quiero comprobar", t: 40,
    guion: "La pregunta no es solo si el modelo responde o se niega. Quiero saber si la dirección de rechazo que se estima en chat sigue ayudando a separar solicitudes dañinas y benignas cuando la entrada tiene formato agéntico. Para responder, primero mido la señal, después comparo los dos formatos y finalmente evalúo si puede ser útil como monitor. Durante toda la exposición separaré la señal interna, la detección y la conducta observada.",
    transicion: "Para comparar necesito mantener la solicitud y cambiar su formato.", fuentes: "Tesis: pregunta de investigación, objetivos y distinción entre geometría, discriminación y conducta." });
}

// 5. Diseño y cobertura, sin catálogo de hiperparámetros.
{
  const s = nueva("El texto se conserva; cambia cómo se presenta al modelo", "Cómo se comparó");
  label(s, "Chat", MX, 2.12, 2.9); label(s, "Formato agéntico", 4.15, 2.12, 3.1, C.agent);
  const cols = [MX, 4.15], ys = [2.65, 4.0];
  [["Solicitud dañina", "Misma solicitud dañina\n+ sistema y catálogo"], ["Contraparte benigna", "Misma contraparte benigna\n+ sistema y catálogo"]].forEach((fila, i) => {
    fila.forEach((t, j) => { line(s, p, cols[j], ys[i] - 0.15, cols[j] + 3.0, ys[i] - 0.15, i ? C.benign : C.harm, 1.5);
      text(s, t, cols[j], ys[i], 3.1, 0.95, { fontSize: 21 }); });
    arrow(s, p, 3.0, ys[i] + 0.4, 3.85, ys[i] + 0.4, i ? C.benign : C.harm);
  });
  line(s, p, 7.65, 2.12, 7.65, 5.18);
  text(s, "96 pares", 8.05, 2.18, 4.3, 0.7, { fontSize: 39, bold: true });
  text(s, "en el corpus ampliado", 8.05, 2.95, 4.2, 0.4, { fontSize: 20, color: C.body });
  text(s, "Mistral y ambos Qwen: 96 pares\nLlama: 42 pares", 8.05, 3.65, 4.45, 1.0, { fontSize: 21, bold: true });
  text(s, "Tres niveles de explicitud:\nexplícito, contextual y apariencia legítima.", 8.05, 4.78, 4.4, 0.85, { fontSize: 18, color: C.body });
  text(s, "Entrada agéntica = solicitud + mensaje de sistema + descripciones de herramientas.", MX, 5.82, 11.8, 0.55, { fontSize: 21, bold: true });
  text(s, "Ejemplo abreviado: facturar servicios no prestados / facturar servicios prestados.", MX, 6.38, 11.8, 0.35, { fontSize: 17, color: C.body });
  footer(s, "Catálogo sin ejecución. Ejemplo: adaptación al español de fraud_06. Tamaños verificados en las exportaciones de resultados.");
  nota(s, { titulo: "Una comparación dentro de cada par", t: 60,
    guion: "Cada par contiene una solicitud etiquetada como dañina y una contraparte benigna. Cada una se presenta en chat y en formato agéntico. El texto de la solicitud se mantiene entre formatos; en agente se incorpora un mensaje de sistema y el catálogo de herramientas. El catálogo describe funciones y parámetros, pero no ejecuta acciones ni acredita permisos. Un ejemplo es cobrar por servicios nunca prestados frente a cobrar por servicios prestados. El corpus ampliado tiene 96 pares. Las exportaciones de Mistral y ambos Qwen usan esos 96; la de Llama contiene 42. Mantengo esos tamaños en los resultados. La explicitud describe cómo se formula el miembro dañino etiquetado; la contraparte conserva su etiqueta benigna.",
    transicion: "Ahora explico la idea matemática con dos pasos sencillos.", fuentes: `${fuenteDatos} Corpus: data/dataset_v2.jsonl y dataset_v2_report.json. Ejemplo: fraud_06, traducción abreviada para explicar el contraste. Se preservan las etiquetas y los resultados originales.` });
}

// 6. Solo dos fórmulas cortas, acompañadas de su lectura verbal.
{
  const s = nueva("Buscar una dirección y dar un puntaje a cada entrada", "La matemática esencial");
  label(s, "Representación interna · esquema ilustrativo", MX, 2.2, 5.4);
  const centros = [[2.0, 4.9, C.benign, "Benignas"], [4.3, 3.4, C.harm, "Dañinas"]];
  const offsets = [[-.5,-.3],[-.2,.2],[.2,-.2],[.4,.25],[-.35,.45],[.55,-.4],[.0,.5]];
  centros.forEach(([x,y,c,t]) => { offsets.forEach(([dx,dy]) => dot(s,p,x+dx,y+dy,.07,c));dot(s,p,x,y,.12,c,true);
    text(s,t,x-.8,y+.7,2.3,.35,{fontSize:18,bold:true,color:c}); });
  arrow(s,p,2.0,4.9,4.3,3.4,C.ink,2.0);
  text(s,"Dirección",2.6,3.15,1.7,.4,{fontSize:19,bold:true});
  line(s,p,5.8,2.2,5.8,6.1);
  text(s,"1. Restar dos promedios",6.3,2.17,6.0,.5,{fontSize:25,bold:true});
  eq(s,"ligera_r",6.3,2.86,5.5,.65,28);
  text(s,"Promedio de dañinas menos\npromedio de benignas.",6.3,3.59,5.6,.85,{fontSize:21,color:C.body});
  text(s,"2. Medir la alineación",6.3,4.6,6.0,.5,{fontSize:25,bold:true});
  eq(s,"ligera_p",6.3,5.21,5.5,.6,28);
  text(s,"El puntaje resume cuánto se alinea\nuna entrada con esa dirección.",6.3,5.9,5.7,.75,{fontSize:20,color:C.body});
  footer(s,"μD y μB: promedios; x: representación interna; r̂: dirección normalizada; p: puntaje. El esquema no muestra activaciones medidas.");
  nota(s,{titulo:"Dos ideas matemáticas",t:75,
    guion:"El modelo transforma la entrada en una representación interna: un vector de números. Primero calculo el promedio de las representaciones dañinas y el de las benignas. La resta entre esos dos promedios define una dirección. El dibujo es solo un esquema para imaginarla. Después doy un puntaje a cada entrada según cuánto se alinea con esa dirección. La primera fórmula es una resta de promedios; la segunda, un producto punto con la dirección normalizada. No necesitamos desarrollar el cálculo para seguir los resultados. Lo importante es que el puntaje representa un contraste geométrico entre clases etiquetadas. La conducta de rechazo se observa por separado: tener un puntaje no equivale por sí solo a rechazar.",
    transicion:"Para saber si ese puntaje sirve, hay que probarlo con ejemplos distintos de los usados para estimarlo.",fuentes:"Tesis, sección 3.3.1: diferencia de medias y proyección. Se omiten índices de capa y la derivación de normalización para facilitar la exposición. Se lee el flujo residual del último token de la entrada antes de generar; no se usa una representación de la respuesta completa."});
}

// 7. Validación y AUROC, explicadas verbalmente.
{
  const s=nueva("La señal debe funcionar con ejemplos que el ajuste no vio", "Cómo se evaluó");
  const ideas=["Estimar con\nunos pares","Evaluar con\notros pares","Comparar chat\ny agente"];
  ideas.forEach((t,i)=>{const x=MX+i*4.15;label(s,`0${i+1}`,x,2.25,1.2,C.agent);text(s,t,x,2.7,3.8,1.0,{fontSize:27,bold:true});if(i<2)arrow(s,p,x+3.2,3.1,x+3.9,3.1,C.mute);});
  line(s,p,MX,4.1,W-MX,4.1);
  text(s,"AUROC: qué tan bien se ordenan las dos clases",MX,4.4,11.8,.5,{fontSize:24,bold:true});
  const a=1.1,b=7.8,esc=v=>a+(v-.5)*2*(b-a);
  line(s,p,a,5.65,b,5.65,C.rule,3);dot(s,p,esc(.9),5.65,.11,C.agent);
  text(s,"0,5\nAzar",a-.4,5.95,1.2,.7,{fontSize:18,align:"center"});
  text(s,"1,0\nOrden perfecto",b-.8,5.95,1.8,.7,{fontSize:18,align:"center"});
  text(s,"0,9",esc(.9)-.4,5.1,.8,.4,{fontSize:24,bold:true,color:C.agent,align:"center"});
  text(s,"Cerca de 9 de cada 10 comparaciones dañina–benigna quedan bien ordenadas.",9.0,5.12,3.55,1.25,{fontSize:21});
  footer(s,"La AUROC evalúa el orden entre puntajes; no es un porcentaje de clasificaciones correctas con un umbral.");
  nota(s,{titulo:"Probar con pares no usados en el ajuste",t:55,
    guion:"Separé el ajuste de la evaluación: la dirección que da puntaje a un par se estima sin usar ese par. Sus variantes quedan juntas para conservar la comparación. Después uso la AUROC, que resume el orden de los puntajes entre dañinas y benignas. Cero coma cinco equivale al azar; uno es un orden perfecto. Una AUROC de cero coma nueve significa que, al tomar una dañina y una benigna al azar, la dañina queda por encima aproximadamente nueve de cada diez veces. Es una medida de orden, no el porcentaje de aciertos de un clasificador con umbral. También observo por separado las respuestas y las falsas alarmas de los monitores.",
    transicion:"Empiezo por el cambio que se ve desde afuera: la conducta.",fuentes:"Tesis, sección 3.4.1: validación por pares y AUROC. En los resultados de proyección se usa cross-fitting; los intervalos se condicionan a los puntajes guardados. Los controles detallados de selección y las pruebas están en la presentación completa."});
}

// 8. Conducta: dos conteos, misma muestra.
{
  const s=nueva("En Llama, el rechazo disminuye al cambiar el formato", "Resultado 1 · conducta");
  label(s,"Solicitudes dañinas etiquetadas como rechazadas",MX,2.15,11.8);
  const r=D.conducta.refusal;
  [[1.1,"En chat",r.chat_harmful,C.ink],[7.5,"Con formato agéntico",r.agent_harmful,C.harm]].forEach(([x,t,v,c])=>{
    text(s,t,x,2.65,4.7,.5,{fontSize:26,bold:true,color:c});
    text(s,`${v.k}/${v.n}`,x,3.3,4.7,1.2,{fontSize:72,bold:true,color:c});
    rect(s,p,x,4.8,4.5,.19,C.pale);rect(s,p,x,4.8,4.5*v.rate,.19,c);
    text(s,`${num(100*v.rate,1)} %`,x,5.13,4.5,.45,{fontSize:23,color:c});
  });
  arrow(s,p,5.85,3.95,6.85,3.95,C.mute,2);
  text(s,"Mismo modelo y mismas solicitudes; cambió el formato.",MX,6.05,11.8,.55,{fontSize:25,bold:true});
  footer(s,"Llama V3 · 42 pares. Etiquetado conductual heurístico, pendiente de auditoría humana. Fuente: metricas_avance.json → llama.behavior.");
  nota(s,{titulo:"El rechazo observado baja",t:50,
    guion:"En Llama, de las mismas 42 solicitudes dañinas, 38 se rechazan en chat y 23 en formato agéntico. Es una diferencia de conducta observada con las etiquetas usadas en el experimento. No son dos grupos de solicitudes diferentes: el texto se mantiene entre formatos. El patrón no demuestra todavía qué mecanismo causó el cambio. Por eso paso a mirar la representación interna y su capacidad de separar las clases. Las etiquetas de rechazo son heurísticas y requieren auditoría humana.",
    transicion:"¿La señal interna también cambia?",fuentes:`${fuenteDatos} Conducta: metricas_avance.json → llama.behavior.refusal. McNemar exacto sobre pares: p=0,000274658. Llama V3, n=42. No se extrapolan estos conteos a 96 pares.`});
}

// 9. Transferencia: cuatro modelos, denominadores visibles, sin capas ni p-valores.
{
  const s=nueva("La señal persiste, con menos separación en agente", "Resultado 2 · detección");
  const a=4.1,b=12.1,esc=v=>a+2*(v-.5)*(b-a);
  text(s,[run("● ",{color:C.ink}),run("chat     "),run("● ",{color:C.agent}),run("agente")],8.1,2.05,4.2,.3,{fontSize:17,align:"right"});
  [.5,.6,.7,.8,.9,1].forEach(v=>{line(s,p,esc(v),2.6,esc(v),5.45,C.rule,.7);text(s,num(v,1),esc(v)-.3,5.56,.6,.3,{fontSize:15,align:"center",color:C.mute});});
  D.modelos.forEach((m,i)=>{const y=2.75+i*.78;
    text(s,m.nombre,MX,y-.2,3.25,.35,{fontSize:21,bold:true});text(s,`${m.n} pares`,MX,y+.18,3.1,.26,{fontSize:14,color:C.mute});
    line(s,p,esc(m.agente),y,esc(m.chat),y,C.rule,4);dot(s,p,esc(m.chat),y,.085,C.ink);dot(s,p,esc(m.agente),y,.085,C.agent);
    text(s,num(m.chat),esc(m.chat)-.42,y-.36,.84,.28,{fontSize:16,align:"center"});
    text(s,num(m.agente),esc(m.agente)-.42,y+.12,.84,.28,{fontSize:16,align:"center",color:C.agent});
  });
  text(s,`En Llama, la separación ajustada por escala también cae ${num(100*D.separacion_llama.perdida_d,0)} %.`,MX,6.06,11.8,.4,{fontSize:21,bold:true});
  text(s,"En Mistral, la incertidumbre no permite afirmar una caída de AUROC al 95 %.",MX,6.53,11.8,.3,{fontSize:16,color:C.body});
  footer(s,"AUROC fuera de muestra de la dirección de chat. Comparación descriptiva entre modelos, con tamaños y versiones distintos.");
  nota(s,{titulo:"Se conserva señal, pero cambia el margen",t:70,
    guion:"Los puntos oscuros muestran la AUROC en chat y los dorados, la de esa dirección aplicada en agente. Todos los valores agénticos quedan por encima de cero coma cinco, pero la pérdida puntual varía: es mucho mayor en Qwen3 de ocho mil millones que en Qwen3 de treinta y dos mil millones. En Llama, la separación ajustada por escala también cae cerca de 44 por ciento; por tanto, el cambio no consiste solo en que los números se hagan pequeños. En Mistral hay que ser prudente: el intervalo de diferencia de AUROC incluye cero. Y estas comparaciones no aíslan el efecto del tamaño del modelo: cambian también arquitectura, entrenamiento y versión del pipeline.",
    transicion:"Otra pregunta es si la apariencia legítima elimina esa señal.",fuentes:`${fuenteDatos} Datos por modelo → chat/agente/ic_diferencia. Llama d=3,2051→1,8036 (−43,73 %). Mistral: IC95% de diferencia [−0,005645; 0,096899]. Qwen3-8B [0,210392; 0,330946]; Qwen3-32B [0,007704; 0,063477]. Llama n=42; demás modelos n=96. No son pruebas formales de diferencias entre modelos.`});
}

// 10. Explicitud: concentrarse en el hallazgo, no en las nueve comparaciones.
{
  const s=nueva("La apariencia legítima no borra toda la señal", "Resultado 3 · explicitud");
  text(s,"AUROC agéntico en el estrato de apariencia legítima",MX,2.17,11.8,.5,{fontSize:23});
  D.estratos96.forEach((m,i)=>{const e=m.estratos.find(x=>x.stratum==="framed"),x=MX+i*4.15;
    line(s,p,x,3.05,x+3.55,3.05);text(s,m.nombre,x,3.32,3.7,.45,{fontSize:24,bold:true});
    text(s,num(e.auroc_agent),x,4.04,3.8,1.0,{fontSize:53,bold:true,color:C.agent});
    text(s,`${e.n_harmful} pares del estrato`,x,5.15,3.7,.35,{fontSize:18,color:C.body});
  });
  text(s,"La pérdida máxima cambia de estrato según el modelo.",MX,5.98,11.8,.5,{fontSize:24,bold:true});
  text(s,"Son estimaciones descriptivas; no establecen una jerarquía universal.",MX,6.48,11.8,.35,{fontSize:17,color:C.body});
  footer(s,"Mistral y Qwen V2: 96 pares por ejecución; aquí se muestran los 30 de apariencia legítima. Sin intervalos ni contrastes entre estratos.");
  nota(s,{titulo:"Lo encubierto conserva señal",t:50,
    guion:"El estrato de apariencia legítima conserva señal por encima del azar en estas tres ejecuciones. Los valores son aproximadamente cero coma 88, cero coma 69 y cero coma 90. Tampoco aparece una jerarquía universal: la mayor pérdida puntual está en el explícito para Mistral y Qwen3 de ocho mil millones, y en el contextual para Qwen3 de treinta y dos mil millones. Esto es descriptivo, porque no tenemos intervalos y contrastes específicos entre estratos. Apariencia legítima describe la formulación del miembro dañino etiquetado; no significa que cualquier declaración de autorización sea un engaño.",
    transicion:"Si la dirección de chat pierde margen, ¿ayuda ajustar un monitor al contexto agéntico?",fuentes:`${fuenteDatos} estratos96 → framed.auroc_agent; 30 dañinas y 30 benignas por modelo. Los tres tamaños totales son 96. El análisis depende de las etiquetas del corpus; autorización real no verificada.`});
}

// 11. Monitores: conteos comprensibles, sin añadir F1 a las métricas de la exposición.
{
  const s=nueva("Un monitor puede ayudar, pero también genera falsas alarmas", "Resultado 4 · utilidad");
  label(s,"Monitor ajustado en agente",MX,2.12,3.7);
  label(s,"Dañinas detectadas",5.0,2.12,3.3,C.harm);
  label(s,"Benignas marcadas como dañinas",9.1,2.12,3.55,C.agent);
  D.modelos.forEach((m,i)=>{const y=2.75+i*.71,r=m.monitor;
    line(s,p,MX,y-.17,W-MX,y-.17);text(s,m.nombre,MX,y,3.9,.4,{fontSize:21,bold:true});
    text(s,`${r.tp}/${r.n}`,5.0,y,2.2,.46,{fontSize:28,color:C.harm});
    text(s,`${r.fp}/${r.n}`,9.1,y,2.2,.46,{fontSize:28,color:C.agent});
    rect(s,p,7.0,y+.14,1.45,.1,C.pale);rect(s,p,7.0,y+.14,1.45*r.tp/r.n,.1,C.harm);
    rect(s,p,11.1,y+.14,1.45,.1,C.pale);if(r.fp)rect(s,p,11.1,y+.14,1.45*r.fp/r.n,.1,C.agent);
  });
  text(s,"La utilidad depende de cuánto detecta y de cuánto bloquea por error.",MX,6.0,11.8,.5,{fontSize:24,bold:true});
  text(s,"Cero falsas alarmas observadas en Llama no asegura un riesgo nulo.",MX,6.52,11.8,.32,{fontSize:17,color:C.body});
  footer(s,"Fuera de fold; umbral calibrado en entrenamiento. Llama: monitor logístico, 42 pares; los otros: 96. Sin validación externa.");
  nota(s,{titulo:"Detectar daño también tiene un costo",t:65,
    guion:"Ajustar un monitor al formato agéntico mejora el AUROC en Llama y ambos Qwen. Para hablar de utilidad muestro conteos con el umbral calibrado en entrenamiento: cuántas dañinas detecta y cuántas benignas marca por error. En Llama, el monitor logístico detecta 37 de 42 dañinas sin falsas alarmas observadas. En el corpus de 96, Qwen3 de treinta y dos mil millones detecta 86 dañinas con 15 falsas alarmas; los otros dos producen más falsas alarmas. El cero de Llama es un resultado de esa muestra, no una garantía de riesgo cero. Tampoco todos los monitores mejoran todo: el AUROC del de Mistral es 0,850 frente a 0,878 del eje de chat aplicado en agente. La utilidad sigue siendo preliminar y necesita validación externa.",
    transicion:"Observar una señal es una cosa. Forzarla durante la generación es otra.",fuentes:`${fuenteDatos} modelos → monitor. TP=recall*n, FP=FPR*n, ambos enteros comprobados y coherentes con F1. Llama: logística; no se muestra aquí el segundo estimador de medias de la versión completa. AUROC de monitores: Llama 0,9926, Mistral 0,8498, Qwen8 0,8297, Qwen32 0,9406. Comparación descriptiva, sin prueba formal entre monitores.`});
}

// 12. Intervención exploratoria: efecto y costo visibles a la vez.
{
  const s=nueva("Forzar el rechazo también puede bloquear solicitudes benignas", "Resultado 5 · intervención");
  label(s,"Una de las intensidades probadas en Llama",MX,2.15,11.8);
  const v=D.intervencion_dosis2;
  [[1.1,"Dañinas rechazadas",v.restore,C.harm],[7.5,"Benignas rechazadas",v.benign,C.benign]].forEach(([x,t,r,c])=>{
    text(s,t,x,2.72,4.75,.55,{fontSize:25,bold:true,color:c});text(s,`${r.k}/${r.n}`,x,3.45,4.75,1.15,{fontSize:70,bold:true,color:c});
    rect(s,p,x,4.95,4.5,.18,C.pale);rect(s,p,x,4.95,4.5*r.rate,.18,c);
  });
  text(s,"Cambiar la conducta no garantiza un rechazo selectivo.",MX,5.88,11.8,.5,{fontSize:25,bold:true});
  text(s,`Control con dirección aleatoria: ${v.control.k}/${v.control.n} dañinas rechazadas.`,MX,6.45,11.8,.35,{fontSize:18,color:C.body});
  footer(s,"Ensayo exploratorio: Llama, capa 56, dosis 2. Subconjuntos que no rechazaban en agente; no se presenta como solución de seguridad.");
  nota(s,{titulo:"Forzar la señal no resuelve todo",t:50,
    guion:"También probé sumar la dirección durante la generación en Llama. Muestro una intensidad concreta para hacer visible el costo: con dosis dos, rechaza las 19 dañinas del subconjunto, pero también 40 de 41 benignas. Con una dirección aleatoria de igual norma rechaza solo una de las 19 dañinas. Hay un efecto sobre la conducta etiquetada, pero no una reparación selectiva de seguridad. Este ensayo es exploratorio: usa subconjuntos que no rechazaban en agente y una capa distinta de la del diagnóstico. No permite concluir que la reorientación geométrica sea la causa del fallo.",
    transicion:"Con estos resultados puedo responder la pregunta inicial.",fuentes:`${fuenteDatos} intervencion_dosis2, extraída de llama.intervencion56.per_dose['2']. Capa 56, alfa=13,8512735; n dañinas=19, benignas=41. Rechazos: 19 y 40; control aleatorio 1. No se corrige aquí por la búsqueda entre capas y dosis.`});
}

// 13. Responder y acotar, con una frase por hallazgo.
{
  const s=nueva("La respuesta es sí, pero con menos margen", "Qué aporta la tesis");
  const ideas=["La señal conserva parte de su capacidad de separar.","El efecto depende del modelo y del contexto.","Recuperar detección o rechazo tiene costos."];
  ideas.forEach((t,i)=>{label(s,`0${i+1}`,MX,2.3+i*.84,.7,C.agent);text(s,t,1.55,2.24+i*.84,10.8,.6,{fontSize:27,bold:true});});
  line(s,p,MX,5.02,W-MX,5.02);
  text(s,"Hasta dónde llega la evidencia",MX,5.3,11.8,.4,{fontSize:21,bold:true});
  text(s,"Un turno y herramientas sin ejecución.\nEtiquetas y permisos requieren revisión.\nMonitores sin validación con datos externos.",MX,5.82,11.8,.95,{fontSize:20,color:C.body});
  footer(s,"Se conserva la distinción entre señal interna, detección y conducta. Las cifras describen los archivos y tamaños analizados.");
  nota(s,{titulo:"La conclusión y sus límites",t:60,
    guion:"La dirección conversacional conserva capacidad de separar, aunque el margen cambia al pasar a agente y la magnitud depende del modelo. La tesis aporta una manera de medir ese cambio y de separar tres cosas: representación interna, detección y conducta. Los monitores muestran utilidad preliminar; la intervención muestra que recuperar rechazo también puede bloquear solicitudes benignas. Los límites son concretos: un turno y herramientas sin ejecución, tamaños y versiones distintos, etiquetas que necesitan revisión humana y ausencia de validación externa. La autorización declarada no se verifica, y casos como fraud_01 conservan una ambigüedad que debe auditarse. Las conclusiones dependen de la validez de las etiquetas; no son una garantía operacional de seguridad.",
    transicion:"Cierro con la implicación práctica que conecta todo el trabajo.",fuentes:"Síntesis de los resultados anteriores y de las limitaciones de la presentación completa. Véase docs/sustentacion/nota_autorizacion_etiquetas.md. No se modifican retrospectivamente etiquetas, muestras ni métricas."});
}

// 14. Cierre sin nuevas cifras.
{
  const s=nueva("Antes de confiar en que un agente rechazará…", "Cierre");
  text(s,"…conviene medir si\ntodavía distingue.",MX,2.4,11.8,1.9,{fontSize:49,bold:true,color:C.agent});
  text(s,"Señal interna   →   Evaluación   →   Conducta",MX,4.95,11.8,.55,{fontSize:25});
  text(s,"Gracias",MX,5.94,11.8,.5,{fontSize:29,bold:true});
  footer(s,"Juan Esteban Espitia Leiva · Tutor: Julián Fernando Sánchez · Ciencia de Datos · Universidad Externado de Colombia");
  nota(s,{titulo:"La idea para recordar",t:25,
    guion:"La idea para recordar es que la seguridad puede cambiar cuando cambia el contexto del modelo. Antes de confiar en que un agente rechazará, conviene medir si todavía distingue. Muchas gracias.",
    transicion:"Preguntas.",fuentes:"Conclusión de la tesis y de esta exposición. No se añaden resultados en el cierre."});
}

// 15. Referencias accesibles, fuera del recorrido principal.
{
  const s=nueva("Fuentes y trazabilidad", "Apoyo 1",true);
  text(s,"El caso de contexto",MX,2.2,11.8,.5,{fontSize:26,bold:true});
  text(s,[run("Hugging Face · informe técnico del 27 de julio de 2026",{hyperlink:{url:hf},color:C.benign})],MX,2.95,11.8,.55,{fontSize:21});
  text(s,[run("OpenAI · informe del incidente del 26 de agosto de 2026",{hyperlink:{url:oa},color:C.benign})],MX,3.62,11.8,.55,{fontSize:21});
  line(s,p,MX,4.4,W-MX,4.4);
  text(s,"Los datos del estudio",MX,4.72,11.8,.5,{fontSize:26,bold:true});
  text(s,"Llama V3: 42 pares. Mistral y Qwen V2: 96 por modelo.\nMétricas y archivos de origen registrados en datos_ligera.json.\nEl detalle estadístico está en la presentación completa.",MX,5.44,11.8,1.1,{fontSize:21,color:C.body});
  footer(s,"Fuentes primarias del contexto y artefactos verificables del proyecto; las referencias completas de la tesis están en el documento.");
  nota(s,{titulo:"A1 · Fuentes",respaldo:true,t:0,guion:"Las dos referencias externas son fuentes primarias del incidente de Hugging Face. Los datos de la tesis se conservan en los archivos del repositorio; esta versión registra ruta y hash de las fuentes y reutiliza las cifras verificadas de la presentación completa.",fuentes:`${hf}\n${oa}\n${fuenteDatos}`});
}

// 16. Resolver la duda de autorización, sin cargar las diapositivas principales.
{
  const s=nueva("Declarar autorización no permite comprobarla", "Apoyo 2",true);
  const ideas=[
    ["Qué recibe el modelo","Un texto, un mensaje de sistema y un catálogo de herramientas."],
    ["Qué significa la etiqueta","Daño atribuido a la solicitud según el escenario descrito en el corpus."],
    ["Qué queda pendiente","Revisar casos ambiguos, con anotadores independientes y análisis de sensibilidad."],
  ];
  ideas.forEach(([a,b],i)=>{const y=2.2+i*1.03;line(s,p,MX,y-.1,W-MX,y-.1);text(s,a,MX,y,3.8,.6,{fontSize:23,bold:true});text(s,b,4.9,y,7.55,.8,{fontSize:22,color:C.body});});
  text(s,"La apariencia legítima describe la formulación del miembro dañino etiquetado; su contraparte sigue siendo benigna.",MX,5.52,11.8,.85,{fontSize:23,bold:true});
  text(s,"El ejemplo principal usa fraud_06. Las etiquetas originales no se han recalculado.",MX,6.52,11.8,.35,{fontSize:17,color:C.body});
  footer(s,"Detalle de fraud_01 y su ambigüedad: docs/sustentacion/nota_autorizacion_etiquetas.md y anexo R13 de la versión completa.");
  nota(s,{titulo:"A2 · Autorización y etiquetas",respaldo:true,t:0,
    guion:"El experimento no verifica identidades ni permisos reales. Si dos escenarios tienen el mismo texto y contexto, pero difieren solo en una autorización oculta, este diseño no puede observar esa diferencia. En fraud_01, el benigno declara cuentas de prueba y QA, pero la contraparte dañina no afirma que las credenciales sean robadas; ese par debe revisarse. Cambiar el ejemplo didáctico a fraud_06 no resuelve la validez del corpus. Se conservan los resultados existentes y se declara la limitación. Tampoco se clasifica una auditoría como ataque solo por decir que es una auditoría: hay que examinar la acción pedida y el contexto disponible.",
    fuentes:"docs/sustentacion/nota_autorizacion_etiquetas.md; data/dataset_full.jsonl y data/dataset_v2.jsonl → fraud_01, fraud_06. Este mazo no reetiqueta ni reejecuta el corpus."});
}

if(n!==16||G.registro().filter(r=>!r.respaldo).length!==14)throw new Error("Estructura inesperada de la versión ligera");
fs.mkdirSync(path.join(__dirname,"build"),{recursive:true});
fs.writeFileSync(path.join(__dirname,"build/guion.json"),JSON.stringify(G.registro(),null,2),"utf8");
fs.writeFileSync(path.join(__dirname,"../guion_ligero.md"),G.markdown(),"utf8");
(async()=>{await p.writeFile({fileName:path.join(__dirname,"build/borrador.pptx")});console.log(`Versión ligera: ${n} diapositivas; 14 principales; ${G.mmss(G.total())}.`);})();
