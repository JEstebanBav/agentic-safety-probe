# Presentación de sustentación

| Archivo | Contenido |
|---|---|
| `Sustentacion_Espitia_Leiva.pptx` | Presentación editable, 16:9: 23 diapositivas de exposición (≈ 20 min) y 13 de respaldo (R1–R13). Notas del presentador en cada diapositiva. |
| `Sustentacion_Espitia_Leiva.pdf` | Exportación de PowerPoint, una página por diapositiva. |
| `Sustentacion_Espitia_Leiva_guion.pdf` | Guion imprimible: miniatura, texto oral, transición, tiempo y fuentes de cada diapositiva. |
| `guion_sustentacion.md` | El mismo guion en texto plano. |
| `fuente/` | Código que genera todo lo anterior a partir de los resultados del repositorio. |
| `nota_autorizacion_etiquetas.md` | Revisión del ejemplo de fraude, autorización declarada y límites de validez del etiquetado. |

## Qué es editable

- Ecuaciones: OMML nativo; doble clic abre el editor de ecuaciones de PowerPoint. Cada una lleva una imagen de respaldo para visores sin soporte de ecuaciones (Google Slides, Keynote).
- Curvas por capa (diap. 14 y R2) y respuesta a la dosis (diap. 20): gráficos nativos con sus datos; la diap. 20 lleva barras de error con los IC de Wilson guardados. Las etiquetas del eje Y de los gráficos por capa están escritas aparte, con coma decimal.
- Proyecciones, enjambres, pendientes, dumbbells y conteos: formas dibujadas desde los datos (no imágenes).
- Tipografía: Palatino Linotype (títulos), Calibri (texto, con negrita nativa) y Cambria Math (notación).
- Diseño: fondo claro cálido, composiciones abiertas y reglas discretas; se sustituyeron las tarjetas y las diapositivas oscuras. Se conservan las figuras, ecuaciones y el relato científico.

## Cobertura del estudio y procedencia de cifras

Las exportaciones verificadas contienen **96 pares en Mistral-7B, Qwen3-8B y Qwen3-32B**, y **42 pares en Llama-3.3-70B**. El corpus ampliado tiene 96 pares: 42 explícitos, 24 contextuales y 30 de apariencia legítima. La presentación indica el tamaño efectivo de cada evaluación.

Los resultados completos incorporados desde la carpeta entregada por el autor están en `results/corridas/mistral7b/v2`, `results/corridas/qwen3_8b/v2` y `results/corridas/qwen3_32b/v2`. Actualizan las comparativas por explicitud, los monitores y el respaldo entre modelos; Mistral V2 también actualiza los paneles de desplazamiento y escala. Su intervalo de diferencia de AUROC incluye cero, por lo que no se presenta como una pérdida estadísticamente significativa al 95 %.

Los análisis de Llama V3 conservan sus 42 pares. Los ensayos complementarios de conducta, PCA y selección de Mistral V3 también usan 42 y se identifican por separado. `fuente/cobertura_estudio.json` registra los tamaños verificados; `datos_presentacion.json → evaluacion96` registra archivos y hashes SHA-256 de las nuevas fuentes. Los resultados originales permanecen intactos.

## Regenerar

Requiere Windows con PowerPoint y Microsoft Edge, Node.js y Python con `numpy`, `Pillow` y `comtypes`. No ejecuta modelos.

```bash
cd docs/sustentacion/fuente
npm install pptxgenjs@3.12.0   # solo la primera vez
bash build.sh
```

La variable `PY` permite seleccionar el intérprete de Python del entorno local; Node debe estar en `PATH`. Se puede ejecutar `build.sh` con Git Bash en Windows.

`datos.py` extrae las cifras a `datos_presentacion.json`, conserva las comprobaciones de los análisis de 42 pares y verifica los tamaños de las nuevas exportaciones de 96. Las nuevas comparativas amplían los resultados del documento de tesis. El resto del proceso: `construir.js` (diapositivas y notas) → `postproceso.py` (ecuaciones y barras de error) → `respaldo_ecuaciones.py` (imágenes de respaldo) → `exportar.py` (PowerPoint guarda el `.pptx` y exporta el PDF) → `guion_pdf.py`.

## Para decidir antes de sustentar

- **R8 (factorial sistema/herramientas)** muestra cifras que están en `factorial_results.json` de la corrida V3, pero no en el documento de tesis. Está marcada como exploratoria y no concluyente; si no quiere presentar resultados fuera del documento, elimínela.
- La diapositiva 3 usa el incidente de Hugging Face de julio de 2026, con los informes primarios de [Hugging Face](https://huggingface.co/blog/agent-intrusion-technical-timeline) y [OpenAI](https://openai.com/index/hugging-face-incident-and-the-road-ahead/). La 2 usa cifras de Gartner (2025) y del International AI Safety Report (2026). Son referencias externas al documento, identificadas en R11 con asterisco.
- El anexo **R13** explica por qué la autorización declarada no se verifica en este diseño y señala la ambigüedad de `fraud_01`. El ejemplo principal se sustituyó por `fraud_06`, sin modificar los datos ni las métricas. Véase [la nota de revisión](nota_autorizacion_etiquetas.md).
