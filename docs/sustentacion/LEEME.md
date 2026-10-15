# Presentación de sustentación

| Archivo | Contenido |
|---|---|
| `Sustentacion_Espitia_Leiva.pptx` | Presentación editable, 16:9: 23 diapositivas de exposición (≈ 19 min) y 12 de respaldo (R1–R12). Notas del presentador en cada diapositiva. |
| `Sustentacion_Espitia_Leiva.pdf` | Exportación de PowerPoint, una página por diapositiva. |
| `Sustentacion_Espitia_Leiva_guion.pdf` | Guion imprimible: miniatura, texto oral, transición, tiempo y fuentes de cada diapositiva. |
| `guion_sustentacion.md` | El mismo guion en texto plano. |
| `fuente/` | Código que genera todo lo anterior a partir de los resultados del repositorio. |

## Qué es editable

- Ecuaciones: OMML nativo; doble clic abre el editor de ecuaciones de PowerPoint. Cada una lleva una imagen de respaldo para visores sin soporte de ecuaciones (Google Slides, Keynote).
- Curvas por capa (diap. 14 y R2) y respuesta a la dosis (diap. 20): gráficos nativos con sus datos; la diap. 20 lleva barras de error con los IC de Wilson guardados. Las etiquetas del eje Y de los gráficos por capa están escritas aparte, con coma decimal.
- Proyecciones, enjambres, pendientes, dumbbells y conteos: formas dibujadas desde los datos (no imágenes).
- Tipografía: Georgia (títulos), Franklin Gothic (texto), Cambria Math (notación). Vienen con Windows y Office.

## Regenerar

Requiere Windows con PowerPoint y Microsoft Edge, Node.js y Python con `numpy`, `Pillow` y `comtypes`. No ejecuta modelos.

```bash
cd docs/sustentacion/fuente
npm install pptxgenjs@3.12.0   # solo la primera vez
bash build.sh
```

`datos.py` extrae todas las cifras a `datos_presentacion.json` y comprueba con aserciones que coinciden con las tablas de la tesis (AUROC, desplazamientos, interacción, separación estandarizada, conteos). El resto del proceso: `construir.js` (diapositivas y notas) → `postproceso.py` (ecuaciones y barras de error) → `respaldo_ecuaciones.py` (imágenes de respaldo) → `exportar.py` (PowerPoint guarda el `.pptx` y exporta el PDF) → `guion_pdf.py`.

## Para decidir antes de sustentar

- **R8 (factorial sistema/herramientas)** muestra cifras que están en `factorial_results.json` de la corrida V3, pero no en el documento de tesis. Está marcada como exploratoria y no concluyente; si no quiere presentar resultados fuera del documento, elimínela.
- La diapositiva 3 usa un caso externo (Anthropic, nov. 2025) y la 2 cifras de Gartner (2025) y del International AI Safety Report (2026). No están en `references.bib`; se citan en R11 con asterisco.
