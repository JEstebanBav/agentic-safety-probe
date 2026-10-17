# Tesis: versión de avances

Esta versión usa únicamente evidencia disponible: corpus construido de 96 pares y ejecuciones de 42. Resultados principales: Llama V3 y Mistral V3 (registro). Qwen histórico se identifica por separado.

Abrir `main.tex` como documento principal en Overleaf. Compilador: XeLaTeX; bibliografía: Biber, estilo APA. Conservar la estructura de carpetas. Compilar hasta resolver las referencias. El PDF compilado se entrega también junto al ZIP.

`figures/avance_*.pdf` y `figures/avance_*.png` son las figuras actualizadas. Las figuras antiguas extraídas del ZIP original no se utilizan en el documento revisado.

`evidencia/metricas_avance.json` registra las cifras y su procedencia; `manifiesto_fuentes.json` identifica con SHA-256 los archivos originales. No incluye activaciones completas. La reproducción de métricas requiere el repositorio y sus resultados, no solo este ZIP.

Para regenerar desde el repositorio: `python scripts/actualizar_tesis_avances.py`. El script sobrescribe las fuentes de esta versión a partir del ZIP original y los resultados locales. El archivo `regenerar_avances.py` es una copia de consulta; debe ubicarse en `scripts/` del repositorio original para ejecutarlo. No ejecuta modelos.

Se incorporaron PCA, inferencia separada de selección de capa, estimadores agénticos diferenciados, discusión, conclusiones del avance y anexos. Permanecen abiertos la réplica de 96 pares, doble anotación, auditoría humana de generaciones y validación externa.
