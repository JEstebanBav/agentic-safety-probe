# Archivo externo de tesis

El 28 de octubre de 2026 se trasladaron **966 archivos** de tesis y sustentacion a la carpeta **Tesis_AgentHarm_2026-10-28** del escritorio de Windows (incluido el escritorio redirigido a OneDrive, si corresponde).

Se conserva la estructura anterior dentro de esa carpeta: agentic-safety-probe/docs/tesis, docs/sustentacion, docs/sustentacion_ligera, los PDF y HTML de presentaciones, y los dos scripts exclusivos de generacion de tesis. La copia adicional workspace/docs/tesis tambien se traslado, conservando su ruta relativa.

El [manifiesto](thesis_archive_manifest.json) registra las rutas de origen y destino, tamanos y SHA-256. Cada archivo movido se verifico contra su huella anterior. Se conservaron los cambios locales que ya existian. Los archivos de src/ y data/ conservan sus hashes; el inventario de resultados experimentales conserva rutas, tamanos y fechas.

En el repositorio permanecen el [articulo cientifico en ingles](paper/README.md), el [top 10 de investigaciones](paper/top_10_related_research.md), el codigo, los datos y los resultados experimentales. Los documentos tecnicos de diseno, metodologia y analisis del proyecto tambien se conservan.

El retiro de los documentos del arbol versionado corresponde a este traslado al archivo externo. Las versiones antiguas siguen existiendo en el historial de Git; no se reescribe ese historial. El manifiesto describe la operacion local original del 28 de octubre de 2026, anterior al commit que publica la reorganizacion.
