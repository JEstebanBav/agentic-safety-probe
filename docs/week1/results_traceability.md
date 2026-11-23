# Trazabilidad local de resultados

Se mapearon 16 tablas del articulo a sus fuentes. Se reconstruyo el objeto de evidencia de los cuatro modelos usando model_metrics, sin regenerar figuras ni modificar el manuscrito.

| Modelo | Extraccion coincide con evidencia guardada | Ruta |
|---|---|---|
| llama | True | `results/ultimos_resultados/llama3380B_v4` |
| mistral | True | `results/ultimos_resultados/mistral7b_v4` |
| qwen8b | True | `results/ultimos_resultados/qwen3_8b_v3/v3` |
| qwen32b | True | `results/ultimos_resultados/qwen32B_v3` |

Coinciden 27 de 27 hashes de archivos fuente registrados; 27 estan versionados en el Git local. No se consulto el remoto de GitHub.

La corrida Qwen3-8B reciente, incluidas sus generaciones y resultados causales, ya esta rastreada en el Git local de este directorio. La observacion del mentor sobre GitHub no debe asumirse automaticamente vigente para el remoto actual sin consultarlo.

El mapa detallado esta en table_sources.csv. La tabla de arquitectura requiere comprobar configuraciones de cabezas y FFN por separado. Esta auditoria verifica la extraccion de evidencia; no se declara una nueva comprobacion celda por celda de todas las tablas redondeadas.

Las tablas conductuales 14 y 15 necesitan considerar la correccion del clasificador. Los informes de week1_rescore conservan las diferencias sin reemplazar resultados originales. La coincidencia con la evidencia antigua acredita trazabilidad, no valida por si misma etiquetas, grupos, generalizacion ni causalidad.
