# Presentación ligera de la tesis

Segunda versión independiente: **14 diapositivas principales y 2 de apoyo**, con una duración orientativa de **11:50**. Con pausas y explicación de las figuras puede ocupar unos 13–15 minutos. La presentación completa se conserva en [sustentacion](../sustentacion/LEEME.md).

| Archivo | Contenido |
|---|---|
| [Sustentacion_Espitia_Leiva_ligera.pptx](Sustentacion_Espitia_Leiva_ligera.pptx) | PowerPoint editable, con notas para exponer. |
| [Sustentacion_Espitia_Leiva_ligera.pdf](Sustentacion_Espitia_Leiva_ligera.pdf) | Las 16 diapositivas exportadas desde PowerPoint. |
| [Sustentacion_Espitia_Leiva_ligera_guion.pdf](Sustentacion_Espitia_Leiva_ligera_guion.pdf) | Guion con miniaturas, explicación oral, transiciones y tiempos. |
| [guion_ligero.md](guion_ligero.md) | Las mismas notas, en texto editable. |

## Cómo está contada

- **1–5:** pregunta, contexto de los agentes, incidente de Hugging Face y comparación experimental.
- **6–7:** dos ideas matemáticas en palabras —restar promedios y medir alineación— y una explicación sencilla de la AUROC y la evaluación con pares no usados en el ajuste.
- **8–12:** rechazo observado, detección entre formatos, explicitud, monitores e intervención con su costo de sobrerrechazo.
- **13–14:** conclusión, límites y mensaje final.
- **15–16:** fuentes y aclaración sobre autorización y etiquetas, solo para preguntas.

Los textos son grandes y las figuras se pueden editar. Las únicas dos ecuaciones están en una misma diapositiva y son nativas de PowerPoint, con imagen alternativa para otros visores. El guion permite practicar la explicación sin memorizar derivaciones ni listas de métricas.

## Datos y alcance

La versión usa las cifras verificadas de la presentación completa: **Llama V3, 42 pares; Mistral y ambos Qwen V2, 96 por modelo**. Los conteos de monitores se comprueban a partir de sensibilidad y tasa de falsos positivos, y se contrastan con F1. La intervención mostrada es dosis 2 en la capa 56 de Llama: **19/19 dañinas y 40/41 benignas rechazadas**, con **1/19** en el control aleatorio.

En Mistral, el intervalo de diferencia de AUROC incluye cero. Las comparaciones entre modelos y estratos son descriptivas. Las etiquetas conductuales son heurísticas, la autorización real no se verifica y los monitores no tienen validación externa. Se conserva la limitación del par `fraud_01`; el ejemplo didáctico principal usa `fraud_06`. Véase [la nota de revisión](../sustentacion/nota_autorizacion_etiquetas.md).

El extracto reproducible está en [fuente/datos_ligera.json](fuente/datos_ligera.json), con rutas y hashes SHA-256 de las fuentes. No se reejecutaron modelos ni se modificaron etiquetas o métricas para crear esta versión.

## Regenerar

Requiere Windows, PowerPoint, Microsoft Edge, Node.js y Python con `numpy`, `Pillow`, `comtypes` y `pymupdf`. Reutiliza los motores de ecuaciones y exportación de la carpeta de la presentación completa, sin escribir en sus archivos.

```bash
cd docs/sustentacion_ligera/fuente
npm install
bash build.sh
```

`PY` permite seleccionar el intérprete de Python; Node debe estar en `PATH`. En Windows se puede usar Git Bash. La construcción escribe únicamente los archivos de esta versión y su carpeta de trabajo `fuente/build`, excluida de Git.
