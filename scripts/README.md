# Scripts

Todos se ejecutan desde la raíz del repositorio, por ejemplo `python scripts/analisis/ver_pares.py`.

| Carpeta | Contenido |
|---|---|
| `runpod/` | Corridas en GPU: `cola_v03.sh` (las 10 corridas del corpus v0.3, por máquina), `run_modelo_v03.sh` (una corrida v0.3), `run_modelo.sh` (base), `cola.sh` (varias corridas seguidas), `resume_pod.sh` y `runpod_setup.sh` (entorno del pod), `estado.sh` (estado del pod). |
| `analisis/` | Sin GPU: `check_token_budget.py` (antes de alquilar), `validate_corrections.py` (verifica los estimadores), `generar_fichas.py` (fichas e índice de resultados), `comparar.py`, `bootstrap_delta_p.py`, `reanalisis_observaciones.py`, `tabla_cap4.py`, `ver_pares.py`. |
| `paper/` | Construcción y validación del artículo y la presentación. |
| `week1/` | Auditoría, acuerdo entre anotadores, reclasificación y traducción de la semana 1. |
| `diagnostico/` | Diagnósticos puntuales de corridas anteriores. |
