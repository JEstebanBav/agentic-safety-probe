# Ficha · Qwen3-8B · V2 (96 pares)

> Generada por `scripts/generar_fichas.py` a partir de los archivos de esta carpeta y de `results/registro.json`. No la edites a mano: corrige el registro y regenera.

## Identificación

| | |
|---|---|
| Carpeta | `results/corridas/qwen3_8b/v2/` |
| Tipo | corrida principal |
| Fecha | no registrada (carpeta no estaba en git) |
| Carpeta anterior | `results/qwen8b_v2` |
| Relación con otras versiones | Primera corrida de Qwen3-8B con el corpus ampliado. |
| Citada en | Sustentación (comparativa de 96 pares, capa 22) |

## El modelo en esta corrida

| | |
|---|---|
| Modelo | `Qwen/Qwen3-8B` |
| Parámetros | 8.20 mil millones |
| Arquitectura | qwen3, 36 capas, dimensión 4096 |
| Precisión de carga | FP16 — deducida: ocupaba 16.4 GB para 8.2 mil millones de parámetros |
| Memoria ocupada | 16.4 GB |
| Posición de lectura | último token del prompt (cabecera del asistente), antes de generar |
| Capas analizadas | 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34 |
| Capa de referencia (máximo ΔP) | 22 |
| Capa por selección en muestra dividida | 22 |

## Corpus

| | |
|---|---|
| Archivo | data/dataset_v2.jsonl — deducido: 96 pares, y la sustentación la identifica como V2 de 96 |
| Pares | 96 |

## Parámetros de la corrida

Esta corrida es de una versión del pipeline que no guardaba su configuración en `results.json`.

## Etapas ejecutadas

- [ ] Activaciones guardadas (activations.npz, solo local)
- [x] Dirección de rechazo y ΔP por capa
- [ ] Validación PCA del contraste
- [x] Evaluación fuera de muestra (held-out)
- [x] Sondas lineales con permutación
- [x] Sondas MLP
- [x] Análisis robusto y selección de capa
- [ ] Factorial sistema × herramientas
- [ ] Línea base de comportamiento
- [ ] Test causal calibrado (d_chat vs. aleatoria)
- [x] Intervención antigua con α arbitrarios (no calibrada)

## Resultados clave

| Medida | Valor |
|---|---|
| ΔP dañinas (capa 22) | 42.273 [IC 95 % 38.401, 45.846] |
| Prueba t pareada | p = 2.59e-39, d_z = 2.24 |
| Control benigno | ΔP benignas -0.122; interacción 42.394 (p = 0.0000) |
| AUROC fuera de muestra | chat 0.934, agente 0.659 |

## Notas

- Versión anterior del pipeline: sin línea base de comportamiento, test causal calibrado, factorial ni PCA, y sin activations.npz. Para tener esas etapas hay que repetirla (V3).
- Qwen3 razona antes de responder; esta versión del pipeline no lo manejaba. Las activaciones no se ven afectadas.

## Archivos

- `agent_direction_results.json`
- `agent_intervention_results.json`
- `directions.npz`
- `format_effect_results.json`
- `results.json`
- `robust_results.json`
- `stratified_results.json`
