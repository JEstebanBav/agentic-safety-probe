# Ficha · Mistral-7B · V2 (96 pares)

> Generada por `scripts/generar_fichas.py` a partir de los archivos de esta carpeta y de `results/registro.json`. No la edites a mano: corrige el registro y regenera.

## Identificación

| | |
|---|---|
| Carpeta | `results/corridas/mistral7b/v2/` |
| Tipo | corrida principal |
| Fecha | no registrada (carpeta no estaba en git) |
| Carpeta anterior | `results/mistral7b_v2` |
| Relación con otras versiones | Primera corrida de Mistral con el corpus ampliado. |
| Citada en | Sustentación (comparativa de 96 pares, capa 18) |

## El modelo en esta corrida

| | |
|---|---|
| Modelo | `mistralai/Mistral-7B-Instruct-v0.3` |
| Parámetros | 7.25 mil millones |
| Arquitectura | mistral, 32 capas, dimensión 4096 |
| Precisión de carga | FP16 — deducida: ocupaba 14.5 GB para 7.25 mil millones de parámetros |
| Memoria ocupada | 14.5 GB |
| Posición de lectura | último token del prompt (cabecera del asistente), antes de generar |
| Capas analizadas | 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30 |
| Capa de referencia (máximo ΔP) | 18 |
| Capa por selección en muestra dividida | 18 |

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
| ΔP dañinas (capa 18) | 2.327 [IC 95 % 1.937, 2.706] |
| Prueba t pareada | p = 1.53e-20, d_z = 1.20 |
| Control benigno | ΔP benignas -1.009; interacción 3.336 (p = 0.0000) |
| AUROC fuera de muestra | chat 0.924, agente 0.878 |

## Notas

- Versión anterior del pipeline: sin línea base de comportamiento, test causal calibrado, factorial ni PCA, y sin activations.npz. Para tener esas etapas hay que repetirla (V4).
- Sin log: la precisión se deduce de la memoria ocupada.

## Archivos

- `agent_direction_results.json`
- `agent_intervention_results.json`
- `directions.npz`
- `format_effect_results.json`
- `results.json`
- `robust_results.json`
- `stratified_results.json`
