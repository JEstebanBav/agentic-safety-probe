# Resultados

Cada versión de cada modelo vive en su propia carpeta y lleva una
**`FICHA.md`**: qué modelo era, cómo se cargó, con qué corpus y parámetros,
qué etapas corrieron y qué salió. Empieza por la ficha, no por los JSON.

```
results/
├── corridas/<modelo>/<versión>/   resultados reales
├── pruebas/<modelo>/<versión>/    corridas de prueba (4 pares): solo
│                                  comprueban que el pipeline corre; sus
│                                  estadísticas no significan nada
├── comparativa/                   figuras y tablas entre modelos
└── registro.json                  lo que ningún archivo registra
```

**Versiones.** `v1`, `v2`, `v3`… son las versiones que cita la tesis. Una
carpeta `vN_<variante>` es una corrida complementaria hecha después de la
versión `vN` (otra capa causal, una verificación, el token inicial antiguo);
su ficha dice qué complementa. El número de pares no va en el nombre: está
en la ficha y en esta tabla, porque una misma versión no siempre usó el mismo
corpus en todos los modelos.

**Una corrida nueva** va en la siguiente versión libre del modelo
(`results/corridas/<modelo>/vN/`). `run_experiment.py` escribe su ficha al
terminar; después añade su entrada en `registro.json` (etiqueta, origen,
relación, uso) y corre `python scripts/generar_fichas.py`.

`activations.npz` no se sube a git (pesa cientos de MB): existe solo en la
máquina donde se descargó. La ficha marca si está presente.

## Corridas

| Versión | Modelo | Precisión | Pares | Capa ref. | Etapas | Uso |
|---|---|---|---|---|---|---|
| [`llama31_70b/v0_hackathon`](corridas/llama31_70b/v0_hackathon/FICHA.md) | Llama-3.1-70B-Instruct | FP16 | 511 | 20 | - | Demo (app.py) |
| [`llama33_70b/v1`](corridas/llama33_70b/v1/FICHA.md) | Llama-3.3-70B-Instruct | FP16 | 42 | 56 | - | Cifras de la tesis original; sustituida por V3 |
| [`llama33_70b/v2`](corridas/llama33_70b/v2/FICHA.md) | Llama-3.3-70B-Instruct | FP16 | 42 | 56 | PCA, held-out, factorial, conducta, causal | Sustituida por V3 |
| [`llama33_70b/v2_bos_antiguo`](corridas/llama33_70b/v2_bos_antiguo/FICHA.md) | Llama-3.3-70B-Instruct | FP16 | 42 | no registrado | - | Solo log |
| [`llama33_70b/v2_causal_capa56`](corridas/llama33_70b/v2_causal_capa56/FICHA.md) | Llama-3.3-70B-Instruct | FP16 | 42 | 56 | PCA, conducta, causal | Complementaria |
| [`llama33_70b/v3`](corridas/llama33_70b/v3/FICHA.md) | Llama-3.3-70B-Instruct | FP16 | 42 | 76 | PCA, held-out, factorial, conducta, causal | **Referencia de Llama en la tesis y la sustentación** |
| [`llama33_70b/v3_bos_antiguo`](corridas/llama33_70b/v3_bos_antiguo/FICHA.md) | Llama-3.3-70B-Instruct | FP16 | 42 | 76 | PCA, held-out | Complementaria |
| [`llama33_70b/v3_causal_capa76`](corridas/llama33_70b/v3_causal_capa76/FICHA.md) | Llama-3.3-70B-Instruct | FP16 | 42 | 76 | PCA, conducta, causal | Complementaria |
| [`llama33_70b/v3_verificacion`](corridas/llama33_70b/v3_verificacion/FICHA.md) | Llama-3.3-70B-Instruct | FP16 | 42 | 56 | PCA | Complementaria |
| [`mistral7b/v1_4bit`](corridas/mistral7b/v1_4bit/FICHA.md) | Mistral-7B-Instruct-v0.3 | 4-bit | 42 | 16 | held-out, interv. antigua | Control de precisión |
| [`mistral7b/v1_estadistica`](corridas/mistral7b/v1_estadistica/FICHA.md) | Mistral-7B-Instruct-v0.3 | FP16 | 42 | 18 | held-out, interv. antigua | Histórica |
| [`mistral7b/v1_fp16`](corridas/mistral7b/v1_fp16/FICHA.md) | Mistral-7B-Instruct-v0.3 | FP16 | 42 | 16 | held-out, interv. antigua | Histórica |
| [`mistral7b/v1_intervencion`](corridas/mistral7b/v1_intervencion/FICHA.md) | Mistral-7B-Instruct-v0.3 | FP16 | 42 | 18 | held-out, interv. antigua | Histórica |
| [`mistral7b/v2`](corridas/mistral7b/v2/FICHA.md) | Mistral-7B-Instruct-v0.3 | FP16 | 96 | 18 | held-out, interv. antigua | **Sustentación, 96 pares** |
| [`mistral7b/v3`](corridas/mistral7b/v3/FICHA.md) | Mistral-7B-Instruct-v0.3 | FP16 | 42 | no registrado | - | Tesis, desde el log |
| [`qwen3_32b/v1`](corridas/qwen3_32b/v1/FICHA.md) | Qwen3-32B | 4-bit | 42 | 46 | held-out | Tesis (avances) |
| [`qwen3_32b/v2`](corridas/qwen3_32b/v2/FICHA.md) | Qwen3-32B | 4-bit | 96 | 46 | held-out, interv. antigua | **Sustentación, 96 pares** |
| [`qwen3_8b/v1`](corridas/qwen3_8b/v1/FICHA.md) | Qwen3-8B | FP16 | 42 | 32 | held-out, interv. antigua | Tesis (avances) |
| [`qwen3_8b/v2`](corridas/qwen3_8b/v2/FICHA.md) | Qwen3-8B | FP16 | 96 | 22 | held-out, interv. antigua | **Sustentación, 96 pares** |

## Corridas de prueba

| Versión | Modelo | Precisión | Pares | Capa ref. | Etapas | Uso |
|---|---|---|---|---|---|---|
| [`llama33_70b/v2`](pruebas/llama33_70b/v2/FICHA.md) | Llama-3.3-70B-Instruct | FP16 | 4 | 56 | PCA, held-out, factorial, conducta, causal | Prueba |
| [`mistral7b/v1`](pruebas/mistral7b/v1/FICHA.md) | Mistral-7B-Instruct-v0.3 | FP16 | 4 | 18 | held-out | Prueba |
| [`qwen3_32b/v1`](pruebas/qwen3_32b/v1/FICHA.md) | Qwen3-32B | 4-bit | 4 | 46 | held-out | Prueba |
