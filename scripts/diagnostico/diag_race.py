import numpy as np, sys
A = np.load("/workspace/results_llama33_v2/activations.npz")
B = np.load("/workspace/results_llama33_causal56/activations.npz")
common = sorted(k for k in A.files if k in B.files and k != "__meta__")
print("clave                         filas  distintas  max|dif|   repetidas(A) repetidas(B)")
for k in common:
    a, b = A[k], B[k]
    diff = np.abs(a - b).max(axis=1)
    n_diff = int((diff > 1e-3).sum())
    rep = lambda x: int(sum(np.allclose(x[i], x[i-1], atol=1e-4) for i in range(1, len(x))))
    print(f"{k:28s}  {len(a):5d}  {n_diff:9d}  {diff.max():9.4f}   {rep(a):11d}  {rep(b):11d}")
