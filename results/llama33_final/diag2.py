import numpy as np, sys
A = np.load(f"/workspace/{sys.argv[1]}/activations.npz")
B = np.load(f"/workspace/{sys.argv[2]}/activations.npz")
print(f"A={sys.argv[1]}  B={sys.argv[2]}")
common = sorted((k for k in A.files if k in B.files and k != "__meta__"),
                key=lambda k: (int(k.split("__L")[1]), k))
print("clave                         filas  distintas  max|dif|")
for k in common:
    d = np.abs(A[k] - B[k]).max(axis=1)
    print(f"{k:28s}  {len(d):5d}  {int((d > 1e-3).sum()):9d}  {d.max():9.4f}")
