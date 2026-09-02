import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("C:\\Users\\deves\\Conjugate-Gradient\\Conjugate-Gradient\\insurance.csv")
x, y = df["age"].values, df["charges"].values
A = np.column_stack([np.ones_like(x), x])
M, v = A.T @ A, A.T @ y
w = np.zeros(2)
r = d = v - M @ w

for _ in range(2):
  Md = M @ d
  alpha = (r @ r) / (d @ Md)
  w += alpha * d
  r_next = r - alpha * Md
  if np.linalg.norm(r_next) < 1e-6:
    break
  d = r_next + ((r_next @ r_next) / (r @ r)) * d
  r = r_next

b0, b1 = w

# 3. Plot fit
plt.figure(figsize=(6, 4))
plt.scatter(x, y, s=10, alpha=0.25, color="gray", label="Kaggle data")
plt.plot(
    [x.min(), x.max()],
    [b0 + b1 * x.min(), b0 + b1 * x.max()],
    color="red",
    lw=2,
    label=f"CG Fit: y = {b0:.0f} + {b1:.1f}x",
)
plt.xlabel("Age")
plt.ylabel("Charges ($)")
plt.legend()
plt.tight_layout()
plt.show()