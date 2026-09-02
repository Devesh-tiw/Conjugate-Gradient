import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

FEATURE = "age"
df = pd.read_csv("insurance.csv")
x = df[[FEATURE]].values.astype(float)
y = df["charges"].values.astype(float)
m = len(y)

sc_x, sc_y = StandardScaler(), StandardScaler()
xs = sc_x.fit_transform(x)
ys = sc_y.fit_transform(y.reshape(-1, 1)).ravel()

A = np.hstack([np.ones((m, 1)), xs])
b = ys
M = A.T @ A
v = A.T @ b

w = np.zeros(2)
r = d = v - M @ w

for k in range(len(w)):
    Md = M @ d
    alpha = (r @ r) / (d @ Md)
    w = w + alpha * d
    r_new = r - alpha * Md
    if np.linalg.norm(r_new) < 1e-12:
        break
    beta = (r_new @ r_new) / (r @ r)
    d = r_new + beta * d
    r = r_new

b1 = sc_y.scale_[0] * w[1] / sc_x.scale_[0]
b0 = sc_y.mean_[0] + sc_y.scale_[0]*w[0] - b1 * sc_x.mean_[0]

plt.figure(figsize=(5, 4)) 
plt.scatter(x, y, s=8, alpha=0.3, color="gray", label="Kaggle data")

x_line = np.linspace(x.min(), x.max(), 100)
plt.plot(x_line, b0 + b1 * x_line, lw=2, color="red", label="CG fit")

plt.xlabel(FEATURE, fontsize=9)
plt.ylabel("charges", fontsize=9)
plt.title("Linear Regression Fit", fontsize=10, fontweight="bold")
plt.legend(fontsize=8, loc="upper left")
plt.grid(alpha=0.2)
plt.tick_params(labelsize=8)
plt.tight_layout()
plt.show()