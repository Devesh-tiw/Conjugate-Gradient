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
r = v - M @ w   
d = r.copy()    

path = [w.copy()] 

for k in range(len(w)): 
    Md = M @ d
    alpha = (r @ r) / (d @ Md)     
    w = w + alpha * d              
    r_new = r - alpha * Md         
    
    path.append(w.copy())
    
    if np.linalg.norm(r_new) < 1e-12:
        break
        
    beta = (r_new @ r_new) / (r @ r) 
    d = r_new + beta * d             
    r = r_new                        

path = np.array(path)

b1 = sc_y.scale_[0] * w[1] / sc_x.scale_[0] 
b0 = sc_y.mean_[0] + sc_y.scale_[0]*w[0] - b1 * sc_x.mean_[0]
y_hat = b0 + b1 * x.ravel()

print(f"Y = {b0:,.2f} + {b1:,.3f}*{FEATURE}")
print(f"CG iterations : {k+1} (n = 2 -> theory: <= n)")
print(f"R^2 = {r2_score(y, y_hat):.4f}")
print(f"RMSE = {np.sqrt(mean_squared_error(y, y_hat)):,.2f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

ax1.scatter(x, y, s=12, alpha=0.35, label="Kaggle data")
x_line = np.linspace(x.min(), x.max(), 100)
ax1.plot(x_line, b0 + b1*x_line, lw=2.5, color="red", label=f"CG fit: Y = {b0:,.0f} + {b1:,.2f}*{FEATURE}")
ax1.set_xlabel(FEATURE)
ax1.set_ylabel("charges")
ax1.set_title("Linear Regression Fit")
ax1.legend()
ax1.grid(alpha=0.3)

margin = 0.5
w0_min, w0_max = path[:, 0].min() - margin, path[:, 0].max() + margin
w1_min, w1_max = path[:, 1].min() - margin, path[:, 1].max() + margin
W0, W1 = np.meshgrid(np.linspace(w0_min, w0_max, 100), np.linspace(w1_min, w1_max, 100))

Z = 0.5 * (M[0, 0] * W0**2 + (M[0, 1] + M[1, 0]) * W0 * W1 + M[1, 1] * W1**2) - (v[0] * W0 + v[1] * W1)

ax2.contour(W0, W1, Z, levels=30, cmap='viridis', alpha=0.8)
ax2.plot(path[:, 0], path[:, 1], marker='o', color='red', label='CG Path', linewidth=2)
ax2.plot(path[0, 0], path[0, 1], 'bo', label='Start $(w_0, w_1)=(0,0)$')
ax2.plot(path[-1, 0], path[-1, 1], 'k*', markersize=12, label='Minimum (Optimal Weights)')
ax2.set_xlabel("Scaled Intercept Weight ($w_0$)")
ax2.set_ylabel("Scaled Slope Weight ($w_1$)")
ax2.set_title("Conjugate Gradient Cost Contour")
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()