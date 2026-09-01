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
y_hat = b0 + b1 * x.ravel()

print(f"Y = {b0:,.2f} + {b1:,.3f}*{FEATURE}")
print(f"CG iterations : {k+1} (n = 2 -> theory: <= n)")
print(f"R^2 = {r2_score(y, y_hat):.4f}")
print(f"RMSE = {np.sqrt(mean_squared_error(y, y_hat)):,.2f}")

plt.figure(figsize=(9, 5.5))
plt.scatter(x, y, s=12, alpha=0.35, label="Kaggle data")
x_line = np.linspace(x.min(), x.max(), 100)
plt.plot(x_line, b0 + b1*x_line, lw=2.5, color="red", label=f"CG fit: Y = {b0:,.0f} + {b1:,.2f}*{FEATURE}")
plt.xlabel(FEATURE)
plt.ylabel("charges")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("regression_line.png", dpi=140)
plt.show()
