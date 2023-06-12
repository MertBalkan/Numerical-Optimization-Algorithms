import numpy as np
import math
from dataset_1 import ti, yi
def polinomIO(t, x):
    yhat = [x[0] + x[1]*ti + x[2]*ti**2 for ti in t ]
    return yhat
numofdata = len(ti)
J = -np.ones((numofdata, 1)) #birinci sutun -1lerden oluşuyo
J = np.hstack((J, -np.ones((numofdata, 1))*np.array(ti).reshape(numofdata, 1))) #-ti lerden oluşan
J = np.hstack((J, -np.ones((numofdata, 1))*np.array(ti).reshape(numofdata, 1)**2)) #-ti^2 lerden oluşur
A = np.linalg.inv(J.transpose().dot(J)) #J^t nin tersini hallettik
B = J.transpose().dot(yi) #J^ty
x = -A.dot(B)
T = np.arange(-3, 3, 0.1) #-3 ile 3 arasında 0.1 aralıklarla
yhat = polinomIO(T, x)
print("J: ", J)
print("x: ", x)
print("T: ", T)


import matplotlib.pyplot as plt
plt.scatter(ti, yi, color="darkred", marker="x")
plt.plot(T, yhat, color="green", linestyle="solid", linewidth=1)
plt.xlabel("ti")
plt.ylabel("yi")
plt.title("Polinom Modeli")
plt.grid(color="green", linestyle="--", linewidth=0.1)
plt.legend(["polinom model", "gercek veri"])
plt.show()