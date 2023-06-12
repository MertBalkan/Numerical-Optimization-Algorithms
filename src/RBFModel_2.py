import numpy as np
import math
from dataset_4 import ti, yi
def gaussianfunction(t, c, s):
    h = math.exp(-(t-c)**2/(s**2))
    return h
def RBFIO(t, x, c, s):
    yhat = []
    for ti in t:
        toplam = 0
        for i in range(0, len(x)):
            toplam += x[i]*gaussianfunction(ti, c[i], s[i])
        yhat.append(toplam)
    return yhat

def findxcs(ti,yi,RBFsayisi):
    lengthofsegment = (max(ti) - min(ti))/RBFsayisi
    s = [lengthofsegment for tmp in range(0, RBFsayisi)]
    c = [min(ti) + lengthofsegment/2 + lengthofsegment*i for i in range(0, RBFsayisi)]
    numofdata = len(ti)
    J = np.zeros((numofdata, RBFsayisi))
    for i in range(0, numofdata):
        for j in range(0, RBFsayisi):
            J[i, j] = -gaussianfunction(ti[i], c[j], s[j])

    A = np.linalg.inv(J.transpose().dot(J))
    B = J.transpose().dot(yi)
    x = -A.dot(B)
    return x, c, s

def plotresult(ti, yi, x, c, s, fvalidation):
    import matplotlib.pyplot as plt
    T = np.arange(min(ti), max(ti), 0.1)
    yhat = RBFIO(T, x, c, s)
    plt.scatter(ti, yi, color="darkred", marker="x")
    plt.plot(T, yhat, color="green", linestyle="solid", linewidth=1)
    plt.xlabel("ti")
    plt.ylabel("yi")
    plt.title(str(len(x)) + "-düğümlü RBF modeli / FV:" + str(fvalidation))
    plt.grid(color="green", linestyle="--", linewidth=0.1)
    plt.legend(["RBF modeli", "gercek veri"])
    plt.show()


trainingindices = np.arange(0, len(ti), 2)
traininginput = np.array(ti)[trainingindices]
trainingoutput = np.array(yi)[trainingindices]
validationindices = np.arange(1, len(ti), 2)
validatioinput = np.array(ti)[validationindices]
validationoutput = np.array(yi)[validationindices]


RBF = [] ; FV = []

for RBFsayisi in range(1, 12):
    x, c, s = findxcs(traininginput, trainingoutput, RBFsayisi)
    yhat = RBFIO(validatioinput, x, c, s)
    e = np.array(validationoutput)-np.array(yhat)
    fvalidation = sum(e**2)
    RBF.append(RBFsayisi)
    FV.append(fvalidation)
    print(RBFsayisi, fvalidation)
    plotresult(ti, yi, x, c, s, fvalidation)


import matplotlib.pyplot as plt
plt.bar(RBF, FV, color="darkred")
plt.xlabel("RBF sayısı")
plt.ylabel("valdation performans")
plt.title("RBF modeli")
plt.grid(color="green", linestyle="--", linewidth=0.1)
plt.show()
