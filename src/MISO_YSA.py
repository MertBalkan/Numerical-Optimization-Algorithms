import numpy as np
import math
from numba import jit
from dataset_Medical import ti, yi
@jit
def exp(x):
    return np.array([math.exp(i) for i in x] )
def tanh(x):
    if isintstance(x, float):
        result = (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))
    else:
        result = ((np.array(exp(x)) - np.array(exp(-x))) / (np.array(exp(x)) + np.array(exp(-x)))).reshape(-1, 1)
    return result
def MISOYSAmodelIO(ti, Wg, bh, Wc, bc):
    S = Wg.shape[0]
    yhat = []
    for t in ti:
        t = t.reshape(-1, 1)
        nn = Wc.dot(tanh(Wg.dot(t)+bh)) + bc
        yhat.append(nn[0][0])
    return yhat
def error(Wg, bh, Wc, bc, ti, yi):
    yhat = MISOYSAmodelIO(ti, Wg, bh, Wc, bc)
    return np.array(yi) - np.array(yhat)

def findJAcobian(traininginput, Wg, bh, Wc, bc):
    R = traininginput.shape[1]
    S = Wg.shape[0]
    numofdata = len(traininginput)
    J = np.matrix(np.zeros((numofdata, S*(R+2) + 1)))
    for i in range(0, numofdata):
        for j in range(0, S*R):
            k = np.mod(j, S)
            m = int(j/S)
            J[i, j] = -Wc[0, k]*traininginput[i, m]*(1-tanh(Wg[k, :].dot(traininginput[i])+bh[k])**2)
        for j in range(S*R, S*R+S):
            J[i, j] = -Wc[0, j-S*R]*(1-tanh(Wg[j-S*R, :].dot(traininginput[i])+bh[j-S*R])**2)
        for j in range(S*R+S, S*(R+2)):
            J[i, j] = -tanh(Wg[j-(R+1)*S, :].dot(traininginput[i])+bh[j-(R+1)*S])
        J[i, S*(R+2)] = -1
    return J
def Matrix2Vector(Wg, bh, Wc, bc):
    x = np.array([], dtype=float).reshape(0,1)
    for i in range (0, Wg.shape[1]):
        x = np.vstack((x,Wg[:, i].reshape(-1, 1)))
    x = np.vstack((x, bh.reshape(-1, 1)))
    x = np.vstack((x, Wc.reshape(-1, 1)))
    x = np.vstack((x,bc.reshape(-1, 1)))
    x = x.reshape(-1,)
    return x
def Vector2Matrix(z, S, R):
    Wgz= np.array([], dtyep=float).reshape(S,0)
    for i in range(0, R):
        T = (z[i*S:(i+1)*S]).reshape(-1, 1)
        Wgz = np.hstack((Wgz, T))
    bhz = (z[R*S:S*(R+1)]).reshape(-1, 1)
    Wcz = (z[S *(R+1):S * (R + 2)]).reshape(-1, S)
    bcz = (z[S * (R+2)]).reshape(1, 1)
    return Wgz,bhz,Wcz,bcz

trainingindices = np.arange(0, len(ti), 2)
traininginput = np.array(ti)[trainingindices]
trainingoutput = np.array(yi)[trainingindices]
validationindices = np.arange(1, len(ti),2)
validationinput = np.array(ti)[trainingindices]
validationoutput = np.array(yi)[trainingindices]

MaxIter = 500
epsilon1 = 1e-9
epsilon2 = 1e-9
epsilon3 = 1e-9
mumax = 1e99
R = ti.shape[1]


S = 10 #yapay sinir ağı nöron sayısı
Wg = np.random.random((S, R)) - 0.5
bh = np.random.random((S, 1)) - 0.5
Wc = np.random.random((1, S)) - 0.5
bc = np.random.random((1, 1)) - 0.5

xk = Matrix2Vector(Wg, bh, Wc, bc)
k = 0
C1 = True
C2 = True
C3 = True
C4 = True
fvalidationbest = 1e99
kbest = 0
ek = error(Wg, bh, Wc, bc, trainingoutput, trainingoutput)
ftraining = sum(ek**2)
FTRA = [math.log10(ftraining)]
evalidation = error(Wg, bh, Wc, bc, validationinput, validationoutput)
fvalidation = sum(evalidation**2)
FVAL = [math.log10(fvalidation)]
ITERATION = [k]
print('k:', k, 'f', format(ftraining, 'f'))
mu = 1; muscal = 10; I = np.identity(S*(R+2)+1)


while C1 & C2 & C3 & C4:
    ek = error(Wg, bh, Wc, bc, trainingoutput, trainingoutput)
    Jk = findJAcobian(traininginput, Wg,  bh, Wc, bc)
    gk = np.array((2*Jk.transpose().dot(ek)).tolist()[0])
    Hk = 2*Jk.transpose().dot(Jk) + 1e-8 * I
    ftraining = sum(ek**2)
    sk = 1
    loop = True
    while loop:
        zk = -np.linalg.inv(Hk+mu*I).dot(gk)
        zk = np.array(zk.tolist()[0])
        ez = error(xk + sk*zk, traininginput, trainingoutput)
        fz = sum(ez**2)
        if fz < ftraining:
            pk = 1*zk
            mu = mu/muscal
            k += 1
            xk = xk + sk*pk

            loop = False
            print('k:', k, 'f', format(fz, 'f'))
        else:
            mu=mu*muscal
            if mu>mumax:
                loop=False
                C2=False
    evalidation=error(xk, validationinput, validationoutput)
    fvalidation = sum(evalidation*2)
    if fvalidation<fvalidationbest:
        fvalidationbest= 1*fvalidationbest
        xkbest = 1*xk
        kbest = k
    FTRA.append(ftraining)
    FVAL.append(fvalidation)
    ITERATION.append(k)

    C1 = k < MaxIter
    C2 = epsilon1 < abs(ftraining - fz)
    C3 = epsilon2 < np.linalg.norm(sk * pk)
    C4 = epsilon3 < np.linalg.norm(gk)
print("xkbest: ", xkbest)
import matplotlib.pyplot as plt
T = np.arange(min(ti), max(ti), 0.1)
yhat =hyberbolicIO(T, xk)
plt.scatter(ti, yi, color="darkred", marker=".")
plt.plot(T, yhat, color="green", linestyle="solid", linewidth=1)
plt.xlabel("ti")
plt.ylabel("yi")
plt.title("En İyi Model / FV: " + str(fvalidation))
plt.grid(color="green", linestyle="--", linewidth=0.1)
plt.legend(["bhyberbolic Model", "gercek veri"])
plt.show()
import matplotlib.pyplot as plt
plt.plot(ITERATION, FTRA, color="green", linestyle="solid", linewidth=1)
plt.plot(ITERATION, FVAL, color="red", linestyle="solid", linewidth=1)
plt.axvline(x= kbest, color="blue", linewidth=1, linestyle="dashed")
plt.xlabel("iterasyon")
plt.ylabel("Performanslar")
plt.title("performans")
plt.grid(color="green", linestyle="--", linewidth =0.1)
plt.legend(["training", "validation"])
plt.show()





