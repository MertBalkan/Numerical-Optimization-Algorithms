import numpy as np
import math
from example_function import f,error,jacobian

# sonlandırma kriterleri-------
MaxIter = 500
epsilon1 = 1e-9
epsilon2 = 1e-9
epsilon3 = 1e-9
mumax = 1e99

x1 = [np.random.random()]
x2 = [np.random.random()]
xk = np.array([x1[0], x2[0]])
k = 0
C1 = True
C2 = True
C3 = True
C4 = True
ek=error(xk)
fk=sum(ek**2) #fk elerin kareleri olarak yzıldığından

print('k:', k, 'x1:', format(xk[0], 'f'), 'x2:', format(xk[1], 'f'), 'f', format(f(xk), 'f'))
mu=1
muscal = 10
I=np.identity(2)

while C1 & C2 & C3 & C4:
    ek = error(xk)
    Jk = jacobian(xk)
    gk = np.array((2*Jk.transpose().dot(ek)).tolist()[0])
    Hk = 2*Jk.transpose().dot(Jk)
    sk = 1
    loop = True
    while loop:
        zk = -np.linalg.inv(Hk+mu*I).dot(gk)
        zk = np.array(zk.tolist()[0])
        ez = error(xk+sk*zk)
        fz = sum(ez**2)
        if fz < fk:
            pk = 1*zk
            mu = mu/muscal
            k += 1
            xk = xk + sk*pk
            x1.append(xk[0])
            x2.append(xk[1])
            loop = False
            print('k:', k, 'x1:', format(xk[0], 'f'), 'x2:', format(xk[1], 'f'), 'f', format(f(xk), 'f'))
        else:
            mu=mu*muscal
            if mu>mumax:
                loop=False
                C2=False
    C1 = k < MaxIter
    C2 = epsilon1 < abs(f(xk) - f(xk + sk * pk))
    C3 = epsilon2 < np.linalg.norm(sk * pk)
    C4 = epsilon3 < np.linalg.norm(gk)

if not C1:
    print('max iterasyon aşıldı')

if not C2:
    print('foksiyonun değeri değişmiyor')
if not C3:
    print('ilerleme yönü bulunamıyor')
if not C4:
    print('gradyant sıfıra çok yakın')


