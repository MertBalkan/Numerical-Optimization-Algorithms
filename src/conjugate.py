import numpy as np
import math
from example_function import f, gradient


def f(x):  # vektör olarak girdiğimizi düşündük
    f = 3 + (x[0] - 1.5 * x[1]) ** 2 + (x[1] - 2) ** 2
    return f


def g(x):
    g = np.array([2 * (x[0] - 1.5 * x[1]), -3 * (x[0] - 1.5 * x[1]) + 2 * (
                x[1] - 2)])  # gradyant vektörleri yani önce x1 göre türev sonra x2ye göre türev
    return g


def GSf(sk, xk, pk):
    sonuc = f(xk + sk * pk)
    return sonuc


def GS(xk, pk, k=None):  # golden section
    salt = 0;
    sust = 1
    ds = 0.0001
    alpha = (1 + math.sqrt(5)) / 2;
    tau = 1 - 1 / alpha;
    epsilon = ds / (sust - salt);
    N = (round(-2.078 * math.log(epsilon)));
    s1 = salt + tau * (sust - salt);
    f1 = GSf(s1, xk, pk);  # orijinal değil de Golden Section için kullanmak için Ayrı bir GS fonk ekledik
    s2 = sust - tau * (sust - salt);
    f2 = GSf(s2, xk, pk);

    while abs(s1 - s2) > ds:
        k += 1
        if f1 > f2:
            salt = 1 * s1;
            s1 = 1 * s2;
            f1 = 1 * f2;
            s2 = sust - tau * (sust - salt);
            f2 = GSf(s2, xk, pk);
        else:
            sust = 1 * s2;
            s2 = 1 * s1;
            f2 = 1 * f1;
            s1 = salt + tau * (sust - salt);
            f1 = GSf(s1), xk, pk;

    s = np.mean([s1, s2])
    return s



# sonlandırma kriterleri-------
MaxIter = 500
epsilon1 = 1e-9
epsilon2 = 1e-9
epsilon3 = 1e-9

x1 = [0];
x2 = [0];
xk = np.array([x1[0], x2[0]])
k = 0;
C1 = True;
C2 = True;
C3 = True;
C4 = True;
prevp = -gradient(xk)
prevg = gradient(xk)
print('k:', k, 'x1:', format(xk[0], 'f'), 'x2:', format(xk[1], 'f'), 'f', format(f(xk), 'f'))

while C1 & C2 & C3 & C4:
    if k == 0:
        pk = -gradient(xk)
    else:
        beta = np.dot(gradient(xk), gradient(xk)) / np.dot(prevg, prevg)
        pk = -gradient(xk) + beta * prevp
        prevp = 1 * pk
        prevg = 1 * gradient(xk)
    k += 1
    sk = GS(xk, pk, 0)
    xk = xk + sk * pk
    x1.append(xk[0])
    x2.append(xk[1])
    print('k:', k, 'x1:', format(xk[0], 'f'), 'x2:', format(xk[1], 'f'), 'f', format(f(xk), 'f'))

    C1 = k < MaxIter
    C2 = epsilon1 < abs(f(xk) - f(xk + sk * pk))
    C3 = epsilon2 < np.linalg.norm(sk * pk)
    C4 = epsilon3 < np.linalg.norm(gradient(xk))

if not C1:
    print('max iterasyon aşıldı')

if not C2:
    print('foksiyonun değeri değişmiyor')
if not C3:
    print('ilerleme yönü bulunamıyor')
if not C4:
    print('gradyant sıfıra çok yakın')



