import numpy as np
import math
from example_function import f, gradient

# this function is using in golden section method.
def golden_section_f(sk, xk, pk):
    result = f(xk + sk * pk)
    return result


def golden_section(xk, pk):
    salt = 0
    sust = 1
    ds = 0.0001
    alpha = (1 + math.sqrt(5)) / 2
    tau = 1 - 1 / alpha
    epsilon = ds / (sust - salt)
    N = round(-2.078 * math.log(epsilon))

    k = 0
    s1 = salt + tau * (sust - salt)
    f1 = golden_section_f(s1, xk, pk)
    s2 = sust - tau * (sust - salt)
    f2 = golden_section_f(s2, xk, pk)

    while abs(s1 - s2) > ds:
        k = 0
        if f1 > f2:
            salt = 1 * s1
            s1 = 1 * s2
            f1 = 1 * f2
            s2 = sust - tau * (sust - salt)
            f2 = golden_section_f(s2, xk, pk)
        else:
            sust = 1 * s2
            s2 = 1 * s1
            f2 = 1 * f1
            s1 = salt + tau * (sust - salt)
            f1 = golden_section_f(s1, xk, pk)
        print(k + 1, s1, s2, f1, f2)
    s = np.mean([s1, s2])
    return s


# sonlandırma kriterleri-------
MaxIter = 500
epsilon1 = 1e-9
epsilon2 = 1e-9
epsilon3 = 1e-9

x1 = [0]
x2 = [0]
xk = np.array([x1[0], x2[0]])
k = 0
C1 = True
C2 = True
C3 = True
C4 = True

while C1 & C2 & C3 & C4:
    k += 1
    pk = -gradient(xk)
    sk = golden_section(xk, pk)
    xk = xk + sk * pk
    x1.append(xk[0])
    x2.append(xk[1])
    print('k:', k, 'x1:', format(xk[0], 'f'), 'x2:', format(xk[1], 'f'), 'f', format(f(xk), 'f'))

    C1 = k < MaxIter
    C2 = epsilon1 < abs(f(xk) - f(xk + sk * pk))
    C3 = epsilon2 < np.linalg.norm(sk * pk)
    C4 = epsilon3 < np.linalg.norm(gradient(xk))

if not C1:
    print('max iterasyon asildi')

if not C2:
    print('foksiyonun degeri degismiyor')
if not C3:
    print('ilerleme yonu bulunamiyor')
if not C4:
    print('gradyant sifira çok yakin')
