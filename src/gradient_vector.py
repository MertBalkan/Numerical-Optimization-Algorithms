import numpy as np

def f(x1, x2):
    f = x1 ** 2 - x2 ** 2 + x1 * x2
    return f


def g(x1, x2):
    g = np.array([2 * x1 + x2, -2 * x2 + x1])
    return g


x1 = 3
x2 = 2
print(f(x1, x2))
print(g(x1, x2))
x = np.array([x1, x2])
s = 0.001
for i in range(0, 10):
    p = -g(x[0], x[1])
    x = x + s * p
    print(i, x[0], x[1], f(x[0], x[1]))
