import numpy as np
import math

def f(x):
    f = 3 + (x[0] - 1.5 * x[1]) ** 2 + (x[1] - 2) ** 2
    return f
def g(x):
    g = np.array([2 * (x[0] - 1.5 * x[1]), -3 * (x[0] - 1.5 * x[1]) + (2 * (x[1] - 2))])
    return g

# this function is using in golden section method.
def golden_section_f(s, x, p):
    result = f(x + s * p)
    return result


def golden_section(x, p, salt, sust, ds):
    alpha = (1 + math.sqrt(5)) / 2
    tau = 1 - 1 / alpha
    epsilon = ds / (sust - salt)
    N = round(-2.078 * math.log(epsilon))

    k = 0
    s1 = salt + tau * (sust - salt)
    f1 = golden_section_f(s1, x, p)
    s2 = sust - tau * (sust - salt)
    f2 = golden_section_f(s2, x, p)

    while abs(s1 - s2) > ds:
        k += 1
        if f1 > f2:
            salt = 1 * s1
            s1 = 1 * s2
            f1 = 1 * f2
            s2 = sust - tau * (sust - salt)
            f2 = golden_section_f(s2, x, p)
        else:
            sust = 1 * s2
            s2 = 1 * s1
            f2 = 1 * f1
            s1 = salt + tau * (sust - salt)
            f1 = golden_section_f(s1, x, p)
        print(k + 1, s1, s2, f1, f2)
    s = np.mean([s1, s2])
    return x


x1 = -4.5
x2 = -3.5

print(f([x1, x2]))
print(g([x1, x2]))

x = np.array([x1, x2])

# executes steepest descent algorithm.
# using golden section method for finding s value.
for i in range(0, 100):
    p = -g(x)
    s = golden_section(x, p, 0, 1, 0.0001)
    s = 0.2
    x = x + s * p
    print(i, x[0], x[1], f(x))