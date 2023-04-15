import numpy as np
import math


def f(xk):
    x1 = xk[0]
    x2 = xk[1]
    x3 = xk[2]
    fk = (x1 - 2 * x2 + x3 - 1) ** 2 + (-x1 + x2 - x3 + 3) ** 2 + (2*x1 + x2 - x3) ** 2
    return fk


def g(xk):
    x1 = xk[0]
    x2 = xk[1]
    g = np.array([2 * (x1 - 1.5 * x2), -3 * (x1 - 1.5 * x2) + (2 * (x2 - 2))])
    return g

def gradient(xk):
    x1 = xk[0]
    x2 = xk[1]
    gk = np.array([2 * (x1 - 1.5 * x2), -3 * (x1 - 1.5 * x2) + 2 * (x2 - 2)])
    return gk


def hessian(xk):
    x1 = xk[0]
    x2 = xk[1]
    Hk = np.matrix([2, -3], [-3, 6.5])
    return Hk


def error(xk):
    x1 = xk[0]
    x2 = xk[1]
    ek = np.array([np.sqrt(3), (x1 - 1.5 * x2), (x2 - 2)])
    return ek


def jacobian(xk):
    x1 = xk[0]
    x2 = xk[1]
    Jk = np.matrix([0, 0], [1, -1.5], [0, 1])
    return Jk
