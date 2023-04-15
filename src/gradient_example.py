import numpy as np

def f(x):
    x1, x2, x3 = x
    return (x1 - 2 * x2 + x3 - 1) ** 2 + (-x1 + x2 - x3 + 3) ** 2 + (2 * x1 + x2 - x3) ** 2


x0 = np.array([0, 0, 0])

alpha = 0.000000001
epsilon = 0.00001
max_iter = 10000
iter_count = 0
grad_norm = epsilon + 1

while grad_norm > epsilon and iter_count < max_iter:

    grad_f = np.array([
        2 * (2 * x0[0] + x0[1] - x0[2]) + 2 * (x0[0] - 2 * x0[1] + x0[2] - 1),
        2 * (-x0[0] + 2 * x0[1] - x0[2] + 3) + 2 * (x0[0] + 2 * x0[1] - 2 * x0[2]),
        2 * (x0[2] - x0[1] - x0[0]) - 2 * (x0[0] - x0[1] + x0[2] - 1)
    ])

    x1 = x0 - alpha * grad_f

    grad_norm = np.linalg.norm(grad_f)

    x0 = x1

    iter_count += 1

print("Yerel minimum noktası:", x0)
print("Minimum değeri:", f(x0))