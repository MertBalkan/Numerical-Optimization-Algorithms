x = 2
f = (x-1) ** 2 * (x-2) * (x-3)
f1 = 2 * (x-1) * (x-2) * (x-3) + (x-1) ** 2 * (2 * x-5)
f2 = 2 ** (x-2) * (x-3) + 4 * (x-1) * (2 * x-5) + 2 * (x-1) ** 2
dx = -f1 / f2

iteration = 0
print (iteration, 'x : ', x, ' f:', f, ' f1:', f1, ' f2:', f2, ' dx:', dx)

while abs(f1) > 1e-10:
    iteration += 1
    x = x + dx
    f = (x - 1) ** 2 * (x - 2) * (x - 3)
    f1 = 2 * (x - 1) * (x - 2) * (x - 3) + (x - 1) ** 2 * (2 * x - 5)
    f2 = 2 ** (x - 2) * (x - 3) + 4 * (x - 1) * (2 * x - 5) + 2 * (x - 1) ** 2
    dx = -f1 / f2
    print (iteration, 'x : ', x, ' f:', f, ' f1:', f1, ' f2:', f2, ' dx:', dx)

