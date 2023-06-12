import numpy as np
import math
ti = np.arange(-10, 10, 1.05)
yi = [-0.0116, .2860, 0.3363, 0.0980, -0.0604, -0.1570, -0.1363, 0.3304, 0.9160, 1.1664, 1.1409, 0.9474, 0.2325, 0.1845, -0.1749, 0.0926, 0.1228, 0.3232, 0.0579, 0.2179]

import matplotlib.pyplot as plt
plt.scatter(ti, yi, color="darkred")
plt.xlabel("ti")
plt.ylabel("yi")
plt.title("Data set 4")
plt.grid(color="green", linestyle="--", linewidth=0.1)
plt.show()