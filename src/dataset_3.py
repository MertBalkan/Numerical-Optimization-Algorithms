import numpy as np
import math
ti = np.arange(-10, 10, 20/14)
yi = [0.17+math.sin(t)/(t) for t in ti]

import matplotlib.pyplot as plt
plt.scatter(ti, yi, color="darkred")
plt.xlabel("ti")
plt.ylabel("yi")
plt.title("Data set 3")
plt.grid(color="green", linestyle="--", linewidth=0.1)
plt.show()