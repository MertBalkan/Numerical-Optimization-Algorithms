import numpy as np
import math
ti = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
yi = [-1.2557, -0.1074, 0.6040, 0.9605, 1.0730, 1.1060, 1.4090, 1.0223, 0.7559]
import matplotlib.pyplot as plt
plt.plot(ti, yi, 'r.')
plt.xlabel('ti')
plt.ylabel('yi')
plt.title('Dataset 1', fontstyle='italic')
plt.grid(color="black", linewidth=0.1)

plt.show()